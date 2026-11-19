"""Session-owned aggregator. Only closed, settled rounds enter the disk queue."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
import queue
import threading
import time
from uuid import uuid4

from .audio import AudioRecorder
from .policy import SimpleRoundBoundaryPolicy

_log = logging.getLogger(__name__)


@dataclass
class Round:
    id: str
    seq: int
    started: float
    utterances: dict = field(default_factory=dict)
    generations: dict = field(default_factory=dict)
    group: str | None = None
    question_revision: int = 0
    ended: float | None = None
    closed_mono: float | None = None
    version: int = 0
    audio_version: int = 0
    saved_version: int = -1
    saved_audio_version: int = -1
    queued_version: int = -1
    failed_version: int = -1
    checkpoint: int | None = None
    checkpoint_at: float | None = None
    media: dict = field(default_factory=lambda: {"interviewer": [], "candidate": []})
    audio_status: dict = field(default_factory=dict)


class InterviewArchive:
    def __init__(self, session_id, *, candidate_enabled, policy=None, save=None, broadcast=None, on_complete=None,
                 wait_timeout=300, audio_budget=64 * 1024 * 1024):
        from services.storage import review_archive

        self.session_id = session_id
        self.candidate_enabled = candidate_enabled
        self.candidate_unavailable = False
        self.policy = policy or SimpleRoundBoundaryPolicy()
        self._save = save or review_archive.save_round
        self._broadcast = broadcast or (lambda _value: None)
        self._complete = on_complete or (lambda: None)
        self._wait_timeout = wait_timeout
        self._lock = threading.RLock()
        self._changed = threading.Event()
        self._writes = queue.Queue(maxsize=4)
        self._rounds: dict[str, Round] = {}
        self._keys: dict[str, str] = {}
        self._candidate_finals: dict[str, tuple[int, float | None]] = {}
        self._runs: dict[str, str] = {}
        self._current: Round | None = None
        self._warnings: list[str] = []
        self._revision = 0
        self._last_status = None
        self._stopping = False
        self._stop_checkpoint: int | None = None
        self._input_ended: float | None = None
        self._stt_timeouts: set[str] = set()
        self._finished = threading.Event()
        self._finalized = threading.Event()
        self._wall_anchor = time.time() - time.monotonic()
        self.audio = AudioRecorder(budget=audio_budget)
        self._coordinator = threading.Thread(target=self._coordinate, name="archive.rounds", daemon=True)
        self._writer = threading.Thread(target=self._write, name="archive.writer", daemon=True)
        self._coordinator.start()
        self._writer.start()

    def warning(self, message):
        # Called from capture too: do not perform IO or broadcast here.
        if message not in self._warnings:
            self._warnings.append(message)
        self._changed.set()

    def capture(self, samples, speaker):
        self.audio.offer(samples, speaker)

    def end_capture(self, speaker):
        self.audio.end_capture(speaker)

    def seal_capture(self):
        with self._lock:
            if self._stop_checkpoint is None:
                self._stop_checkpoint = self.audio.seal()
                self._input_ended = time.monotonic()

    def stt_timeout(self, speaker):
        with self._lock:
            self._stt_timeouts.add(speaker)

    def settle_transcriptions(self):
        with self._lock:
            for value in self._rounds.values():
                for key, utterance in value.utterances.items():
                    if not utterance["is_final"]:
                        self.discard(key)

    def _new(self, started):
        value = Round(uuid4().hex, len(self._rounds) + 1, started)
        self._rounds[value.id] = value
        self._current = value
        return value

    def _touch(self, value, *, audio_changed=False):
        value.version += 1
        if audio_changed:
            value.audio_version += 1
        self._changed.set()

    def _candidate_before(self, at):
        return max((u for r in self._rounds.values() for u in r.utterances.values()
                    if u["speaker"] == "candidate" and u["started_mono"] < at),
                   key=lambda u: u["started_mono"], default=None)

    def _candidate_round_at(self, started):
        return next((r for r in reversed(self._rounds.values())
                     if r.started <= started and (r.ended is None or started < r.ended)), self._current)

    def transcript(self, speaker, text, key, started_mono, is_final=True, group=None, *,
                   audio_end=None, cloud=False, source_capture_id=None):
        with self._lock:
            if self._stopping:
                return self._keys.get(key)
            value = self._rounds.get(self._keys.get(key))
            existing = value.utterances.get(key) if value else None
            reason = existing.get("boundary_reason") if existing else None
            if existing and existing["is_final"] and not is_final:
                return value.id
            observed = existing["observed_start"] if existing else started_mono
            capture_id = existing.get("capture_id") if existing else None
            if cloud:
                final_only = speaker == "candidate" and is_final and existing is None
                prior = self._candidate_before(observed) if final_only else None
                after = prior["started_mono"] if prior else None
                started_mono, capture_id = self.audio.speech_start(
                    speaker, observed, capture_id, after=after, final_only=final_only,
                )
                if final_only and started_mono == observed:
                    self._candidate_finals[key] = self.audio.checkpoint(), after
            if value is None:
                value = self._candidate_round_at(started_mono) if speaker == "candidate" else self._current
                value = value or self._new(started_mono)
                self._keys[key] = value.id
            if speaker == "interviewer" and is_final and text and (not existing or not existing["is_final"]):
                has_question = any(u["speaker"] == "interviewer" and u["text"] for u in value.utterances.values())
                same_group = bool(group and group == value.group)
                # An upstream group can be reused by MMPG after a reply; a new
                # explicit question must still close the business round.
                if any(u["speaker"] == "candidate" and u["text"] for u in value.utterances.values()):
                    same_group = False
                decision = self.policy.decide(text, has_question=has_question, same_group=same_group)
                reason = decision.reason
                if decision.action == "start_next_turn" and value is self._current:
                    start = max(value.started, started_mono)
                    previous = value
                    previous.ended = start
                    previous.closed_mono = time.monotonic()
                    previous.checkpoint = self.audio.checkpoint()
                    previous.checkpoint_at = previous.closed_mono
                    previous.utterances.pop(key, None)
                    value = self._new(start)
                    # A candidate can start before the final for the new
                    # question reaches us. Correct those provisional bindings
                    # using the now-confirmed source audio boundary.
                    for candidate_key, utterance in list(previous.utterances.items()):
                        if utterance["speaker"] == "candidate" and utterance["started_mono"] >= start:
                            value.utterances[candidate_key] = previous.utterances.pop(candidate_key)
                            self._keys[candidate_key] = value.id
                    self._touch(previous, audio_changed=True)
                    self._keys[key] = value.id
                    started_mono = start
                elif not has_question and value.seq == 1:
                    value.started = min(value.started, started_mono)
                if decision.action != "append_context":
                    value.question_revision += 1
                if group:
                    value.group = group
            if existing and existing["text"] == text and existing["is_final"] == is_final:
                return value.id
            if not cloud and existing:
                started_mono = existing["started_mono"]
                audio_end = audio_end or existing.get("audio_ended_mono")
            observed_end = time.monotonic()
            value.utterances[key] = dict(
                id=key, speaker=speaker, text=text, is_final=is_final,
                started_mono=started_mono, ended_mono=audio_end or observed_end,
                audio_ended_mono=audio_end, alignment="local_vad" if cloud else "capture",
                observed_start=observed, observed_end=observed_end, capture_id=capture_id,
                source_capture_id=source_capture_id or (existing or {}).get("source_capture_id"), boundary_reason=reason,
            )
            if value.ended is not None:
                value.checkpoint = self.audio.checkpoint()
            self._touch(value, audio_changed=True)
            return value.id

    def _resolve_candidate_finals(self):
        for key, (checkpoint, after) in list(self._candidate_finals.items()):
            if not self.audio.encoded(checkpoint):
                continue
            del self._candidate_finals[key]
            old = self._rounds[self._keys[key]]
            utterance = old.utterances[key]
            started, capture_id = self.audio.speech_start(
                "candidate", utterance["observed_start"], utterance["capture_id"], after=after, final_only=True,
            )
            if started == utterance["started_mono"]:
                continue
            utterance.update(started_mono=started, capture_id=capture_id)
            owner = self._candidate_round_at(started)
            if owner is not None and owner is not old:
                owner.utterances[key] = old.utterances.pop(key)
                self._keys[key] = owner.id
                owner.checkpoint = checkpoint
                self._touch(owner, audio_changed=True)
            self._touch(old, audio_changed=True)

    def discard(self, key, *, audio_end=None):
        with self._lock:
            value = self._rounds.get(self._keys.get(key))
            if value and key in value.utterances and not value.utterances[key]["is_final"]:
                utterance = value.utterances[key]
                utterance.update(is_final=True, transcription_failed=True)
                if audio_end:
                    utterance["audio_ended_mono"] = audio_end
                utterance["observed_end"] = time.monotonic()
                if value.ended is not None:
                    value.checkpoint = self.audio.checkpoint()
                self._touch(value, audio_changed=True)

    def bind_task(self, meta, question):
        with self._lock:
            rid = self._keys.get(meta.get("transcript_anchor"))
            if rid is None:
                rid = self._current.id if self._current else None
            if rid is None and not self._stopping:
                rid = self.transcript("interviewer", question, "manual-" + uuid4().hex, time.monotonic(), True)
            return rid

    def start_ai(self, round_id, run_id, model):
        with self._lock:
            value = self._rounds.get(round_id)
            if value is None:
                return None
            self._runs[run_id] = round_id
            run = value.generations.setdefault(run_id, dict(id=run_id, status="pending", answer="", thinking="", attempts=[],
                started_at=time.time(), started_mono=time.monotonic(), first_token_at=None, first_token_ms=None, total_ms=None,
                question_revision=value.question_revision))
            run["model"] = model or run.get("model", {})
            self._touch(value)
            return run_id

    def ai_event(self, run_id, event):
        with self._lock:
            value = self._rounds.get(self._runs.get(run_id))
            if value is None:
                return
            run = value.generations[run_id]
            kind = event["type"]
            if kind == "answer_chunk" and event.get("chunk"):
                if run["first_token_at"] is None and event["chunk"].strip():
                    run["first_token_at"] = time.time()
                    run["first_token_ms"] = (time.monotonic() - run["started_mono"]) * 1000
                run["answer"] += event["chunk"]
            elif kind == "answer_think_chunk":
                run["thinking"] += event.get("chunk", "")
            elif kind == "answer_done":
                run["answer"] = event.get("answer", run["answer"])
                self.finish_ai(run_id, "completed")
            elif kind in ("answer_error", "answer_cancelled"):
                self.finish_ai(run_id, "failed" if kind == "answer_error" else "cancelled", error=event.get("message", ""))
            self._touch(value)

    def observe_attempt(self, run_id, event):
        with self._lock:
            value = self._rounds.get(self._runs.get(run_id))
            if value is None:
                return
            run = value.generations[run_id]
            if event.get("event") == "request":
                if not run["attempts"]:
                    run["started_at"] = time.time()
                    run["started_mono"] = time.monotonic()
            else:
                previous = next((a for a in run["attempts"] if a["attempt_id"] == event["attempt_id"]), None)
                if previous is None:
                    run["attempts"].append(copy.deepcopy(event))
            self._touch(value)

    def finish_ai(self, run_id, status, *, error=""):
        with self._lock:
            value = self._rounds.get(self._runs.get(run_id))
            if not value:
                return
            run = value.generations[run_id]
            if status == run["status"] or (status == "cancelled" and run["status"] not in ("pending", "timed_out")):
                return
            # Explicit later commit failure may replace an apparent success.
            finished_at, finished_mono = run.pop("generated_at", (time.time(), time.monotonic()))
            run.update(status=status, error=error, finished_at=finished_at, total_ms=(finished_mono - run["started_mono"]) * 1000)
            self._touch(value)

    def mark_generation_ended(self, run_id):
        with self._lock:
            value = self._rounds.get(self._runs.get(run_id))
            if value:
                value.generations[run_id]["generated_at"] = time.time(), time.monotonic()

    def settle_unreported_ai(self, run_id, cancelled=False):
        with self._lock:
            value = self._rounds.get(self._runs.get(run_id))
            run = value.generations.get(run_id) if value else None
            if run and run["status"] == "pending":
                self.finish_ai(run_id, "cancelled" if cancelled else "failed")

    def _ready(self, value):
        if value.ended is None:
            return False
        if any(key in self._candidate_finals for key in value.utterances):
            return False
        expired = time.monotonic() - value.closed_mono >= self._wait_timeout
        pending = [u for u in value.utterances.values() if not u["is_final"]]
        running = [r for r in value.generations.values() if r["status"] == "pending"]
        if expired:
            for u in pending:
                u.update(is_final=True, transcription_failed=True)
            if pending:
                self._touch(value, audio_changed=True)
            for run in running:
                self.finish_ai(run["id"], "timed_out", error="归档等待超时")
            return True
        # The ASR grouper dispatches just after the final transcript hook.
        # Give it one short scheduling window to attach a queued generation.
        if not self._stopping and time.monotonic() - value.closed_mono < 0.25:
            return False
        return not pending and not running

    def _snapshot(self, value):
        utterances = sorted(value.utterances.values(), key=lambda u: u["started_mono"])
        timeouts = sorted(self._stt_timeouts) if self._stopping and value is self._current else []
        if value.audio_version == value.saved_audio_version:
            media, assets, audio_status = value.media, {}, value.audio_status
        else:
            media, assets, audio_status = self.audio.media(
                utterances, start=value.started, end=value.ended,
                previous=value.media, previous_status=value.audio_status,
                timed_out=timeouts,
                previous_candidate=self._candidate_before(value.started),
            )
        generations = list(value.generations.values())
        current = [g for g in generations if g["question_revision"] == value.question_revision]
        success = [g for g in current if g["status"] == "completed" and g["answer"]]
        latest = success[-1] if success else (current[-1] if current else None)
        candidate = [u for u in utterances if u["speaker"] == "candidate"]
        answer = "\n".join(u["text"] for u in candidate if u["text"])
        candidate_status = ("recorded" if answer else "transcription_failed" if candidate or media["candidate"] else
                            "unanswered" if self.candidate_enabled else "unavailable" if self.candidate_unavailable else "disabled")
        return dict(session_id=self.session_id, round_id=value.id, seq=value.seq, policy_version=self.policy.version,
                    started_at=self._wall_anchor + value.started, ended_at=self._wall_anchor + value.ended,
                    question_text="\n".join(u["text"] for u in utterances if u["speaker"] == "interviewer" and u["text"]),
                    candidate_answer_text=answer, reference_answer_text=latest["answer"] if success else "",
                    candidate_status=candidate_status, ai_status=latest["status"] if latest else "not_requested",
                    model_summary={**latest.get("model", {}), "first_token_ms": latest["first_token_ms"], "total_ms": latest["total_ms"]} if latest else None,
                    utterances=[{**u, "started_at": self._wall_anchor + u["started_mono"],
                                 "ended_at": self._wall_anchor + u["ended_mono"]} for u in utterances],
                    generations=copy.deepcopy(generations), media=media, assets=assets, audio_status=audio_status,
                    transcription_timeouts=timeouts)

    def snapshot(self):
        with self._lock:
            saved = [r for r in self._rounds.values() if r.saved_version >= 0]
            pending = [r for r in self._rounds.values() if r.ended is not None and r.saved_version < r.version and r.failed_version != r.version]
            gaps = [f"第 {r.seq} 题{label}{'部分录音缺失' if state['state'] == 'partial' else '录音缺失'}"
                    for r in saved for speaker, label in (("interviewer", "面试官"), ("candidate", "候选人"))
                    if (state := r.audio_status.get(speaker)) and state["state"] in ("partial", "missing")]
            warnings = self._warnings + gaps
            return dict(session_id=self.session_id, revision=self._revision, archive_status=(
                ("partial" if warnings else "complete") if self._finished.is_set() else "draining" if self._stopping else "recording"),
                persisted_turn_count=len(saved), pending_round_count=len(pending), last_saved_seq=max((r.seq for r in saved), default=0),
                warning="；".join(warnings), memory_bytes=self.audio.memory_bytes)

    def _emit(self):
        state = self.snapshot()
        comparison = {k: v for k, v in state.items() if k not in ("revision", "memory_bytes")}
        if comparison == self._last_status:
            return
        self._last_status = comparison
        self._revision += 1
        state["revision"] = self._revision
        try:
            self._broadcast({"type": "review_archive_status", **state})
        except Exception:
            _log.exception("archive status broadcast failed")

    def _coordinate(self):
        while not self._finished.is_set():
            self._changed.wait(0.05)
            self._changed.clear()
            with self._lock:
                self._resolve_candidate_finals()
                self._preserve_untranscribed_session()
                for value in self._rounds.values():
                    if self._writes.full():
                        break
                    if (value.saved_version >= value.version or value.queued_version >= value.version
                            or value.failed_version == value.version or not self._ready(value)):
                        continue
                    if value.queued_version > value.saved_version and value.queued_version != value.failed_version:
                        continue
                    if value.checkpoint is None:
                        value.checkpoint = self.audio.checkpoint()
                    if not self.audio.encoded(value.checkpoint):
                        continue
                    version = value.version
                    snapshot = self._snapshot(value)
                    self._writes.put_nowait((value.id, version, value.audio_version, snapshot))
                    value.queued_version = version
            self._emit()
            if self._stopping:
                with self._lock:
                    settled = all(r.saved_version >= r.version or r.failed_version == r.version for r in self._rounds.values())
                if settled and self.audio.encoded(self._stop_checkpoint) and not self._writes.unfinished_tasks:
                    self._finished.set()
                    self.audio.close()
                    self._writer.join()
                    try:
                        self._complete()
                    except Exception:
                        self.warning("面试归档状态写入失败")
                        _log.exception("archive completion callback failed")
                    finally:
                        self._emit()
                        self._finalized.set()

    def _preserve_untranscribed_session(self):
        """After the final checkpoint, keep detected speech even if ASR never opened a turn."""
        if self._stopping and not self._rounds and self.audio.encoded(self._stop_checkpoint):
            start = self.audio.first_speech(self._stt_timeouts)
            if start is not None:
                value = self._new(start)
                value.ended = self._input_ended
                value.closed_mono = time.monotonic()
                value.checkpoint = self._stop_checkpoint

    def _write(self):
        while not self._finished.is_set() or self._writes.unfinished_tasks:
            try:
                rid, version, audio_version, snapshot = self._writes.get(timeout=0.1)
            except queue.Empty:
                continue
            error = None
            for attempt in range(3):
                try:
                    self._save(snapshot)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt < 2:
                        time.sleep(0.1 * (attempt + 1))
            with self._lock:
                value = self._rounds[rid]
                if error:
                    value.failed_version = version
                    self.warning(f"第 {value.seq} 题保存失败，请检查磁盘空间或权限")
                    _log.error("round archive write failed: %s", error)
                else:
                    value.saved_version = version
                    value.saved_audio_version = audio_version
                    value.media = snapshot["media"]
                    value.audio_status = snapshot["audio_status"]
                    active = [r.started for r in self._rounds.values() if r.saved_version < r.version]
                    self.audio.prune((min(active) if active else time.monotonic()) - 0.3)
            if error is None:
                _log.info("Archived session=%s round=%s encoding_and_save_ms=%.0f audio=%s",
                          self.session_id, value.seq, (time.monotonic() - (value.checkpoint_at or time.monotonic())) * 1000,
                          snapshot["audio_status"])
                _log.debug("Archived sample ranges session=%s round=%s ranges=%s", self.session_id, value.seq, {
                    speaker: [(clip["capture_id"], clip["sample_start"], clip["sample_count"]) for clip in playlist]
                    for speaker, playlist in snapshot["media"].items()
                })
            self._writes.task_done()
            self._changed.set()

    def wait_saved(self, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = any(r.ended is not None and r.saved_version < r.version for r in self._rounds.values())
            if not pending and not self._writes.unfinished_tasks:
                return True
            time.sleep(0.01)
        return False

    def stop(self, timeout=5):
        with self._lock:
            if not self._stopping:
                self.seal_capture()
                if self._current and self._current.ended is None:
                    self._current.ended = max(self._current.started, self._input_ended)
                    self._current.closed_mono = time.monotonic()
                    self._current.checkpoint = self._stop_checkpoint
                    self._current.checkpoint_at = self._current.closed_mono
                    self._touch(self._current, audio_changed=True)
                self.settle_transcriptions()
                for value in self._rounds.values():
                    for run in value.generations.values():
                        if run["status"] == "pending":
                            self.finish_ai(run["id"], "cancelled", error="面试已停止")
                self._stopping = True
        self._changed.set()
        return self.wait_finalized(timeout)

    def wait_finalized(self, timeout=None):
        return self._finalized.wait(timeout)
