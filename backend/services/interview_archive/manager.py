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
    saved_version: int = -1
    queued_version: int = -1
    failed_version: int = -1
    media: dict = field(default_factory=lambda: {"interviewer": [], "candidate": []})


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
        self._runs: dict[str, str] = {}
        self._current: Round | None = None
        self._warnings: list[str] = []
        self._revision = 0
        self._last_status = None
        self._stopping = False
        self._finished = threading.Event()
        self._finalized = threading.Event()
        self._wall_anchor = time.time() - time.monotonic()
        self.audio = AudioRecorder(self.warning, budget=audio_budget)
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
        if not self._stopping:
            self.audio.offer(samples, speaker)

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

    def _touch(self, value):
        value.version += 1
        self._changed.set()

    def transcript(self, speaker, text, key, started_mono, is_final=True, group=None, *, audio_end=None, cloud=False):
        with self._lock:
            value = self._rounds.get(self._keys.get(key))
            existing = value.utterances.get(key) if value else None
            reason = existing.get("boundary_reason") if existing else None
            if existing and existing["is_final"] and not is_final:
                return value.id
            if value is None:
                if self._stopping:
                    return None
                value = self._current
                if speaker == "candidate":
                    started_mono, matched_end = self.audio.align(speaker, started_mono, key, final=is_final) if cloud else (started_mono, None)
                    audio_end = audio_end or matched_end
                    eligible = [round_ for round_ in self._rounds.values()
                                if round_.started <= started_mono and (round_.ended is None or started_mono < round_.ended)]
                    if eligible:
                        value = eligible[-1]
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
                    start, end = self.audio.align(speaker, started_mono, key, final=True) if cloud else (started_mono, audio_end)
                    previous = value
                    previous.ended = start
                    previous.closed_mono = time.monotonic()
                    previous.utterances.pop(key, None)
                    value = self._new(start)
                    # A candidate can start before the final for the new
                    # question reaches us. Correct those provisional bindings
                    # using the now-confirmed source audio boundary.
                    for candidate_key, utterance in list(previous.utterances.items()):
                        if utterance["speaker"] == "candidate" and utterance["started_mono"] >= start:
                            value.utterances[candidate_key] = previous.utterances.pop(candidate_key)
                            self._keys[candidate_key] = value.id
                    self._touch(previous)
                    self._keys[key] = value.id
                    started_mono = start
                    audio_end = audio_end or end
                if decision.action != "append_context":
                    value.question_revision += 1
                if group:
                    value.group = group
            if existing and existing["text"] == text and existing["is_final"] == is_final:
                return value.id
            if cloud:
                aligned, end = self.audio.align(speaker, started_mono, key, final=is_final)
                started_mono = aligned
                audio_end = audio_end or end
            elif existing:
                started_mono = existing["started_mono"]
                audio_end = audio_end or existing.get("audio_ended_mono")
            value.utterances[key] = dict(
                id=key, speaker=speaker, text=text, is_final=is_final,
                started_mono=started_mono, ended_mono=audio_end or time.monotonic(),
                audio_ended_mono=audio_end, alignment="local_vad" if cloud else "capture",
                boundary_reason=reason,
            )
            self._touch(value)
            return value.id

    def discard(self, key, *, audio_end=None):
        with self._lock:
            value = self._rounds.get(self._keys.get(key))
            if value and key in value.utterances and not value.utterances[key]["is_final"]:
                utterance = value.utterances[key]
                utterance.update(is_final=True, transcription_failed=True)
                if audio_end:
                    utterance["audio_ended_mono"] = audio_end
                elif utterance["alignment"] == "local_vad":
                    start, end = self.audio.align(utterance["speaker"], utterance["started_mono"], key, final=True)
                    utterance.update(started_mono=start, audio_ended_mono=end)
                self._touch(value)

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
        expired = time.monotonic() - value.closed_mono >= self._wait_timeout
        pending = [u for u in value.utterances.values() if not u["is_final"]]
        running = [r for r in value.generations.values() if r["status"] == "pending"]
        if expired:
            for u in pending:
                u.update(is_final=True, transcription_failed=True)
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
        media, assets = self.audio.media(utterances)
        for speaker in media:
            # Rebuilt manifests replace the ranges for still-buffered assets.
            # Keep ranges already committed and evicted from the PCM cache.
            new_asset_ids = {clip["asset_id"] for clip in media[speaker]}
            kept = [clip for clip in value.media[speaker] if clip["asset_id"] not in new_asset_ids]
            media[speaker] = sorted(kept + media[speaker], key=lambda clip: clip["capture_mono"])
        generations = list(value.generations.values())
        current = [g for g in generations if g["question_revision"] == value.question_revision]
        success = [g for g in current if g["status"] == "completed" and g["answer"]]
        latest = success[-1] if success else (current[-1] if current else None)
        candidate = [u for u in utterances if u["speaker"] == "candidate"]
        answer = "\n".join(u["text"] for u in candidate if u["text"])
        candidate_status = ("recorded" if answer else "transcription_failed" if candidate else
                            "unanswered" if self.candidate_enabled else "unavailable" if self.candidate_unavailable else "disabled")
        return dict(session_id=self.session_id, round_id=value.id, seq=value.seq, policy_version=self.policy.version,
                    started_at=self._wall_anchor + value.started, ended_at=self._wall_anchor + value.ended,
                    question_text="\n".join(u["text"] for u in utterances if u["speaker"] == "interviewer" and u["text"]),
                    candidate_answer_text=answer, reference_answer_text=latest["answer"] if success else "",
                    candidate_status=candidate_status, ai_status=latest["status"] if latest else "not_requested",
                    model_summary={**latest.get("model", {}), "first_token_ms": latest["first_token_ms"], "total_ms": latest["total_ms"]} if latest else None,
                    utterances=[{**u, "started_at": self._wall_anchor + u["started_mono"],
                                 "ended_at": self._wall_anchor + u["ended_mono"]} for u in utterances],
                    generations=copy.deepcopy(generations), media=media, assets=assets)

    def snapshot(self):
        with self._lock:
            saved = [r for r in self._rounds.values() if r.saved_version >= 0]
            pending = [r for r in self._rounds.values() if r.ended is not None and r.saved_version < r.version and r.failed_version != r.version]
            return dict(session_id=self.session_id, revision=self._revision, archive_status=(
                "partial" if self._warnings else "complete" if self._finished.is_set() else "draining" if self._stopping else "recording"),
                persisted_turn_count=len(saved), pending_round_count=len(pending), last_saved_seq=max((r.seq for r in saved), default=0),
                warning="；".join(self._warnings), memory_bytes=self.audio.memory_bytes)

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
                ready = [r for r in self._rounds.values() if r.saved_version < r.version and r.queued_version < r.version
                         and r.failed_version != r.version and self._ready(r)]
            if ready and not self.audio.flush():
                self.warning("音频编码尚未完成，部分录音可能缺失")
            for value in ready:
                with self._lock:
                    if self._writes.full():
                        break
                    if not self._ready(value) or value.saved_version >= value.version:
                        continue
                    version = value.version
                    snapshot = self._snapshot(value)
                    self._writes.put_nowait((value.id, version, snapshot))
                    value.queued_version = version
            self._emit()
            if self._stopping:
                with self._lock:
                    settled = all(r.saved_version >= r.version or r.failed_version == r.version for r in self._rounds.values())
                if settled and not self._writes.unfinished_tasks:
                    self._finished.set()
                    self._emit()
                    try:
                        self._complete()
                    except Exception:
                        _log.exception("archive completion callback failed")
                    finally:
                        self._finalized.set()

    def _write(self):
        while not self._finished.is_set() or self._writes.unfinished_tasks:
            try:
                rid, version, snapshot = self._writes.get(timeout=0.1)
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
                    self.warning("保存失败，请检查磁盘空间或权限")
                    _log.error("round archive write failed: %s", error)
                else:
                    value.saved_version = version
                    value.media = snapshot["media"]
                    active = [r.started for r in self._rounds.values() if r.saved_version < r.version]
                    self.audio.prune((min(active) if active else time.monotonic()) - 0.3)
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
        self._stopping = True
        self.audio.close(min(timeout, 1))
        with self._lock:
            if self._current and self._current.ended is None:
                self._current.ended = time.monotonic()
                self._current.closed_mono = time.monotonic()
                self._touch(self._current)
            self.settle_transcriptions()
            for value in self._rounds.values():
                for run in value.generations.values():
                    if run["status"] == "pending":
                        self.finish_ai(run["id"], "cancelled", error="面试已停止")
        self._changed.set()
        return self._finalized.wait(timeout)
