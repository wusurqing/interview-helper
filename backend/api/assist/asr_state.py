"""ASR merge and question-group state machine for interview assist."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

from core.session import conversation_lock
from services.stt import (
    build_asr_question_group_text,
    classify_asr_question_candidate,
    is_asr_question_constraint_tail,
    is_viable_asr_question_group,
    join_transcription_fragments,
    split_question_like_text,
    transcription_for_publish,
)
from api.assist.scheduler import TaskPayload


@dataclass
class PendingASRGroup:
    source: str
    utterances: list[str] = field(default_factory=list)
    first_mono: float = 0.0
    last_mono: float = 0.0
    has_promote: bool = False
    transcript_anchor: str | None = None
    transcript_anchors: list[str] = field(default_factory=list)


class AssistAsrStateMachine:
    def __init__(
        self,
        *,
        broadcast: Callable[[dict], None],
        submit_answer_task: Callable[[TaskPayload], bool],
        begin_asr_turn: Callable[[], int],
        logger,
        append_late_constraint_tail: Callable[[str, str, float, str | None], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        allow_answer: Callable[[], bool] | None = None,
    ):
        self.broadcast = broadcast
        self.submit_answer_task = submit_answer_task
        self.begin_asr_turn = begin_asr_turn
        self.append_late_constraint_tail = append_late_constraint_tail or (
            lambda _text, _source, _now, _anchor: False
        )
        self.logger = logger
        self.clock = clock
        self._allow_answer = allow_answer
        self.merge_parts: list[str] = []
        self.merge_mono_first: Optional[float] = None
        self.merge_mono_last: Optional[float] = None
        self.merge_transcript_anchor: str | None = None
        self.merge_transcript_anchors: list[str] = []
        self.pending_group: Optional[PendingASRGroup] = None

    def reset_merge_buffer(self):
        self.merge_parts = []
        self.merge_mono_first = None
        self.merge_mono_last = None
        self.merge_transcript_anchor = None
        self.merge_transcript_anchors = []

    def reset_pending_group(self):
        self.pending_group = None

    def _automatic_answers_enabled(self, cfg) -> bool:
        if self._allow_answer is not None:
            return self._allow_answer()
        return cfg.auto_detect

    def accept_final_turn(self, cfg, session, text: str, mmpg_trace: str | None = None) -> str:
        """Publish a complete cloud turn and evaluate it without a local timer."""
        text = text.strip()
        if not text:
            return "empty"
        with conversation_lock:
            session.add_transcription(text)
        self.broadcast({"type": "transcription", "text": text})
        return self.submit_final_question(cfg, session, text, mmpg_trace)

    def submit_final_question(self, cfg, session, text, mmpg_trace=None, request=None) -> str:
        if not self._automatic_answers_enabled(cfg):
            return "auto_answer_disabled"
        if not is_transcript_question(cfg, text):
            return "filtered"
        turn = self.begin_asr_turn() if request is None else 0
        source = "conversation_loopback" if session.capture_is_loopback else "conversation_mic"
        queued = self.submit_answer_task((text, None, False, source, {
            "origin": "asr" if request is None else "mmpg",
            "asr_turn_id": turn,
            "utterances": text.splitlines() if request is not None else [text],
            "mmpg_trace": mmpg_trace,
            **({"mmpg_request": request} if request is not None else {}),
            **({"transcript_anchor": request.transcript_anchor} if request and request.transcript_anchor else {}),
        }))
        return "queued" if queued else "no_answer_model"

    def flush_question_group_now(self, cfg, session, *, immediate: bool = False) -> None:
        group = self.pending_group
        if group is None:
            return
        self.pending_group = None
        if not self._automatic_answers_enabled(cfg):
            return
        if not is_viable_asr_question_group(
            group.utterances,
            getattr(cfg, "transcription_min_sig_chars", 2),
        ):
            return
        question_text = build_asr_question_group_text(group.utterances)
        if not question_text:
            return
        if group.transcript_anchors:
            session.group_transcript_question(group.transcript_anchors)
        now_mono = self.clock()
        grace_sec = 0.0 if immediate else _asr_late_constraint_grace_sec(cfg)
        turn_id = self.begin_asr_turn()
        self.logger.info(
            "ASR_QUESTION turn=%d utterances=%d text=%r",
            turn_id, len(group.utterances),
            question_text[:150],
        )
        self.submit_answer_task(
            (
                question_text,
                None,
                False,
                group.source,
                {
                    "origin": "asr",
                    "asr_turn_id": turn_id,
                    "utterances": list(group.utterances),
                    "dispatch_after_mono": 0.0 if immediate else now_mono + grace_sec,
                    "asr_tail_grace_until_mono": 0.0 if immediate else now_mono + grace_sec,
                    **({"transcript_anchor": group.transcript_anchor} if group.transcript_anchor else {}),
                    **({"transcript_anchors": tuple(group.transcript_anchors)} if group.transcript_anchors else {}),
                },
            )
        )

    def try_flush_question_group(self, cfg, session, now_mono: float, force: bool = False) -> None:
        group = self.pending_group
        if group is None:
            return
        if not self._automatic_answers_enabled(cfg):
            self.reset_pending_group()
            return
        confirm = _asr_confirm_window_sec(cfg)
        fast_confirm = _asr_fast_confirm_sec(cfg)
        max_wait = _asr_group_max_wait_sec(cfg)
        if group.has_promote and len(group.utterances) == 1:
            max_wait = max(max_wait, fast_confirm + confirm)
        since_last = now_mono - group.last_mono
        age = now_mono - group.first_mono
        if force:
            self.flush_question_group_now(cfg, session, immediate=True)
        elif age >= max_wait:
            self.flush_question_group_now(cfg, session)
        elif group.has_promote and len(group.utterances) == 1 and since_last >= fast_confirm:
            self.flush_question_group_now(cfg, session)
        elif group.has_promote and since_last >= confirm:
            self.flush_question_group_now(cfg, session)
        elif not group.has_promote and since_last >= confirm * 2:
            self.flush_question_group_now(cfg, session)

    def handle_auto_detect_asr_text(
        self, cfg, session, pub: str, source: str, now_mono: float,
        transcript_anchor: str | None = None, transcript_anchors: tuple[str, ...] = (),
    ) -> None:
        if not self._automatic_answers_enabled(cfg):
            self.reset_pending_group()
            return
        kind, cleaned = classify_asr_question_candidate(
            pub,
            getattr(cfg, "transcription_min_sig_chars", 2),
        )
        if not cleaned:
            return
        if (
            self.pending_group is None
            and kind == "candidate"
            and is_asr_question_constraint_tail(cleaned)
        ):
            try:
                merged_tail = self.append_late_constraint_tail(
                    cleaned, source, now_mono, transcript_anchor, transcript_anchors,
                )
            except TypeError:
                merged_tail = self.append_late_constraint_tail(
                    cleaned, source, now_mono, transcript_anchor,
                )
            if merged_tail:
                return
        if (
            self.pending_group is not None
            and kind == "candidate"
            and is_asr_question_constraint_tail(cleaned)
        ):
            self.pending_group.utterances.append(cleaned)
            self.pending_group.last_mono = now_mono
            self.pending_group.source = source
            self.pending_group.transcript_anchor = transcript_anchor or self.pending_group.transcript_anchor
            self.pending_group.transcript_anchors.extend(transcript_anchors)
            return
        self.try_flush_question_group(cfg, session, now_mono, False)
        if kind == "ignore" and self.pending_group is None:
            return
        if self.pending_group is None:
            self.pending_group = PendingASRGroup(
                source=source,
                utterances=[cleaned],
                first_mono=now_mono,
                last_mono=now_mono,
                has_promote=(kind == "promote"),
                transcript_anchor=transcript_anchor,
                transcript_anchors=list(transcript_anchors),
            )
        else:
            self.pending_group.utterances.append(cleaned)
            self.pending_group.last_mono = now_mono
            self.pending_group.source = source
            self.pending_group.has_promote = self.pending_group.has_promote or kind == "promote"
            if kind != "ignore" and transcript_anchor:
                self.pending_group.transcript_anchor = transcript_anchor
                self.pending_group.transcript_anchors.extend(transcript_anchors)

    def flush_merge_buffer_now(self, cfg, session) -> None:
        if not self.merge_parts:
            return
        parts = list(self.merge_parts)
        transcript_anchor = self.merge_transcript_anchor
        transcript_anchors = tuple(self.merge_transcript_anchors)
        self.reset_merge_buffer()
        merged_raw = join_transcription_fragments(parts)
        min_sig = getattr(cfg, "transcription_min_sig_chars", 2)
        pub = transcription_for_publish(merged_raw, min_sig)
        if not pub:
            return
        with conversation_lock:
            session.add_transcription(pub)
        self.broadcast({"type": "transcription", "text": pub})
        if self._automatic_answers_enabled(cfg):
            source = (
                "conversation_loopback"
                if session.capture_is_loopback
                else "conversation_mic"
            )
            self.handle_auto_detect_asr_text(cfg, session, pub, source, self.clock(), transcript_anchor, transcript_anchors)

    def try_flush_merge_buffer(self, cfg, session, now_mono: float, force: bool = False) -> None:
        if not self.merge_parts:
            return
        gap = float(getattr(cfg, "assist_transcription_merge_gap_sec", 2.0) or 0.0)
        max_wait = float(getattr(cfg, "assist_transcription_merge_max_sec", 12.0) or 12.0)
        if max_wait < 1.0:
            max_wait = 12.0
        if force:
            self.flush_merge_buffer_now(cfg, session)
            return
        if gap <= 0:
            return
        if self.merge_mono_last is None:
            return
        since_last = now_mono - self.merge_mono_last
        burst_age = (now_mono - self.merge_mono_first) if self.merge_mono_first is not None else 0.0
        if since_last >= gap or burst_age >= max_wait:
            self.flush_merge_buffer_now(cfg, session)

    def append_transcription_fragment(
        self,
        cfg,
        session,
        pub: str,
        now_mono: float,
        force_flush_tail: bool = False,
        *,
        started_at: float | None = None,
        transcript_key: str = "",
        transcript_epoch: int | None = None,
        publish: bool = True,
        raw_text: str | None = None,
        audio_end: float | None = None,
        source_capture_id: str | None = None,
    ) -> None:
        transcript_key = transcript_key or uuid4().hex
        transcript_anchor = session.record_transcript(
            "interviewer", pub, started_at=now_mono if started_at is None else started_at,
            utterance_key=transcript_key, epoch=transcript_epoch,
            question_group=transcript_key if is_transcript_question(cfg, pub) else None,
            raw_text=raw_text, audio_end=audio_end, source_capture_id=source_capture_id,
        )
        if transcript_anchor is None or not publish:
            return
        gap = float(getattr(cfg, "assist_transcription_merge_gap_sec", 2.0) or 0.0)
        if gap <= 0:
            with conversation_lock:
                session.add_transcription(pub)
            self.broadcast({"type": "transcription", "text": pub})
            if self._automatic_answers_enabled(cfg):
                source = (
                    "conversation_loopback"
                    if session.capture_is_loopback
                    else "conversation_mic"
                )
                self.handle_auto_detect_asr_text(cfg, session, pub, source, now_mono, transcript_anchor, (transcript_anchor,))
            return
        if not self.merge_parts:
            self.merge_mono_first = now_mono
        self.merge_parts.append(pub)
        self.merge_transcript_anchors.append(transcript_anchor)
        if self.merge_transcript_anchor is None or classify_asr_question_candidate(pub)[0] != "ignore":
            self.merge_transcript_anchor = transcript_anchor
        self.merge_mono_last = now_mono
        if force_flush_tail:
            self.flush_merge_buffer_now(cfg, session)
        else:
            self.try_flush_merge_buffer(cfg, session, now_mono, False)


def is_transcript_question(cfg, text: str) -> bool:
    minimum = getattr(cfg, "transcription_min_sig_chars", 2)
    return classify_asr_question_candidate(text, minimum)[0] == "promote" or any(
        classify_asr_question_candidate(part, minimum)[0] == "promote"
        for line in text.splitlines()
        for part in (line, *split_question_like_text(line))
    )


def _asr_confirm_window_sec(cfg) -> float:
    confirm = float(getattr(cfg, "assist_asr_confirm_window_sec", 0.45) or 0.0)
    return max(0.0, min(5.0, confirm))


def _asr_group_max_wait_sec(cfg) -> float:
    max_wait = float(getattr(cfg, "assist_asr_group_max_wait_sec", 1.2) or 0.0)
    return max(0.2, min(8.0, max_wait))


def _asr_fast_confirm_sec(cfg) -> float:
    fast = float(getattr(cfg, "assist_asr_fast_confirm_sec", 1.15) or 0.0)
    return max(0.1, min(2.0, fast))


def _asr_late_constraint_grace_sec(cfg) -> float:
    grace = float(getattr(cfg, "assist_asr_late_constraint_grace_sec", 1.2) or 0.0)
    return max(0.0, min(3.0, grace))


def asr_interrupt_running(cfg) -> bool:
    return bool(getattr(cfg, "assist_asr_interrupt_running", True))
