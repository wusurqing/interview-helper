"""Interview assist pipeline: ASR buffering, task dispatch, parallel answer workers."""

import copy
import gc
import numpy as np
import queue
import time
import threading
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

from core.background import BoundedTaskWorker
from core.config import MMPG_STT_PROVIDER, SEGMENT_STT_PROVIDERS, get_config
from core.logger import get_interview_logger, get_logger
from core.session import get_session, reset_session, conversation_lock
from services import review_integration

_ilog = get_interview_logger()
_elog = get_logger("pipeline")
from services.audio import AudioCapture, VADBuffer, _is_sc_id, audio_capture
from services.stt import (
    STTError,
    build_asr_question_group_text,
    stt_configuration_error,
    transcribe_segment,
    transcription_for_publish,
    postprocess_interview_transcription,
)
from services.stt.mmpg import refresh_manual_rtc_token
from api.common import get_model_health
from api.realtime.ws import broadcast
from api.assist.answer_worker import (
    AnswerCommit,
    AnswerWorkerDeps,
    process_question_parallel,
    prompt_mode_for_task as answer_prompt_mode_for_task,
)
from api.assist.asr_state import (
    AssistAsrStateMachine,
    PendingASRGroup,
    asr_interrupt_running,
    is_transcript_question,
)
from api.assist.single_mic import RoleSwitch, SingleMicConflict, SingleMicSession
from api.assist.scheduler import (
    AnswerRouting,
    EXAM_MODEL_STALE,
    EXAM_MODEL_UNAVAILABLE,
    InFlightTasks,
    ModelReservation,
    TaskPayload,
    begin_asr_turn as scheduler_begin_asr_turn,
    claim_next_dispatch,
    dispatch_model_order as scheduler_dispatch_model_order,
    dispatch_snapshot as scheduler_dispatch_snapshot,
    drain_commit_queue,
    eligible_exam_model_index,
    exam_model_order,
    is_asr_task,
    is_stale_inflight_asr_task,
    key_ok,
    live_exam_model_index,
    max_parallel_slots as scheduler_max_parallel_slots,
    model_eligible as scheduler_model_eligible,
    model_reservation,
    physical_busy_models as scheduler_physical_busy_models,
    pick_model_index as scheduler_pick_model_index,
    priority_model_index as scheduler_priority_model_index,
    task_is_written_exam,
    task_meta,
)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_interview_thread: Optional[threading.Thread] = None
_interviewer_asr_thread: Optional[threading.Thread] = None
_candidate_thread: Optional[threading.Thread] = None
_interviewer_streaming = False
_interviewer_config = None
_candidate_audio_capture = AudioCapture()
_stop_event = threading.Event()
_pause_event = threading.Event()
_candidate_flush_event = threading.Event()

# H4: 添加独立 flush 线程，避免阻塞音频采集主循环
_flush_thread: Optional[threading.Thread] = None
_flush_queue: queue.Queue = queue.Queue(maxsize=10)
_flush_stop_event = threading.Event()
_interviewer_segment_queue_maxsize = 12
_interviewer_segment_drain_timeout_sec = 1.5
_interviewer_vad_preroll_sec = 0.24
_interviewer_vad_rollover_sec = 0.28
_vad_feed_chunk_samples = 320

_answer_generation = 0
_gen_lock = threading.Lock()

_pending: list[tuple[TaskPayload, int, int]] = []
_dispatch_lock = threading.Lock()
_dispatch_changed = threading.Condition(_dispatch_lock)
_asr_state_lock = threading.RLock()
_in_flight_tasks: InFlightTasks = {}
_task_session_version = 0
_latest_asr_turn_id = 0

_commit_buffer: dict[int, Callable[[], None]] = {}
_skipped_commit_seqs: set[int] = set()
_next_commit_seq = 0
_next_submit_seq = 0
_commit_lock = threading.Lock()
_exam_cancel_lock = threading.Lock()
_cancelled_exam_preflight_ids: deque[str] = deque(maxlen=128)

_knowledge_worker: Optional[BoundedTaskWorker] = None
_asr_state = AssistAsrStateMachine(
    broadcast=lambda data: broadcast(data),
    allow_answer=lambda: bool(getattr(get_config(), "auto_detect", True)),
    submit_answer_task=lambda task: submit_answer_task(task),
    begin_asr_turn=lambda: _begin_asr_turn(),
    append_late_constraint_tail=lambda text, source, now_mono, anchor: _append_late_asr_constraint_tail(
        text, source, now_mono, anchor,
    ),
    logger=_ilog,
)


@dataclass
class InterviewerSegment:
    audio: Any
    sample_rate: int
    started_mono: float
    ended_mono: float
    audio_sec: float
    flush_reason: Literal["silence", "max_speech", "pause_flush", "stop_flush", "role_switch"]
    capture_is_loopback: bool
    transcript_key: str = ""
    transcript_reservation: str | None = None


class _InterviewerRuntime:
    def __init__(self):
        self.transcript_epoch: int | None = None
        self.segment_queue: queue.Queue[InterviewerSegment] = queue.Queue(
            maxsize=_interviewer_segment_queue_maxsize
        )
        self.drain_event = threading.Event()
        self.flush_queue: queue.Queue = queue.Queue(maxsize=10)
        self.flush_stop_event = threading.Event()
        self.flush_thread: Optional[threading.Thread] = None
        self.asr_thread: Optional[threading.Thread] = None
        self.stop_managed_by_caller = False
        self.raw_queue_drop_count = 0
        self.segment_emitted = 0
        self.segment_dropped = 0
        self.segment_failures = 0
        self.reported_role_failures = 0
        self.last_chunk_mono: Optional[float] = None
        self.last_chunk_audio_sec = 0.0
        self.max_observed_segment_queue_depth = 0
        self.loopback_discontinuity_count = 0
        self.max_raw_chunk_samples = 0


_interviewer_runtime: Optional[_InterviewerRuntime] = None
_last_interviewer_runtime_snapshot: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# ASR merge / question grouping
# ---------------------------------------------------------------------------

def _reset_asr_merge_buffer_locked():
    _asr_state.reset_merge_buffer()


def _reset_asr_merge_buffer():
    with _asr_state_lock:
        _reset_asr_merge_buffer_locked()


def _reset_pending_asr_group_locked():
    _asr_state.reset_pending_group()


def _reset_pending_asr_group():
    with _asr_state_lock:
        _reset_pending_asr_group_locked()


def _asr_confirm_window_sec(cfg) -> float:
    confirm = float(getattr(cfg, "assist_asr_confirm_window_sec", 0.45) or 0.0)
    return max(0.0, min(5.0, confirm))


def _asr_group_max_wait_sec(cfg) -> float:
    max_wait = float(getattr(cfg, "assist_asr_group_max_wait_sec", 1.2) or 0.0)
    return max(0.2, min(8.0, max_wait))


def _asr_interrupt_running(cfg) -> bool:
    return asr_interrupt_running(cfg)


def _task_meta(task: TaskPayload) -> dict[str, Any]:
    return task_meta(task)


def _is_asr_task(task: TaskPayload) -> bool:
    return is_asr_task(task)


def _get_latest_asr_turn_id() -> int:
    with _dispatch_lock:
        return _latest_asr_turn_id


def _is_stale_inflight_asr_task(task: TaskPayload) -> bool:
    return is_stale_inflight_asr_task(task, _latest_asr_turn_id)


def _assist_vad_min_speech_sec(cfg) -> float:
    return max(0.1, float(getattr(cfg, "assist_vad_min_speech_sec", 0.3) or 0.3))


def _assist_vad_min_samples(cfg) -> int:
    return int(AudioCapture.SAMPLE_RATE * _assist_vad_min_speech_sec(cfg))


def _new_interviewer_runtime() -> _InterviewerRuntime:
    return _InterviewerRuntime()


def _set_interviewer_runtime(runtime: Optional[_InterviewerRuntime]) -> None:
    global _interviewer_runtime
    _interviewer_runtime = runtime


def _set_last_interviewer_runtime_snapshot(snapshot: Optional[dict[str, Any]]) -> None:
    global _last_interviewer_runtime_snapshot
    _last_interviewer_runtime_snapshot = dict(snapshot) if snapshot is not None else None


def _snapshot_interviewer_runtime(
    runtime: _InterviewerRuntime,
    *,
    live: bool,
    drained: Optional[bool] = None,
    remaining_after_drain: Optional[int] = None,
) -> dict[str, Any]:
    snapshot = {
        "live": bool(live),
        "raw_queue_drop_count": int(runtime.raw_queue_drop_count),
        "loopback_discontinuity_count": int(runtime.loopback_discontinuity_count),
        "segment_emitted": int(runtime.segment_emitted),
        "segment_dropped": int(runtime.segment_dropped),
        "segment_queue_depth": int(runtime.segment_queue.qsize()),
        "segment_backlog": int(_interviewer_segment_backlog(runtime)),
        "max_observed_segment_queue_depth": int(runtime.max_observed_segment_queue_depth),
        "max_raw_chunk_samples": int(runtime.max_raw_chunk_samples),
    }
    if drained is not None:
        snapshot["drained"] = bool(drained)
        snapshot["drain_timed_out"] = not bool(drained)
    if remaining_after_drain is not None:
        snapshot["remaining_after_drain"] = int(remaining_after_drain)
    return snapshot


def get_interviewer_runtime_snapshot() -> Optional[dict[str, Any]]:
    runtime = _interviewer_runtime
    if runtime is not None:
        return _snapshot_interviewer_runtime(runtime, live=True)
    if _last_interviewer_runtime_snapshot is None:
        return None
    return dict(_last_interviewer_runtime_snapshot)


def _log_interviewer_chunk_gap(
    runtime: _InterviewerRuntime,
    batch_started_mono: float,
    batch_audio_sec: float,
) -> None:
    if runtime.last_chunk_mono is not None:
        expected_next_started = runtime.last_chunk_mono + max(0.0, runtime.last_chunk_audio_sec)
        gap_ms = (batch_started_mono - expected_next_started) * 1000.0
        if gap_ms >= 250:
            _ilog.warning("INTERVIEWER_CHUNK_GAP gap_ms=%.0f", gap_ms)
    runtime.last_chunk_mono = batch_started_mono
    runtime.last_chunk_audio_sec = max(0.0, float(batch_audio_sec or 0.0))


def _refresh_interviewer_raw_drop_count(runtime: _InterviewerRuntime, audio_input=None) -> None:
    audio_input = audio_capture if audio_input is None else audio_input
    snapshot_fn = getattr(audio_input, "capture_stats_snapshot", None)
    if callable(snapshot_fn):
        try:
            capture_stats = snapshot_fn()
        except Exception:
            capture_stats = None
        if isinstance(capture_stats, dict):
            runtime.raw_queue_drop_count = int(
                capture_stats.get("dropped_chunks_count", runtime.raw_queue_drop_count)
            )
            runtime.loopback_discontinuity_count = int(
                capture_stats.get("loopback_discontinuity_count", runtime.loopback_discontinuity_count)
            )
            runtime.max_raw_chunk_samples = int(
                capture_stats.get("max_queue_chunk_samples", runtime.max_raw_chunk_samples)
            )
            return
    dropped = getattr(audio_input, "dropped_chunks_count", None)
    if dropped is None:
        return
    runtime.raw_queue_drop_count = int(dropped)


def _interviewer_segment_backlog(runtime: _InterviewerRuntime) -> int:
    unfinished = getattr(runtime.segment_queue, "unfinished_tasks", None)
    if unfinished is None:
        return runtime.segment_queue.qsize()
    return max(0, int(unfinished))


def _emit_interviewer_segment(
    runtime: _InterviewerRuntime,
    segment: InterviewerSegment,
) -> None:
    try:
        runtime.segment_queue.put_nowait(segment)
        runtime.segment_emitted += 1
        depth = runtime.segment_queue.qsize()
        runtime.max_observed_segment_queue_depth = max(
            runtime.max_observed_segment_queue_depth, depth
        )
        _ilog.info(
            "INTERVIEWER_SEGMENT_EMITTED flush_reason=%s audio_sec=%.2f queue_depth=%d",
            segment.flush_reason,
            segment.audio_sec,
            depth,
        )
    except queue.Full:
        dropped = None
        try:
            dropped = runtime.segment_queue.get_nowait()
            runtime.segment_queue.task_done()
        except queue.Empty:
            dropped = None
        if dropped is not None:
            runtime.segment_dropped += 1
            get_session().discard_unfinished_transcript(dropped.transcript_reservation)
        try:
            runtime.segment_queue.put_nowait(segment)
            runtime.segment_emitted += 1
            depth = runtime.segment_queue.qsize()
            runtime.max_observed_segment_queue_depth = max(
                runtime.max_observed_segment_queue_depth, depth
            )
        except queue.Full:
            runtime.segment_dropped += 1
            get_session().discard_unfinished_transcript(segment.transcript_reservation)
            depth = runtime.segment_queue.qsize()
        _elog.warning(
            "INTERVIEWER_SEGMENT_QUEUE_FULL flush_reason=%s audio_sec=%.2f dropped_total=%d queue_depth=%d",
            segment.flush_reason,
            segment.audio_sec,
            runtime.segment_dropped,
            depth,
        )


def _maybe_build_interviewer_segment(
    cfg,
    runtime: _InterviewerRuntime,
    session,
    speech_audio,
    *,
    flush_reason: Literal["silence", "max_speech", "pause_flush", "stop_flush", "role_switch"],
    segment_started_mono: float,
    segment_ended_mono: float,
) -> bool:
    if speech_audio is None:
        return False
    min_interviewer_samples = _assist_vad_min_samples(cfg)
    if len(speech_audio) < min_interviewer_samples:
        return False
    audio_sec = len(speech_audio) / AudioCapture.SAMPLE_RATE
    ended_mono = max(0.0, float(segment_ended_mono))
    started_mono = max(0.0, min(float(segment_started_mono), ended_mono))
    if ended_mono - started_mono + 1e-6 < audio_sec:
        started_mono = max(0.0, ended_mono - audio_sec)
    transcript_key = uuid4().hex
    transcript_reservation = session.record_transcript(
        "interviewer", "", started_at=started_mono, utterance_key=transcript_key,
        is_final=False, epoch=runtime.transcript_epoch,
    )
    if not transcript_reservation:
        return False
    segment = InterviewerSegment(
        audio=speech_audio,
        sample_rate=AudioCapture.SAMPLE_RATE,
        started_mono=started_mono,
        ended_mono=ended_mono,
        audio_sec=audio_sec,
        flush_reason=flush_reason,
        capture_is_loopback=bool(getattr(session, "capture_is_loopback", False)),
        transcript_key=transcript_key,
        transcript_reservation=transcript_reservation,
    )
    _emit_interviewer_segment(runtime, segment)
    return True


def _drain_interviewer_segments(
    timeout_sec: float,
    runtime: Optional[_InterviewerRuntime] = None,
) -> tuple[bool, int]:
    target = _interviewer_runtime if runtime is None else runtime
    if target is None:
        return True, 0
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while time.monotonic() < deadline:
        if _interviewer_segment_backlog(target) == 0:
            return True, 0
        time.sleep(0.02)
    remaining = _interviewer_segment_backlog(target)
    return remaining == 0, remaining


def _interviewer_segment_drain_timeout() -> float:
    return max(
        0.5,
        min(
            30.0,
            float(
                getattr(
                    get_config(),
                    "assist_interviewer_asr_drain_timeout_sec",
                    _interviewer_segment_drain_timeout_sec,
                )
                or _interviewer_segment_drain_timeout_sec
            ),
        ),
    )


def _stop_capture_compat(capture, *, owner: str, clear_queue: bool) -> None:
    try:
        capture.stop(owner=owner, clear_queue=clear_queue)
    except TypeError:
        capture.stop(owner=owner)


def _start_interview_worker_thread_if_needed() -> None:
    global _interview_thread
    if _interview_thread and _interview_thread.is_alive():
        return
    session = get_session()
    args = (_interviewer_config, session.transcript_epoch)
    if session.single_mic is not None:
        args += (session.single_mic,)
    _interview_thread = threading.Thread(target=_interview_worker, args=args, daemon=True)
    _interview_thread.start()


def _drain_remaining_interviewer_audio_chunks(
    runtime: _InterviewerRuntime,
    vad: VADBuffer,
    cfg,
    session,
    audio_input=None,
) -> None:
    audio_input = audio_capture if audio_input is None else audio_input
    while True:
        chunks = audio_input.drain_audio_chunks(timeout=0.0, max_chunks=6)
        if not chunks:
            break
        now = time.monotonic()
        offset_samples = 0
        total_samples = sum(len(chunk) for chunk in chunks)
        batch_end_mono = now
        batch_start_mono = max(0.0, batch_end_mono - (total_samples / AudioCapture.SAMPLE_RATE))
        _log_interviewer_chunk_gap(
            runtime,
            batch_start_mono,
            total_samples / AudioCapture.SAMPLE_RATE,
        )
        for chunk in chunks:
            offset_samples += len(chunk)
            chunk_ended_mono = batch_start_mono + (offset_samples / AudioCapture.SAMPLE_RATE)
            speech_audio = vad.feed(chunk)
            if speech_audio is None:
                continue
            flush_reason = getattr(vad, "last_flush_reason", None) or "silence"
            segment_ended_mono = chunk_ended_mono
            segment_started_mono = max(
                0.0,
                segment_ended_mono - (len(speech_audio) / AudioCapture.SAMPLE_RATE),
            )
            _maybe_build_interviewer_segment(
                cfg,
                runtime,
                session,
                speech_audio,
                flush_reason=(
                    "max_speech"
                    if flush_reason == "max_speech"
                    else "silence"
                ),
                segment_started_mono=segment_started_mono,
                segment_ended_mono=segment_ended_mono,
            )


def _iter_vad_feed_chunks(audio_chunk: np.ndarray):
    frame = max(1, int(_vad_feed_chunk_samples))
    total = len(audio_chunk)
    for start in range(0, total, frame):
        yield audio_chunk[start:start + frame]


# ---------------------------------------------------------------------------
# Generation control
# ---------------------------------------------------------------------------

def _bump_generation():
    global _answer_generation
    with _gen_lock:
        _answer_generation += 1


def _capture_generation() -> int:
    with _gen_lock:
        return _answer_generation


def _reset_answer_state(reset_session_data: bool = False, *, invalidate_capture: bool = False):
    global _pending, _in_flight_tasks, _commit_buffer, _skipped_commit_seqs, _next_commit_seq, _task_session_version, _latest_asr_turn_id
    with _asr_state_lock:
        with _dispatch_lock:
            tasks = [item[0] for item in _pending] + [item[1] for item in _in_flight_tasks.values()]
            _pending.clear()
            _in_flight_tasks.clear()
            _latest_asr_turn_id = 0
            _task_session_version += 1
            next_commit_seq = _next_submit_seq
            _dispatch_changed.notify_all()
        _reset_asr_merge_buffer_locked()
        _reset_pending_asr_group_locked()
        if reset_session_data:
            reset_session()
        elif invalidate_capture:
            with conversation_lock:
                get_session().stop_transcript_capture()
    for task in tasks:
        if request := task_meta(task).get("mmpg_request"):
            _cancel_mmpg_request(request, reason="stopped")
    # commit 相关操作单独处理
    with _commit_lock:
        discarded = [_commit_buffer.pop(seq) for seq in list(_commit_buffer) if seq < next_commit_seq]
        _skipped_commit_seqs.difference_update({seq for seq in _skipped_commit_seqs if seq < next_commit_seq})
        _next_commit_seq = max(_next_commit_seq, next_commit_seq)
        _drain_commit_queue_locked()
    for commit in discarded:
        if isinstance(commit, AnswerCommit):
            commit.cancel()


def cancel_answer_work(reset_session_data: bool = False, *, invalidate_capture: bool = False):
    _bump_generation()
    _reset_answer_state(reset_session_data, invalidate_capture=invalidate_capture)


def _is_exam_preflight_cancelled(preflight_id: str) -> bool:
    if not preflight_id:
        return False
    with _exam_cancel_lock:
        return preflight_id in _cancelled_exam_preflight_ids


def cancel_exam_preflight_task(preflight_id: str) -> bool:
    """Cancel a timed-out or obsolete exam preflight task and release its slot & commit seq."""
    target = (preflight_id or "").strip()
    if not target:
        return False
    with _exam_cancel_lock:
        if target not in _cancelled_exam_preflight_ids:
            _cancelled_exam_preflight_ids.append(target)
    skipped_seqs: list[int] = []
    with _dispatch_lock:
        kept: list[tuple[TaskPayload, int, int]] = []
        for task, seq, tv in _pending:
            if str(task_meta(task).get("exam_preflight_id") or "") == target:
                skipped_seqs.append(seq)
                continue
            kept.append((task, seq, tv))
        _pending[:] = kept
        for seq, (_reservation, task) in list(_in_flight_tasks.items()):
            if str(task_meta(task).get("exam_preflight_id") or "") == target:
                _in_flight_tasks.pop(seq, None)
                skipped_seqs.append(seq)
        _dispatch_changed.notify_all()
    for seq in skipped_seqs:
        _mark_seq_skipped(seq)
    if skipped_seqs:
        _try_dispatch()
    return bool(skipped_seqs)


def init_background_workers():
    global _knowledge_worker
    if _knowledge_worker is None:
        _knowledge_worker = BoundedTaskWorker(
            "assist.knowledge_worker",
            _save_knowledge_record,
            maxsize=64,
        )
    _knowledge_worker.start()


def shutdown_background_workers():
    global _knowledge_worker
    if _knowledge_worker is None:
        return
    _knowledge_worker.stop()


def _submit_knowledge_record(
    question: str,
    answer: str,
    qa_id: str = "",
    candidate_answer: str = "",
) -> bool:
    worker = _knowledge_worker
    if worker is None:
        _save_knowledge_record("save", question, answer, qa_id, candidate_answer)
        return True
    return worker.submit("save", question, answer, qa_id, candidate_answer)


def _submit_candidate_knowledge_update(qa_id: str, candidate_answer: str) -> bool:
    if not (qa_id or "").strip() or not (candidate_answer or "").strip():
        return False
    worker = _knowledge_worker
    if worker is None:
        _save_knowledge_record("candidate_update", qa_id, candidate_answer)
        return True
    return worker.submit("candidate_update", qa_id, candidate_answer)


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

def _key_ok(m) -> bool:
    return key_ok(m)


def _model_eligible(i: int, m, need_vision: bool) -> bool:
    return scheduler_model_eligible(i, m, need_vision, get_model_health)


def _prompt_mode_for_task(source: str, manual_input: bool, written_exam: bool = False):
    return answer_prompt_mode_for_task(source, manual_input, written_exam=written_exam)


def _priority_model_index(cfg) -> int:
    return scheduler_priority_model_index(cfg)


def _dispatch_model_order(cfg) -> list[int]:
    return scheduler_dispatch_model_order(cfg)


def _dispatch_snapshot_locked() -> tuple[set[ModelReservation], int]:
    return scheduler_dispatch_snapshot(_in_flight_tasks, _latest_asr_turn_id)


def _physical_busy_models_locked() -> set[ModelReservation]:
    return scheduler_physical_busy_models(_in_flight_tasks)


def _drain_commit_queue_locked():
    global _next_commit_seq
    _next_commit_seq = drain_commit_queue(
        _commit_buffer,
        _skipped_commit_seqs,
        _next_commit_seq,
    )


def _mark_seq_skipped(seq: int):
    with _commit_lock:
        if seq < _next_commit_seq:
            return
        discarded = _commit_buffer.pop(seq, None)
        _skipped_commit_seqs.add(seq)
        _drain_commit_queue_locked()
    if isinstance(discarded, AnswerCommit):
        discarded.cancel()


def _answer_work_idle() -> bool:
    with _dispatch_lock:
        pending = bool(_pending)
        in_flight = bool(_in_flight_tasks)
    with _commit_lock:
        commits = bool(_commit_buffer)
    return not pending and not in_flight and not commits


def _wait_for_answer_work_idle(timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while time.monotonic() < deadline:
        if _answer_work_idle():
            return True
        time.sleep(0.05)
    return _answer_work_idle()


def _begin_asr_turn() -> int:
    global _latest_asr_turn_id
    cfg = get_config()
    with _dispatch_lock:
        _latest_asr_turn_id, skipped = scheduler_begin_asr_turn(
            _pending,
            _latest_asr_turn_id,
            interrupt_pending_asr=_should_interrupt_stale_asr(cfg),
        )
        turn_id = _latest_asr_turn_id
        _dispatch_changed.notify_all()
    for seq in skipped:
        _mark_seq_skipped(seq)
    return turn_id


def pick_model_index(
    task: TaskPayload,
    busy: set[ModelReservation],
    avoid_models: Optional[set[ModelReservation]] = None,
    *,
    config_snapshot=None,
) -> Optional[int]:
    return scheduler_pick_model_index(
        task,
        busy,
        config_snapshot if config_snapshot is not None else get_config(),
        get_model_health,
        avoid_models=avoid_models,
    )


def _max_parallel_slots() -> int:
    return scheduler_max_parallel_slots(get_config(), get_model_health)


def _should_interrupt_stale_asr(cfg=None) -> bool:
    cfg = cfg or get_config()
    return bool(
        asr_interrupt_running(cfg)
        and scheduler_max_parallel_slots(cfg, get_model_health, include_screenshots=False) <= 1
    )


def submit_answer_task(task: TaskPayload, *, config_snapshot=None) -> bool:
    global _next_submit_seq
    cfg = config_snapshot if config_snapshot is not None else get_config()
    written_exam = task_is_written_exam(task, cfg)
    model_index = pick_model_index(task, set(), config_snapshot=cfg)
    if model_index is None:
        broadcast(
            {
                "type": "error",
                "message": EXAM_MODEL_UNAVAILABLE if written_exam else (
                    "没有可用的答题模型，请在实时辅助中选择已配置且支持该题型的模型。"
                ),
            }
        )
        return False
    routing = AnswerRouting(written_exam=False)
    if written_exam:
        routing = AnswerRouting(
            written_exam=True,
            exam_model_ids=tuple(cfg.models[index].id for index in exam_model_order(cfg)),
            exam_config=copy.deepcopy(cfg),
        )
    text, image, manual, source, meta = task
    task = (text, image, manual, source, {**meta, "answer_routing": routing})
    request = task_meta(task).get("mmpg_request")
    with _dispatch_lock:
        seq = _next_submit_seq
        _next_submit_seq += 1
        tv = _task_session_version
        if request:
            request.seq = seq
        _pending.append((task, seq, tv))
        delay = max(0.0, float(_task_meta(task).get("dispatch_after_mono", 0.0) or 0.0) - time.monotonic())
    if trace := _task_meta(task).get("mmpg_trace"):
        if request:
            get_logger("mmpg.diagnostics").info(
                "turn=%s group=%s revision=%s event=answer_queued seq=%s since_final_ms=%.1f",
                trace, request.group_id, request.revision, seq, (time.monotonic() - request.final_at) * 1000,
            )
        else:
            get_logger("mmpg.diagnostics").info("turn=%s event=answer_queued seq=%s", trace, seq)
    _try_dispatch()
    if delay > 0:
        timer = threading.Timer(delay, _try_dispatch)
        timer.daemon = True
        timer.start()
    return True


def _cancel_mmpg_request(request, *, reason):
    with request.lock:
        if request.cancel():
            get_logger("mmpg.diagnostics").info(
                "turn=%s group=%s revision=%s event=request_cancelled reason=%s",
                request.traces[-1], request.group_id, request.revision, reason,
            )
            if request.shown_qa_id:
                with conversation_lock:
                    session = get_session()
                    if session.current_candidate_qa_id == request.shown_qa_id:
                        session.close_candidate_answer_window()
                broadcast({"type": "answer_cancelled", "id": request.shown_qa_id, "reason": reason})
            return True
    return False


def cancel_mmpg_answer(request, *, reason="continuation"):
    if not _cancel_mmpg_request(request, reason=reason):
        return request.completed_qa_id
    with _dispatch_lock:
        for index, (task, _seq, _version) in enumerate(_pending):
            if task_meta(task).get("mmpg_request") is request:
                _pending.pop(index)
                break
    if request.seq is not None:
        _mark_seq_skipped(request.seq)
    _try_dispatch()
    return None


# ---------------------------------------------------------------------------
# ASR question group handling
# ---------------------------------------------------------------------------

def _flush_asr_question_group_now(cfg, session, *, immediate: bool = False) -> None:
    with _asr_state_lock:
        _asr_state.flush_question_group_now(cfg, session, immediate=immediate)


def _try_flush_asr_question_group(cfg, session, now_mono: float, force: bool = False) -> None:
    with _asr_state_lock:
        _asr_state.try_flush_question_group(cfg, session, now_mono, force)


def _handle_auto_detect_asr_text(cfg, session, pub: str, source: str, now_mono: float) -> None:
    with _asr_state_lock:
        _asr_state.handle_auto_detect_asr_text(cfg, session, pub, source, now_mono)


def _append_late_asr_constraint_tail(
    text: str, source: str, now_mono: float, transcript_anchor: str | None = None,
) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    with _dispatch_lock:
        for idx in range(len(_pending) - 1, -1, -1):
            task, seq, session_version = _pending[idx]
            question, image, manual_input, task_source, meta = task
            if not _is_asr_task(task) or task_source != source:
                continue
            grace_until = float(meta.get("asr_tail_grace_until_mono", 0.0) or 0.0)
            if grace_until and now_mono > grace_until:
                continue
            utterances = [str(item).strip() for item in (meta.get("utterances") or []) if str(item).strip()]
            if not utterances:
                utterances = [str(question or "").strip()]
            if cleaned in utterances:
                return True
            utterances.append(cleaned)
            updated_question = build_asr_question_group_text(utterances) or f"{question}\n{cleaned}".strip()
            updated_meta = {**meta, "utterances": utterances, "late_constraint_tail": True}
            if transcript_anchor:
                updated_meta["transcript_anchor"] = transcript_anchor
            _pending[idx] = (
                (updated_question, image, manual_input, task_source, updated_meta),
                seq,
                session_version,
            )
            _ilog.info(
                "ASR_LATE_CONSTRAINT_MERGED seq=%d text=%r question=%r",
                seq,
                cleaned[:80],
                updated_question[:150],
            )
            return True
    return False


# ---------------------------------------------------------------------------
# ASR merge buffer
# ---------------------------------------------------------------------------

def _flush_asr_merge_buffer_now(cfg, session) -> None:
    with _asr_state_lock:
        _asr_state.flush_merge_buffer_now(cfg, session)


def _try_flush_asr_merge_buffer(cfg, session, now_mono: float, force: bool = False) -> None:
    with _asr_state_lock:
        _asr_state.try_flush_merge_buffer(cfg, session, now_mono, force)


def _append_transcription_fragment(
    cfg, session, pub: str, now_mono: float, force_flush_tail: bool = False, **transcript_metadata,
) -> None:
    with _asr_state_lock:
        _asr_state.append_transcription_fragment(cfg, session, pub, now_mono, force_flush_tail, **transcript_metadata)


# ---------------------------------------------------------------------------
# Dispatch / worker
# ---------------------------------------------------------------------------

def _try_dispatch():
    while True:
        # Keep model selection and the model object passed to the worker tied
        # to one immutable configuration snapshot. Settings updates replace
        # the global config while a worker may still be waiting to start.
        cfg = get_config()

        def _pick_from_snapshot(
            task: TaskPayload,
            busy: set[ModelReservation],
            avoid_models: Optional[set[ModelReservation]] = None,
            cfg_snapshot=cfg,
        ) -> Optional[int]:
            return scheduler_pick_model_index(
                task,
                busy,
                cfg_snapshot,
                get_model_health,
                avoid_models=avoid_models,
            )

        with _dispatch_lock:
            practice_pending = any(task_meta(task).get("purpose") == "knowledge_practice" for task, _, _ in _pending)
            step = claim_next_dispatch(
                _pending,
                _in_flight_tasks,
                _latest_asr_turn_id,
                scheduler_max_parallel_slots(cfg, get_model_health, include_practice=practice_pending),
                _pick_from_snapshot,
                interrupt_stale_asr=_should_interrupt_stale_asr(cfg),
                now_mono=time.monotonic(),
                reserve_model=lambda index, snapshot=cfg: model_reservation(snapshot, index),
                discard_unavailable=True,
            )
            model_cfg_snapshot = None
            config_snapshot = None
            worker_model_idx = step.claim.model_idx if step.claim is not None else None
            if step.claim is not None:
                selected_idx = step.claim.model_idx
                if 0 <= selected_idx < len(cfg.models):
                    # Keep model selection and every prompt/generation option
                    # tied to the same immutable configuration generation.
                    # update_config() replaces the global object while a
                    # worker may be queued; copying only the model used to
                    # let written-exam/KB/token settings drift underneath it.
                    model_cfg_snapshot = copy.deepcopy(cfg.models[selected_idx])
                    routing = task_meta(step.claim.task).get("answer_routing")
                    if routing is not None and routing.written_exam:
                        config_snapshot = routing.exam_config
                        worker_model_idx = eligible_exam_model_index(config_snapshot, model_cfg_snapshot.id)
                    else:
                        config_snapshot = copy.deepcopy(cfg)
        if step.skipped_seq is not None:
            _mark_seq_skipped(step.skipped_seq)
            if step.unavailable_task is not None:
                unavailable_task = step.unavailable_task
                message = EXAM_MODEL_STALE if task_is_written_exam(unavailable_task, cfg) else (
                    "选中的答题模型已不可用，请在实时辅助中重新选择。"
                )
                if preflight_id := task_meta(unavailable_task).get("exam_preflight_id"):
                    from .exam_test import record_exam_preflight_answer_event

                    record_exam_preflight_answer_event({
                        "type": "answer_error", "exam_preflight_id": preflight_id, "message": message,
                    })
                else:
                    broadcast({"type": "error", "message": message})
            continue
        if step.claim is None:
            return
        assert worker_model_idx is not None
        threading.Thread(
            target=_run_answer_worker,
            args=(
                step.claim.task,
                step.claim.seq,
                worker_model_idx,
                step.claim.session_version,
                model_cfg_snapshot,
                config_snapshot,
            ),
            daemon=True,
        ).start()


def _run_answer_worker(
    task: TaskPayload,
    seq: int,
    model_idx: int,
    sess_v: int,
    model_cfg_snapshot=None,
    config_snapshot=None,
):
    try:
        _process_question_parallel(
            task,
            seq,
            model_idx,
            sess_v,
            model_cfg_snapshot,
            config_snapshot,
        )
    finally:
        with _dispatch_changed:
            _in_flight_tasks.pop(seq, None)
            _dispatch_changed.notify_all()
        request = task_meta(task).get("mmpg_request")
        if request and request.on_finished:
            request.on_finished()
        _try_dispatch()


def _reserve_answer_retry(
    task: TaskPayload, seq: int, cfg, attempted: set[int], aborted: Callable[[], bool],
    *, on_wait: Callable[[], None] | None = None,
) -> int | None:
    routing = task_meta(task).get("answer_routing")

    def pick(busy):
        if routing is not None and routing.written_exam:
            current = get_config()
            for model_id in routing.exam_model_ids:
                index = eligible_exam_model_index(cfg, model_id)
                if index is None or index in attempted or model_id in busy:
                    continue
                if live_exam_model_index(routing, current, model_id) is not None:
                    return index
            return None
        return scheduler_pick_model_index(task, busy, cfg, lambda _index: None, excluded_indices=attempted)

    with _dispatch_changed:
        if seq not in _in_flight_tasks:
            return None
        # Keep the same question/commit slot and release its failed model so
        # two concurrent questions can each try the other's available model.
        _in_flight_tasks[seq] = (None, task)
        _dispatch_changed.notify_all()
    announced_wait = False
    while not aborted():
        with _dispatch_changed:
            if seq not in _in_flight_tasks:
                return None
            busy = scheduler_physical_busy_models(_in_flight_tasks)
            index = pick(busy)
            if index is not None:
                _in_flight_tasks[seq] = (model_reservation(cfg, index), task)
                return index
            if not busy or pick(set()) is None:
                return None
            if announced_wait or on_wait is None:
                _dispatch_changed.wait(timeout=0.1)
                continue
        on_wait()
        announced_wait = True
    return None


def _flush_commit(seq: int, apply_fn: Callable[[], None]):
    with _commit_lock:
        if seq >= _next_commit_seq and seq not in _skipped_commit_seqs:
            _commit_buffer[seq] = apply_fn
            _drain_commit_queue_locked()
            return
    if isinstance(apply_fn, AnswerCommit):
        apply_fn.cancel()


# ---------------------------------------------------------------------------
# Interview loop
# ---------------------------------------------------------------------------

# PortAudio device indices are 0-based. Never reuse 0 as "unset".
_UNSET_DEVICE_ID = -1


def _device_is_loopback(device_id: Optional[int]) -> bool:
    if device_id is None:
        return False
    if _is_sc_id(device_id):
        return True
    for d in AudioCapture.list_devices():
        if d["id"] == device_id:
            return bool(d["is_loopback"])
    return False


def _normalize_stored_device_id(value: Any, *, fallback: int = _UNSET_DEVICE_ID) -> int:
    """Coerce session/API device ids; negative or invalid → unset sentinel."""
    if value is None:
        return fallback
    try:
        device_id = int(value)
    except (TypeError, ValueError):
        return fallback
    return device_id if device_id >= 0 else fallback


def _is_valid_device_id(value: Any) -> bool:
    try:
        return value is not None and int(value) >= 0
    except (TypeError, ValueError):
        return False


def get_capture_config():
    """Use the session's ASR selection and credentials until that session ends."""
    return _interviewer_config if _interviewer_config is not None else get_config()


def candidate_capture_provider() -> str:
    cfg = _interviewer_config
    return cfg.candidate_stt_provider if cfg is not None and getattr(cfg, "candidate_asr_enabled", False) else ""


def discard_pending_auto_answer() -> None:
    with _asr_state_lock:
        _asr_state.reset_pending_group()


def report_stt_error(provider: str, role: Literal["interviewer", "candidate"], message: str) -> None:
    _elog.warning("ASR failed provider=%s role=%s: %s", provider, role, message)
    broadcast({"type": "stt_error", "provider": provider, "role": role, "message": message})


def validate_single_mic_start(device_id: int, cfg) -> None:
    if bool(getattr(cfg, "written_exam_mode", False)):
        raise SingleMicConflict("单人测试适用于语音面试")
    if not bool(getattr(cfg, "candidate_asr_enabled", False)):
        raise SingleMicConflict("请先在实时辅助中开启「我的回答」")
    if detail := stt_configuration_error(cfg, cfg.candidate_stt_provider):
        raise SingleMicConflict(detail)
    if not _is_valid_device_id(device_id) or _device_is_loopback(device_id):
        raise SingleMicConflict("请选择麦克风作为单人测试的音频输入")
    if (getattr(cfg, "stt_provider", None) == MMPG_STT_PROVIDER
            and getattr(cfg, "candidate_stt_provider", None) == MMPG_STT_PROVIDER
            and cfg.mmpg.token_mode == "manual"):
        raise SingleMicConflict("两路豆包 MMPG 同时使用时需要自动 Token")


def validate_single_mic_resume(single_mic: SingleMicSession, cfg) -> None:
    current_streams = {
        role for role, provider in (
            ("interviewer", cfg.stt_provider), ("candidate", cfg.candidate_stt_provider),
        ) if provider == MMPG_STT_PROVIDER
    }
    if cfg.written_exam_mode or not cfg.candidate_asr_enabled or current_streams != single_mic.required_streams:
        raise SingleMicConflict("语音配置已变更，请结束本次单人测试后重新开始")
    validate_single_mic_start(single_mic.device_id, cfg)


def broadcast_single_mic(single_mic: SingleMicSession | None) -> None:
    broadcast({"type": "single_mic", "value": single_mic.snapshot() if single_mic is not None else None})


def _open_single_mic(single_mic: SingleMicSession, cfg) -> None:
    try:
        audio_capture.start(
            single_mic.device_id, on_audio=single_mic.offer_audio, owner="assist",
            mic_compatibility_mode=bool(getattr(cfg, "candidate_mic_compatibility_mode", True)),
            queue_audio=False,
        )
    except Exception:
        _stop_capture_compat(audio_capture, owner="assist", clear_queue=True)
        raise


def drain_single_mic_worker(single_mic: SingleMicSession, operation: RoleSwitch, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    if not operation.wait(timeout):
        return False
    if operation.from_role == "candidate":
        return True
    session = get_session()
    while session.single_mic is single_mic and single_mic.owns_switch(operation):
        if time.monotonic() >= deadline:
            break
        runtime = _interviewer_runtime
        if runtime is not None and _interviewer_segment_backlog(runtime) == 0:
            with _asr_state_lock:
                if session.single_mic is not single_mic or not single_mic.owns_switch(operation):
                    return False
                cfg = _interviewer_config or get_config()
                _flush_asr_merge_buffer_now(cfg, session)
                _flush_asr_question_group_now(cfg, session, immediate=True)
                failures = runtime.segment_failures + runtime.segment_dropped
                success = failures == runtime.reported_role_failures
                runtime.reported_role_failures = failures
            return success
        if _stop_event.wait(0.02):
            break
    return False


def _start_candidate_capture(device_id: int, cfg, *, single_mic: SingleMicSession | None = None) -> None:
    global _candidate_thread
    if detail := stt_configuration_error(cfg, cfg.candidate_stt_provider):
        raise STTError(detail)
    if getattr(cfg, "candidate_stt_provider", None) == MMPG_STT_PROVIDER:
        # The independent RTC publisher owns this capture once its transport is ready.
        return
    if single_mic is None:
        _candidate_audio_capture.start(
            device_id,
            owner="assist-candidate",
            mic_compatibility_mode=bool(getattr(cfg, "candidate_mic_compatibility_mode", True)),
        )
    if not _candidate_thread or not _candidate_thread.is_alive():
        _candidate_thread = threading.Thread(
            target=_candidate_worker,
            args=(get_session().transcript_epoch, single_mic, cfg), daemon=True,
        )
        _candidate_thread.start()


def start_nonblocking(
    device_id: Optional[int] = None,
    candidate_mic_device_id: Optional[int] = None,
    *,
    stream_audio: Optional[Callable[[np.ndarray], None]] = None,
    config_snapshot=None,
    single_mic_id: str | None = None,
):
    global _interview_thread, _interviewer_streaming, _interviewer_config
    start_cfg = config_snapshot if config_snapshot is not None else get_config()
    if getattr(start_cfg, "written_exam_mode", False):
        device_id = None
        candidate_mic_device_id = None
    if device_id is not None:
        if detail := stt_configuration_error(start_cfg, start_cfg.stt_provider):
            raise STTError(detail)
    single_mic = None
    if single_mic_id is not None:
        validate_single_mic_start(device_id, start_cfg)
        required_streams = set()
        if stream_audio is not None:
            required_streams.add("interviewer")
        if getattr(start_cfg, "candidate_stt_provider", None) == MMPG_STT_PROVIDER:
            required_streams.add("candidate")
        single_mic = SingleMicSession(
            single_mic_id, device_id=device_id,
            required_streams=frozenset(required_streams),
        )
        candidate_mic_device_id = device_id
    stop_interview_loop()
    _set_last_interviewer_runtime_snapshot(None)
    _stop_event.clear()
    _pause_event.clear()
    _candidate_flush_event.clear()

    session = get_session()
    capture_is_loopback = bool(getattr(session, "capture_is_loopback", False))
    review_candidate_device_id: Optional[int] = None
    with conversation_lock:
        existing_qa_ids = frozenset(qa.id for qa in session.qa_pairs) if single_mic is not None else frozenset()

    if device_id is not None:
        capture_is_loopback = _device_is_loopback(device_id)

        if single_mic is not None:
            _open_single_mic(single_mic, start_cfg)
        elif stream_audio is None:
            audio_capture.start(device_id, owner="assist")
        else:
            audio_capture.start(device_id, on_audio=stream_audio, owner="assist", queue_audio=False)

        with conversation_lock:
            session.last_device_id = int(device_id)
            session.capture_is_loopback = capture_is_loopback
        _ilog.info("INTERVIEW_START device=%s loopback=%s", device_id, capture_is_loopback)
    else:
        with conversation_lock:
            session.last_device_id = _UNSET_DEVICE_ID
            session.last_candidate_mic_device_id = _UNSET_DEVICE_ID
            session.capture_is_loopback = False
        _ilog.info("INTERVIEW_START no_device (written_exam_mode)")

    _interviewer_streaming = stream_audio is not None and device_id is not None
    _interviewer_config = copy.deepcopy(start_cfg) if device_id is not None else None
    with _asr_state_lock, conversation_lock:
        session.begin_transcript_capture()
        session.single_mic = single_mic
        session.is_recording = True
        session.is_paused = False

    recording = {"type": "recording", "value": True, "candidate_capture_provider": candidate_capture_provider()}
    if _interviewer_streaming:
        recording["interviewer_streaming"] = True
    broadcast(recording)
    broadcast({"type": "paused", "value": False})

    if device_id is not None and not _interviewer_streaming:
        _start_interview_worker_thread_if_needed()

    cfg = get_capture_config()
    if (
        device_id is not None
        and candidate_mic_device_id is not None
        and bool(getattr(cfg, "candidate_asr_enabled", False))
        and (single_mic is not None or int(candidate_mic_device_id) != int(device_id))
    ):
        try:
            with conversation_lock:
                session.last_candidate_mic_device_id = int(candidate_mic_device_id)
            if single_mic is None:
                _start_candidate_capture(int(candidate_mic_device_id), cfg)
            else:
                _start_candidate_capture(int(candidate_mic_device_id), cfg, single_mic=single_mic)
            review_candidate_device_id = int(candidate_mic_device_id)
            _ilog.info("CANDIDATE_ASR_START device=%s", candidate_mic_device_id)
        except Exception as exc:
            if single_mic is not None:
                stop_interview_loop()
                raise
            with conversation_lock:
                session.last_candidate_mic_device_id = _UNSET_DEVICE_ID
            _elog.warning("CANDIDATE_ASR_START_FAIL device=%s err=%s", candidate_mic_device_id, exc)
            broadcast({
                "type": "candidate_asr_status",
                "loaded": False,
                "loading": False,
                "provider": "off",
                "error": str(exc)[:160],
                "reason": "mic_unavailable",
                "safe_degraded": True,
            })
    elif device_id is not None:
        with conversation_lock:
            session.last_candidate_mic_device_id = _UNSET_DEVICE_ID
        reason = "disabled" if not bool(getattr(cfg, "candidate_asr_enabled", False)) else "missing_or_same_device"
        broadcast({"type": "candidate_asr_status", "loaded": False, "loading": False, "provider": "off", "reason": reason})

    if single_mic is not None:
        broadcast_single_mic(single_mic)

    # 创建 review session（如果满足条件）
    review_integration.on_assist_start(
        interviewer_device_id=device_id,
        candidate_device_id=review_candidate_device_id,
        candidate_asr_enabled=bool(getattr(cfg, "candidate_asr_enabled", False)),
        written_exam_mode=not _interviewer_streaming and bool(getattr(cfg, "written_exam_mode", False)),
        **({
            "single_mic": True, "existing_qa_ids": existing_qa_ids,
        } if single_mic is not None else {}),
    )


def stop_interview_loop():
    global _interview_thread, _interviewer_asr_thread, _candidate_thread, _flush_thread, _interviewer_streaming
    global _interviewer_config
    session = get_session()
    single_mic = session.single_mic
    runtime = _interviewer_runtime
    if runtime is not None:
        runtime.stop_managed_by_caller = True
        runtime.flush_stop_event.set()
    _flush_stop_event.set()
    _stop_event.set()
    if single_mic is not None:
        single_mic.close()
    _pause_event.clear()
    _candidate_flush_event.set()
    _stop_capture_compat(audio_capture, owner="assist", clear_queue=False)
    _stop_capture_compat(_candidate_audio_capture, owner="assist-candidate", clear_queue=True)
    runtime_snapshot: Optional[dict[str, Any]] = None

    current_thread = threading.current_thread()
    interview_thread = _interview_thread
    _interview_thread = None
    if interview_thread and interview_thread.is_alive() and interview_thread is not current_thread:
        interview_thread.join(timeout=5)
    asr_thread = (runtime.asr_thread if runtime is not None else None) or _interviewer_asr_thread
    flush_thread = (runtime.flush_thread if runtime is not None else None) or _flush_thread
    if runtime is not None:
        try:
            _refresh_interviewer_raw_drop_count(runtime, single_mic.interviewer_input if single_mic else None)
            runtime.flush_stop_event.set()
            runtime.drain_event.set()
            drain_timeout_sec = _interviewer_segment_drain_timeout()
            drain_deadline = time.monotonic() + drain_timeout_sec
            if asr_thread and asr_thread.is_alive() and asr_thread is not current_thread:
                asr_thread.join(timeout=max(0.0, drain_deadline - time.monotonic()))
            else:
                _drain_interviewer_segments(max(0.0, drain_deadline - time.monotonic()), runtime=runtime)
            remaining = _interviewer_segment_backlog(runtime)
            drained = remaining == 0
            _ilog.info(
                "INTERVIEWER_ASR_DRAIN drained=%s remaining=%d segment_emitted=%d segment_dropped=%d raw_queue_drop_count=%d loopback_discontinuity_count=%d max_raw_chunk_samples=%d max_segment_queue_depth=%d",
                drained,
                remaining,
                runtime.segment_emitted,
                runtime.segment_dropped,
                runtime.raw_queue_drop_count,
                runtime.loopback_discontinuity_count,
                runtime.max_raw_chunk_samples,
                runtime.max_observed_segment_queue_depth,
            )
            if not drained:
                _elog.warning("INTERVIEWER_ASR_DRAIN_TIMEOUT remaining=%d", remaining)
            runtime_snapshot = _snapshot_interviewer_runtime(
                runtime,
                live=False,
                drained=drained,
                remaining_after_drain=remaining,
            )
        except Exception:
            _elog.error("interviewer ASR drain failed", exc_info=True)
        if runtime_snapshot is None:
            runtime_snapshot = _snapshot_interviewer_runtime(runtime, live=False)
        _set_last_interviewer_runtime_snapshot(runtime_snapshot)
        runtime.asr_thread = None
        runtime.flush_thread = None
    elif asr_thread and asr_thread.is_alive() and asr_thread is not current_thread:
        asr_thread.join(timeout=0.2)
    if _interviewer_asr_thread is asr_thread:
        _interviewer_asr_thread = None
    if flush_thread and flush_thread.is_alive() and flush_thread is not current_thread:
        flush_thread.join(timeout=1)
    if _flush_thread is flush_thread:
        _flush_thread = None
    candidate_thread = _candidate_thread
    _candidate_thread = None
    if candidate_thread and candidate_thread.is_alive() and candidate_thread is not current_thread:
        candidate_thread.join(timeout=5)

    if not _interviewer_streaming:
        try:
            cfg = get_capture_config()
            now = time.monotonic()
            _try_flush_asr_merge_buffer(cfg, session, now, True)
            _try_flush_asr_question_group(cfg, session, now, True)
        except Exception:
            _elog.debug("final ASR flush during stop skipped", exc_info=True)
    _interviewer_streaming = False
    _interviewer_config = None
    with _asr_state_lock, conversation_lock:
        session.stop_transcript_capture()

    wait_sec = float(getattr(get_config(), "assist_stop_answer_wait_sec", 3.0))
    if wait_sec > 0 and not _wait_for_answer_work_idle(min(wait_sec, 20.0)):
        _elog.warning("ANSWER_STOP_WAIT_TIMEOUT pending/inflight work will be cancelled")
    cancel_answer_work(reset_session_data=False)

    # 结束 review session（如果存在）。必须在音频 worker 最后 flush、候选人 ASR final、
    # 以及可等待的答案 commit 之后执行，否则关闭应用时复盘会漏掉末尾问题。
    review_integration.on_assist_stop(session)

    with conversation_lock:
        session.is_recording = False
        session.is_paused = False
        session.single_mic = None
        if hasattr(session, "close_candidate_answer_window"):
            session.close_candidate_answer_window()
        if len(session.transcription_history) > 30:
            session.transcription_history = session.transcription_history[-30:]
        if hasattr(session, "candidate_transcription_history") and len(session.candidate_transcription_history) > 30:
            session.candidate_transcription_history = session.candidate_transcription_history[-30:]
    broadcast({"type": "recording", "value": False, "candidate_capture_provider": ""})
    broadcast({"type": "paused", "value": False})
    if single_mic is not None:
        broadcast_single_mic(None)
    _candidate_flush_event.clear()
    if _interviewer_runtime is runtime:
        _set_interviewer_runtime(None)
    gc.collect()
    _ilog.info("INTERVIEW_STOP qa_count=%d", len(session.qa_pairs))


def _finalize_unexpected_interview_stop(
    session,
    transcript_epoch: int,
    error_message: str,
    *,
    single_mic: SingleMicSession | None = None,
    runtime: Optional[_InterviewerRuntime] = None,
) -> None:
    """Unified teardown when _interview_worker exits abnormally (device lost or crash)."""
    global _candidate_thread, _interview_thread, _interviewer_streaming, _interviewer_config
    with conversation_lock:
        if session.transcript_epoch != transcript_epoch:
            return
    _stop_event.set()
    _pause_event.clear()
    _candidate_flush_event.set()
    if single_mic is not None:
        single_mic.close()
    try:
        _stop_capture_compat(_candidate_audio_capture, owner="assist-candidate", clear_queue=False)
    except Exception:
        _elog.warning("candidate capture stop during unexpected exit failed", exc_info=True)
    current_thread = threading.current_thread()
    candidate_thread = _candidate_thread
    if candidate_thread and candidate_thread.is_alive() and candidate_thread is not current_thread:
        candidate_thread.join(timeout=3.0)
    if _candidate_thread is candidate_thread:
        _candidate_thread = None
    if _interview_thread is current_thread:
        _interview_thread = None

    if not _interviewer_streaming:
        try:
            cfg = get_capture_config()
            now = time.monotonic()
            _try_flush_asr_merge_buffer(cfg, session, now, True)
            _try_flush_asr_question_group(cfg, session, now, True)
        except Exception:
            _elog.debug("final ASR flush during unexpected stop skipped", exc_info=True)
    _interviewer_streaming = False
    _interviewer_config = None
    with _asr_state_lock, conversation_lock:
        if session.transcript_epoch != transcript_epoch:
            return
        session.stop_transcript_capture()

    wait_sec = float(getattr(get_config(), "assist_stop_answer_wait_sec", 3.0))
    if wait_sec > 0 and not _wait_for_answer_work_idle(min(wait_sec, 20.0)):
        _elog.warning("ANSWER_STOP_WAIT_TIMEOUT pending/inflight work will be cancelled")
    cancel_answer_work(reset_session_data=False)

    try:
        review_integration.on_assist_stop(session)
    except Exception:
        _elog.error("review_integration.on_assist_stop failed during unexpected stop", exc_info=True)

    with conversation_lock:
        session.is_recording = False
        session.is_paused = False
        session.single_mic = None
        if hasattr(session, "close_candidate_answer_window"):
            session.close_candidate_answer_window()
        if len(session.transcription_history) > 30:
            session.transcription_history = session.transcription_history[-30:]
        if hasattr(session, "candidate_transcription_history") and len(session.candidate_transcription_history) > 30:
            session.candidate_transcription_history = session.candidate_transcription_history[-30:]

    if runtime is not None:
        remaining = _interviewer_segment_backlog(runtime)
        _set_last_interviewer_runtime_snapshot(
            _snapshot_interviewer_runtime(
                runtime,
                live=False,
                drained=remaining == 0,
                remaining_after_drain=remaining,
            )
        )
        if _interviewer_runtime is runtime:
            _set_interviewer_runtime(None)
    _candidate_flush_event.clear()

    broadcast({"type": "error", "message": error_message})
    broadcast({"type": "recording", "value": False, "candidate_capture_provider": ""})
    broadcast({"type": "paused", "value": False})
    if single_mic is not None:
        broadcast_single_mic(None)


def pause_interview():
    session = get_session()
    _pause_event.set()
    if session.single_mic is not None:
        session.single_mic.pause()
        broadcast_single_mic(session.single_mic)
    _candidate_flush_event.set()
    _stop_capture_compat(audio_capture, owner="assist", clear_queue=False)
    _stop_capture_compat(_candidate_audio_capture, owner="assist-candidate", clear_queue=False)
    with conversation_lock:
        session.is_paused = True
    broadcast({"type": "paused", "value": True})


def unpause_interview(
    device_id: Optional[int] = None,
    candidate_mic_device_id: Optional[int] = None,
    *,
    stream_audio: Optional[Callable[[np.ndarray], None]] = None,
):
    global _interviewer_config
    session = get_session()
    cfg = get_capture_config()
    if (
        _interviewer_config is not None
        and getattr(cfg, "candidate_asr_enabled", False)
        and cfg.candidate_stt_provider == MMPG_STT_PROVIDER
        and cfg.mmpg.token_mode == "manual"
    ):
        refreshed_mmpg = refresh_manual_rtc_token(cfg.mmpg, get_config().mmpg)
        if refreshed_mmpg is not cfg.mmpg:
            cfg = _interviewer_config = cfg.model_copy(update={"mmpg": refreshed_mmpg})
    single_mic = session.single_mic
    if single_mic is not None:
        validate_single_mic_resume(single_mic, cfg)
        _open_single_mic(single_mic, cfg)
        single_mic.resume()
        if "interviewer" not in single_mic.required_streams:
            _start_interview_worker_thread_if_needed()
        if "candidate" not in single_mic.required_streams:
            _start_candidate_capture(single_mic.device_id, cfg, single_mic=single_mic)
        with conversation_lock:
            session.is_paused = False
        _pause_event.clear()
        broadcast({"type": "paused", "value": False})
        return
    written_exam = stream_audio is None and bool(getattr(cfg, "written_exam_mode", False))
    if written_exam:
        # A written exam has no audio workers. Ignore stale device IDs from a
        # previous interview and requests from older clients.
        device_id = None
        candidate_mic_device_id = None
    capture_is_loopback = bool(getattr(session, "capture_is_loopback", False))
    last_id = (
        _UNSET_DEVICE_ID
        if written_exam
        else _normalize_stored_device_id(getattr(session, "last_device_id", _UNSET_DEVICE_ID))
    )
    next_device_id = last_id
    if device_id is not None:
        next_device_id = int(device_id)
        capture_is_loopback = _device_is_loopback(next_device_id)
    # Mirror candidate resume: prefer last_device_id when the worker/capture died.
    # Valid capture targets are PortAudio indices >= 0 (including device 0).
    should_resume_interviewer = bool(
        not written_exam
        and _is_valid_device_id(next_device_id)
        and (
            device_id is not None
            or _is_valid_device_id(last_id)
            or (_interview_thread and _interview_thread.is_alive())
            or bool(getattr(audio_capture, "is_running", False))
        )
    )
    if should_resume_interviewer:
        if stream_audio is None:
            audio_capture.start(next_device_id, owner="assist")
            _start_interview_worker_thread_if_needed()
        else:
            audio_capture.start(next_device_id, on_audio=stream_audio, owner="assist", queue_audio=False)
    last_candidate_id = (
        _UNSET_DEVICE_ID
        if written_exam
        else _normalize_stored_device_id(
            getattr(session, "last_candidate_mic_device_id", _UNSET_DEVICE_ID)
        )
    )
    next_candidate_id = last_candidate_id
    should_resume_candidate = bool(
        candidate_mic_device_id is not None
        or _is_valid_device_id(last_candidate_id)
        or (_candidate_thread and _candidate_thread.is_alive())
        or bool(getattr(_candidate_audio_capture, "is_running", False))
    )
    if candidate_mic_device_id is not None:
        next_candidate_id = int(candidate_mic_device_id)
    candidate_enabled = bool(getattr(cfg, "candidate_asr_enabled", False))
    candidate_device_valid = bool(
        _is_valid_device_id(next_candidate_id)
        and candidate_enabled
        and int(next_candidate_id) != int(next_device_id)
    )
    if (
        should_resume_candidate
        and candidate_device_valid
    ):
        try:
            _start_candidate_capture(next_candidate_id, cfg)
        except Exception as exc:
            next_candidate_id = _UNSET_DEVICE_ID
            _elog.warning("CANDIDATE_ASR_RESUME_FAIL err=%s", exc)
            broadcast({
                "type": "candidate_asr_status",
                "loaded": False,
                "loading": False,
                "provider": "off",
                "error": str(exc)[:160],
                "reason": "mic_unavailable",
                "safe_degraded": True,
            })
    elif _is_valid_device_id(next_candidate_id) and not candidate_device_valid:
        reason = "disabled" if not candidate_enabled else "missing_or_same_device"
        next_candidate_id = _UNSET_DEVICE_ID
        if should_resume_candidate or candidate_mic_device_id is not None:
            broadcast({
                "type": "candidate_asr_status",
                "loaded": False,
                "loading": False,
                "provider": "off",
                "reason": reason,
            })
    _candidate_flush_event.clear()
    _pause_event.clear()
    with conversation_lock:
        session.last_device_id = (
            _UNSET_DEVICE_ID if written_exam else _normalize_stored_device_id(next_device_id)
        )
        session.last_candidate_mic_device_id = (
            _UNSET_DEVICE_ID if written_exam else _normalize_stored_device_id(next_candidate_id)
        )
        session.capture_is_loopback = False if written_exam else capture_is_loopback
        session.is_paused = False
    broadcast({"type": "paused", "value": False})


def is_paused() -> bool:
    return _pause_event.is_set()


def is_streaming_interview() -> bool:
    return _interviewer_streaming and get_session().is_recording


def publish_streaming_asr_turn(
    text: str, *, trace: str, started_at: float, question_group: str | None, transcript_epoch: int | None,
) -> str | None:
    session = get_session()
    with conversation_lock:
        if not text or not is_streaming_interview() or session.is_paused:
            return None
        anchor = session.record_transcript(
            "interviewer", text, started_at=started_at, utterance_key=trace, epoch=transcript_epoch,
            question_group=question_group if is_transcript_question(get_capture_config(), text) else None,
        )
        if not anchor:
            return None
        session.add_transcription(text)
    broadcast({"type": "transcription", "text": text})
    return anchor


def submit_mmpg_question(request, *, transcript_epoch: int | None = None) -> bool:
    session = get_session()
    trace = request.traces[-1]
    with _asr_state_lock:
        if (not is_streaming_interview() or session.is_paused or not request.transcript_anchor
                or transcript_epoch != session.transcript_epoch):
            return False
        cfg = get_capture_config()
        if is_transcript_question(cfg, request.text):
            session.group_transcript_question(request.transcript_anchors, split_after_reply=True)
        outcome = _asr_state.submit_final_question(cfg, session, request.text, trace, request)
    get_logger("mmpg.diagnostics").info(
        "turn=%s event=answer_decision outcome=%s", trace, outcome,
    )
    return outcome == "queued"


# H4: 独立 flush worker 线程，避免阻塞音频采集主循环
def _flush_worker(runtime: Optional[_InterviewerRuntime] = None):
    """独立线程处理 ASR buffer flush，避免阻塞音频采集"""
    stop_event = runtime.flush_stop_event if runtime is not None else _flush_stop_event
    flush_queue = runtime.flush_queue if runtime is not None else _flush_queue
    while not stop_event.is_set():
        try:
            flush_signal = flush_queue.get(timeout=0.1)
            if flush_signal:
                cfg, session, now, epoch = flush_signal
                with _asr_state_lock:
                    if session.transcript_epoch == epoch:
                        _try_flush_asr_merge_buffer(cfg, session, now, False)
                        _try_flush_asr_question_group(cfg, session, now, False)
        except queue.Empty:
            continue
        except Exception as e:
            _elog.error("flush_worker error: %s", e, exc_info=True)


def _interviewer_asr_worker(runtime: _InterviewerRuntime, session, cfg) -> None:
    last_error = ""
    while True:
        if runtime.drain_event.is_set() and _interviewer_segment_backlog(runtime) == 0:
            break
        try:
            segment = runtime.segment_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            if session.transcript_epoch != runtime.transcript_epoch:
                continue
            queue_delay_ms = max(0.0, (time.monotonic() - segment.ended_mono) * 1000.0)
            broadcast({"type": "transcribing", "value": True})
            t0 = time.monotonic()
            provider = cfg.stt_provider
            text = transcribe_segment(
                segment.audio,
                segment.sample_rate,
                position=cfg.position,
                language=cfg.language,
                provider=provider,
                config_snapshot=cfg,
                is_current=lambda: session.transcript_epoch == runtime.transcript_epoch,
            )
            if session.transcript_epoch != runtime.transcript_epoch:
                continue
            if last_error:
                last_error = ""
                broadcast({"type": "stt_status", "loaded": True, "loading": False, "provider": provider})
            text = postprocess_interview_transcription(text)
            stt_ms = (time.monotonic() - t0) * 1000
            pub = transcription_for_publish(
                text,
                getattr(cfg, "transcription_min_sig_chars", 2),
            )
            captured = pub or transcription_for_publish(text, 1)
            _ilog.info(
                "INTERVIEWER_ASR_SEGMENT flush_reason=%s queue_delay_ms=%.0f raw=%.1fs stt=%.0fms ratio=%.2f publish_chars=%d",
                segment.flush_reason,
                queue_delay_ms,
                segment.audio_sec,
                stt_ms,
                (stt_ms / 1000.0 / segment.audio_sec) if segment.audio_sec > 0 else 0.0,
                len(pub) if pub else 0,
            )
            if captured:
                if pub:
                    _ilog.info("ASR publish=%r", pub[:120])
                _append_transcription_fragment(
                    cfg, session, captured, time.monotonic(), False, started_at=segment.started_mono,
                    transcript_key=segment.transcript_key, transcript_epoch=runtime.transcript_epoch,
                    publish=bool(pub),
                )
        except STTError as error:
            runtime.segment_failures += 1
            if session.transcript_epoch == runtime.transcript_epoch and last_error != str(error):
                last_error = str(error)
                report_stt_error(cfg.stt_provider, "interviewer", last_error)
        except Exception as e:
            runtime.segment_failures += 1
            _elog.error("interviewer ASR segment error: %s", e, exc_info=True)
        finally:
            session.discard_unfinished_transcript(segment.transcript_reservation)
            runtime.segment_queue.task_done()
            if session.transcript_epoch == runtime.transcript_epoch:
                broadcast({"type": "transcribing", "value": False})


def _flush_interviewer_tail(runtime, vad, cfg, session, reason):
    remaining = vad.flush()
    if remaining is not None:
        ended_at = time.monotonic()
        _maybe_build_interviewer_segment(
            cfg, runtime, session, remaining, flush_reason=reason,
            segment_started_mono=max(0.0, ended_at - len(remaining) / AudioCapture.SAMPLE_RATE),
            segment_ended_mono=ended_at,
        )


def _interview_worker(config_snapshot=None, transcript_epoch=None, single_mic=None):
    session = get_session()
    audio_input = single_mic.interviewer_input if single_mic is not None else audio_capture
    if transcript_epoch is None:
        transcript_epoch = session.transcript_epoch
    if transcript_epoch != session.transcript_epoch:
        return
    cfg = config_snapshot if config_snapshot is not None else get_config()
    # 每次开新会话都用全新 runtime：上次异常退出遗留的 last_chunk_mono 会把
    # 跨会话的间隔当成音频空洞，打出 gap_ms=几千万 的假警报（日志已多次出现）。
    runtime = _new_interviewer_runtime()
    runtime.transcript_epoch = transcript_epoch
    _set_interviewer_runtime(runtime)

    broadcast({"type": "stt_status", "loaded": True, "loading": False, "provider": cfg.stt_provider})

    def new_vad():
        return VADBuffer(
            sample_rate=AudioCapture.SAMPLE_RATE,
            silence_threshold=getattr(cfg, "silence_threshold", 0.01),
            silence_duration=getattr(cfg, "silence_duration", 1.2),
            max_speech_duration=getattr(cfg, "assist_vad_max_speech_sec", 18.0),
            min_speech_duration=getattr(cfg, "assist_vad_min_speech_sec", 0.3),
            preroll_duration=_interviewer_vad_preroll_sec,
            rollover_duration=_interviewer_vad_rollover_sec,
        )

    vad = new_vad()
    _reset_asr_merge_buffer()
    _reset_pending_asr_group()
    pause_flushed = False

    # gc.collect() 之前直接放在主 ASR 循环里 (每 60s 同步执行),
    # 大堆下单次 50~500ms, 期间无法读音频可能丢块。改成独立 daemon 线程,
    # 主循环零阻塞; 线程靠 _stop_event 退出, 与 worker 生命周期对齐。
    _gc_stop = threading.Event()

    def _gc_periodic_worker() -> None:
        while not _gc_stop.wait(60.0):
            try:
                gc.collect()
            except Exception:
                pass

    _gc_thread = threading.Thread(
        target=_gc_periodic_worker, daemon=True, name="assist-gc"
    )
    _gc_thread.start()

    # H4: 启动独立 flush 线程（封装在 per-session runtime 内，防止跨会话泄漏或误 join）
    global _flush_thread, _interviewer_asr_thread
    runtime.flush_stop_event.clear()
    runtime.flush_thread = threading.Thread(
        target=_flush_worker, args=(runtime,), daemon=True, name="assist-flush"
    )
    if _interviewer_runtime is runtime:
        _flush_thread = runtime.flush_thread
    runtime.flush_thread.start()

    runtime.asr_thread = threading.Thread(
        target=_interviewer_asr_worker,
        args=(runtime, session, cfg),
        daemon=True,
        name="assist-interviewer-asr",
    )
    if _interviewer_runtime is runtime:
        _interviewer_asr_thread = runtime.asr_thread
    runtime.asr_thread.start()

    unexpected_error: Optional[str] = None
    try:
        while not _stop_event.is_set() and session.transcript_epoch == transcript_epoch:
            if _pause_event.is_set():
                if not pause_flushed:
                    _refresh_interviewer_raw_drop_count(runtime, audio_input)
                    _drain_remaining_interviewer_audio_chunks(runtime, vad, cfg, session, audio_input)
                    _flush_interviewer_tail(runtime, vad, cfg, session, "pause_flush")
                    if single_mic is not None:
                        vad = new_vad()
                    pause_flushed = True
                now = time.monotonic()
                try:
                    runtime.flush_queue.put_nowait((cfg, session, now, transcript_epoch))
                except queue.Full:
                    pass
                time.sleep(0.1)
                continue
            pause_flushed = False
            if single_mic is not None:
                operation = audio_input.take_barrier()
                if operation is not None:
                    _flush_interviewer_tail(runtime, vad, cfg, session, "role_switch")
                    vad = new_vad()
                    runtime.last_chunk_mono = None
                    operation.complete()
                if _pause_event.is_set() or _stop_event.is_set():
                    continue
            if not audio_capture.is_running:
                _elog.error("Interview worker: audio capture stopped unexpectedly, exiting loop")
                unexpected_error = "音频采集异常中断，面试录音已停止"
                break
            now = time.monotonic()
            _refresh_interviewer_raw_drop_count(runtime, audio_input)
            # H4: 将 flush 逻辑移到独立线程，避免阻塞音频采集
            try:
                runtime.flush_queue.put_nowait((cfg, session, now, transcript_epoch))
            except queue.Full:
                pass  # 如果队列满，跳过本次 flush

            chunks = audio_input.drain_audio_chunks(timeout=0.1, max_chunks=6)
            if not chunks:
                time.sleep(0.05)
                continue
            energy = AudioCapture.compute_energy(np.concatenate(chunks))
            broadcast({"type": "audio_level", "value": round(energy, 4)})

            offset_samples = 0
            total_samples = sum(len(chunk) for chunk in chunks)
            batch_end_mono = now
            batch_start_mono = max(0.0, batch_end_mono - (total_samples / AudioCapture.SAMPLE_RATE))
            _log_interviewer_chunk_gap(
                runtime,
                batch_start_mono,
                total_samples / AudioCapture.SAMPLE_RATE,
            )
            for chunk in chunks:
                for vad_chunk in _iter_vad_feed_chunks(chunk):
                    offset_samples += len(vad_chunk)
                    sub_ended_mono = batch_start_mono + (offset_samples / AudioCapture.SAMPLE_RATE)
                    speech_audio = vad.feed(vad_chunk)
                    if speech_audio is None:
                        continue
                    flush_reason = getattr(vad, "last_flush_reason", None) or "silence"
                    segment_ended_mono = sub_ended_mono
                    segment_started_mono = max(
                        0.0,
                        segment_ended_mono - (len(speech_audio) / AudioCapture.SAMPLE_RATE),
                    )
                    _maybe_build_interviewer_segment(
                        cfg,
                        runtime,
                        session,
                        speech_audio,
                        flush_reason=(
                            "max_speech"
                            if flush_reason == "max_speech"
                            else "silence"
                        ),
                        segment_started_mono=segment_started_mono,
                        segment_ended_mono=segment_ended_mono,
                    )

        if session.transcript_epoch != transcript_epoch:
            return
        _refresh_interviewer_raw_drop_count(runtime, audio_input)
        _drain_remaining_interviewer_audio_chunks(runtime, vad, cfg, session, audio_input)
        _flush_interviewer_tail(runtime, vad, cfg, session, "stop_flush")
    except Exception as e:
        _elog.error("Interview worker crashed: %s", e, exc_info=True)
        unexpected_error = f"\u9762\u8bd5\u5faa\u73af\u5f02\u5e38: {e}"
    finally:
        # 保证 worker 任何退出路径 (正常 / 异常 / 早退) 都释放音频设备。
        # AudioCapture.stop 是幂等的: 即使外部 stop_interview_loop 已经先调过,
        # 重复调用也是 no-op (内部用 _lock + _running 标志位防御)。
        # 这能修复 worker 异常崩溃后麦克风/系统音频设备一直被占用的泄漏。
        try:
            if single_mic is None and _interviewer_runtime is runtime:
                _stop_capture_compat(audio_capture, owner="assist", clear_queue=True)
        except Exception:
            _elog.error("audio_capture.stop in worker finally failed", exc_info=True)
        # 通知 GC daemon 退出并 join, 让 worker 生命周期完全确定 (避免测试需要
        # sleep 等收敛, 也不让旧 daemon 与下一轮 worker 的 daemon 短暂并存)。
        # threading.Event.set() 不会抛, 不需要 try; join(timeout) 只兜个上限。
        _gc_stop.set()
        try:
            _gc_thread.join(timeout=0.5)
        except Exception:
            pass
        runtime.flush_stop_event.set()
        flush_thread = runtime.flush_thread
        try:
            if flush_thread and flush_thread.is_alive():
                flush_thread.join(timeout=0.5)
        except Exception:
            pass
        if _flush_thread is flush_thread:
            _flush_thread = None
        if not runtime.stop_managed_by_caller:
            asr_thread = runtime.asr_thread
            try:
                if asr_thread and asr_thread.is_alive():
                    runtime.drain_event.set()
                    asr_thread.join(timeout=_interviewer_segment_drain_timeout())
            except Exception:
                pass
            if _interviewer_asr_thread is asr_thread:
                _interviewer_asr_thread = None
        _refresh_interviewer_raw_drop_count(runtime, audio_input)

    if unexpected_error is not None and not runtime.stop_managed_by_caller and not _stop_event.is_set():
        _finalize_unexpected_interview_stop(
            session,
            transcript_epoch,
            unexpected_error,
            single_mic=single_mic,
            runtime=runtime,
        )


def _publish_candidate_transcription(
    session,
    text: str,
    provider: str,
    qa_id: str | None = None,
    *,
    segment_id: str = "",
    is_final: bool = True,
    started_at: float | None = None,
    transcript_epoch: int | None = None,
) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    recent_interviewer = session.transcription_history[-3:]
    for item in recent_interviewer:
        interviewer_text = (item or "").strip()
        if len(cleaned) < 12 or len(interviewer_text) < 12:
            continue
        threshold = 0.95 if len(cleaned) < 30 else 0.88
        similarity = SequenceMatcher(None, cleaned, interviewer_text).ratio()
        if similarity >= threshold:
            _ilog.info("CANDIDATE_ASR_SKIP_ECHO ratio=%.2f text=%r", similarity, cleaned[:80])
            return
    with conversation_lock:
        anchor = session.record_transcript(
            "candidate", cleaned, started_at=time.monotonic() if started_at is None else started_at,
            utterance_key=segment_id, is_final=is_final, epoch=transcript_epoch,
        )
        if not anchor:
            return
        segment = session.add_candidate_transcription(
            cleaned,
            qa_id=qa_id,
            provider=provider,
            segment_id=segment_id,
            is_final=is_final,
        )
    if segment is None:
        return
    if segment.is_final and segment.qa_id:
        with conversation_lock:
            candidate_answer = session.get_candidate_answer_for_qa(segment.qa_id, max_chars=2400)
        if candidate_answer:
            _submit_candidate_knowledge_update(segment.qa_id, candidate_answer)
    _ilog.info(
        "CANDIDATE_ASR_PUBLISH segment=%s qa_id=%s final=%s provider=%s chars=%d",
        segment.segment_id,
        segment.qa_id,
        segment.is_final,
        provider,
        len(cleaned),
    )
    broadcast(
        {
            "type": "candidate_transcription",
            "scope": "assist",
            "text": cleaned,
            "qa_id": segment.qa_id,
            "provider": provider,
            "segment_id": segment.segment_id,
            "is_final": segment.is_final,
        }
    )
    return anchor


def _candidate_worker(transcript_epoch=None, single_mic=None, config_snapshot=None):
    session = get_session()
    if transcript_epoch is None:
        transcript_epoch = session.transcript_epoch
    audio_input = single_mic.candidate_input if single_mic is not None else _candidate_audio_capture
    cfg = config_snapshot if config_snapshot is not None else get_capture_config()
    provider = cfg.candidate_stt_provider
    if provider not in SEGMENT_STT_PROVIDERS:
        return
    _ilog.info("CANDIDATE_ASR_WORKER_START provider=%s", provider)
    broadcast({"type": "candidate_asr_status", "loaded": True, "loading": False, "provider": provider})

    def is_current():
        return session.transcript_epoch == transcript_epoch

    def new_vad():
        return VADBuffer(
            sample_rate=AudioCapture.SAMPLE_RATE,
            silence_threshold=cfg.silence_threshold,
            silence_duration=cfg.silence_duration,
            min_speech_duration=0.3,
        )

    vad = new_vad()
    current_segment_id = ""
    segment_started_at = 0.0
    segment_qa_id = ""
    candidate_final_failed = False
    last_error = ""

    def finalize_candidate_audio(final_audio, log_kind: str) -> bool:
        nonlocal current_segment_id, last_error
        try:
            if final_audio is None or len(final_audio) <= AudioCapture.SAMPLE_RATE * 0.3:
                return True
            with conversation_lock:
                if not is_current():
                    return False
                session.mark_candidate_asr_busy(segment_qa_id)
            t0 = time.monotonic()
            text = transcribe_segment(
                final_audio,
                AudioCapture.SAMPLE_RATE,
                position=cfg.position,
                language=cfg.language,
                provider=provider,
                scope="candidate",
                config_snapshot=cfg,
                is_current=is_current,
            )
            if not is_current():
                return False
            if last_error:
                last_error = ""
                broadcast({"type": "candidate_asr_status", "loaded": True, "loading": False, "provider": provider})
            pub = transcription_for_publish(postprocess_interview_transcription(text), 1)
            if pub:
                _ilog.info(
                    "CANDIDATE_ASR_%s segment=%s qa_id=%s provider=%s raw=%.1fs stt=%.0fms chars=%d",
                    log_kind, current_segment_id, segment_qa_id, provider,
                    len(final_audio) / AudioCapture.SAMPLE_RATE, (time.monotonic() - t0) * 1000, len(pub),
                )
                _publish_candidate_transcription(
                    session, pub, provider, qa_id=segment_qa_id,
                    segment_id=current_segment_id, started_at=segment_started_at,
                    transcript_epoch=transcript_epoch,
                )
            return True
        except STTError as error:
            if is_current() and last_error != str(error):
                last_error = str(error)
                report_stt_error(provider, "candidate", last_error)
            return False
        finally:
            with conversation_lock:
                if is_current():
                    session.mark_candidate_asr_idle()
            current_segment_id = ""

    def feed_candidate_chunks(chunks):
        nonlocal current_segment_id, segment_qa_id, segment_started_at, candidate_final_failed
        sample_rate = AudioCapture.SAMPLE_RATE
        chunk_started_at = time.monotonic() - sum(len(chunk) for chunk in chunks) / sample_rate
        for chunk in chunks:
            for vad_chunk in _iter_vad_feed_chunks(chunk):
                flushed = vad.feed(vad_chunk)
                pending = getattr(vad, "has_pending_audio", False)
                if not current_segment_id and (pending or flushed is not None):
                    with conversation_lock:
                        if not is_current():
                            return False
                        segment_qa_id = session.current_candidate_qa_id
                    current_segment_id = f"cand-{uuid4().hex}"
                    segment_started_at = chunk_started_at
                    _ilog.info("CANDIDATE_ASR_SEGMENT_START segment=%s qa_id=%s", current_segment_id, segment_qa_id)
                if flushed is not None:
                    if not finalize_candidate_audio(flushed, "FINAL"):
                        candidate_final_failed = True
                elif not pending and current_segment_id:
                    finalize_candidate_audio(None, "EMPTY")
                chunk_started_at += len(vad_chunk) / sample_rate
        return True

    def drain_candidate_input():
        while is_current():
            chunks = audio_input.drain_audio_chunks(timeout=0.0, max_chunks=6)
            if not chunks or not feed_candidate_chunks(chunks):
                break

    try:
        while not _stop_event.is_set() and is_current():
            if _pause_event.is_set():
                if _candidate_flush_event.is_set():
                    _candidate_flush_event.clear()
                    if single_mic is not None:
                        drain_candidate_input()
                    finalize_candidate_audio(vad.flush(), "PAUSE_FINAL")
                    if single_mic is not None:
                        vad = new_vad()
                        candidate_final_failed = False
                time.sleep(0.1)
                continue
            if single_mic is not None:
                operation = audio_input.take_barrier()
                if operation is not None:
                    success = finalize_candidate_audio(vad.flush(), "ROLE_SWITCH")
                    vad = new_vad()
                    operation.complete(success and not candidate_final_failed)
                    candidate_final_failed = False
            chunks = audio_input.drain_audio_chunks(timeout=0.1, max_chunks=6)
            if not chunks:
                continue
            if not feed_candidate_chunks(chunks):
                return
            if getattr(vad, "has_pending_audio", False):
                with conversation_lock:
                    if not is_current():
                        return
                    session.mark_candidate_asr_busy(segment_qa_id)

        if single_mic is not None and is_current():
            drain_candidate_input()
        finalize_candidate_audio(vad.flush(), "FLUSH_FINAL")
    except Exception:
        _elog.exception("Candidate ASR worker crashed")
        if is_current():
            report_stt_error(provider, "candidate", "我的回答转写中断，请检查麦克风和语音配置")
    finally:
        if single_mic is None and is_current():
            try:
                _stop_capture_compat(_candidate_audio_capture, owner="assist-candidate", clear_queue=True)
            except Exception:
                _elog.exception("candidate audio_capture.stop failed")


# ---------------------------------------------------------------------------
# Answer generation worker
# ---------------------------------------------------------------------------

def _process_question_parallel(
    task: TaskPayload,
    seq: int,
    model_idx: int,
    sess_v: int,
    model_cfg_snapshot=None,
    config_snapshot=None,
):
    cfg = config_snapshot if config_snapshot is not None else get_config()
    my_gen = _capture_generation()
    my_asr_turn = int(_task_meta(task).get("asr_turn_id", 0)) if _is_asr_task(task) else 0
    request = _task_meta(task).get("mmpg_request")
    exam_preflight_id = str(_task_meta(task).get("exam_preflight_id") or "")
    routing = task_meta(task).get("answer_routing")

    def resolve_exam_model(index):
        current = get_config()
        submitted = cfg.models[index]
        live_index = live_exam_model_index(routing, current, submitted.id)
        if live_index is None:
            return None
        # Preserve every submitted transport/payload setting. A renamed row
        # and rotated credentials are valid for the same provider destination.
        model = copy.copy(submitted)
        model.name = current.models[live_index].name
        model.api_key = current.models[live_index].api_key
        return model

    def session_stale() -> bool:
        return sess_v != _task_session_version

    def aborted() -> bool:
        if session_stale() or (request and request.cancelled):
            return True
        if exam_preflight_id and _is_exam_preflight_cancelled(exam_preflight_id):
            return True
        if my_gen != _answer_generation:
            return True
        if (
            _is_asr_task(task)
            and _should_interrupt_stale_asr(cfg)
            and my_asr_turn
            and my_asr_turn < _get_latest_asr_turn_id()
        ):
            return True
        return False

    return process_question_parallel(
        task,
        seq,
        model_idx,
        sess_v,
        AnswerWorkerDeps(
            abort_check=aborted,
            is_session_current=lambda version: version == _task_session_version,
            flush_commit=_flush_commit,
            mark_seq_skipped=_mark_seq_skipped,
            submit_knowledge_record=_submit_knowledge_record,
            broadcast=broadcast,
            logger=_ilog,
            error_logger=_elog,
            start_abort_check=aborted,
            model_cfg_snapshot=model_cfg_snapshot,
            config_snapshot=config_snapshot,
            resolve_exam_model=resolve_exam_model if routing is not None and routing.written_exam else None,
            reserve_retry=lambda attempted, on_wait: _reserve_answer_retry(
                task, seq, cfg, attempted, aborted, on_wait=on_wait,
            ),
        ),
    )


def _save_knowledge_record(action: str, *args: object):
    try:
        from services.storage.knowledge import save_record, update_candidate_answer_for_qa

        if action == "candidate_update":
            qa_id = str(args[0] if len(args) > 0 else "")
            candidate_answer = str(args[1] if len(args) > 1 else "")
            update_candidate_answer_for_qa(qa_id, candidate_answer)
            return

        question = str(args[0] if len(args) > 0 else "")
        answer = str(args[1] if len(args) > 1 else "")
        qa_id = str(args[2] if len(args) > 2 else "")
        candidate_answer = str(args[3] if len(args) > 3 else "")
        save_record("assist", question, answer, qa_id=qa_id, candidate_answer=candidate_answer)
    except Exception as exc:
        _elog.warning("_save_knowledge_record failed: %s", exc)
