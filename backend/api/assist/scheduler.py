"""Answer task scheduling primitives for the assist pipeline.

This module owns the queueing rules and stale-ASR arbitration.  The pipeline
module keeps the public functions and worker side effects, while this file
keeps the mutable scheduling decisions testable and isolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from core.config import written_exam_model_available

TaskImage = Optional[str | list[str]]
TaskPayload = Tuple[str, TaskImage, bool, str, dict[str, Any]]
PendingTask = Tuple[TaskPayload, int, int]
ModelReservation = str | int
InFlightTasks = dict[int, tuple[ModelReservation | None, TaskPayload]]
EXAM_MODEL_UNAVAILABLE = "请在「笔试配置」中选择已启用、已配置 API Key 且支持识图的模型。"
EXAM_MODEL_STALE = "本题的笔试模型已不可用或接口配置已变更，请重新提交。"


@dataclass(frozen=True)
class DispatchClaim:
    task: TaskPayload
    seq: int
    model_idx: int
    session_version: int


@dataclass(frozen=True)
class DispatchStep:
    claim: DispatchClaim | None = None
    skipped_seq: int | None = None
    skipped_task: TaskPayload | None = None
    unavailable_task: TaskPayload | None = None


@dataclass(frozen=True)
class AnswerRouting:
    written_exam: bool
    exam_model_ids: tuple[str, ...] = ()
    exam_config: Any = field(default=None, repr=False)


def task_meta(task: TaskPayload) -> dict[str, Any]:
    return task[4]


def task_is_written_exam(task: TaskPayload, cfg) -> bool:
    meta = task_meta(task)
    routing = meta.get("answer_routing")
    if routing is not None:
        return routing.written_exam
    return bool(meta.get("exam_preflight") or (
        task[2] and meta.get("purpose") != "knowledge_practice" and getattr(cfg, "written_exam_mode", False)
    ))


def eligible_exam_model_index(cfg, model_id: str | None) -> int | None:
    if not model_id:
        return None
    return next((
        index for index, model in enumerate(cfg.models)
        if getattr(model, "id", None) == model_id and written_exam_model_available(model)
    ), None)


def exam_model_order(cfg) -> list[int]:
    """Return selected catalog-ready vision models in exam preference order."""
    indices = (
        eligible_exam_model_index(cfg, model_id)
        for model_id in getattr(cfg, "written_exam_model_ids", ())
    )
    return list(dict.fromkeys(index for index in indices if index is not None))


def _exam_model_destination(model) -> tuple[str, str, str]:
    return (
        getattr(model, "api_base_url", "").strip().rstrip("/"),
        getattr(model, "api_type", ""),
        getattr(model, "model", ""),
    )


def live_exam_model_index(routing: AnswerRouting, cfg, model_id: str) -> int | None:
    index = eligible_exam_model_index(cfg, model_id)
    submitted_index = eligible_exam_model_index(routing.exam_config, model_id)
    if index is None or submitted_index is None:
        return None
    if _exam_model_destination(cfg.models[index]) != _exam_model_destination(routing.exam_config.models[submitted_index]):
        return None
    return index


def exam_primary_index(task: TaskPayload, cfg) -> int | None:
    routing = task_meta(task).get("answer_routing")
    if routing is not None:
        return live_exam_model_index(routing, cfg, routing.exam_model_ids[0])
    return next(iter(exam_model_order(cfg)), None)


def is_asr_task(task: TaskPayload) -> bool:
    return task_meta(task).get("origin") == "asr"


def is_stale_inflight_asr_task(task: TaskPayload, latest_asr_turn_id: int) -> bool:
    return is_asr_task(task) and int(task_meta(task).get("asr_turn_id", 0)) < latest_asr_turn_id


def key_ok(model) -> bool:
    return bool(model.api_key and model.api_key not in ("", "sk-your-api-key-here"))


def model_eligible(i: int, model, need_vision: bool, get_model_health: Callable[[int], Optional[str]]) -> bool:
    if not getattr(model, "enabled", True):
        return False
    if get_model_health(i) == "error":
        return False
    if not key_ok(model):
        return False
    if need_vision and not model.supports_vision:
        return False
    return True


def priority_model_index(cfg) -> int:
    n = len(cfg.models)
    if n <= 0:
        return 0
    fallback = max(0, min(int(getattr(cfg, "active_model", 0) or 0), n - 1))
    try:
        active_model = cfg.get_active_model()
    except Exception:
        return fallback
    for i, model in enumerate(cfg.models):
        if model is active_model:
            return i
    return fallback


def _legacy_model_order(cfg) -> list[int]:
    n = len(cfg.models)
    if n == 0:
        return []
    priority = priority_model_index(cfg)
    return [priority] + [i for i in range(n) if i != priority]


def dispatch_model_order(cfg, task: TaskPayload | None = None) -> list[int]:
    if task is not None and task_is_written_exam(task, cfg):
        primary = exam_primary_index(task, cfg)
        return [] if primary is None else [primary]
    selected = getattr(cfg, "assist_model_ids", None)
    if selected is None or (task is not None and task_meta(task).get("purpose") == "knowledge_practice"):
        return _legacy_model_order(cfg)
    if task is not None and task[3].startswith("server_screen_"):
        selected = [getattr(cfg, "assist_screenshot_model_id", None)]
    by_id = {model.id: index for index, model in enumerate(cfg.models)}
    return [by_id[model_id] for model_id in selected if model_id in by_id]


def model_reservation(cfg, index: int) -> ModelReservation:
    return getattr(cfg.models[index], "id", "") or index


def pick_model_index(
    task: TaskPayload,
    busy: set[ModelReservation],
    cfg,
    get_model_health: Callable[[int], Optional[str]],
    avoid_models: Optional[set[ModelReservation]] = None,
    *,
    excluded_indices: set[int] | None = None,
) -> Optional[int]:
    _text, image, _manual, _source, _meta = task
    need_vision = bool(image)
    avoid = avoid_models or set()
    excluded = excluded_indices or set()

    def ok_basic(i: int, model) -> bool:
        if i in excluded or i in busy or model_reservation(cfg, i) in busy:
            return False
        if not getattr(model, "enabled", True):
            return False
        if not key_ok(model):
            return False
        if need_vision and not model.supports_vision:
            return False
        return True

    available = [i for i in dispatch_model_order(cfg, task) if ok_basic(i, cfg.models[i])]
    if task_is_written_exam(task, cfg):
        return available[0] if available else None
    preferred = [i for i in available if i not in avoid and model_reservation(cfg, i) not in avoid]
    candidates = preferred or available
    return next((i for i in candidates if get_model_health(i) != "error"), candidates[0] if candidates else None)


def max_parallel_slots(
    cfg, get_model_health: Callable[[int], Optional[str]], *, include_screenshots: bool = True, include_practice: bool = False,
) -> int:
    order = _legacy_model_order(cfg) if include_practice else dispatch_model_order(cfg)
    if include_screenshots:
        screenshot = eligible_exam_model_index(cfg, getattr(cfg, "assist_screenshot_model_id", None))
        if screenshot is not None:
            order.append(screenshot)
        order = list(dict.fromkeys((*order, *exam_model_order(cfg))))
    n_ok = sum(1 for i in order if model_eligible(i, cfg.models[i], False, get_model_health))
    cap = max(1, getattr(cfg, "max_parallel_answers", 2))
    return max(1, min(cap, max(n_ok, 1)))


def dispatch_snapshot(
    in_flight_tasks: InFlightTasks,
    latest_asr_turn_id: int,
) -> tuple[set[ModelReservation], int]:
    return physical_busy_models(in_flight_tasks), len(in_flight_tasks)


def physical_busy_models(in_flight_tasks: InFlightTasks) -> set[ModelReservation]:
    return {model for model, _task in in_flight_tasks.values() if model is not None}


def begin_asr_turn(
    pending: list[PendingTask],
    latest_asr_turn_id: int,
    *,
    interrupt_pending_asr: bool = True,
) -> tuple[int, list[int]]:
    latest_asr_turn_id += 1
    skipped: list[int] = []
    if not interrupt_pending_asr:
        return latest_asr_turn_id, skipped
    kept: list[PendingTask] = []
    for task, seq, session_version in pending:
        if is_asr_task(task):
            skipped.append(seq)
            continue
        kept.append((task, seq, session_version))
    pending[:] = kept
    return latest_asr_turn_id, skipped


def claim_next_dispatch(
    pending: list[PendingTask],
    in_flight_tasks: InFlightTasks,
    latest_asr_turn_id: int,
    max_parallel_slots: int,
    pick_model_index: Callable[[TaskPayload, set[ModelReservation], Optional[set[ModelReservation]]], Optional[int]],
    *,
    interrupt_stale_asr: bool = True,
    now_mono: float | None = None,
    reserve_model: Callable[[int], ModelReservation] = lambda index: index,
    discard_unavailable: bool = False,
) -> DispatchStep:
    busy_models, effective_slots = dispatch_snapshot(in_flight_tasks, latest_asr_turn_id)
    physical_busy = physical_busy_models(in_flight_tasks)
    if effective_slots >= max_parallel_slots:
        return DispatchStep()

    idx = 0
    while idx < len(pending):
        task, seq, session_version = pending[idx]
        meta = task_meta(task)
        dispatch_after = float(meta.get("dispatch_after_mono", 0.0) or 0.0)
        if now_mono is not None and dispatch_after > now_mono:
            idx += 1
            continue
        if interrupt_stale_asr and is_asr_task(task) and int(meta.get("asr_turn_id", 0)) < latest_asr_turn_id:
            pending.pop(idx)
            return DispatchStep(skipped_seq=seq, skipped_task=task)
        avoid_models = physical_busy if is_asr_task(task) else None
        model_idx = pick_model_index(task, busy_models, avoid_models)
        if model_idx is not None:
            pending.pop(idx)
            in_flight_tasks[seq] = (reserve_model(model_idx), task)
            return DispatchStep(
                claim=DispatchClaim(
                    task=task,
                    seq=seq,
                    model_idx=model_idx,
                    session_version=session_version,
                )
            )
        if discard_unavailable and pick_model_index(task, set(), None) is None:
            pending.pop(idx)
            return DispatchStep(skipped_seq=seq, unavailable_task=task)
        idx += 1
    return DispatchStep()


def drain_commit_queue(
    commit_buffer: dict[int, Callable[[], None]],
    skipped_commit_seqs: set[int],
    next_commit_seq: int,
) -> int:
    while True:
        while next_commit_seq in skipped_commit_seqs:
            skipped_commit_seqs.discard(next_commit_seq)
            next_commit_seq += 1
        apply_fn = commit_buffer.pop(next_commit_seq, None)
        if apply_fn is None:
            return next_commit_seq
        apply_fn()
        next_commit_seq += 1
