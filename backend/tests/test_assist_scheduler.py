from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.assist import scheduler  # noqa: E402


def _asr_task(text: str, turn_id: int) -> scheduler.TaskPayload:
    return (text, None, False, "conversation_mic", {"origin": "asr", "asr_turn_id": turn_id})


def _manual_task(text: str) -> scheduler.TaskPayload:
    return (text, None, True, "manual_text", {"origin": "manual"})


def test_begin_asr_turn_drops_pending_asr_only():
    pending: list[scheduler.PendingTask] = [
        (_asr_task("旧 ASR", 1), 0, 3),
        (_manual_task("手动问题"), 1, 3),
    ]

    latest_turn, skipped = scheduler.begin_asr_turn(pending, 1)

    assert latest_turn == 2
    assert skipped == [0]
    assert pending == [(_manual_task("手动问题"), 1, 3)]


def test_begin_asr_turn_keeps_pending_asr_when_interrupt_disabled():
    pending: list[scheduler.PendingTask] = [
        (_asr_task("旧 ASR", 1), 0, 3),
        (_manual_task("手动问题"), 1, 3),
    ]

    latest_turn, skipped = scheduler.begin_asr_turn(
        pending,
        1,
        interrupt_pending_asr=False,
    )

    assert latest_turn == 2
    assert skipped == []
    assert pending == [
        (_asr_task("旧 ASR", 1), 0, 3),
        (_manual_task("手动问题"), 1, 3),
    ]


def test_claim_next_dispatch_avoids_physical_busy_model_for_asr():
    pending: list[scheduler.PendingTask] = [(_asr_task("最新问题", 2), 2, 7)]
    in_flight = {1: (2, _asr_task("旧问题", 1))}
    calls: list[tuple[set[int], set[int] | None]] = []

    def pick_model(_task, busy, avoid_models=None):
        calls.append((set(busy), set(avoid_models or set())))
        return 0

    step = scheduler.claim_next_dispatch(
        pending,
        in_flight,
        latest_asr_turn_id=2,
        max_parallel_slots=2,
        pick_model_index=pick_model,
    )

    assert step.claim is not None
    assert step.claim.model_idx == 0
    assert step.claim.seq == 2
    assert pending == []
    assert in_flight[2] == (0, _asr_task("最新问题", 2))
    assert calls == [({2}, {2})]


def test_claim_next_dispatch_can_keep_stale_asr_when_interrupt_disabled():
    pending: list[scheduler.PendingTask] = [(_asr_task("旧问题", 1), 2, 7)]
    in_flight: dict[int, tuple[int, scheduler.TaskPayload]] = {}

    step = scheduler.claim_next_dispatch(
        pending,
        in_flight,
        latest_asr_turn_id=2,
        max_parallel_slots=2,
        pick_model_index=lambda _task, _busy, _avoid_models=None: 0,
        interrupt_stale_asr=False,
    )

    assert step.claim is not None
    assert step.claim.seq == 2
    assert step.skipped_seq is None
    assert pending == []


def test_claim_next_dispatch_waits_for_deferred_task():
    task = (
        "稍后分发",
        None,
        False,
        "conversation_mic",
        {"origin": "asr", "asr_turn_id": 2, "dispatch_after_mono": 10.0},
    )
    pending: list[scheduler.PendingTask] = [(task, 2, 7)]
    in_flight: dict[int, tuple[int, scheduler.TaskPayload]] = {}

    early = scheduler.claim_next_dispatch(
        pending,
        in_flight,
        latest_asr_turn_id=2,
        max_parallel_slots=2,
        pick_model_index=lambda _task, _busy, _avoid_models=None: 0,
        now_mono=9.5,
    )
    ready = scheduler.claim_next_dispatch(
        pending,
        in_flight,
        latest_asr_turn_id=2,
        max_parallel_slots=2,
        pick_model_index=lambda _task, _busy, _avoid_models=None: 0,
        now_mono=10.0,
    )

    assert early.claim is None
    assert pending == []
    assert ready.claim is not None
    assert ready.claim.seq == 2


def test_drain_commit_queue_skips_then_drains_in_order():
    committed: list[str] = []
    commit_buffer = {
        1: lambda: committed.append("seq1"),
        2: lambda: committed.append("seq2"),
    }
    skipped_commit_seqs = {0}

    next_seq = scheduler.drain_commit_queue(
        commit_buffer,
        skipped_commit_seqs,
        next_commit_seq=0,
    )

    assert committed == ["seq1", "seq2"]
    assert next_seq == 3
    assert commit_buffer == {}
    assert skipped_commit_seqs == set()


def test_pick_model_index_respects_priority_and_vision():
    cfg = SimpleNamespace(
        models=[
            SimpleNamespace(enabled=True, api_key="k0", supports_vision=False),
            SimpleNamespace(enabled=True, api_key="k1", supports_vision=True),
        ],
        active_model=0,
        get_active_model=lambda: None,
    )
    cfg.get_active_model = lambda: cfg.models[cfg.active_model]
    task: scheduler.TaskPayload = (
        "看图题",
        "data:image/png;base64,xxx",
        False,
        "server_screen_code",
        {},
    )

    picked = scheduler.pick_model_index(
        task,
        busy=set(),
        cfg=cfg,
        get_model_health=lambda _idx: None,
    )

    assert picked == 1


def test_realtime_selection_limits_pasted_images_and_separates_screenshots():
    from core.config import AppConfig, ModelConfig

    cfg = AppConfig(
        models=[
            ModelConfig(id="outside", api_key="test", supports_vision=True),
            ModelConfig(id="answer", api_key="test"),
            ModelConfig(id="paste", api_key="test", supports_vision=True),
            ModelConfig(id="screen", api_key="test", supports_vision=True),
        ],
        assist_model_ids=["answer", "paste"],
        assist_screenshot_model_id="screen",
    )

    def pick(image=None, source="manual_text", busy=None, exclude=None):
        return scheduler.pick_model_index(
            ("问题", image, True, source, {}), busy or set(), cfg, lambda _index: None,
            excluded_indices=exclude,
        )

    assert pick() == 1
    assert pick(busy={"answer"}) == 2
    assert pick(exclude={1, 2}) is None
    assert pick("image", "manual_image") == 2
    assert pick("image", "manual_image", {"paste"}) is None
    assert pick("image", "server_screen_left") == 3
    assert pick("image", "server_screen_left", {"screen"}) is None


def test_retry_releases_its_failed_model_and_waits_for_an_occupied_candidate(monkeypatch):
    from threading import Event, Thread

    from api.assist import pipeline
    from core.config import AppConfig, ModelConfig

    task = _manual_task("待重试的题")
    cfg = AppConfig(
        models=[ModelConfig(id="primary", api_key="test"), ModelConfig(id="fallback", api_key="test")],
        assist_model_ids=["primary", "fallback"],
    )
    in_flight = {0: ("primary", task), 1: ("fallback", _manual_task("并行的题"))}
    monkeypatch.setattr(pipeline, "_in_flight_tasks", in_flight)
    released = Event()
    result = []

    worker = Thread(
        target=lambda: result.append(
            pipeline._reserve_answer_retry(task, 0, cfg, {0}, lambda: False, on_wait=released.set)
        ),
        daemon=True,
    )
    worker.start()
    try:
        assert released.wait(2)
        with pipeline._dispatch_changed:
            assert in_flight[0] == (None, task)
            assert in_flight[1][0] == "fallback"
            del in_flight[1]
            pipeline._dispatch_changed.notify_all()
        worker.join(2)
        assert result == [1]
        assert in_flight[0] == ("fallback", task)
    finally:
        with pipeline._dispatch_changed:
            in_flight.clear()
            pipeline._dispatch_changed.notify_all()
        worker.join(2)
