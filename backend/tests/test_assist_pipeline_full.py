from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.session import get_session, reset_session  # noqa: E402
from api.assist import answer_worker, pipeline  # noqa: E402


class _DeferredThread:
    started: list["_DeferredThread"] = []

    def __init__(self, target, args=(), kwargs=None, daemon=None, name=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name

    def start(self):
        self.started.append(self)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None

    def run(self):
        self.target(*self.args, **self.kwargs)


class _DeferredTimer(_DeferredThread):
    started: list["_DeferredTimer"] = []

    def __init__(self, interval, function, args=None, kwargs=None):
        super().__init__(function, args=tuple(args or ()), kwargs=kwargs or {}, daemon=True)
        self.interval = interval


def _cfg():
    models = [
        SimpleNamespace(
            name="模型一",
            api_key="k1",
            model="fake-1",
            enabled=True,
            supports_vision=False,
        ),
        SimpleNamespace(
            name="模型二",
            api_key="k2",
            model="fake-2",
            enabled=True,
            supports_vision=False,
        ),
    ]
    cfg = SimpleNamespace(
        models=models,
        active_model=0,
        max_parallel_answers=2,
        written_exam_mode=False,
        stt_provider="doubao",
        candidate_stt_provider="doubao",
        doubao_stt_api_key="test-asr",
        written_exam_think=False,
        screen_capture_region="left_half",
        kb_enabled=False,
        kb_trigger_modes=[],
        assist_asr_interrupt_running=False,
    )
    cfg.get_active_model = lambda: cfg.models[cfg.active_model]
    return cfg


@pytest.fixture(autouse=True)
def reset_pipeline_state(monkeypatch: pytest.MonkeyPatch):
    reset_session()
    pipeline._answer_generation = 0
    pipeline._pending.clear()
    pipeline._in_flight_tasks.clear()
    pipeline._latest_asr_turn_id = 0
    pipeline._reset_asr_merge_buffer()
    pipeline._reset_pending_asr_group()
    pipeline._commit_buffer.clear()
    pipeline._skipped_commit_seqs.clear()
    pipeline._next_commit_seq = 0
    pipeline._next_submit_seq = 0
    pipeline._task_session_version = 0
    monkeypatch.setattr(pipeline, "_interview_thread", None)
    monkeypatch.setattr(pipeline, "_candidate_thread", None)
    pipeline._stop_event.clear()
    pipeline._pause_event.clear()
    pipeline._candidate_flush_event.clear()
    pipeline._flush_stop_event.clear()
    monkeypatch.setattr(pipeline, "_interviewer_config", None)
    _DeferredThread.started = []
    _DeferredTimer.started = []

    cfg = _cfg()
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "get_model_health", lambda _idx: None)
    monkeypatch.setattr(pipeline.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(pipeline.threading, "Timer", _DeferredTimer)
    monkeypatch.setattr(pipeline, "_submit_knowledge_record", lambda _q, _a, *_rest: True)
    monkeypatch.setattr(answer_worker, "build_system_prompt", lambda **_kwargs: "system")
    monkeypatch.setattr(
        answer_worker,
        "get_token_stats",
        lambda: {"prompt": 1, "completion": 2, "total": 3, "by_model": {}},
    )


def test_submit_answer_task_runs_through_dispatch_worker_and_commit(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    def fake_stream(model_cfg, messages, **_kwargs):
        question = messages[-1]["content"]
        yield ("text", f"{model_cfg.name}:{question}")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    accepted = pipeline.submit_answer_task(
        ("Redis 怎么持久化？", None, True, "manual_text", {"origin": "manual"})
    )

    assert accepted is True
    assert len(_DeferredThread.started) == 1

    _DeferredThread.started[0].run()

    event_types = [event["type"] for event in broadcasts]
    assert event_types == ["answer_start", "answer_chunk", "answer_done", "token_update"]
    assert pipeline._pending == []
    assert pipeline._in_flight_tasks == {}
    assert pipeline._next_submit_seq == 1
    assert pipeline._next_commit_seq == 1

    session = get_session()
    assert [qa.question for qa in session.qa_pairs] == ["Redis 怎么持久化？"]
    assert [qa.answer for qa in session.qa_pairs] == ["模型一:Redis 怎么持久化？"]


def test_dispatched_worker_keeps_selected_model_when_settings_reorder_models(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pipeline, "broadcast", lambda _data: None)

    def fake_stream(model_cfg, messages, **_kwargs):
        question = messages[-1]["content"]
        yield ("text", f"{model_cfg.name}:{question}")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    cfg = pipeline.get_config()
    assert pipeline.submit_answer_task(
        ("配置更新期间的问题", None, True, "manual_text", {"origin": "manual"})
    )
    assert len(_DeferredThread.started) == 1

    cfg.models = [cfg.models[1], cfg.models[0]]
    cfg.active_model = 0
    _DeferredThread.started[0].run()

    session = get_session()
    assert [qa.answer for qa in session.qa_pairs] == ["模型一:配置更新期间的问题"]


def test_exam_text_waits_for_primary_then_uses_frozen_vision_fallback_and_prompt(monkeypatch):
    import copy
    from services.llm.streaming import LLMConnectionError

    cfg = pipeline.get_config()
    cfg.written_exam_mode = True
    cfg.written_exam_model_ids = ["text", "exam", "fallback", "unused"]
    cfg.assist_model_ids = ["screenshot"]
    cfg.assist_screenshot_model_id = "screenshot"
    cfg.system_prompts = {"assist_exam": "提交时的笔试指令"}
    cfg.assist_generation = {"assist_exam": SimpleNamespace(temperature=0.2, max_output_tokens=660)}
    cfg.models = [
        SimpleNamespace(id=model_id, name=model_id, api_key="test", enabled=True, supports_vision=vision)
        for model_id, vision in [
            ("text", False), ("screenshot", True), ("exam", True), ("unused", True), ("fallback", True),
        ]
    ]
    from services.llm import build_system_prompt
    monkeypatch.setattr(answer_worker, "build_system_prompt", build_system_prompt)
    events = []
    monkeypatch.setattr(pipeline, "broadcast", events.append)
    calls = []

    def transport(model, _messages, **kwargs):
        calls.append((model.id, model.api_key, kwargs["system_prompt"], kwargs["override_max_tokens"]))
        if model.id == "exam":
            raise LLMConnectionError("fixture")
        yield "text", "备用模型的答案"

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", transport)
    pipeline._in_flight_tasks[99] = ("exam", ("", None, True, "manual_text", {}))
    assert pipeline.submit_answer_task(("一道文字笔试题", None, True, "manual_text", {"origin": "manual"}))
    assert _DeferredThread.started == []
    next_cfg = copy.deepcopy(cfg)
    next_cfg.written_exam_mode = False
    next_cfg.written_exam_model_ids = ["screenshot"]
    next_cfg.assist_model_ids = ["screenshot"]
    next_cfg.system_prompts = {"assist_exam": "更新后的笔试指令"}
    next_cfg.assist_generation["assist_exam"].max_output_tokens = 900
    next_cfg.models[-1].name = "备用模型（已改名）"
    next_cfg.models[-1].api_key = "rotated-test-key"
    next_cfg.models.reverse()
    monkeypatch.setattr(pipeline, "get_config", lambda: next_cfg)
    pipeline._in_flight_tasks.pop(99)
    pipeline._try_dispatch()
    _DeferredThread.started[0].run()

    assert calls == [("exam", "test", "提交时的笔试指令", 660), ("fallback", "rotated-test-key", "提交时的笔试指令", 660)]
    answer_events = [event for event in events if event["type"] in ("answer_start", "answer_retry", "answer_done")]
    assert len({event["id"] for event in answer_events}) == 1
    assert [event["model_name"] for event in answer_events] == ["exam", "备用模型（已改名）", "备用模型（已改名）"]


@pytest.mark.parametrize("stage", ["before_submission", "queued", "before_request", "endpoint_changed"])
def test_exam_stops_when_primary_is_unavailable_before_any_provider_request(monkeypatch, stage):
    cfg = pipeline.get_config()
    cfg.written_exam_mode = True
    cfg.written_exam_model_ids = ["exam", "fallback"]
    cfg.assist_model_ids = ["fallback"]
    for model, model_id in zip(cfg.models, ("exam", "fallback")):
        model.id = model_id
        model.supports_vision = True
        model.api_base_url = f"https://{model_id}.invalid/v1"
    events = []
    monkeypatch.setattr(pipeline, "broadcast", events.append)

    def unexpected_transport(*_args, **_kwargs):
        pytest.fail("no model may be invoked before a viable primary is attempted")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", unexpected_transport)
    task = ("文字题", None, True, "manual_text", {"origin": "manual"})
    if stage == "before_submission":
        cfg.models[0].api_key = " "
        cfg.models[1].supports_vision = False
        assert not pipeline.submit_answer_task(task)
    elif stage in ("queued", "endpoint_changed"):
        pipeline._in_flight_tasks[99] = ("exam", ("", None, True, "manual_text", {}))
        assert pipeline.submit_answer_task(task)
        if stage == "queued":
            cfg.models.pop(0)
        else:
            cfg.models[0].api_base_url = "https://replacement.invalid/v1"
        pipeline._in_flight_tasks.pop(99)
        pipeline._try_dispatch()
        assert _DeferredThread.started == []
    else:
        assert pipeline.submit_answer_task(task)
        cfg.models[0].enabled = False
        _DeferredThread.started[0].run()

    errors = [event for event in events if event["type"] in {"error", "answer_error"}]
    assert len(errors) == 1 and "笔试" in errors[0]["message"]
    assert pipeline._pending == []
    assert get_session().qa_pairs == []


def test_dispatched_worker_keeps_full_config_snapshot_when_settings_change(
    monkeypatch: pytest.MonkeyPatch,
):
    """Queued work must not switch prompt/generation settings mid-flight."""
    from services.llm import build_system_prompt

    cfg = pipeline.get_config()
    cfg.written_exam_mode = True
    cfg.models[0].id = "exam"
    cfg.written_exam_model_ids = ["exam"]
    cfg.written_exam_think = False
    cfg.language = "Rust"
    cfg.system_prompts = {"assist_exam": "按段落提交笔试答案。"}
    cfg.max_tokens = 777
    cfg.models[0].supports_vision = True

    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "broadcast", lambda _data: None)

    def fake_stream(_model_cfg, messages, **kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["question"] = messages[-1]["content"][0]["text"]
        captured["override_think_mode"] = kwargs.get("override_think_mode")
        captured["override_max_tokens"] = kwargs.get("override_max_tokens")
        yield ("text", "截图答案")

    monkeypatch.setattr(answer_worker, "build_system_prompt", build_system_prompt)
    monkeypatch.setattr("services.vision_verify.schedule_self_verify", lambda **_kw: None)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    assert pipeline.submit_answer_task(
        (
            "截图题",
            "data:image/png;base64,queued",
            True,
            "server_screen_left",
            {"origin": "server_screen", "image_count": 1},
        )
    )
    assert len(_DeferredThread.started) == 1

    # Simulate a settings save while the worker is queued.
    cfg.written_exam_mode = False
    cfg.written_exam_think = True
    cfg.language = "Python"
    cfg.system_prompts["assist_exam"] = "改用表格。"
    cfg.max_tokens = 2048

    _DeferredThread.started[0].run()

    assert captured["system_prompt"] == "按段落提交笔试答案。"
    assert captured["question"] == "[本轮任务资料]\n首选编程语言：Rust\n\n[当前输入]\n截图题"
    assert captured["override_think_mode"] is False
    assert captured["override_max_tokens"] == 777


def test_parallel_answers_commit_in_submit_order_when_workers_finish_out_of_order(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pipeline, "broadcast", lambda _data: None)

    def fake_stream(model_cfg, messages, **_kwargs):
        question = messages[-1]["content"]
        yield ("text", f"{model_cfg.name}:{question}")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    assert pipeline.submit_answer_task(
        ("第一个问题", None, True, "manual_text", {"origin": "manual"})
    )
    assert pipeline.submit_answer_task(
        ("第二个问题", None, True, "manual_text", {"origin": "manual"})
    )
    assert len(_DeferredThread.started) == 2

    _DeferredThread.started[1].run()
    assert get_session().qa_pairs == []
    assert pipeline._next_commit_seq == 0

    _DeferredThread.started[0].run()

    session = get_session()
    assert [qa.question for qa in session.qa_pairs] == ["第一个问题", "第二个问题"]
    assert [qa.answer for qa in session.qa_pairs] == [
        "模型一:第一个问题",
        "模型二:第二个问题",
    ]
    assert pipeline._next_commit_seq == 2
    assert pipeline._commit_buffer == {}


def test_cancelling_a_plain_answer_waiting_for_an_earlier_commit_emits_terminal_event(monkeypatch):
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", lambda *_args, **_kwargs: iter([("text", "已显示。")]))
    for question in ("第一个问题", "第二个问题"):
        assert pipeline.submit_answer_task((question, None, True, "manual_text", {"origin": "manual"}))

    _DeferredThread.started[1].run()
    qa_id = broadcasts[0]["id"]
    assert [event["type"] for event in broadcasts] == ["answer_start", "answer_chunk"]
    pipeline.cancel_answer_work()
    _DeferredThread.started[0].run()

    assert broadcasts[-1] == {"type": "answer_cancelled", "id": qa_id, "attempt": 1}
    assert get_session().qa_pairs == []


def test_cancelling_between_plain_stream_and_commit_discards_it_and_emits_terminal_event(monkeypatch):
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", lambda *_args, **_kwargs: iter([("text", "已显示。")]))
    flush_commit = pipeline._flush_commit

    def cancel_then_commit(seq, apply_fn):
        pipeline.cancel_answer_work()
        flush_commit(seq, apply_fn)

    monkeypatch.setattr(pipeline, "_flush_commit", cancel_then_commit)
    assert pipeline.submit_answer_task(("已取消", None, True, "manual_text", {"origin": "manual"}))
    _DeferredThread.started[0].run()

    assert [event["type"] for event in broadcasts] == ["answer_start", "answer_chunk", "answer_cancelled"]
    assert get_session().qa_pairs == []


def test_cancel_after_dispatch_before_worker_start_suppresses_stale_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    def fail_stream(*_args, **_kwargs):
        raise AssertionError("cancelled worker must not call the LLM")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fail_stream)

    assert pipeline.submit_answer_task(
        ("取消前的问题", None, True, "manual_text", {"origin": "manual"})
    )
    assert len(_DeferredThread.started) == 1

    pipeline.cancel_answer_work(reset_session_data=False)
    _DeferredThread.started[0].run()

    assert broadcasts == []
    assert get_session().qa_pairs == []
    assert pipeline._pending == []
    assert pipeline._in_flight_tasks == {}


def test_running_asr_worker_is_not_cancelled_by_new_asr_turn_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    def fake_stream(*_args, **_kwargs):
        yield ("text", "旧回答继续提交")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    pipeline._latest_asr_turn_id = 1
    assert pipeline.submit_answer_task(
        (
            "旧 ASR 问题",
            None,
            False,
            "conversation_mic",
            {"origin": "asr", "asr_turn_id": 1},
        )
    )
    assert len(_DeferredThread.started) == 1

    assert pipeline._begin_asr_turn() == 2
    _DeferredThread.started[0].run()

    event_types = [event["type"] for event in broadcasts]
    assert event_types == ["answer_start", "answer_chunk", "answer_done", "token_update"]
    assert [qa.answer for qa in get_session().qa_pairs] == ["旧回答继续提交"]
    assert pipeline._next_commit_seq == 1
    assert pipeline._in_flight_tasks == {}


def test_running_asr_worker_can_still_be_cancelled_when_interrupt_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg()
    cfg.assist_asr_interrupt_running = True
    cfg.max_parallel_answers = 1
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    def fake_stream(*_args, **_kwargs):
        # Simulate a newer ASR turn arriving while this worker is actively streaming.
        assert pipeline._begin_asr_turn() == 2
        yield ("text", "旧回答不应提交")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    pipeline._latest_asr_turn_id = 1
    assert pipeline.submit_answer_task(
        (
            "旧 ASR 问题",
            None,
            False,
            "conversation_mic",
            {"origin": "asr", "asr_turn_id": 1},
        )
    )
    _DeferredThread.started[0].run()

    event_types = [event["type"] for event in broadcasts]
    assert event_types == ["answer_start", "answer_cancelled"]
    assert get_session().qa_pairs == []


def test_preempted_asr_worker_before_start_suppresses_ghost_answer_start(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg()
    cfg.assist_asr_interrupt_running = True
    cfg.max_parallel_answers = 1
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)
    monkeypatch.setattr(
        answer_worker,
        "chat_stream_single_model",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("preempted worker must not stream")),
    )

    pipeline._latest_asr_turn_id = 1
    assert pipeline.submit_answer_task(
        (
            "已被抢占的旧 ASR 问题",
            None,
            False,
            "conversation_mic",
            {"origin": "asr", "asr_turn_id": 1},
        )
    )
    assert pipeline._begin_asr_turn() == 2
    _DeferredThread.started[0].run()

    assert broadcasts == []
    assert get_session().qa_pairs == []
    assert pipeline._next_commit_seq == 1


def test_running_asr_worker_is_not_cancelled_when_parallel_slots_available(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg()
    cfg.assist_asr_interrupt_running = True
    cfg.max_parallel_answers = 2
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    def fake_stream(*_args, **_kwargs):
        yield ("text", "旧回答应继续提交")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    pipeline._latest_asr_turn_id = 1
    assert pipeline.submit_answer_task(
        (
            "旧 ASR 问题",
            None,
            False,
            "conversation_mic",
            {"origin": "asr", "asr_turn_id": 1},
        )
    )
    assert pipeline._begin_asr_turn() == 2
    _DeferredThread.started[0].run()

    event_types = [event["type"] for event in broadcasts]
    assert event_types == ["answer_start", "answer_chunk", "answer_done", "token_update"]
    assert [qa.answer for qa in get_session().qa_pairs] == ["旧回答应继续提交"]


def test_late_asr_constraint_tail_updates_deferred_pending_task(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline, "broadcast", lambda _data: None)
    now_values = iter([100.0, 100.0])
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(now_values))

    task = (
        "rules 和 skills 的区别是什么",
        None,
        False,
        "conversation_loopback",
        {
            "origin": "asr",
            "asr_turn_id": 1,
            "utterances": ["rules 和 skills 的区别是什么"],
            "dispatch_after_mono": 101.2,
            "asr_tail_grace_until_mono": 101.2,
        },
    )

    assert pipeline.submit_answer_task(task)
    assert _DeferredThread.started == []
    assert len(_DeferredTimer.started) == 1

    merged = pipeline._append_late_asr_constraint_tail(
        "不要结合项目",
        "conversation_loopback",
        now_mono=100.6,
    )

    assert merged is True
    assert len(pipeline._pending) == 1
    updated_task = pipeline._pending[0][0]
    assert "rules 和 skills 的区别是什么" in updated_task[0]
    assert "不要结合项目" in updated_task[0]
    assert updated_task[4]["utterances"] == [
        "rules 和 skills 的区别是什么",
        "不要结合项目",
    ]


def test_stop_waits_for_worker_flush_before_review_archive(monkeypatch: pytest.MonkeyPatch):
    session = get_session()
    session.is_recording = True
    archived_counts: list[int] = []

    class JoinAddsQa:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            session.add_qa(
                "最后 flush 出来的问题",
                "最后生成完成的答案",
                qa_id="qa-final",
            )

    pipeline._interview_thread = JoinAddsQa()
    monkeypatch.setattr(pipeline, "broadcast", lambda _data: None)
    monkeypatch.setattr(pipeline.audio_capture, "stop", lambda owner=None: None)
    monkeypatch.setattr(pipeline._candidate_audio_capture, "stop", lambda owner=None: None)
    monkeypatch.setattr(
        pipeline.review_integration,
        "on_assist_stop",
        lambda archived_session: archived_counts.append(len(archived_session.qa_pairs)),
    )

    pipeline.stop_interview_loop()

    assert archived_counts == [1]


def test_candidate_audio_start_failure_does_not_block_interviewer_chain(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    review_calls: list[dict] = []

    class _MainAudio:
        SAMPLE_RATE = 16000

        def __init__(self):
            self.start_calls = []
            self.stop_calls = []

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))

        def stop(self, owner=None):
            self.stop_calls.append(owner)

    class _BrokenCandidateAudio(_MainAudio):
        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))
            raise RuntimeError("mic unavailable")

    cfg = _cfg()
    cfg.candidate_asr_enabled = True
    main_audio = _MainAudio()
    candidate_audio = _BrokenCandidateAudio()
    session = get_session()

    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _device_id: True)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)
    monkeypatch.setattr(
        pipeline.review_integration,
        "on_assist_start",
        lambda **kwargs: review_calls.append(kwargs),
    )

    pipeline.start_nonblocking(10, 11)

    assert session.is_recording is True
    assert main_audio.start_calls == [(10, "assist", {})]
    assert candidate_audio.start_calls == [(11, "assist-candidate", {"mic_compatibility_mode": True})]
    assert len(_DeferredThread.started) == 1
    assert _DeferredThread.started[0].target is pipeline._interview_worker
    assert any(
        event.get("type") == "candidate_asr_status"
        and event.get("provider") == "off"
        and event.get("safe_degraded") is True
        and "mic unavailable" in event.get("error", "")
        for event in broadcasts
    )
    assert review_calls == [
        {
            "interviewer_device_id": 10,
            "candidate_device_id": 11,
            "candidate_asr_enabled": True,
            "written_exam_mode": False,
            "single_mic": False,
            "existing_qa_ids": frozenset(),
        }
    ]


def test_unpause_restarts_candidate_audio_from_last_device_when_worker_stopped(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    class _Audio:
        def __init__(self):
            self.start_calls = []
            self._is_running = False

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))
            self._is_running = True

        @property
        def is_running(self):
            return self._is_running

    class _AliveThread:
        def is_alive(self):
            return True

    cfg = _cfg()
    cfg.candidate_asr_enabled = True
    main_audio = _Audio()
    candidate_audio = _Audio()
    session = get_session()
    session.last_device_id = 10
    session.last_candidate_mic_device_id = 11

    pipeline._interview_thread = _AliveThread()
    pipeline._candidate_thread = None
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _device_id: True)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    pipeline.unpause_interview()

    assert main_audio.start_calls == [(10, "assist", {})]
    assert candidate_audio.start_calls == [(11, "assist-candidate", {"mic_compatibility_mode": True})]
    assert session.last_candidate_mic_device_id == 11
    assert _DeferredThread.started[-1].target is pipeline._candidate_worker


def test_unpause_written_exam_ignores_audio_devices_and_clears_stale_state(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    class _Audio:
        def __init__(self):
            self.start_calls = []
            self._is_running = True

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))

        @property
        def is_running(self):
            return self._is_running

    cfg = _cfg()
    cfg.written_exam_mode = True
    cfg.candidate_asr_enabled = True
    main_audio = _Audio()
    candidate_audio = _Audio()
    session = get_session()
    session.last_device_id = 20
    session.last_candidate_mic_device_id = 21
    session.capture_is_loopback = True
    session.is_paused = True

    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    pipeline.unpause_interview(10, 11)

    assert main_audio.start_calls == []
    assert candidate_audio.start_calls == []
    assert session.last_device_id == -1
    assert session.last_candidate_mic_device_id == -1
    assert session.capture_is_loopback is False
    assert session.is_paused is False
    assert broadcasts[-1] == {"type": "paused", "value": False}


def test_unpause_restarts_interviewer_from_last_device_when_worker_stopped(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    class _Audio:
        def __init__(self):
            self.start_calls = []
            self._is_running = False

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))
            self._is_running = True

        @property
        def is_running(self):
            return self._is_running

    cfg = _cfg()
    cfg.candidate_asr_enabled = False
    main_audio = _Audio()
    candidate_audio = _Audio()
    session = get_session()
    session.last_device_id = 10
    session.last_candidate_mic_device_id = -1
    session.is_paused = True

    pipeline._interview_thread = None
    pipeline._candidate_thread = None
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _device_id: True)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    pipeline.unpause_interview()

    assert main_audio.start_calls == [(10, "assist", {})]
    assert candidate_audio.start_calls == []
    assert session.last_device_id == 10
    assert session.is_paused is False
    assert any(t.target is pipeline._interview_worker for t in _DeferredThread.started)
    assert broadcasts[-1] == {"type": "paused", "value": False}


def test_unpause_skips_interviewer_when_last_device_is_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    class _Audio:
        def __init__(self):
            self.start_calls = []
            self._is_running = False

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))
            self._is_running = True

        @property
        def is_running(self):
            return self._is_running

    cfg = _cfg()
    main_audio = _Audio()
    candidate_audio = _Audio()
    session = get_session()
    session.last_device_id = -1
    session.last_candidate_mic_device_id = -1
    session.is_paused = True

    pipeline._interview_thread = None
    pipeline._candidate_thread = None
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    pipeline.unpause_interview()

    assert main_audio.start_calls == []
    assert candidate_audio.start_calls == []
    assert session.last_device_id == -1
    assert session.is_paused is False
    assert _DeferredThread.started == []
    assert broadcasts[-1] == {"type": "paused", "value": False}


def test_unpause_restarts_interviewer_from_device_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    """PortAudio index 0 is a real device; resume must not treat it as unset."""
    broadcasts: list[dict] = []

    class _Audio:
        def __init__(self):
            self.start_calls = []
            self._is_running = False

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))
            self._is_running = True

        @property
        def is_running(self):
            return self._is_running

    cfg = _cfg()
    cfg.candidate_asr_enabled = False
    main_audio = _Audio()
    candidate_audio = _Audio()
    session = get_session()
    session.last_device_id = 0
    session.last_candidate_mic_device_id = -1
    session.is_paused = True

    pipeline._interview_thread = None
    pipeline._candidate_thread = None
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _device_id: True)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    pipeline.unpause_interview()

    assert main_audio.start_calls == [(0, "assist", {})]
    assert candidate_audio.start_calls == []
    assert session.last_device_id == 0
    assert session.is_paused is False
    assert any(t.target is pipeline._interview_worker for t in _DeferredThread.started)
    assert broadcasts[-1] == {"type": "paused", "value": False}


@pytest.mark.parametrize(
    ("candidate_enabled", "last_candidate_id", "expected_reason"),
    [
        (False, 11, "disabled"),
        (True, 10, "missing_or_same_device"),
    ],
)
def test_unpause_clears_invalid_candidate_audio_state(
    monkeypatch: pytest.MonkeyPatch,
    candidate_enabled: bool,
    last_candidate_id: int,
    expected_reason: str,
):
    broadcasts: list[dict] = []

    class _Audio:
        def __init__(self):
            self.start_calls = []
            self._is_running = False

        def start(self, device_id, owner=None, **kwargs):
            self.start_calls.append((device_id, owner, kwargs))
            self._is_running = True

        @property
        def is_running(self):
            return self._is_running

    class _AliveThread:
        def is_alive(self):
            return True

    cfg = _cfg()
    cfg.candidate_asr_enabled = candidate_enabled
    main_audio = _Audio()
    candidate_audio = _Audio()
    session = get_session()
    session.last_device_id = 10
    session.last_candidate_mic_device_id = last_candidate_id

    pipeline._interview_thread = _AliveThread()
    pipeline._candidate_thread = None
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "audio_capture", main_audio)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_audio)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _device_id: True)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    pipeline.unpause_interview()

    assert main_audio.start_calls == [(10, "assist", {})]
    assert candidate_audio.start_calls == []
    assert session.last_candidate_mic_device_id == -1
    assert any(
        event.get("type") == "candidate_asr_status"
        and event.get("provider") == "off"
        and event.get("reason") == expected_reason
        for event in broadcasts
    )
