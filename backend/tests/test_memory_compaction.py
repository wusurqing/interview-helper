"""Session memory compaction reliability & session isolation tests (P1-5)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from core.config import AppConfig, ModelConfig
from core.session import Session, reset_session
from services import memory
from services.llm.streaming import _THINK_DISABLED_BASE_PARAMS


def _seed_history(session: Session, count: int, prefix: str = "msg") -> list[dict]:
    msgs = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{prefix}-{i}"}
        for i in range(count)
    ]
    session.conversation_history = list(msgs)
    return msgs


def test_compaction_failure_preserves_history_and_prior_summary(monkeypatch: pytest.MonkeyPatch):
    session = Session()
    session.system_summary = "existing summary"
    original = _seed_history(session, 10)

    def fail_summarize(_snapshot, _prior):
        raise RuntimeError("upstream timeout")

    monkeypatch.setattr(memory, "summarize_messages", fail_summarize)

    thread = memory.schedule_compaction(session)
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert session.conversation_history == original
    assert session.system_summary == "existing summary"
    assert session._compaction_running is False


@pytest.mark.parametrize(
    "fake_response",
    [
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="   "))], usage=None),
    ],
)
def test_empty_llm_summary_raises_and_does_not_truncate_history(
    monkeypatch: pytest.MonkeyPatch,
    fake_response,
):
    session = Session()
    session.system_summary = "prior notes"
    original = _seed_history(session, 10)

    cfg = AppConfig(
        models=[
            ModelConfig(
                name="Test",
                api_base_url="https://api.example.test/v1",
                api_key="sk-test",
                model="gpt-4o-mini",
            )
        ]
    )
    monkeypatch.setattr("core.config.get_config", lambda: cfg)
    monkeypatch.setattr(
        "services.llm.streaming.chat_completion_create",
        lambda *_args, **_kwargs: fake_response,
    )

    with pytest.raises(RuntimeError):
        memory.summarize_messages(original[:6], "prior notes")

    thread = memory.schedule_compaction(session)
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert session.conversation_history == original
    assert session.system_summary == "prior notes"
    assert session._compaction_running is False


def test_session_clear_during_compaction_blocks_stale_writeback_and_flag_clobber(
    monkeypatch: pytest.MonkeyPatch,
):
    session = Session()
    _seed_history(session, 10, prefix="old")
    entered = threading.Event()
    release = threading.Event()

    def slow_summarize(_snapshot, _prior):
        entered.set()
        assert release.wait(timeout=5)
        return "- old session summary"

    monkeypatch.setattr(memory, "summarize_messages", slow_summarize)

    thread = memory.schedule_compaction(session)
    assert thread is not None
    assert entered.wait(timeout=5)

    # User clears session and starts a new conversation (even with another compaction flag active).
    old_version = session.session_version
    session.clear()
    assert session.session_version == old_version + 1
    new_msgs = _seed_history(session, 8, prefix="new")
    session._compaction_running = True  # simulate a new compaction in flight for new session

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    # Stale worker must not overwrite new session summary, history, or its _compaction_running flag.
    assert session.system_summary == ""
    assert session.conversation_history == new_msgs
    assert session._compaction_running is True


def test_reset_session_singleton_during_compaction_blocks_stale_writeback(
    monkeypatch: pytest.MonkeyPatch,
):
    session = reset_session()
    try:
        _seed_history(session, 10, prefix="old")
        entered = threading.Event()
        release = threading.Event()

        def slow_summarize(_snapshot, _prior):
            entered.set()
            assert release.wait(timeout=5)
            return "- stale singleton summary"

        monkeypatch.setattr(memory, "summarize_messages", slow_summarize)

        thread = memory.schedule_compaction(session)
        assert thread is not None
        assert entered.wait(timeout=5)

        reset_session()
        new_msgs = _seed_history(session, 6, prefix="fresh")

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        assert session.system_summary == ""
        assert session.conversation_history == new_msgs
        assert session._compaction_running is False
    finally:
        reset_session()


def test_prefix_drift_skips_writeback_instead_of_dropping_unrelated_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    session = Session()
    _seed_history(session, 10, prefix="v1")
    entered = threading.Event()
    release = threading.Event()

    def slow_summarize(_snapshot, _prior):
        entered.set()
        assert release.wait(timeout=5)
        return "- summary of v1"

    monkeypatch.setattr(memory, "summarize_messages", slow_summarize)

    thread = memory.schedule_compaction(session)
    assert thread is not None
    assert entered.wait(timeout=5)

    # Simulate _trim_history shifting the front of conversation_history while worker runs.
    drifted = _seed_history(session, 10, prefix="shifted")
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert session.conversation_history == drifted
    assert session.system_summary == ""
    assert session._compaction_running is False


def test_prefix_drift_with_identical_messages_skips_writeback(
    monkeypatch: pytest.MonkeyPatch,
):
    session = Session()
    session.conversation_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "same"}
        for i in range(18)
    ]
    entered = threading.Event()
    release = threading.Event()

    def slow_summarize(_snapshot, _prior):
        entered.set()
        assert release.wait(timeout=5)
        return "- summary of repeated msgs"

    monkeypatch.setattr(memory, "summarize_messages", slow_summarize)

    thread = memory.schedule_compaction(session)
    assert thread is not None
    assert entered.wait(timeout=5)

    # Append 4 identical messages to force _trim_history to drop oldest 2 dict objects.
    for i in range(4):
        if i % 2 == 0:
            session.add_user_message("same")
        else:
            session.add_assistant_message("same")
    assert len(session.conversation_history) == session.MAX_HISTORY
    expected_after_trim = list(session.conversation_history)

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    # Identity check must detect the shift even when message contents are equal.
    assert session.conversation_history == expected_after_trim
    assert session.system_summary == ""
    assert session._compaction_running is False


def test_prefix_in_place_mutation_skips_writeback(monkeypatch: pytest.MonkeyPatch):
    session = Session()
    _seed_history(session, 10, prefix="orig")
    entered = threading.Event()
    release = threading.Event()

    def slow_summarize(_snapshot, _prior):
        entered.set()
        assert release.wait(timeout=5)
        return "- summary before edit"

    monkeypatch.setattr(memory, "summarize_messages", slow_summarize)

    thread = memory.schedule_compaction(session)
    assert thread is not None
    assert entered.wait(timeout=5)

    session.conversation_history[0]["content"] = "mutated-in-place"
    expected = [dict(m) for m in session.conversation_history]

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert session.conversation_history == expected
    assert session.system_summary == ""
    assert session._compaction_running is False


def test_compaction_success_preserves_concurrently_appended_tail(monkeypatch: pytest.MonkeyPatch):
    session = Session()
    original = _seed_history(session, 10, prefix="turn")
    entered = threading.Event()
    release = threading.Event()

    def slow_summarize(snapshot, prior):
        entered.set()
        assert release.wait(timeout=5)
        assert snapshot == original[:6]
        assert prior == ""
        return "- 候选人介绍了分布式锁实现"

    monkeypatch.setattr(memory, "summarize_messages", slow_summarize)

    thread = memory.schedule_compaction(session)
    assert thread is not None
    assert entered.wait(timeout=5)

    extra = {"role": "user", "content": "追问：锁续期怎么做的？"}
    session.conversation_history.append(extra)
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert session.system_summary == "- 候选人介绍了分布式锁实现"
    assert session.conversation_history == original[6:] + [extra]
    assert session._compaction_running is False


@pytest.mark.parametrize(
    "model_kwargs,expected_think_params",
    [
        (
            {
                "name": "DeepSeek R1",
                "api_base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-ds-test",
                "api_type": "openai",
                "model": "deepseek-reasoner",
                "supports_think": True,
            },
            _THINK_DISABLED_BASE_PARAMS,
        ),
        (
            {
                "name": "Claude Explicit Disable",
                "api_base_url": "https://api.anthropic.com",
                "api_key": "sk-ant-test",
                "api_type": "anthropic",
                "model": "claude-3-7-sonnet-20250219",
                "supports_think": True,
                "think_disabled_params": {"thinking": {"type": "disabled"}},
            },
            {"thinking": {"type": "disabled"}},
        ),
        (
            {
                "name": "Claude Default Off",
                "api_base_url": "https://api.anthropic.com",
                "api_key": "sk-ant-test",
                "api_type": "anthropic",
                "model": "claude-3-7-sonnet-20250219",
                "supports_think": True,
            },
            {},
        ),
    ],
)
def test_summarize_messages_disables_thinking_even_when_global_think_mode_enabled(
    monkeypatch: pytest.MonkeyPatch,
    model_kwargs: dict,
    expected_think_params: dict,
):
    model = ModelConfig(**model_kwargs)
    cfg = AppConfig(models=[model], think_mode=True, think_effort="high")
    monkeypatch.setattr("core.config.get_config", lambda: cfg)

    captured: dict = {}

    def fake_create(model_cfg, messages, *, temperature, max_tokens, think_params=None, trace_context=None):
        captured["model_cfg"] = model_cfg
        captured["max_tokens"] = max_tokens
        captured["think_params"] = think_params
        captured["trace_context"] = trace_context
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="- 摘要条目"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    monkeypatch.setattr("services.llm.streaming.chat_completion_create", fake_create)
    monkeypatch.setattr("services.llm._add_tokens", lambda *_a, **_kw: None)

    result = memory.summarize_messages([{"role": "user", "content": "介绍你的项目"}], "")
    assert result == "- 摘要条目"
    assert captured["max_tokens"] == 600
    assert captured["think_params"] == expected_think_params
    assert captured["trace_context"] == {"scene": "memory_compaction"}


def test_schedule_compaction_thread_start_failure_resets_flag_and_maybe_compact_survives(
    monkeypatch: pytest.MonkeyPatch,
):
    session = Session()
    _seed_history(session, session.SUMMARY_TRIGGER, prefix="msg")

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(memory.threading, "Thread", FailingThread)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        memory.schedule_compaction(session)
    assert session._compaction_running is False

    # Adding a message above SUMMARY_TRIGGER triggers _maybe_compact(), which must not crash caller.
    session.add_user_message("new question")
    assert session._compaction_running is False
    assert session.conversation_history[-1] == {"role": "user", "content": "new question"}


def test_schedule_compaction_skips_when_below_keep_or_already_running():
    session = Session()
    _seed_history(session, session.SUMMARY_KEEP_RECENT, prefix="short")

    assert memory.schedule_compaction(session) is None
    assert session._compaction_running is False

    _seed_history(session, 10, prefix="long")
    session._compaction_running = True
    assert memory.schedule_compaction(session) is None
    assert session._compaction_running is True
