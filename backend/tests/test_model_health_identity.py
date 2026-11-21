from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

model_health = importlib.import_module("api.common.model_health")


def _model(name: str, model: str, key: str = "sk-test") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        api_base_url="https://example.test/v1",
        api_key=key,
        model=model,
        enabled=True,
        supports_vision=False,
        supports_think=False,
        think_enabled_params={},
        think_disabled_params={},
    )


@pytest.fixture(autouse=True)
def clear_health_state():
    model_health._model_health.clear()
    model_health._model_health_detail.clear()
    model_health._model_health_latency.clear()
    model_health._model_health_fingerprint.clear()
    yield
    model_health._model_health.clear()
    model_health._model_health_detail.clear()
    model_health._model_health_latency.clear()
    model_health._model_health_fingerprint.clear()


def test_health_status_is_not_reused_after_models_reorder(monkeypatch: pytest.MonkeyPatch):
    first = _model("first", "model-a")
    second = _model("second", "model-b")
    cfg = SimpleNamespace(models=[first, second])
    monkeypatch.setattr(model_health, "get_config", lambda: cfg)

    fingerprint = model_health.model_health_fingerprint(first)
    assert model_health._store_model_health(0, fingerprint, "error", "old failure", 0)
    assert model_health.get_model_health(0) == "error"

    cfg.models = [second, first]

    assert model_health.get_model_health(0) is None
    snapshot = model_health.get_model_health_snapshot()
    assert 0 not in snapshot["health"]
    assert 0 not in snapshot["detail"]
    assert 0 not in snapshot["fingerprint"]


def test_queued_probe_drops_result_when_index_changes_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    first = _model("first", "model-a")
    second = _model("second", "model-b")
    cfg = SimpleNamespace(models=[first, second])
    submitted: list[tuple[object, tuple[object, ...]]] = []
    events: list[dict] = []

    monkeypatch.setattr(model_health, "get_config", lambda: cfg)
    monkeypatch.setattr(
        model_health,
        "submit_low_priority_background",
        lambda fn, *args: submitted.append((fn, args)) or True,
    )
    monkeypatch.setattr(
        model_health,
        "_probe_basic",
        lambda *_args, **_kwargs: pytest.fail("stale probe must not call provider"),
    )
    ws = importlib.import_module("api.realtime.ws")
    monkeypatch.setattr(ws, "broadcast", events.append)

    assert model_health.start_single_model_check(0) is True
    assert len(submitted) == 1

    cfg.models = [second, first]
    fn, args = submitted[0]
    fn(*args)

    assert events == []
    assert model_health.get_model_health(0) is None


def test_api_key_change_invalidates_existing_health(monkeypatch: pytest.MonkeyPatch):
    model = _model("first", "model-a", key="sk-old")
    cfg = SimpleNamespace(models=[model])
    monkeypatch.setattr(model_health, "get_config", lambda: cfg)

    old_fingerprint = model_health.model_health_fingerprint(model)
    assert model_health._store_model_health(0, old_fingerprint, "ok", "", 123)

    model.api_key = "sk-new"

    assert model_health.get_model_health(0) is None


def test_claude_think_probe_uses_min_1024_budget_and_matching_max_tokens(monkeypatch: pytest.MonkeyPatch):
    claude_model = _model("Claude", "claude-3-7-sonnet")
    candidates = dict(model_health._think_probe_candidates(claude_model))
    assert candidates["claude_thinking_budget"]["thinking"]["budget_tokens"] >= 1024

    seen_payloads: list[dict] = []

    def fake_post_chat(_model_cfg, payload: dict, timeout: int = 12):
        seen_payloads.append(payload)
        return {"choices": [{"message": {"content": "2", "reasoning_content": "1+1=2"}}]}, 10

    monkeypatch.setattr(model_health, "_post_chat", fake_post_chat)
    ok, style, params, _detail = model_health._probe_think(claude_model)
    assert ok is True
    assert style == "claude_thinking_budget"
    budget = params["thinking"]["budget_tokens"]
    assert budget >= 1024
    assert seen_payloads[0]["max_tokens"] > budget


def test_probe_think_rejects_empty_response_body_and_short_circuits_non_param_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _model("DeepSeek", "deepseek-r1")
    calls = {"think": 0, "disable": 0}

    # Empty response body (neither reasoning nor text) must not be treated as think accepted
    monkeypatch.setattr(model_health, "_post_chat", lambda *_a, **_kw: ({}, 5))
    ok, style, params, detail = model_health._probe_think(model)
    assert ok is False
    assert style == ""
    assert params == {}
    assert "未返回有效内容" in detail

    # Non-parameter error (e.g., HTTP 401 / timeout) must break immediately instead of trying all candidates
    def fail_think(*_a, **_kw):
        calls["think"] += 1
        raise RuntimeError("HTTP 401: Unauthorized")

    monkeypatch.setattr(model_health, "_post_chat", fail_think)
    ok, _style, _params, detail = model_health._probe_think(model)
    assert ok is False
    assert calls["think"] == 1
    assert "401" in detail

    def fail_disable(*_a, **_kw):
        calls["disable"] += 1
        raise RuntimeError("Read timed out")

    monkeypatch.setattr(model_health, "_post_chat", fail_disable)
    disabled_params, disable_detail = model_health._probe_disable_think(model)
    assert disabled_params == {}
    assert calls["disable"] == 1
    assert "timed out" in disable_detail


def test_start_all_model_checks_acquires_lock_when_collecting_stale_indexes(
    monkeypatch: pytest.MonkeyPatch,
):
    first = _model("first", "model-a")
    cfg = SimpleNamespace(models=[first])
    monkeypatch.setattr(model_health, "get_config", lambda: cfg)
    monkeypatch.setattr(model_health, "start_single_model_check", lambda _i: True)

    model_health._model_health[3] = "ok"
    model_health._model_health_detail[3] = ""

    lock_held_on_iter = {"ok": False}
    real_lock = model_health._model_health_lock

    class CheckedHealthDict(dict):
        def __iter__(self):
            # RLock._is_owned() is available in CPython threading.RLock
            is_owned = getattr(real_lock, "_is_owned", None)
            if callable(is_owned):
                lock_held_on_iter["ok"] = bool(is_owned())
            else:
                lock_held_on_iter["ok"] = True
            return super().__iter__()

    checked = CheckedHealthDict(model_health._model_health)
    monkeypatch.setattr(model_health, "_model_health", checked)

    assert model_health.start_all_model_checks() is True
    assert lock_held_on_iter["ok"] is True
    assert 3 not in model_health._model_health
