from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

smoke_script = importlib.import_module("scripts.smoke_llm_protocols")


def _model(name: str = "target-model", api_type: str = "anthropic", model: str = "claude-3-7-sonnet"):
    return SimpleNamespace(
        name=name,
        api_type=api_type,
        model=model,
        enabled=True,
        api_key="sk-test",
    )


def test_smoke_model_uses_single_model_streaming_for_target_model(monkeypatch):
    target = _model()
    seen: dict[str, object] = {}

    def fake_create(model_cfg, messages, **_kwargs):
        seen["blocking_model"] = model_cfg
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="成功"))])

    def fake_stream(model_cfg, messages, **kwargs):
        seen["stream_model"] = model_cfg
        seen["stream_kwargs"] = kwargs
        yield ("text", "Redis 是内存数据库。")

    monkeypatch.setattr(smoke_script, "chat_completion_create", fake_create)
    monkeypatch.setattr(smoke_script, "chat_stream_single_model", fake_stream)

    assert smoke_script.smoke_model(target) is True
    assert seen["blocking_model"] is target
    assert seen["stream_model"] is target
    assert seen["stream_kwargs"] == {"override_think_mode": False}


def test_smoke_model_handles_whitespace_only_stream_without_type_error(monkeypatch):
    target = _model()

    monkeypatch.setattr(
        smoke_script,
        "chat_completion_create",
        lambda *_a, **_kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="成功"))]
        ),
    )
    monkeypatch.setattr(
        smoke_script,
        "chat_stream_single_model",
        lambda *_a, **_kw: iter([("text", "   "), ("text", "\n")]),
    )

    # Must not raise TypeError formatting first_ms=None; must report failure cleanly
    assert smoke_script.smoke_model(target) is False


def test_smoke_model_handles_empty_choices_and_zero_chunks(monkeypatch):
    target = _model()

    monkeypatch.setattr(
        smoke_script,
        "chat_completion_create",
        lambda *_a, **_kw: SimpleNamespace(choices=[]),
    )
    monkeypatch.setattr(
        smoke_script,
        "chat_stream_single_model",
        lambda *_a, **_kw: iter([]),
    )

    assert smoke_script.smoke_model(target) is False


def test_smoke_main_rejects_out_of_range_index(monkeypatch):
    monkeypatch.setattr(
        smoke_script,
        "get_config",
        lambda: SimpleNamespace(models=[_model()]),
    )

    assert smoke_script.main(["--index", "5"]) == 1
    assert smoke_script.main(["--index", "-1"]) == 1

    monkeypatch.setattr(
        smoke_script,
        "get_config",
        lambda: SimpleNamespace(models=[]),
    )
    assert smoke_script.main(["--index", "0"]) == 1
