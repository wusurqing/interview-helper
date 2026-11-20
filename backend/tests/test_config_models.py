from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.common.config_payload import build_config_payload  # noqa: E402
from api.common.router import router  # noqa: E402
from core import config  # noqa: E402
from core.config import AppConfig, ModelConfig  # noqa: E402


@pytest.fixture
def model_config_client(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "max_tokens": 4096,
        "models": [
            {"name": "Legacy", "api_key": "sk-legacy"},
            {"name": "Configured", "api_key": "sk-configured", "context_window_tokens": 65536},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", config_path)
    monkeypatch.setattr(config, "_config", None)
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        yield client


def test_model_context_window_round_trips_through_reorder_and_preserves_saved_keys(model_config_client):
    client = model_config_client
    old_models = client.get("/api/config/models-full").json()["models"]
    assert [model["context_window_tokens"] for model in old_models] == [262144, 65536]
    assert all(model["api_key"] == "" and model["has_key"] for model in old_models)
    assert [model["context_window_tokens"] for model in client.get("/api/config").json()["models"]] == [262144, 65536]

    saved = client.post("/api/config", json={"models": [
        {**old_models[1], "model_original_index": 1, "api_key": "__IA_KEEP_EXISTING_API_KEY__"},
        {
            **old_models[0],
            "model_original_index": 0,
            "api_key": "__IA_KEEP_EXISTING_API_KEY__",
            "context_window_tokens": 256,
        },
        {"name": "New"},
    ]})

    assert saved.status_code == 200
    assert saved.json()["max_tokens"] == 4096
    assert [model["context_window_tokens"] for model in saved.json()["models"]] == [65536, 256, 262144]
    config._config = None
    persisted = config.get_config().models
    assert [(model.name, model.api_key, model.context_window_tokens) for model in persisted] == [
        ("Configured", "sk-configured", 65536),
        ("Legacy", "sk-legacy", 256),
        ("New", "", 262144),
    ]


@pytest.mark.parametrize("value", [False, True, 0, -1, "1024", 1.0, 1.5, None, 2**53])
def test_model_context_window_rejects_invalid_values_without_saving(model_config_client, value):
    client = model_config_client
    response = client.post("/api/config", json={"models": [{"name": "Invalid", "context_window_tokens": value}]})

    assert response.status_code == 400
    assert "context_window_tokens" in response.json()["detail"]
    assert [model["name"] for model in client.get("/api/config").json()["models"]] == ["Legacy", "Configured"]


@pytest.mark.parametrize("value", [1, 2**53 - 1])
def test_model_context_window_accepts_positive_safe_integer_bounds(value):
    assert ModelConfig.model_validate({"context_window_tokens": value}).model_dump()["context_window_tokens"] == value


def test_active_model_falls_back_to_first_enabled_model():
    cfg = AppConfig(
        models=[
            ModelConfig(name="disabled", enabled=False),
            ModelConfig(name="enabled", enabled=True),
        ],
        active_model=0,
    )

    assert cfg.active_model == 1
    assert cfg.get_active_model().name == "enabled"


def test_active_model_keeps_clamped_index_when_all_models_are_disabled():
    cfg = AppConfig(
        models=[
            ModelConfig(name="disabled-a", enabled=False),
            ModelConfig(name="disabled-b", enabled=False),
        ],
        active_model=5,
    )

    assert cfg.active_model == 1


def test_candidate_remote_stt_requires_explicit_cost_opt_in():
    cfg = AppConfig(
        candidate_stt_provider="doubao",
        candidate_remote_stt_enabled=False,
    )

    assert cfg.candidate_stt_provider == ""
    assert not cfg.candidate_asr_enabled


def test_candidate_mic_asr_defaults_to_opt_in():
    cfg = AppConfig()

    assert cfg.candidate_asr_enabled is False


def test_candidate_remote_stt_can_be_enabled_explicitly():
    cfg = AppConfig(
        candidate_stt_provider="generic",
        candidate_remote_stt_enabled=True,
    )

    assert cfg.candidate_stt_provider == "generic"


def test_think_effort_accepts_xhigh():
    cfg = AppConfig(think_mode=True, think_effort="xhigh")

    assert cfg.think_mode is True
    assert cfg.think_effort == "xhigh"


def test_generation_params_are_normalized_for_runtime_safety():
    cfg = AppConfig(temperature=float("nan"), max_tokens=999999)

    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 32768

    cfg = AppConfig(temperature=-3, max_tokens=8)

    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 256


def test_invalid_generation_param_strings_fall_back_to_defaults():
    cfg = AppConfig(temperature="not-a-number", max_tokens="not-a-number")

    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 4096


def test_assist_stop_answer_wait_default_is_preserved():
    cfg = AppConfig()

    assert cfg.assist_stop_answer_wait_sec == 3.0


def test_assist_stop_answer_wait_can_be_disabled_explicitly():
    cfg = AppConfig(assist_stop_answer_wait_sec=0)

    assert cfg.assist_stop_answer_wait_sec == 0.0


def test_config_payload_includes_realtime_voice_runtime_knobs():
    cfg = AppConfig(
        assist_realtime_max_tokens=800,
        assist_realtime_high_churn_max_tokens=360,
        assist_stop_answer_wait_sec=2.5,
        assist_interviewer_asr_drain_timeout_sec=8.0,
    )

    payload = build_config_payload(cfg)

    assert payload["assist_realtime_max_tokens"] == 800
    assert payload["assist_realtime_high_churn_max_tokens"] == 360
    assert payload["assist_stop_answer_wait_sec"] == 2.5
    assert payload["assist_interviewer_asr_drain_timeout_sec"] == 8.0


def test_config_payload_model_health_fingerprint_changes_with_credentials():
    cfg = AppConfig()
    cfg.models[0].api_key = "sk-first"
    first = build_config_payload(cfg)["models"][0]["health_fingerprint"]

    cfg.models[0].api_key = "sk-second"
    second = build_config_payload(cfg)["models"][0]["health_fingerprint"]

    assert first
    assert second
    assert first != second
    assert "sk-first" not in first
    assert "sk-second" not in second
