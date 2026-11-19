import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.common.router import router
from core import config


@pytest.fixture
def realtime_settings(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "active_model": 1,
        "models": [
            {"name": "Vision", "api_key": "test-vision", "api_base_url": "https://vision.example.test/v1", "supports_vision": True},
            {"name": "Primary", "api_key": "test-primary", "api_base_url": "https://primary.example.test/v1"},
            {"name": "Disabled", "api_key": "test-disabled", "enabled": False},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "assist_realtime_max_tokens": 900,
        "assist_high_churn_short_answer": True,
        "system_prompts": {
            "assist_realtime": "standard voice",
            "assist_realtime_short": "retired voice",
        },
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", config_path)
    monkeypatch.setattr(config, "_config", None)
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        yield client


def test_migrates_standard_reply_and_legacy_model_order(realtime_settings):
    client = realtime_settings
    payload = client.get("/api/config").json()
    vision_id, primary_id, disabled_id = [model["id"] for model in payload["models"]]
    assert len({vision_id, primary_id, disabled_id}) == 3
    assert payload["models"][0]["has_api_key"]
    assert "test-vision" not in json.dumps(payload["models"])
    assert payload["assist_model_ids"] == [primary_id, vision_id]
    assert payload["assist_screenshot_model_id"] == vision_id
    assert set(payload["assist_generation"]) == {"assist_realtime", "assist_manual", "assist_screen", "assist_exam"}
    assert payload["assist_generation"]["assist_realtime"] == {"temperature": 0.7, "max_output_tokens": 512}
    assert payload["system_prompts"] == {"assist_realtime": "standard voice"}
    assert "assist_realtime_short" not in [scene["id"] for scene in client.get("/api/prompts").json()["scenes"]]
    config._config = None
    assert client.get("/api/config").json()["assist_model_ids"] == [primary_id, vision_id]


def test_scene_updates_are_independent_and_keep_retired_overrides_on_disk(realtime_settings):
    client = realtime_settings
    response = client.post("/api/config", json={
        "assist_generation": {"assist_realtime": {"temperature": 0.2, "max_output_tokens": 128}},
        "system_prompts": {"assist_realtime": "new voice"},
    })
    assert response.status_code == 200
    assert response.json()["assist_generation"]["assist_manual"] == {"temperature": 0.7, "max_output_tokens": 512}
    assert client.post("/api/config", json={
        "assist_generation": {"assist_exam": {"temperature": 1.25, "max_output_tokens": 32768}},
        "assist_model_ids": [],
        "assist_screenshot_model_id": None,
    }).status_code == 200
    config._config = None
    saved = client.get("/api/config").json()
    assert saved["assist_model_ids"] == []
    assert saved["assist_screenshot_model_id"] is None
    assert saved["assist_generation"]["assist_realtime"] == {"temperature": 0.2, "max_output_tokens": 128}
    assert saved["assist_generation"]["assist_exam"] == {"temperature": 1.25, "max_output_tokens": 32768}
    assert saved["system_prompts"]["assist_realtime"] == "new voice"
    assert config.get_config().system_prompts["assist_realtime_short"] == "retired voice"
    assert client.post("/api/config", json={
        "assist_generation": {"assist_exam": {"temperature": 2.1, "max_output_tokens": 512}},
    }).status_code == 422


def test_assist_selection_follows_identity_after_catalog_changes(realtime_settings):
    client = realtime_settings
    vision, primary, disabled = client.get("/api/config/models-full").json()["models"]
    keep = "__IA_KEEP_EXISTING_API_KEY__"
    saved = client.post("/api/config", json={"models": [
        {**primary, "name": "Renamed", "api_key": keep},
        {"name": "New model", "api_key": "test-new", "supports_vision": True},
        {**disabled, "api_key": keep},
    ]})
    assert saved.status_code == 200
    config._config = None
    payload = client.get("/api/config").json()
    assert payload["models"][0]["id"] == primary["id"]
    assert payload["models"][1]["id"] not in {vision["id"], primary["id"], disabled["id"]}
    assert config.get_config().models[0].api_key == "test-primary"
    assert payload["assist_model_ids"] == [primary["id"]]
    assert payload["assist_screenshot_model_id"] is None
    assert client.post("/api/config", json={"assist_screenshot_model_id": primary["id"]}).status_code == 422
    assert client.post("/api/config", json={"models": [vision]}).status_code == 409


def test_remote_model_discovery_keeps_the_original_provider_after_reorder(realtime_settings, monkeypatch):
    import importlib

    route_module = importlib.import_module("api.common.router")
    client = realtime_settings
    original = client.get("/api/config/models-full").json()["models"][0]
    assert client.post("/api/config/models-layout", json={"order": [1, 0, 2]}).status_code == 200
    calls = []

    def list_models(url, key, api_type):
        calls.append((url, key, api_type))
        return {"models": []}

    monkeypatch.setattr(route_module, "_list_remote_models", list_models)
    body = {
        "model_index": 0, "model_id": original["id"],
        "api_base_url": original["api_base_url"], "api_key": "__IA_KEEP_EXISTING_API_KEY__",
    }
    assert client.post("/api/models/list", json=body).status_code == 200
    assert calls == [("https://vision.example.test/v1", "test-vision", "openai")]
    body["api_base_url"] = "https://primary.example.test/v1"
    assert client.post("/api/models/list", json=body).status_code == 409
    assert len(calls) == 1
