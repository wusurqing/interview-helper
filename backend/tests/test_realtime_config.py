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
    assert payload["written_exam_model_id"] == vision_id
    assert payload["written_exam_model_ids"] == [vision_id]
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


def test_written_exam_settings_are_independent_and_survive_reload(realtime_settings):
    client = realtime_settings
    vision, primary, disabled = client.get("/api/config/models-full").json()["models"]
    keep = "__IA_KEEP_EXISTING_API_KEY__"
    response = client.post("/api/config", json={"models": [
        {**primary, "api_key": keep},
        {**vision, "name": "Renamed vision", "api_key": "test-rotated"},
        {**disabled, "api_key": keep},
        {"name": "Other vision", "supports_vision": True, "api_key": "test-other"},
    ]})
    assert response.status_code == 200
    other_id = response.json()["models"][-1]["id"]
    assert response.json()["written_exam_model_id"] == vision["id"]
    assert client.post("/api/config", json={
        "written_exam_model_id": other_id,
        "system_prompts": {"assist_exam": "custom exam"},
        "assist_generation": {"assist_exam": {"temperature": 0.25, "max_output_tokens": 768}},
    }).status_code == 200
    config._config = None
    payload = client.get("/api/config").json()
    assert payload["written_exam_model_id"] == other_id
    assert payload["assist_screenshot_model_id"] == vision["id"]
    assert payload["system_prompts"]["assist_exam"] == "custom exam"
    assert payload["assist_generation"]["assist_exam"] == {"temperature": 0.25, "max_output_tokens": 768}

    catalog = client.get("/api/config/models-full").json()["models"]
    response = client.post("/api/config", json={"models": [
        {**model, "api_key": keep, "supports_vision": False} if model["id"] == other_id
        else {**model, "api_key": keep}
        for model in catalog
    ]})
    assert response.status_code == 200
    assert response.json()["written_exam_model_id"] == other_id
    assert not next(model for model in response.json()["models"] if model["id"] == other_id)["supports_vision"]

    assert client.post("/api/config", json={"written_exam_model_id": None}).status_code == 200
    config._config = None
    assert client.get("/api/config").json()["written_exam_model_id"] is None
    catalog = client.get("/api/config/models-full").json()["models"]
    assert client.post("/api/config", json={
        "models": [
            {**model, "api_key": keep, "supports_vision": True} if model["id"] == other_id
            else {**model, "api_key": keep}
            for model in catalog
        ],
        "written_exam_model_id": other_id,
    }).status_code == 200
    catalog = client.get("/api/config/models-full").json()["models"]
    response = client.post("/api/config", json={"models": [
        {**model, "api_key": keep} for model in catalog if model["id"] != other_id
    ]})
    assert response.status_code == 200
    assert response.json()["written_exam_model_id"] is None
    assert response.json()["assist_screenshot_model_id"] == vision["id"]


def test_written_exam_model_rejects_ineligible_choices(realtime_settings):
    client = realtime_settings
    catalog = client.get("/api/config/models-full").json()["models"]
    keep = "__IA_KEEP_EXISTING_API_KEY__"
    response = client.post("/api/config", json={"models": [
        *[{**model, "api_key": keep} for model in catalog],
        {"name": "Disabled vision", "supports_vision": True, "enabled": False, "api_key": "test-disabled-vision"},
        {"name": "Vision missing key", "supports_vision": True, "api_key": "  "},
    ]})
    assert response.status_code == 200
    models = response.json()["models"]
    assert not models[-1]["has_api_key"]
    for invalid_id in (models[1]["id"], models[-2]["id"], models[-1]["id"], "deleted-model", ""):
        for selection in ({"written_exam_model_id": invalid_id}, {"written_exam_model_ids": [invalid_id]}):
            response = client.post("/api/config", json=selection)
            assert response.status_code == 422
            assert "笔试答题" in response.json()["detail"]
    assert client.get("/api/config").json()["written_exam_model_id"] == catalog[0]["id"]


def test_written_exam_does_not_migrate_unconfigured_screenshot_model():
    cfg = config.AppConfig(
        models=[{"id": "old-screen", "supports_vision": True, "api_key": ""}],
        assist_screenshot_model_id="old-screen",
    )
    assert cfg.assist_screenshot_model_id == "old-screen"
    assert cfg.written_exam_model_id is None
    assert cfg.written_exam_model_ids == []


def test_written_exam_queue_migrates_old_primary_and_only_eligible_fallbacks():
    old_config = {
        "models": [
            {"id": "text-only", "api_key": "test-key"},
            {"id": "vision-a", "supports_vision": True, "api_key": "test-key"},
            {"id": "vision-b", "supports_vision": True, "api_key": "test-key"},
            {"id": "primary", "supports_vision": True, "api_key": "test-key"},
            {"id": "disabled", "supports_vision": True, "api_key": "test-key", "enabled": False},
            {"id": "missing-key", "supports_vision": True, "api_key": ""},
        ],
        "assist_model_ids": ["text-only", "vision-b", "disabled", "missing-key", "primary", "vision-a"],
        "written_exam_model_id": "primary",
    }
    assert config.AppConfig(**old_config).written_exam_model_ids == ["primary", "vision-b", "vision-a"]
    assert config.AppConfig(**{**old_config, "written_exam_model_id": None}).written_exam_model_ids == []
    assert config.AppConfig(**old_config, written_exam_model_ids=[]).written_exam_model_ids == []
    assert config.AppConfig(**{**old_config, "written_exam_model_id": "disabled"}).written_exam_model_ids == ["disabled"]


def test_ordered_exam_queue_saves_and_reloads_independently(realtime_settings):
    client = realtime_settings
    old_catalog = client.get("/api/config/models-full").json()["models"]
    keep = "__IA_KEEP_EXISTING_API_KEY__"
    updated = client.post("/api/config", json={"models": [
        *[{**model, "api_key": keep} for model in old_catalog],
        {"name": "Second vision", "supports_vision": True, "api_key": "test-second"},
    ]}).json()
    original_id = old_catalog[0]["id"]
    second_id = updated["models"][-1]["id"]
    saved = client.post("/api/config", json={"written_exam_model_ids": [second_id, original_id, second_id]}).json()
    assert saved["written_exam_model_ids"] == [second_id, original_id]
    assert saved["written_exam_model_id"] == second_id
    assert saved["assist_model_ids"] == updated["assist_model_ids"]

    catalog = client.get("/api/config/models-full").json()["models"]
    models_by_id = {model["id"]: model for model in catalog}
    models_by_id[second_id]["enabled"] = False
    assert client.post("/api/config", json={"models": [
        {**model, "api_key": keep} for model in reversed(list(models_by_id.values()))
    ]}).status_code == 200
    assert client.post("/api/config", json={"written_exam_model_ids": [original_id, second_id]}).status_code == 200
    config._config = None
    assert client.get("/api/config").json()["written_exam_model_ids"] == [original_id, second_id]

    assert client.post("/api/config", json={"written_exam_model_ids": []}).status_code == 200
    config._config = None
    payload = client.get("/api/config").json()
    assert payload["written_exam_model_ids"] == []
    assert payload["written_exam_model_id"] is None
    assert payload["assist_model_ids"] == updated["assist_model_ids"]


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
