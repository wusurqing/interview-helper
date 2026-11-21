"""Config corruption auto-backup and read-only protection tests (P1-4)."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import config

common_router = import_module("api.common.router")


@pytest.fixture
def isolated_config(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", str(cfg_path))
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.setattr(config, "_config_corrupt_state", None)
    yield cfg_path
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.setattr(config, "_config_corrupt_state", None)


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(common_router.router, prefix="/api")
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))


def test_validation_error_backs_up_corrupt_config_and_blocks_overwrite(isolated_config: Path):
    original_payload = {
        "models": [
            {
                "name": "My Valuable Model",
                "api_base_url": "https://api.example.test/v1",
                "api_key": "sk-super-valuable-key-do-not-lose",
                "model": "gpt-4o",
            }
        ],
        "think_effort": "invalid-effort",  # single corrupt field triggers ValidationError
    }
    raw = json.dumps(original_payload, ensure_ascii=False, indent=2)
    isolated_config.write_text(raw, encoding="utf-8")

    cfg = config.get_config()
    assert isinstance(cfg, config.AppConfig)

    # Backup file created with exact original bytes and original file untouched.
    backups = list(isolated_config.parent.glob("config.json.corrupt.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == raw
    assert isolated_config.read_text(encoding="utf-8") == raw

    # Corrupt read-only state exposed without leaking API key.
    corrupt = config.get_config_corrupt_state()
    assert corrupt is not None
    assert corrupt["corrupt"] is True
    assert corrupt["backup_filename"] == backups[0].name
    assert "sk-super-valuable-key-do-not-lose" not in corrupt["error"]

    # Any accidental update_config refuses to overwrite the user's file.
    with pytest.raises(config.ConfigCorruptError):
        config.update_config({"language": "Go"})
    assert isolated_config.read_text(encoding="utf-8") == raw

    # HTTP GET /api/config exposes warning; unconfirmed POST /api/config returns 409.
    client = _make_client()
    get_resp = client.get("/api/config")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["config_corrupt"] is True
    assert body["config_corrupt_info"]["backup_filename"] == backups[0].name
    assert "sk-super-valuable-key-do-not-lose" not in get_resp.text

    blocked = client.post("/api/config", json={"language": "Go"})
    assert blocked.status_code == 409
    assert backups[0].name in blocked.json()["detail"]
    assert isolated_config.read_text(encoding="utf-8") == raw

    # Explicit confirmation unlocks overwriting and clears corrupt state.
    confirmed = client.post(
        "/api/config",
        json={"language": "Go", "confirm_overwrite_corrupt": True},
    )
    assert confirmed.status_code == 200
    confirmed_body = confirmed.json()
    assert confirmed_body["config_corrupt"] is False
    assert confirmed_body["config_corrupt_info"] is None
    assert confirmed_body["language"] == "Go"
    assert config.get_config_corrupt_state() is None

    # Subsequent normal updates work again.
    follow_up = client.post("/api/config", json={"language": "Java"})
    assert follow_up.status_code == 200
    assert follow_up.json()["language"] == "Java"


def test_malformed_json_backs_up_and_enters_read_only_without_print(isolated_config: Path, capsys):
    raw = "{broken json"
    isolated_config.write_text(raw, encoding="utf-8")

    _ = config.get_config()
    captured = capsys.readouterr()
    assert "[Config]" not in captured.out

    backups = list(isolated_config.parent.glob("config.json.corrupt.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == raw
    assert config.get_config_corrupt_state() is not None

    with pytest.raises(config.ConfigCorruptError):
        config._save_config(config.AppConfig())
    assert isolated_config.read_text(encoding="utf-8") == raw


def test_non_object_json_root_triggers_corrupt_protection(isolated_config: Path):
    isolated_config.write_text('["not", "an", "object"]', encoding="utf-8")

    _ = config.get_config()
    assert config.get_config_corrupt_state() is not None
    with pytest.raises(config.ConfigCorruptError):
        config.update_config({"language": "Go"})


def test_provider_and_analytics_config_writes_return_409_while_corrupt(isolated_config: Path):
    isolated_config.write_text("{invalid", encoding="utf-8")
    _ = config.get_config()

    provider_settings = import_module("api.common.provider_settings")
    analytics_router = import_module("api.analytics.router")

    app = FastAPI()
    app.include_router(provider_settings.router, prefix="/api")
    app.include_router(analytics_router.router, prefix="/api")
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))

    provider_resp = client.put(
        "/api/model-providers",
        json={
            "revision": config.provider_layout_revision(config.get_config()),
            "providers": [],
            "models": [],
            "confirmed_remove_ids": [],
            "active_model_id": None,
        },
    )
    assert provider_resp.status_code == 409

    analytics_resp = client.put(
        "/api/knowledge/settings",
        json={"temperature": 0.4, "max_tokens": 1024, "think_effort": "off"},
    )
    assert analytics_resp.status_code == 409
