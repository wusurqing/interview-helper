"""STT secret masking and repr=False contract tests (P1-1)."""
from __future__ import annotations

from importlib import import_module

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import config

common_router = import_module("api.common.router")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(config, "_config", config.AppConfig())
    app = FastAPI()
    app.include_router(common_router.router, prefix="/api")
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))


def test_app_config_repr_hides_stt_secrets():
    secrets = {
        "doubao_stt_access_token": "tok-super-secret-1",
        "doubao_stt_api_key": "key-super-secret-2",
        "generic_stt_api_key": "gen-super-secret-3",
        "generic_stt_custom_headers": '{"Authorization": "Bearer secret-header-4"}',
    }
    cfg = config.AppConfig(**secrets)
    rendered_repr = repr(cfg)
    rendered_str = str(cfg)
    for value in secrets.values():
        assert value not in rendered_repr
        assert value not in rendered_str


def test_config_api_masks_stt_secrets_and_supports_keep_replace_clear(client, monkeypatch):
    secrets = {
        "doubao_stt_access_token": "doubao-tok-123",
        "doubao_stt_api_key": "doubao-key-456",
        "generic_stt_api_key": "generic-key-789",
        "generic_stt_custom_headers": "Authorization: Bearer hdr-000\nX-Trace: ok",
    }

    # Initially all false, no plaintext keys exposed.
    initial = client.get("/api/config")
    assert initial.status_code == 200
    initial_json = initial.json()
    for field in secrets:
        assert field not in initial_json
        assert initial_json[f"{field}_configured"] is False

    # Save all 4 secrets -> response hides plaintext and marks configured=True.
    saved_resp = client.post("/api/config", json=secrets)
    assert saved_resp.status_code == 200
    saved_json = saved_resp.json()
    for field, value in secrets.items():
        assert value not in saved_resp.text
        assert field not in saved_json
        assert saved_json[f"{field}_configured"] is True

    # GET /api/config also hides plaintext.
    fetched = client.get("/api/config")
    assert fetched.status_code == 200
    for field, value in secrets.items():
        assert value not in fetched.text
        assert field not in fetched.json()
        assert fetched.json()[f"{field}_configured"] is True

    # Blank strings preserve existing secrets, non-blank replaces, null explicitly clears.
    updated_resp = client.post(
        "/api/config",
        json={
            "doubao_stt_access_token": "",
            "doubao_stt_api_key": "   ",
            "generic_stt_api_key": "  generic-key-new  ",
            "generic_stt_custom_headers": None,
        },
    )
    assert updated_resp.status_code == 200
    assert "generic-key-new" not in updated_resp.text
    updated_json = updated_resp.json()
    assert updated_json["doubao_stt_access_token_configured"] is True
    assert updated_json["doubao_stt_api_key_configured"] is True
    assert updated_json["generic_stt_api_key_configured"] is True
    assert updated_json["generic_stt_custom_headers_configured"] is False

    # Reload from disk and verify persisted values.
    monkeypatch.setattr(config, "_config", None)
    reloaded = config.get_config()
    assert reloaded.doubao_stt_access_token == "doubao-tok-123"
    assert reloaded.doubao_stt_api_key == "doubao-key-456"
    assert reloaded.generic_stt_api_key == "generic-key-new"
    assert reloaded.generic_stt_custom_headers == ""


def test_invalid_stt_secret_type_does_not_echo_input(client):
    resp = client.post("/api/config", json={"generic_stt_api_key": ["leaked-secret-value"]})
    assert resp.status_code == 422
    assert "leaked-secret-value" not in resp.text
