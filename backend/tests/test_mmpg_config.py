import json
from importlib import import_module
from unittest.mock import AsyncMock

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


def test_mmpg_can_be_saved_for_both_audio_sources(client):
    options = client.get("/api/options").json()
    assert "doubao_mmpg" in options["stt_providers"]
    assert "doubao_mmpg" in options["candidate_stt_providers"]
    assert client.post("/api/config", json={"stt_provider": "doubao_mmpg"}).status_code == 200
    candidate = client.post("/api/config", json={"candidate_stt_provider": "doubao_mmpg", "candidate_asr_enabled": True})
    assert candidate.status_code == 200
    assert candidate.json()["candidate_stt_provider"] == "doubao_mmpg"
    assert config._load_config().candidate_stt_provider == "doubao_mmpg"


def test_role_edits_do_not_end_the_active_mmpg_session(client, monkeypatch):
    from api.assist import candidate_mmpg

    stop = AsyncMock()
    monkeypatch.setattr(candidate_mmpg, "stop", stop)
    result = client.post("/api/config", json={"candidate_stt_provider": "doubao_mmpg", "candidate_asr_enabled": True})
    assert result.status_code == 200
    stop.assert_not_awaited()

    assert client.post("/api/config", json={"candidate_asr_enabled": False}).status_code == 200
    stop.assert_not_awaited()


def test_previous_config_uses_probe_tuning_defaults(client):
    response = client.get("/api/config").json()["mmpg"]
    assert response["aivad"] is True
    assert response["silence_ms"] == 800
    assert response["answer_delay_ms"] == response["followup_window_ms"] == 1500
    assert response["twopass"] is True
    assert response["enable_prefetch"] is True
    assert response["expire_time_ms"] is None
    assert response["force_end"] is False
    assert response["token_mode"] == "auto"


def test_answer_windows_allow_zero_and_require_nonnegative_integers(client):
    result = client.post("/api/config", json={"mmpg": {"answer_delay_ms": 0, "followup_window_ms": 2000}})
    assert result.status_code == 200
    assert result.json()["mmpg"]["answer_delay_ms"] == 0
    assert result.json()["mmpg"]["followup_window_ms"] == 2000
    for bad in (-1, True, 1.5, "1500"):
        assert client.post("/api/config", json={"mmpg": {"answer_delay_ms": bad}}).status_code == 422


def test_mmpg_patch_keeps_replaces_and_clears_secrets(client, monkeypatch):
    secrets = {
        "volc_access_key": "old-test-access-key",
        "volc_secret_key": "test-secret-key",
        "rtc_app_key": "test-rtc-app-key",
        "rtc_token": "test-rtc-token",
        "doubao_api_key": "test-doubao-key",
    }
    response = client.post("/api/config", json={"mmpg": {**secrets, "expire_time_ms": 1200}})
    assert response.status_code == 200
    for field, value in secrets.items():
        assert value not in response.text
        assert field not in response.json()["mmpg"]
        assert response.json()["mmpg"][f"{field}_configured"] is True

    response = client.post("/api/config", json={"mmpg": {
        "volc_access_key": "new-test-access-key", "volc_secret_key": " ", "rtc_token": None,
        "expire_time_ms": None,
    }})
    assert response.status_code == 200
    assert response.json()["mmpg"]["rtc_token_configured"] is False

    monkeypatch.setattr(config, "_config", None)
    saved = config.get_config().mmpg
    assert saved.volc_access_key == "new-test-access-key"
    assert saved.volc_secret_key == "test-secret-key"
    assert saved.rtc_app_key == "test-rtc-app-key"
    assert not saved.rtc_token
    assert saved.expire_time_ms is None


def test_invalid_mmpg_value_is_rejected_without_echoing_input(client):
    response = client.post("/api/config", json={"mmpg": {
        "volc_secret_key": ["invalid-test-secret"], "silence_ms": 3000,
    }})
    assert response.status_code == 422
    assert "invalid-test-secret" not in response.text
    assert client.get("/api/config").json()["mmpg"]["silence_ms"] == 800


def test_manual_token_expiry_resets_when_token_changes(client):
    response = client.post("/api/config", json={"mmpg": {
        "token_mode": "manual", "rtc_token": "first-test-token", "rtc_token_expires_at": 1900000000,
    }})
    assert response.json()["mmpg"]["rtc_token_expires_at"] == 1900000000

    response = client.post("/api/config", json={"mmpg": {"rtc_token": " first-test-token "}})
    assert response.json()["mmpg"]["rtc_token_expires_at"] == 1900000000

    response = client.post("/api/config", json={"mmpg": {"rtc_token": "second-test-token"}})
    assert response.json()["mmpg"]["rtc_token_expires_at"] is None


def test_config_save_failure_preserves_existing_file_and_in_memory_settings(client, monkeypatch):
    response = client.post("/api/config", json={"language": "Java"})
    assert response.status_code == 200

    def fail_replace(*args):
        raise PermissionError("test filesystem error")

    monkeypatch.setattr(config.os, "replace", fail_replace)
    response = client.post("/api/config", json={"language": "Go", "mmpg": {"silence_ms": 1100}})
    assert response.status_code == 500
    assert client.get("/api/config").json()["language"] == "Java"
    assert config._load_config().language == "Java"
    assert config._load_config().mmpg.silence_ms == 800


def test_probe_import_preserves_token_mode_app_key_and_provider(client, monkeypatch, tmp_path):
    response = client.post("/api/config", json={"stt_provider": "doubao", "mmpg": {
        "rtc_app_key": "existing-test-app-key", "doubao_api_key": "existing-test-asr-key",
    }})
    assert response.status_code == 200
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps({
        "volc_access_key": "probe-test-key", "rtc_app_id": "probe-app-id",
        "rtc_app_key": "ignored-test-app-key", "doubao_api_key": None,
        "rtc_token": "probe-test-token", "token_mode": "manual", "silence_ms": 950,
        "stt_provider": "doubao_mmpg", "answer_delay_ms": 999, "followup_window_ms": 50,
    }), encoding="utf-8-sig")
    monkeypatch.setattr(common_router, "_MMPG_PROBE_CONFIG_PATH", probe_path, raising=False)
    response = client.post("/api/stt/mmpg/import-probe")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    saved = config._load_config()
    assert saved.stt_provider == "doubao"
    assert saved.mmpg.token_mode == "auto"
    assert saved.mmpg.rtc_app_key == "existing-test-app-key"
    assert saved.mmpg.doubao_api_key == "existing-test-asr-key"
    assert saved.mmpg.volc_access_key == "probe-test-key"
    assert saved.mmpg.rtc_token == "probe-test-token"
    assert saved.mmpg.rtc_app_id == "probe-app-id"
    assert saved.mmpg.silence_ms == 950
    assert saved.mmpg.answer_delay_ms == saved.mmpg.followup_window_ms == 1500


def test_probe_import_rejects_invalid_values_without_modifying_saved_config(client, monkeypatch, tmp_path):
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps({
        "volc_secret_key": ["invalid-test-secret"], "silence_ms": 950,
    }), encoding="utf-8")
    monkeypatch.setattr(common_router, "_MMPG_PROBE_CONFIG_PATH", probe_path)
    response = client.post("/api/stt/mmpg/import-probe")
    assert response.status_code == 422
    assert "invalid-test-secret" not in response.text
    assert client.get("/api/config").json()["mmpg"]["silence_ms"] == 800


@pytest.mark.parametrize("remote_host,headers", [
    ("192.168.1.14", {}),
    ("127.0.0.1", {"Origin": "https://example.test"}),
])
def test_mmpg_writes_require_local_same_site_requests(client, remote_host, headers):
    remote = TestClient(client.app, base_url="http://127.0.0.1", client=(remote_host, 40000), headers=headers)
    assert remote.post("/api/stt/mmpg/import-probe").status_code == 403
    assert remote.post("/api/config", json={"mmpg": {"rtc_app_key": "test-rtc-key"}}).status_code == 403
    assert remote.post("/api/config", json={"language": "Java"}).status_code == 200


def test_legacy_stt_test_points_mmpg_to_session_diagnostics(client):
    assert client.post("/api/config", json={"stt_provider": "doubao_mmpg"}).status_code == 200
    response = client.post("/api/stt/test")
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "诊断" in response.json()["detail"]


def test_mmpg_status_reports_missing_credentials_without_loading_legacy_engine(client, monkeypatch):
    def unexpected_engine_call():
        pytest.fail("status should not load a one-shot STT engine for MMPG")

    monkeypatch.setattr(common_router, "get_stt_engine", unexpected_engine_call)
    assert client.post("/api/config", json={"stt_provider": "doubao_mmpg"}).status_code == 200
    response = client.get("/api/stt/status")
    assert response.status_code == 200
    assert response.json()["provider"] == "doubao_mmpg"
    assert response.json()["loaded"] is False
    assert response.json()["configured"] is False
    assert "rtc_app_key" in response.json()["missing"]


def test_mmpg_status_keeps_active_provider_after_saving_future_provider(client, monkeypatch):
    integration = import_module("api.assist.mmpg_integration")
    monkeypatch.setattr(integration.pipeline, "_interviewer_config", config.AppConfig(stt_provider="doubao_mmpg"))
    assert client.post("/api/config", json={"stt_provider": "doubao"}).status_code == 200
    assert client.get("/api/stt/status").json()["provider"] == "doubao_mmpg"
