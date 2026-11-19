import asyncio
import json
from importlib import import_module
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.assist.asr_state import AssistAsrStateMachine
from api.assist import mmpg_integration, routes
from core.config import AppConfig
from core import config

router = import_module("api.common.router")


@pytest.mark.parametrize("provider", ["whisper", "funasr", "qwen3_asr", "iflytek"])
def test_retired_audio_routes_are_unassigned_after_loading(provider):
    cfg = AppConfig(stt_provider=provider, candidate_stt_provider=provider, candidate_asr_enabled=True)
    assert cfg.stt_provider == ""
    assert cfg.candidate_stt_provider == ""
    assert cfg.candidate_asr_enabled is False


def test_cloud_routes_are_preserved_after_loading():
    cfg = AppConfig(stt_provider="doubao_mmpg", candidate_stt_provider="generic", candidate_asr_enabled=True)
    assert (cfg.stt_provider, cfg.candidate_stt_provider, cfg.candidate_asr_enabled) == (
        "doubao_mmpg", "generic", True,
    )


@pytest.mark.parametrize("legacy_opt_in", [None, False, True])
def test_legacy_segmented_candidate_selection_requires_saved_consent(monkeypatch, tmp_path, legacy_opt_in):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", str(path))
    data = {"candidate_stt_provider": "generic", "candidate_asr_enabled": True}
    if legacy_opt_in is not None:
        data["candidate_remote_stt_enabled"] = legacy_opt_in
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = config._load_config()
    assert loaded.candidate_stt_provider == ("generic" if legacy_opt_in else "")
    assert loaded.candidate_asr_enabled is bool(legacy_opt_in)

    config._save_config(AppConfig(candidate_stt_provider="generic", candidate_asr_enabled=True))
    reloaded = config._load_config()
    assert reloaded.candidate_stt_provider == "generic"
    assert reloaded.candidate_asr_enabled


def test_diagnostic_can_test_a_provider_without_changing_audio_roles(monkeypatch):
    cfg = AppConfig(stt_provider="doubao_mmpg", candidate_stt_provider="doubao")
    cfg.generic_stt_api_base_url = "https://asr.example/v1"
    cfg.generic_stt_api_key = "test-key"
    cfg.generic_stt_model = "test-model"
    tested = []

    def factory(provider, *, config_snapshot):
        assert config_snapshot is cfg
        tested.append(provider)
        return SimpleNamespace(transcribe=lambda *_args, **_options: "测试通过")

    monkeypatch.setattr(router, "get_config", lambda: cfg)
    monkeypatch.setattr(router, "get_stt_engine", factory)
    result = asyncio.run(router.api_stt_test(SimpleNamespace(provider="generic")))
    assert result["ok"]
    assert tested == ["generic"]
    assert (cfg.stt_provider, cfg.candidate_stt_provider) == ("doubao_mmpg", "doubao")


def test_mmpg_diagnostic_can_prepare_without_reassigning_cloud_roles(monkeypatch):
    cfg = AppConfig(stt_provider="generic", candidate_stt_provider="doubao")
    monkeypatch.setattr(mmpg_integration, "get_config", lambda: cfg)
    monkeypatch.setattr(mmpg_integration, "get_session", lambda: SimpleNamespace(is_recording=False))
    monkeypatch.setattr(mmpg_integration, "_test_settings", None)
    monkeypatch.setattr(mmpg_integration, "_session_settings", None)
    asyncio.run(mmpg_integration.prepare("test", [0], None))
    assert mmpg_integration._test_settings == cfg.mmpg
    assert mmpg_integration._session_settings is None
    assert (cfg.stt_provider, cfg.candidate_stt_provider) == ("generic", "doubao")


def test_unassigned_provider_cannot_start_capture(monkeypatch):
    monkeypatch.setattr(routes, "get_config", lambda: AppConfig(stt_provider="whisper"))
    monkeypatch.setattr(routes, "get_session", lambda: SimpleNamespace(is_recording=False, single_mic=None))
    monkeypatch.setattr(routes, "start_nonblocking", lambda *_args, **_kwargs: pytest.fail("Unexpected audio capture"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.api_start({"device_id": 0}))
    assert exc.value.status_code == 422
    assert "选择 ASR" in exc.value.detail


@pytest.mark.parametrize("provider", ["generic", "doubao", "doubao_mmpg"])
def test_exam_ignores_both_stale_devices_on_start_and_resume(monkeypatch, provider):
    cfg = AppConfig(written_exam_mode=True, stt_provider=provider)
    monkeypatch.setattr(routes, "get_config", lambda: cfg)
    monkeypatch.setattr(routes, "get_session", lambda: SimpleNamespace(is_recording=True, single_mic=None))
    monkeypatch.setattr(routes, "is_paused", lambda: True)
    monkeypatch.setattr(routes, "is_streaming_interview", lambda: False)
    calls = []

    def capture(device_id=None, candidate_mic_device_id=None, **_kwargs):
        calls.append((device_id, candidate_mic_device_id))

    monkeypatch.setattr(routes, "start_nonblocking", capture)
    monkeypatch.setattr(routes, "unpause_interview", capture)
    assert asyncio.run(routes.api_start({"device_id": 5, "candidate_mic_device_id": 6})) == {"ok": True}
    assert asyncio.run(routes.api_resume_interview({"device_id": 5, "candidate_mic_device_id": 6})) == {"ok": True}
    assert calls == [(None, None), (None, None)]


def test_workspace_cannot_change_during_an_active_recording(monkeypatch):
    from core import session

    monkeypatch.setattr(router, "get_config", lambda: AppConfig(written_exam_mode=False))
    monkeypatch.setattr(session, "get_session", lambda: SimpleNamespace(is_recording=True))
    monkeypatch.setattr(router, "update_config", lambda *_args, **_kwargs: pytest.fail("Unexpected config update"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.api_update_config(router.ConfigUpdate(written_exam_mode=True)))
    assert exc.value.status_code == 409


def test_workspace_change_only_broadcasts_the_saved_mode(monkeypatch):
    from api.realtime import ws
    from core import session

    cfg = AppConfig(written_exam_mode=False)
    sent = []
    monkeypatch.setattr(session, "get_session", lambda: SimpleNamespace(is_recording=False))
    monkeypatch.setattr(router, "get_config", lambda: cfg)
    monkeypatch.setattr(router, "update_config", lambda updates: setattr(cfg, "written_exam_mode", updates["written_exam_mode"]))
    monkeypatch.setattr(ws, "broadcast", sent.append)

    for mode in (True, True, False):
        asyncio.run(router.api_update_config(router.ConfigUpdate(written_exam_mode=mode)))

    assert sent == [
        {"type": "workspace_mode", "written_exam_mode": True},
        {"type": "workspace_mode", "written_exam_mode": False},
    ]


def test_disabling_automatic_answers_drops_pending_question_but_keeps_transcription():
    events = []
    submissions = []
    transcript = []
    state = AssistAsrStateMachine(
        broadcast=events.append,
        submit_answer_task=submissions.append,
        begin_asr_turn=lambda: 1,
        clock=lambda: 0.0,
        logger=SimpleNamespace(info=lambda *_args: None),
    )
    session = SimpleNamespace(
        add_transcription=transcript.append,
        capture_is_loopback=True,
    )
    cfg = AppConfig(auto_detect=True, assist_transcription_merge_gap_sec=0)
    state.handle_auto_detect_asr_text(cfg, session, "请解释什么是线程安全？", "conversation_loopback", 0.0)
    assert state.pending_group is not None
    cfg.auto_detect = False
    state.try_flush_question_group(cfg, session, 3.0, force=True)
    assert state.pending_group is None
    assert submissions == []
    state.accept_final_turn(cfg, session, "请解释什么是进程。")
    assert transcript == ["请解释什么是进程。"]


def test_capture_started_with_auto_answer_off_can_enable_it_live():
    submissions = []
    state = AssistAsrStateMachine(
        broadcast=lambda _event: None,
        submit_answer_task=lambda task: submissions.append(task) or True,
        begin_asr_turn=lambda: 1,
        logger=SimpleNamespace(info=lambda *_args: None),
        allow_answer=lambda: True,
    )
    session = SimpleNamespace(add_transcription=lambda _text: None, capture_is_loopback=True)
    snapshot = AppConfig(auto_detect=False)

    assert state.accept_final_turn(snapshot, session, "请解释线程安全？") == "queued"
    assert len(submissions) == 1
