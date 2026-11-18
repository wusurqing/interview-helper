import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import HTTPException

from api.assist import candidate_mmpg, mmpg_integration, pipeline, routes
from core import config
from core.session import get_session, reset_session
from services.stt.mmpg import MmpgError


SESSION_ID = "test-mic"
INTERVIEWER = "interviewer"
CANDIDATE = "candidate"


def _transport():
    return SimpleNamespace(
        active=False,
        single_mic_id=None,
        offer_audio=Mock(),
        stop=AsyncMock(),
        drain_single_mic_input=AsyncMock(return_value=True),
        finish_role_input=AsyncMock(return_value=True),
        status=lambda: {"state": "listening", "error": None},
    )


@pytest.mark.parametrize("main_rtc,candidate_rtc", [(True, False), (False, True), (True, True)])
def test_single_mic_mmpg_transport_matrix(monkeypatch, main_rtc, candidate_rtc):
    cfg = config.AppConfig(
        stt_provider="doubao_mmpg" if main_rtc else "whisper",
        candidate_stt_provider="doubao_mmpg" if candidate_rtc else "whisper",
        candidate_asr_enabled=True,
        assist_stop_answer_wait_sec=0,
    )
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_turns", None)
    monkeypatch.setattr(mmpg_integration, "_capture_epoch", None)
    monkeypatch.setattr(mmpg_integration, "_session_settings", None)
    monkeypatch.setattr(mmpg_integration, "_speech_starts", {})
    monkeypatch.setattr(candidate_mmpg, "_capture_epoch", None)
    monkeypatch.setattr(candidate_mmpg, "_pending", {})
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _device: False)
    monkeypatch.setattr(pipeline, "review_integration", Mock())
    monkeypatch.setattr(pipeline, "_submit_candidate_knowledge_update", Mock(return_value=True))
    monkeypatch.setattr(pipeline, "_start_interview_worker_thread_if_needed", Mock())
    monkeypatch.setattr(pipeline, "_start_candidate_capture", Mock())

    capture = Mock(is_running=True, dropped_chunks_count=0)
    candidate_capture = Mock()
    monkeypatch.setattr(pipeline, "audio_capture", capture)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_capture)
    main_manager, candidate_manager = _transport(), _transport()
    monkeypatch.setattr(mmpg_integration, "manager", main_manager)
    monkeypatch.setattr(candidate_mmpg, "manager", candidate_manager)

    # The local worker's ASR/VAD is tested separately. Consume its handoff
    # barrier here after the test has read the queued PCM.
    def complete_local_role(single_mic, operation, _timeout):
        audio_input = (
            single_mic.interviewer_input if operation.from_role == INTERVIEWER else single_mic.candidate_input
        )
        assert audio_input.take_barrier() is operation
        operation.complete()
        return True

    monkeypatch.setattr(pipeline, "drain_single_mic_worker", complete_local_role)

    async def exercise():
        session = reset_session()
        microphone_id = 0
        listening = {"state": "listening", "error": None}
        if main_rtc:
            main_manager.single_mic_id = SESSION_ID
            await mmpg_integration.prepare("start", [microphone_id], microphone_id, single_mic_id=SESSION_ID)
            await mmpg_integration.start_capture("start", [microphone_id], microphone_id, single_mic_id=SESSION_ID)
            mmpg_integration._broadcast_state(listening)
        else:
            assert await routes.api_start({"device_id": microphone_id, "single_mic_id": SESSION_ID}) == {"ok": True}

        try:
            if candidate_rtc:
                with pytest.raises(MmpgError) as stale:
                    await candidate_mmpg.prepare("start", [microphone_id], None, single_mic_id="previous")
                assert stale.value.code == "candidate_interview_inactive"
                candidate_manager.single_mic_id = SESSION_ID
                await candidate_mmpg.prepare("start", [microphone_id], None, single_mic_id=SESSION_ID)
                await candidate_mmpg._start_capture("start", [microphone_id], None, single_mic_id=SESSION_ID)
                candidate_mmpg._broadcast_state(listening)

            single_mic = session.single_mic
            assert single_mic is not None and single_mic.ready
            assert single_mic.snapshot()["role"] == INTERVIEWER
            capture.start.assert_called_once()
            candidate_capture.start.assert_not_called()
            microphone = capture.start.call_args.kwargs["on_audio"]
            samples = np.full(1024, 0.1, dtype=np.float32)

            def assert_routing(active_role, connected_rtc=None):
                for manager in (main_manager, candidate_manager):
                    manager.offer_audio.reset_mock()
                microphone(samples)

                legs = (
                    (INTERVIEWER, main_rtc, main_manager, single_mic.interviewer_input),
                    (CANDIDATE, candidate_rtc, candidate_manager, single_mic.candidate_input),
                )
                for role, use_rtc, manager, audio_input in legs:
                    local_chunks = audio_input.drain_audio_chunks(timeout=0)
                    if use_rtc and (connected_rtc is None or role in connected_rtc):
                        manager.offer_audio.assert_called_once()
                        expected = samples if active_role == role else np.zeros_like(samples)
                        np.testing.assert_array_equal(manager.offer_audio.call_args.args[0], expected)
                        assert not local_chunks
                    else:
                        manager.offer_audio.assert_not_called()
                        if active_role == role:
                            np.testing.assert_array_equal(local_chunks, [samples])
                        else:
                            assert not local_chunks

            assert_routing(INTERVIEWER)
            await mmpg_integration.stop_capture("start", "owner_disconnected", single_mic_id="previous")
            await candidate_mmpg._stop_capture("start", "owner_disconnected", single_mic_id="previous")
            assert single_mic.ready
            result = await routes.api_single_mic_role(
                routes.SingleMicRoleRequest(session_id=SESSION_ID, role=CANDIDATE),
            )
            assert result["ok"] is True
            assert_routing(CANDIDATE)
            if main_rtc:
                main_manager.finish_role_input.assert_awaited_once()
                assert main_manager.finish_role_input.call_args.args == (SESSION_ID,)

            if candidate_rtc:
                session.add_qa("请介绍一下你的项目", "参考回答", qa_id="test-qa")
                session.open_candidate_answer_window("test-qa")
                candidate_mmpg._on_turn("started", "candidate-turn")
                text = "我把幂等表的代号定为紫河七号"
                candidate_mmpg._on_transcript(text, "candidate-turn")
                assert session.get_candidate_answer_for_qa("test-qa") == text

            reconnecting_role = CANDIDATE if candidate_rtc else INTERVIEWER
            reconnecting = candidate_mmpg if candidate_rtc else mmpg_integration
            reconnecting._broadcast_state({"state": "reconnecting", "error": None})
            assert single_mic.snapshot()["role"] == CANDIDATE
            assert not single_mic.ready
            with pytest.raises(HTTPException) as unavailable:
                await routes.api_single_mic_role(routes.SingleMicRoleRequest(session_id=SESSION_ID, role=INTERVIEWER))
            assert unavailable.value.status_code == 409
            assert_routing(None, connected_rtc={INTERVIEWER} if main_rtc and candidate_rtc else set())

            reconnecting._broadcast_state(listening)
            assert single_mic.ready
            assert_routing(CANDIDATE)
            result = await routes.api_single_mic_role(
                routes.SingleMicRoleRequest(session_id=SESSION_ID, role=INTERVIEWER),
            )
            assert result["ok"] is True
            assert_routing(INTERVIEWER)
            if candidate_rtc:
                candidate_manager.finish_role_input.assert_awaited_once()
                assert candidate_manager.finish_role_input.call_args.args == (SESSION_ID,)
            capture.start.assert_called_once()
            candidate_capture.start.assert_not_called()
        finally:
            await routes.api_stop()
            assert get_session().single_mic is None
            reset_session()

    asyncio.run(exercise())
