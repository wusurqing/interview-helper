import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.assist import pipeline, routes
from core import config
from core.session import get_session, reset_session

SESSION_ID = "a" * 32


@pytest.fixture
def single_mic_client(monkeypatch):
    settings = config.AppConfig(candidate_asr_enabled=True, assist_stop_answer_wait_sec=0)
    monkeypatch.setattr(config, "_config", settings)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda device_id: device_id == 20000)
    capture = Mock(is_running=True, dropped_chunks_count=0)
    capture.drain_audio_chunks.return_value = []
    monkeypatch.setattr(pipeline, "audio_capture", capture)
    candidate_capture = Mock()
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", candidate_capture)
    review = Mock()
    monkeypatch.setattr(pipeline, "review_integration", review)
    monkeypatch.setattr(pipeline, "get_stt_engine", lambda **_: SimpleNamespace(is_loaded=True))
    monkeypatch.setattr(pipeline, "_preload_candidate_whisper_async", lambda *_: None)

    def transcribe(_samples, _rate, **kwargs):
        return "我把幂等表的代号定为紫河七号" if kwargs.get("scope") == "candidate" else "请介绍一下你的项目"

    monkeypatch.setattr(pipeline, "transcribe_with_fallback", transcribe)
    monkeypatch.setattr(pipeline, "_submit_candidate_knowledge_update", Mock(return_value=True))

    def turn(question, *_args, **_kwargs):
        if not get_session().qa_pairs:
            get_session().add_qa(question[0], "回答", qa_id="first")
            get_session().open_candidate_answer_window("first")
        return True

    auto_answer = Mock(side_effect=turn)
    monkeypatch.setattr(pipeline, "submit_answer_task", auto_answer)
    reset_session()
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    with TestClient(app) as client:
        try:
            yield client, capture, candidate_capture, review, settings, auto_answer
        finally:
            client.post("/api/stop")
            reset_session()


def test_single_mic_route_keeps_voice_roles_and_archive_in_one_session(single_mic_client):
    client, capture, candidate_capture, review, _, auto_answer = single_mic_client
    started = client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID})
    assert started.status_code == 200
    assert client.get("/api/session").json()["single_mic"] == {
        "id": SESSION_ID, "device_id": 0, "role": "interviewer", "switching": False, "revision": 0,
    }
    capture.start.assert_called_once()
    candidate_capture.start.assert_not_called()
    microphone = capture.start.call_args.kwargs["on_audio"]
    for _ in range(10):
        microphone(np.full(1024, 0.1, dtype=np.float32))

    response = client.post("/api/single-mic/role", json={"session_id": SESSION_ID, "role": "candidate"})
    assert response.status_code == 200
    snapshot = client.get("/api/session").json()
    assert response.json()["single_mic"] == snapshot["single_mic"]
    assert snapshot["transcriptions"] == ["请介绍一下你的项目"]
    assert snapshot["candidate_transcriptions"] == []
    assert snapshot["single_mic"]["role"] == "candidate"
    auto_answer.assert_called_once()

    for _ in range(10):
        microphone(np.full(1024, 0.1, dtype=np.float32))
    response = client.post("/api/single-mic/role", json={"session_id": SESSION_ID, "role": "interviewer"})
    assert response.status_code == 200
    snapshot = client.get("/api/session").json()
    assert snapshot["single_mic"]["role"] == "interviewer"
    assert snapshot["candidate_transcriptions"] == ["我把幂等表的代号定为紫河七号"]
    assert get_session().get_candidate_answer_for_qa("first") == "我把幂等表的代号定为紫河七号"
    auto_answer.assert_called_once()
    capture.start.assert_called_once()

    assert client.post("/api/stop").status_code == 200
    assert client.get("/api/session").json()["single_mic"] is None
    assert review.on_assist_start.call_args.kwargs["single_mic"] is True
    assert any(call.args[0].get_candidate_answer_for_qa("first") for call in review.on_assist_stop.call_args_list)


def test_single_mic_rejects_loopback_and_requires_candidate_asr(single_mic_client):
    client, capture, _, _, settings, _ = single_mic_client
    response = client.post("/api/start", json={"device_id": 20000, "single_mic_id": SESSION_ID})
    assert response.status_code == 400

    settings.candidate_asr_enabled = False
    response = client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID})
    assert response.status_code == 400
    capture.start.assert_not_called()


def test_single_mic_role_endpoint_requires_matching_active_session(single_mic_client):
    client, _, _, _, _, _ = single_mic_client
    assert client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID}).status_code == 200
    response = client.post("/api/single-mic/role", json={"session_id": "b" * 32, "role": "candidate"})
    assert response.status_code == 409
    assert client.get("/api/session").json()["single_mic"]["role"] == "interviewer"


def test_single_mic_keeps_role_when_asr_drain_times_out(single_mic_client, monkeypatch):
    client, _, _, _, _, _ = single_mic_client
    assert client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID}).status_code == 200
    monkeypatch.setattr(routes.mmpg, "finish_single_mic_role", AsyncMock(return_value=False))
    response = client.post("/api/single-mic/role", json={"session_id": SESSION_ID, "role": "candidate"})
    assert response.status_code == 504
    state = client.get("/api/session").json()["single_mic"]
    assert state["role"] == "interviewer"
    assert state["switching"] is False
    assert state["revision"] == 2


def test_duplicate_switch_shares_drain_and_stop_cancels_it(single_mic_client, monkeypatch):
    client, _, _, _, _, _ = single_mic_client
    assert client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID}).status_code == 200
    drain_entered, drain_release, duplicate_entered = threading.Event(), threading.Event(), threading.Event()

    async def finish(_single_mic, operation, _timeout):
        drain_entered.set()
        await asyncio.to_thread(drain_release.wait, 4)
        operation.complete()
        return True

    drain = AsyncMock(side_effect=finish)
    monkeypatch.setattr(routes.mmpg, "finish_single_mic_role", drain)
    mic = get_session().single_mic
    start_switch = mic.begin_switch

    def observe_duplicate(role):
        if mic.snapshot()["switching"]:
            duplicate_entered.set()
        return start_switch(role)

    monkeypatch.setattr(mic, "begin_switch", observe_duplicate)
    body = {"session_id": SESSION_ID, "role": "candidate"}
    with ThreadPoolExecutor(max_workers=2) as clients:
        first = clients.submit(client.post, "/api/single-mic/role", json=body)
        try:
            assert drain_entered.wait(2)
            second = clients.submit(client.post, "/api/single-mic/role", json=body)
            assert duplicate_entered.wait(2)
            assert client.post("/api/stop").status_code == 200
            assert client.get("/api/session").json()["single_mic"] is None
        finally:
            drain_release.set()

    assert first.result().status_code == 409
    assert second.result().status_code == 409
    drain.assert_awaited_once()


def test_single_mic_pause_identity_and_resume_configuration(single_mic_client):
    client, capture, _, _, settings, _ = single_mic_client
    assert client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID}).status_code == 200
    assert client.post("/api/pause", json={"single_mic_id": "stale"}).status_code == 409
    assert client.post("/api/pause", json={"single_mic_id": SESSION_ID}).status_code == 200
    settings.candidate_stt_provider = "doubao_mmpg"
    assert client.post("/api/unpause", json={"single_mic_id": SESSION_ID}).status_code == 409
    assert capture.start.call_count == 1

    settings.candidate_stt_provider = "whisper"
    assert client.post("/api/unpause", json={"single_mic_id": SESSION_ID}).status_code == 200
    assert capture.start.call_count == 2
    assert client.get("/api/session").json()["single_mic"]["role"] == "interviewer"

def test_single_mic_watchdog_pauses_and_releases_stalled_microphone(single_mic_client, monkeypatch):
    client, capture, _, _, _, _ = single_mic_client
    assert client.post("/api/start", json={"device_id": 0, "single_mic_id": SESSION_ID}).status_code == 200
    capture.stop.reset_mock()
    monkeypatch.setattr(get_session().single_mic, "seconds_since_audio", lambda: 3.0)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = client.get("/api/session").json()
        if snapshot["is_paused"]:
            break
        time.sleep(0.02)

    assert snapshot["is_paused"]
    assert snapshot["single_mic"]["id"] == SESSION_ID
    assert snapshot["single_mic"]["role"] == "interviewer"
    capture.stop.assert_called_with(owner="assist", clear_queue=False)
