import asyncio
import json
import logging
import struct
import time

import numpy as np
import pytest


CONFIG = {
    "rtc_app_id": "123456781234567812345678",
    "rtc_app_key": "app key",
    "rtc_room_id": "meeting",
    "rtc_user_id": "listener",
    "rtc_bot_id": "agent",
    "volc_access_key": "signature-account",
    "volc_secret_key": "signature-key",
    "doubao_api_key": "asr-key",
    "token_mode": "auto",
}


class CloudSocket:
    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False

    def push(self, event_type, **fields):
        self.events.put_nowait(json.dumps({"type": event_type, **fields}))

    async def recv(self):
        return await self.events.get()

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        self.closed = True


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout=1)


def test_rtc_token_matches_official_python_reference():
    from services.stt.mmpg import sign_rtc_token

    assert sign_rtc_token(
        CONFIG["rtc_app_id"], CONFIG["rtc_app_key"], "meeting", "listener",
        issued_at=1800000000, expires_at=1800003600, nonce=978,
    ) == (
        "001123456781234567812345678"
        "OQDSAwAAANJJaxDgSWsHAG1lZXRpbmcIAGxpc3RlbmVyBAAAABDgSWsBABDgSWsCABDgSWs"
        "DABDgSWsgACbNKJzAO265N7zESr9pNe7j022inxbNMK72EHYtBy/+"
    )


def test_manual_token_infers_expiry_and_rejects_wrong_rtc_scope():
    from services.stt.mmpg import MmpgError, sign_rtc_token, validate_config

    issued = int(time.time())
    expires = issued + 3600
    token = sign_rtc_token(
        CONFIG["rtc_app_id"], CONFIG["rtc_app_key"], "meeting", "listener",
        issued_at=issued, expires_at=expires,
    )
    manual = {**CONFIG, "token_mode": "manual", "rtc_token": token}
    assert validate_config(manual)["rtc_token_expires_at"] == expires
    assert validate_config({**manual, "rtc_token_expires_at": expires + 60})["rtc_token_expires_at"] == expires
    assert validate_config({**manual, "rtc_token_expires_at": expires - 60})["rtc_token_expires_at"] == expires - 60
    for field, value in (("rtc_room_id", "other"), ("rtc_user_id", "someone"),
                         ("rtc_app_id", "100000000000000000000000"), ("rtc_token", token[:-4])):
        with pytest.raises(MmpgError, match="rtc_token_invalid"):
            validate_config({**manual, field: value})
    wildcard = sign_rtc_token(
        CONFIG["rtc_app_id"], CONFIG["rtc_app_key"], "*", "listener",
        issued_at=issued, expires_at=expires,
    )
    assert validate_config({**manual, "rtc_token": wildcard})["rtc_token_expires_at"] == expires
    opaque = {**manual, "rtc_token": "opaque-token", "rtc_token_expires_at": expires}
    assert validate_config(opaque)["rtc_token_expires_at"] == expires


def test_cloud_connector_signs_the_expected_rtc_path_without_wire_logging(monkeypatch):
    pytest.importorskip("volcenginesdkcore")
    from websockets.asyncio import client

    from services.stt.mmpg import connect_mmpg

    async def check():
        socket = object()

        async def connect(url, *, additional_headers, logger, **options):
            assert url == "wss://interactai.volcvideo.com/api/v2/mmpg"
            authorization = additional_headers["Authorization"]
            assert "Credential=signature-account/" in authorization
            assert "/cn-north-1/rtc/request" in authorization
            assert "signature-key" not in authorization
            assert options["open_timeout"] > 0
            assert not logger.isEnabledFor(10)
            return socket

        monkeypatch.setattr(client, "connect", connect)
        assert await connect_mmpg(CONFIG) is socket

    asyncio.run(check())


def test_mmpg_gates_audio_on_both_transports_and_filters_conversations(caplog):
    from services.stt.mmpg import MmpgManager

    async def check():
        cloud = CloudSocket()
        controls, audio, transcripts, capture = [], [], [], []

        async def connect(_):
            return cloud

        async def send(payload):
            (audio if isinstance(payload, bytes) else controls).append(payload)

        async def start(mode, ids, candidate):
            capture.append((mode, ids, candidate))

        async def stop(mode, reason):
            capture.append((mode, reason))

        def on_transcript(text, trace):
            transcripts.append((text, trace))

        manager = MmpgManager(lambda: CONFIG, on_transcript, on_capture_start=start,
                              on_capture_stop=stop, connector=connect)
        owner = object()
        manager.attach(owner, send)
        try:
            manager.start(owner, "start", [-3], 12)
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            join = next(msg for msg in controls if msg["type"] == "rtc_join")
            assert join["token"].startswith("001" + CONFIG["rtc_app_id"])
            assert join["roomId"] != CONFIG["rtc_room_id"]
            assert capture == cloud.sent == []
            assert not manager.offer_audio(np.array([0.25]))
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": join["generation"]})
            await wait_until(lambda: bool(cloud.sent))
            request = cloud.sent[0]
            assert request["type"] == "session.create"
            rtc = request["session"]["AgentConfig"]["TransmitConfig"]
            assert rtc["RoomId"] == join["roomId"]
            assert rtc["TargetUserId"] == [join["userId"]]
            assert capture == []
            cloud.push("session.created")
            await wait_until(lambda: manager.status()["state"] == "listening")
            assert capture == [("start", [-3], 12)]
            assert await asyncio.to_thread(manager.offer_audio, np.array([0.25, -0.25]))
            await wait_until(lambda: bool(audio))
            assert audio == [struct.pack("<ff", 0.25, -0.25)]

            cloud.push("user.audio.transcription.segment", conv_id="a", transcript="partial")
            cloud.push("user.conversation.canceled", conv_id="a")
            cloud.push("user.audio.transcription.done", conv_id="a", transcript="discard")
            cloud.push("user.audio.transcription.done", conv_id="private-cloud-id", transcript=" one-private-transcript ")
            cloud.push("user.audio.transcription.done", conv_id="private-cloud-id", transcript="duplicate")
            cloud.push("user.audio.transcription.done", conv_id="c", transcript="last")
            await wait_until(lambda: len(transcripts) >= 2)
            assert [text for text, _ in transcripts] == ["one-private-transcript", "last"]
            for _, trace in transcripts:
                assert f"turn={trace} event=asr_done" in caplog.text
        finally:
            await manager.stop("paused")
        assert cloud.closed and cloud.sent[-1]["type"] == "session.close"
        assert capture[-1] == ("start", "paused")
        assert not manager.active
        assert manager.status()["reason"] == "paused"
        assert not manager.offer_audio(np.array([0.25]))

    with caplog.at_level("INFO", logger="mmpg.diagnostics"):
        logger = logging.getLogger("mmpg.diagnostics")
        attach_handler = not logger.propagate
        if attach_handler:
            logger.addHandler(caplog.handler)
        try:
            asyncio.run(check())
        finally:
            if attach_handler:
                logger.removeHandler(caplog.handler)
    for private in ("one-private-transcript", "private-cloud-id", CONFIG["doubao_api_key"], CONFIG["rtc_app_key"]):
        assert private not in caplog.text


def test_mmpg_retries_with_a_fresh_generation_and_rejects_stale_owner_events():
    from services.stt.mmpg import MmpgManager

    async def check():
        controls, clouds, transcripts = [], [], []

        async def connect(_):
            socket = CloudSocket()
            clouds.append(socket)
            return socket

        async def send(payload):
            controls.append(payload)

        manager = MmpgManager(lambda: CONFIG, transcripts.append, connector=connect, retry_delays=(0,))
        owner = object()
        manager.attach(owner, send)
        assert not manager.attach(object(), send)
        try:
            manager.start(owner, "test", [-2])
            joins = lambda: [msg for msg in controls if msg["type"] == "rtc_join"]
            await wait_until(lambda: len(joins()) == 1)
            first = joins()[0]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": first["generation"]})
            await wait_until(lambda: len(clouds) == 1)
            clouds[0].push("session.created")
            await wait_until(lambda: manager.status()["state"] == "listening")
            await manager.rtc_event(owner, {"type": "rtc_fault", "generation": first["generation"],
                                            "code": "rtc_error"})
            await wait_until(lambda: len(joins()) == 2)
            assert clouds[0].closed
            second = joins()[1]
            assert first["roomId"] != second["roomId"]
            assert first["generation"] != second["generation"]
            assert not await manager.rtc_event(owner, {"type": "rtc_ready", "generation": first["generation"]})
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": second["generation"]})
            await wait_until(lambda: len(clouds) == 2)
            clouds[1].push("session.created")
            await wait_until(lambda: manager.status()["state"] == "listening")
            clouds[0].push("user.audio.transcription.done", conv_id="late", transcript="stale")
            clouds[1].push("user.audio.transcription.started", conv_id="test")
            clouds[1].push("user.audio.transcription.segment", conv_id="test", transcript="partial")
            clouds[1].push("user.audio.transcription.done", conv_id="test", transcript="heard")
            await wait_until(lambda: any(msg["type"] == "transcript" for msg in controls))
            assert [msg["transcript"] for msg in controls if msg["type"] == "transcript"] == ["heard"]
            diagnostics = [msg for msg in controls if msg["type"] == "diagnostic"]
            assert [msg["event"] for msg in diagnostics] == ["speech_recognized", "speech_paused"]
            assert all("transcript" not in msg for msg in diagnostics)
            assert transcripts == []
        finally:
            await manager.detach(owner)
        assert clouds[1].closed

    asyncio.run(check())


def test_mmpg_retries_each_new_outage_after_stable_recovery(monkeypatch):
    from services.stt import mmpg

    monkeypatch.setattr(mmpg, "RECOVERY_STABLE_SECONDS", 0, raising=False)

    async def check():
        controls = []
        clouds = []

        async def connect(_):
            cloud = CloudSocket()
            clouds.append(cloud)
            cloud.push("session.created")
            return cloud

        async def send(payload):
            controls.append(payload)

        manager = mmpg.MmpgManager(lambda: CONFIG, str, connector=connect, retry_delays=(0,))
        owner = object()
        manager.attach(owner, send)

        async def wait_for_join(generation):
            await wait_until(lambda: any(
                msg["type"] == "rtc_join" and msg["generation"] == generation for msg in controls
            ))

        try:
            manager.start(owner, "start", [1])
            for generation in (1, 2):
                await wait_for_join(generation)
                await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
                await wait_until(lambda: manager.status()["state"] == "listening")
                clouds[-1].push("session.closed")
            await wait_for_join(3)
            # A failed recovery never reached listening, so it consumes the retry budget.
            await manager.rtc_event(owner, {"type": "rtc_fault", "generation": 3, "code": "rtc_join_failed"})
            await wait_until(lambda: manager.status()["state"] == "error")
            assert manager.status()["generation"] == 3
            assert all(cloud.closed for cloud in clouds)
        finally:
            await manager.detach(owner)

    asyncio.run(check())


def test_mmpg_stop_waits_for_partial_capture_start_before_releasing_devices():
    from services.stt.mmpg import MmpgManager

    async def check():
        controls, lifecycle = [], []
        cloud = CloudSocket()
        allow_capture_open = asyncio.Event()

        async def connect(_):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            controls.append(payload)

        async def start(*_):
            lifecycle.append("opening")
            await allow_capture_open.wait()
            lifecycle.append("opened")

        async def stop(_mode, reason):
            lifecycle.append(reason)

        manager = MmpgManager(lambda: CONFIG, str, on_capture_start=start,
                              on_capture_stop=stop, connector=connect)
        owner = object()
        manager.attach(owner, send)
        manager.start(owner, "start", [1])
        await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
        generation = manager.status()["generation"]
        await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
        await wait_until(lambda: lifecycle == ["opening"])
        closing = asyncio.create_task(manager.stop("paused"))
        await asyncio.sleep(0)
        assert manager.active and not closing.done()
        assert lifecycle == ["opening"]
        assert not manager.offer_audio(bytes(4))
        allow_capture_open.set()
        await asyncio.wait_for(closing, timeout=1)
        assert lifecycle == ["opening", "opened", "paused"]
        assert cloud.closed
        assert not any(msg.get("state") == "listening" for msg in controls)

    asyncio.run(check())


def test_mmpg_overrun_discards_pending_pcm_and_closes_the_generation():
    from services.stt.mmpg import MmpgManager

    async def check():
        controls, audio = [], []
        cloud = CloudSocket()

        async def connect(_):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            (audio if isinstance(payload, bytes) else controls).append(payload)

        manager = MmpgManager(lambda: CONFIG, str, connector=connect, retry_delays=())
        owner = object()
        manager.attach(owner, send)
        try:
            manager.start(owner, "start", [1])
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": manager.status()["generation"]})
            await wait_until(lambda: manager.status()["state"] == "listening")
            assert manager.offer_audio(bytes(16_000 * 4))
            assert not manager.offer_audio(bytes(4))
            await wait_until(lambda: manager.status()["state"] == "error")
            assert manager.status()["error"] == "audio_overrun"
            assert manager.status()["audio_dropped"] == 16_001
            assert audio == [] and cloud.closed
        finally:
            await manager.detach(owner)

    asyncio.run(check())


def test_manual_token_keeps_room_and_does_not_retry_expiry():
    from services.stt.mmpg import MmpgManager

    async def check():
        cfg = {
            **CONFIG, "token_mode": "manual", "rtc_app_key": "",
            "rtc_token": "console-issued", "rtc_token_expires_at": int(time.time()) + 1000,
        }
        controls = []
        cloud = CloudSocket()

        async def connect(_):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            controls.append(payload)

        manager = MmpgManager(lambda: cfg, str, connector=connect, retry_delays=(0,))
        owner = object()
        manager.attach(owner, send)
        try:
            manager.start(owner, "test", [1])
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            join = next(msg for msg in controls if msg["type"] == "rtc_join")
            assert join["roomId"] == cfg["rtc_room_id"] and join["token"] == cfg["rtc_token"]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": join["generation"]})
            await wait_until(lambda: manager.status()["state"] == "listening")
            await manager.rtc_event(owner, {"type": "renew", "generation": join["generation"]})
            await manager.rtc_event(owner, {"type": "renew", "generation": join["generation"]})
            assert [msg["code"] for msg in controls if msg["type"] == "notice"] == [
                "manual_token_renewal_required",
            ]
            assert manager.status()["state"] == "listening"
            await manager.rtc_event(owner, {"type": "rtc_fault", "generation": join["generation"],
                                            "code": "rtc_token_expired"})
            await wait_until(lambda: manager.status()["state"] == "error")
            assert manager.status()["error"] == "rtc_token_expired"
            assert [msg["type"] for msg in controls].count("rtc_join") == 1
            assert not any(msg["type"] == "rtc_token" for msg in controls)
        finally:
            await manager.detach(owner)

    asyncio.run(check())


@pytest.mark.parametrize("retrying", (False, True))
def test_mmpg_diagnostic_deadline_spans_retry_and_releases_capture(retrying):
    from services.stt.mmpg import MmpgManager

    async def check():
        cloud = CloudSocket()
        controls, stopped = [], []

        async def connect(_):
            cloud.push("session.created")
            return cloud

        async def send(message):
            controls.append(message)

        async def start(*_):
            pass

        async def stop(_mode, reason):
            stopped.append(reason)

        manager = MmpgManager(lambda: CONFIG, str, connector=connect, on_capture_start=start,
                              on_capture_stop=stop, test_timeout=0.25, retry_delays=(1,))
        owner = object()
        manager.attach(owner, send)
        try:
            manager.start(owner, "test", [1])
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            generation = manager.status()["generation"]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
            await wait_until(lambda: manager.status()["state"] == "listening")
            if retrying:
                await manager.rtc_event(owner, {"type": "rtc_fault", "generation": generation, "code": "rtc_error"})
                await wait_until(lambda: manager.status()["state"] == "reconnecting")
            await wait_until(lambda: manager.status()["state"] == "idle")
            assert manager.status()["reason"] == "test_timeout"
            assert stopped == ["rtc_error" if retrying else "test_timeout"] and cloud.closed
            assert not manager.active
        finally:
            await manager.detach(owner)

    asyncio.run(check())


def test_audio_watchdog_excludes_rtc_recovery_then_detects_missing_audio(monkeypatch):
    from services.stt import mmpg

    monkeypatch.setattr(mmpg, "AUDIO_STALL_TIMEOUT", 0.025)

    async def check():
        cloud = CloudSocket()
        messages = []

        async def connect(_):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            messages.append(payload)

        manager = mmpg.MmpgManager(lambda: CONFIG, str, connector=connect, retry_delays=())
        owner = object()
        manager.attach(owner, send)
        try:
            manager.start(owner, "start", [1])
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in messages))
            generation = manager.status()["generation"]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
            await wait_until(lambda: manager.status()["state"] == "listening")
            await manager.rtc_event(owner, {"type": "rtc_reconnecting", "generation": generation})
            await asyncio.sleep(0.075)
            assert manager.status()["state"] == "reconnecting" and not cloud.closed
            await manager.rtc_event(owner, {"type": "rtc_recovered", "generation": generation})
            await wait_until(lambda: manager.status()["state"] == "error")
            assert manager.status()["error"] == "audio_stalled"
            assert cloud.closed
        finally:
            await manager.detach(owner)

    asyncio.run(check())


@pytest.mark.parametrize(("reports", "expected"), (
    ([{"underruns": 1}, {"underruns": 2}], "audio_underrun"),
    ([{"overrun_samples": 16}], "audio_overrun"),
))
def test_browser_pcm_health_recovers_after_repeated_underruns_or_overrun(reports, expected):
    from services.stt.mmpg import MmpgManager

    async def check():
        controls = []
        cloud = CloudSocket()

        async def connect(_):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            controls.append(payload)

        manager = MmpgManager(lambda: CONFIG, str, connector=connect, retry_delays=(0,))
        owner = object()
        manager.attach(owner, send)
        try:
            manager.start(owner, "test", [1])
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            generation = manager.status()["generation"]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
            await wait_until(lambda: manager.status()["state"] == "listening")
            await manager.rtc_event(owner, {"type": "audio_status", "generation": generation, "underruns": 1})
            assert manager.status()["state"] == "listening"
            for report in reports:
                await manager.rtc_event(owner, {"type": "audio_status", "generation": generation, **report})
            await wait_until(lambda: manager.status()["generation"] > generation)
            assert cloud.closed
            assert any(msg.get("state") == "reconnecting" and msg.get("error") == expected for msg in controls)
        finally:
            await manager.detach(owner)

    asyncio.run(check())


def test_mmpg_ws_rejects_non_loopback_and_cross_origin_owners(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from api.realtime import mmpg
    from services.stt.mmpg import MmpgManager

    class Platform:
        platform = "win32"

    monkeypatch.setattr(mmpg, "sys", Platform())
    manager = MmpgManager(lambda: CONFIG, str)
    app = FastAPI()
    mmpg.bind_mmpg_routes(app, manager, on_control=None)
    url = "ws://127.0.0.1:18080/ws/stt/mmpg"
    good_headers = {"origin": "http://127.0.0.1:18080"}
    local = TestClient(app, base_url="http://127.0.0.1:18080", client=("127.0.0.1", 10_001))
    remote = TestClient(app, base_url="http://127.0.0.1:18080", client=("192.0.2.3", 10_001))
    public = TestClient(app, base_url="http://example.com", client=("127.0.0.1", 10_001))
    denied = [
        (remote, url, good_headers),
        (public, "ws://example.com/ws/stt/mmpg", {"origin": "http://example.com"}),
        (local, url, {"origin": "http://localhost:18080"}),
        (local, url, {"origin": "http://127.0.0.1:18081"}),
        (local, url, {"origin": "https://127.0.0.1:18080"}),
        (local, url, {}),
    ]
    for client, endpoint, headers in denied:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(endpoint, headers=headers):
                pytest.fail("The publisher accepted an untrusted owner")
        assert exc.value.code == 1008

    # TestClient constructs websocket URL with its own default testserver and
    # derives the Host header from the absolute URL when given one.
    with local.websocket_connect(url, headers=good_headers) as owner:
        init = owner.receive_json()
        assert init["state"] == "idle" and init["initial"]
        owner.send_json({"type": "start", "device_ids": [True]})
        assert owner.receive_json() == {"type": "error", "code": "invalid_command"}
        with pytest.raises(WebSocketDisconnect) as exc:
            with local.websocket_connect(url, headers=good_headers) as busy:
                assert busy.receive_json() == {"type": "error", "code": "mmpg_busy"}
                busy.receive_text()
        assert exc.value.code == 1008

def test_candidate_mmpg_routes_final_transcripts_to_qa_and_drains_on_pause(monkeypatch):
    from api.assist import candidate_mmpg, pipeline
    from core.config import AppConfig, Mmpg
    from core.session import Session
    from services.stt.mmpg import MmpgManager

    cfg = AppConfig(stt_provider="doubao_mmpg", candidate_stt_provider="doubao_mmpg",
                    candidate_asr_enabled=True, mmpg=Mmpg(**CONFIG))
    session = Session(is_recording=True, last_device_id=10, last_candidate_mic_device_id=20)
    saved, interviewer_texts, statuses = [], [], []
    monkeypatch.setattr(candidate_mmpg.pipeline, "get_capture_config", lambda: cfg)
    monkeypatch.setattr(candidate_mmpg, "get_session", lambda: session)
    monkeypatch.setattr(candidate_mmpg, "_pending", {})
    monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: True)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _: False)
    monkeypatch.setattr(pipeline, "broadcast", statuses.append)
    monkeypatch.setattr(pipeline, "_submit_candidate_knowledge_update", lambda *args: saved.append(args))

    class Mic:
        callback = None
        stopped = False

        def start(self, device_id, *, on_audio, owner, mic_compatibility_mode, queue_audio):
            assert device_id == 20 and owner == "assist-candidate-mmpg"
            assert mic_compatibility_mode and not queue_audio
            self.callback = on_audio

        def stop(self, *, owner):
            assert owner == "assist-candidate-mmpg"
            self.stopped = True

    mic = Mic()
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", mic)

    async def check():
        primary_cloud, candidate_cloud = CloudSocket(), CloudSocket()

        async def primary_connect(_):
            return primary_cloud

        async def candidate_connect(_):
            return candidate_cloud

        interviewer = MmpgManager(lambda: cfg.mmpg, lambda text, _: interviewer_texts.append(text),
                                  connector=primary_connect)
        candidate = MmpgManager(
            lambda: cfg.mmpg, candidate_mmpg._on_transcript, on_turn=candidate_mmpg._on_turn,
            on_capture_start=candidate_mmpg._start_capture, on_capture_stop=candidate_mmpg._stop_capture,
            connector=candidate_connect,
        )
        monkeypatch.setattr(candidate_mmpg, "manager", candidate)
        owners = [object(), object()]
        controls, pcm = [[], []], [[], []]

        async def primary_send(payload):
            (pcm[0] if isinstance(payload, bytes) else controls[0]).append(payload)

        async def candidate_send(payload):
            (pcm[1] if isinstance(payload, bytes) else controls[1]).append(payload)
            if mic.stopped and isinstance(payload, bytes) and not any(payload):
                candidate_cloud.push("user.audio.transcription.done", conv_id="tail", transcript="最后一句")

        interviewer.attach(owners[0], primary_send)
        candidate.attach(owners[1], candidate_send)
        interviewer.start(owners[0], "start", [10], 20)
        candidate.start(owners[1], "start", [20], preflight=candidate_mmpg.prepare)

        try:
            await wait_until(lambda: all(any(p["type"] == "rtc_join" for p in messages) for messages in controls))
            joins = [next(p for p in messages if p["type"] == "rtc_join") for messages in controls]
            assert joins[0]["roomId"] != joins[1]["roomId"]
            for manager, owner, join, cloud in zip(
                (interviewer, candidate), owners, joins, (primary_cloud, candidate_cloud),
            ):
                await manager.rtc_event(owner, {"type": "rtc_ready", "generation": join["generation"]})
                cloud.push("session.created")
            await wait_until(lambda: all(m.status()["state"] == "listening" for m in (interviewer, candidate)))
            assert interviewer.offer_audio(np.array([0.25]))
            assert mic.callback(np.array([0.75]))
            await wait_until(lambda: all(pcm))
            assert pcm[0] == [struct.pack("<f", 0.25)]
            assert pcm[1] == [struct.pack("<f", 0.75)]

            session.open_candidate_answer_window("qa-1")
            candidate_cloud.push("user.audio.transcription.started", conv_id="reply")
            await wait_until(lambda: session.has_candidate_asr_pending_for_qa("qa-1"))
            session.open_candidate_answer_window("qa-2")
            primary_cloud.push("user.audio.transcription.done", conv_id="question", transcript="下一道问题")
            candidate_cloud.push("user.audio.transcription.done", conv_id="reply", transcript="我的真实回答")
            await wait_until(lambda: bool(saved) and bool(interviewer_texts))
            assert interviewer_texts == ["下一道问题"]
            assert saved == [("qa-1", "我的真实回答")]
            assert statuses[-1]["type"] == "candidate_transcription"
            assert statuses[-1]["provider"] == "doubao_mmpg"
            assert statuses[-1]["is_final"] is True

            candidate_cloud.push("user.audio.transcription.started", conv_id="cleared")
            await wait_until(lambda: session.has_candidate_asr_pending_for_qa("qa-2"))
            candidate.discard_turns()
            candidate_cloud.push("user.audio.transcription.done", conv_id="cleared", transcript="旧记录")
            candidate_cloud.push("user.audio.transcription.started", conv_id="tail")
            await wait_until(lambda: session.has_candidate_asr_pending_for_qa("qa-2"))

            async def keep_interviewer_alive():
                chunk = np.zeros(160, dtype=np.float32)
                while True:
                    interviewer.offer_audio(chunk)
                    await asyncio.sleep(0.2)

            keepalive = asyncio.create_task(keep_interviewer_alive())
            try:
                await candidate_mmpg.stop("paused", drain=True)
            finally:
                keepalive.cancel()
                await asyncio.gather(keepalive, return_exceptions=True)
            assert candidate_cloud.closed and mic.stopped
            assert interviewer.status()["state"] == "listening" and not primary_cloud.closed
            assert saved == [("qa-1", "我的真实回答"), ("qa-2", "最后一句")]
            assert not session.candidate_asr_busy
        finally:
            await candidate.stop()
            await interviewer.stop()

    asyncio.run(check())


def test_candidate_mmpg_does_not_share_the_interviewer_manual_token(monkeypatch):
    from api.assist import candidate_mmpg, pipeline
    from core.config import AppConfig, Mmpg
    from core.session import Session
    from services.stt.mmpg import MmpgError

    cfg = AppConfig(candidate_stt_provider="doubao_mmpg", candidate_asr_enabled=True,
                    mmpg=Mmpg(**{**CONFIG, "token_mode": "manual"}))
    monkeypatch.setattr(candidate_mmpg.pipeline, "get_capture_config", lambda: cfg)
    monkeypatch.setattr(candidate_mmpg, "get_session", lambda: Session(
        is_recording=True, last_device_id=10, last_candidate_mic_device_id=20,
    ))
    monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: True)
    with pytest.raises(MmpgError, match="candidate_mmpg_auto_token_required"):
        asyncio.run(candidate_mmpg.prepare("start", [20], None))

    monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: False)
    asyncio.run(candidate_mmpg.prepare("start", [20], None))



@pytest.mark.parametrize("mode, value", [
    ("start", None),
    ("resume", True),
    ("start", ""),
    ("start", "with spaces"),
    ("start", "a" * 65),
    ("test", "single-mic-run"),
])
def test_mmpg_ws_rejects_invalid_single_mic_id(monkeypatch, mode, value):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.realtime import mmpg
    from services.stt.mmpg import MmpgManager

    class Platform:
        platform = "win32"

    monkeypatch.setattr(mmpg, "sys", Platform())
    manager = MmpgManager(lambda: CONFIG, str)
    app = FastAPI()
    mmpg.bind_mmpg_routes(app, manager, on_control=None)
    url = "ws://127.0.0.1:18080/ws/stt/mmpg"
    client = TestClient(app, base_url="http://127.0.0.1:18080", client=("127.0.0.1", 10_001))
    with client.websocket_connect(url, headers={"origin": "http://127.0.0.1:18080"}) as owner:
        assert owner.receive_json()["state"] == "idle"
        owner.send_json({"type": mode, "device_ids": [8], "single_mic_id": value})
        assert owner.receive_json() == {"type": "error", "code": "invalid_command"}
        assert not manager.active


def test_mmpg_single_mic_switch_waits_for_delayed_finals_without_restarting(monkeypatch):
    from services.stt import mmpg

    monkeypatch.setattr(mmpg, "ROLE_DRAIN_PLAYBACK_GRACE_SECONDS", 0.02)

    async def check():
        cloud = CloudSocket()
        controls, audio, finals, lifecycle, started = [], [], [], [], []
        session_id = "one-mic-session"

        async def connector(_):
            cloud.push("session.created")
            return cloud

        async def send(message):
            (audio if isinstance(message, bytes) else controls).append(message)

        async def prepare(mode, devices, candidate, *, single_mic_id):
            lifecycle.append(("prepare", mode, devices, candidate, single_mic_id))

        async def start(mode, devices, candidate, *, single_mic_id):
            lifecycle.append(("start", mode, devices, candidate, single_mic_id))

        async def stop(mode, reason, *, single_mic_id):
            lifecycle.append(("stop", mode, reason, single_mic_id))

        manager = mmpg.MmpgManager(
            lambda: {**CONFIG, "silence_ms": 500},
            lambda text, _: finals.append(text),
            on_turn=lambda kind, trace: started.append(trace) if kind == "started" else None,
            on_capture_start=start, on_capture_stop=stop, connector=connector,
        )
        owner = object()
        assert manager.attach(owner, send)
        manager.start(owner, "start", [8], 8, preflight=prepare, single_mic_id=session_id)
        assert manager.single_mic_id == session_id
        try:
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            generation = manager.status()["generation"]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
            await wait_until(lambda: manager.status()["state"] == "listening")
            assert lifecycle == [("prepare", "start", [8], 8, session_id), ("start", "start", [8], 8, session_id)]
            pcm = np.ones(640, dtype=np.float32).tobytes()
            assert manager.offer_audio(pcm)

            switching = asyncio.create_task(manager.finish_role_input(session_id, timeout=4))
            await asyncio.sleep(0)
            await wait_until(lambda: audio == [pcm])
            assert not switching.done()

            # The provider can report speech only after the server sent the last spoken PCM.
            cloud.push("user.conversation.created", conv_id="late-turn")
            await wait_until(lambda: len(started) == 1)
            silence = bytes(8_000 * 4)
            assert manager.offer_audio(silence)
            await wait_until(lambda: audio == [pcm, silence])
            # Once enough silence is sent, the open provider turn still blocks the switch.
            await asyncio.sleep(0.1)
            assert not switching.done()

            cloud.push("user.audio.transcription.done", conv_id="late-turn", transcript="候选人最后几个字")
            assert await asyncio.wait_for(switching, timeout=3)
            assert finals == ["候选人最后几个字"]
            assert audio == [pcm, silence]
            assert manager.status()["generation"] == generation
            assert manager.status()["state"] == "listening"
        finally:
            await manager.detach(owner)
        assert lifecycle[-1] == ("stop", "start", "owner_disconnected", session_id)
        assert manager.single_mic_id is None

    asyncio.run(check())


def test_mmpg_single_mic_switch_rejects_stale_id_timeout_and_recovered_audio_gate():
    from services.stt.mmpg import MmpgManager

    async def check():
        cloud, controls = CloudSocket(), []

        async def connector(_):
            cloud.push("session.created")
            return cloud

        async def send(message):
            if isinstance(message, dict):
                controls.append(message)

        owner = object()
        manager = MmpgManager(lambda: CONFIG, str, connector=connector)
        assert manager.attach(owner, send)
        manager.start(owner, "resume", [8], single_mic_id="current")
        try:
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            generation = manager.status()["generation"]
            await manager.rtc_event(owner, {"type": "rtc_ready", "generation": generation})
            await wait_until(lambda: manager.status()["state"] == "listening")
            assert not await manager.finish_role_input("stale", timeout=0.1)
            assert not await manager.finish_role_input("current", timeout=0.03)
            assert manager.status()["state"] == "listening"

            switching = asyncio.create_task(manager.finish_role_input("current", timeout=2))
            await asyncio.sleep(0)
            await manager.rtc_event(owner, {"type": "rtc_reconnecting", "generation": generation})
            await manager.rtc_event(owner, {"type": "rtc_recovered", "generation": generation})
            assert manager.status()["state"] == "listening"
            assert not await asyncio.wait_for(switching, timeout=0.2)
        finally:
            await manager.detach(owner)

    asyncio.run(check())


def test_mmpg_offer_audio_catches_closed_loop_runtime_error_and_monotonic_renew(monkeypatch, caplog):
    from services.stt import mmpg
    from services.stt.mmpg import MmpgError, MmpgManager

    manager = MmpgManager(lambda: CONFIG, str)
    manager._set_audio_gate(1)

    class ClosedLoop:
        def call_soon_threadsafe(self, _cb):
            raise RuntimeError("Event loop is closed")

    manager._loop = ClosedLoop()
    assert manager.offer_audio(np.array([0.1, -0.1], dtype=np.float32)) is False
    assert manager._wake_pending is False

    async def check_renew():
        cloud = CloudSocket()
        controls = []

        async def connector(_):
            cloud.push("session.created")
            return cloud

        async def send(msg):
            if isinstance(msg, dict):
                controls.append(msg)

        mgr = MmpgManager(lambda: CONFIG, str, connector=connector)
        owner = object()
        mgr.attach(owner, send)
        mgr.start(owner, "test", [1])
        try:
            await wait_until(lambda: any(m["type"] == "rtc_join" for m in controls))
            gen = mgr.status()["generation"]
            await mgr.rtc_event(owner, {"type": "rtc_ready", "generation": gen})
            await wait_until(lambda: mgr.status()["state"] == "listening")
            mgr._current.token_issued_at = time.monotonic() - 2.0
            await mgr.rtc_event(owner, {"type": "renew", "generation": gen})
            assert any(m["type"] == "rtc_token" for m in controls)
        finally:
            await mgr.detach(owner)

    asyncio.run(check_renew())

    pytest.importorskip("volcenginesdkcore")
    from websockets.asyncio import client

    async def failing_connect(*_a, **_kw):
        raise OSError("network unreachable")

    monkeypatch.setattr(client, "connect", failing_connect)
    logger = logging.getLogger("mmpg.diagnostics")
    attach_handler = not logger.propagate
    if attach_handler:
        logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING", logger="mmpg.diagnostics"):
            with pytest.raises(MmpgError, match="cloud_connection_failed"):
                asyncio.run(mmpg.connect_mmpg(CONFIG))
    finally:
        if attach_handler:
            logger.removeHandler(caplog.handler)
    assert "connect_mmpg failed kind=OSError" in caplog.text


def test_mmpg_test_timeout_tracks_and_cleans_up_timeout_task():
    from services.stt.mmpg import MmpgManager

    async def check_timeout():
        cloud = CloudSocket()
        controls = []

        async def connector(_):
            cloud.push("session.created")
            return cloud

        async def send(msg):
            if isinstance(msg, dict):
                controls.append(msg)

        mgr = MmpgManager(lambda: CONFIG, str, connector=connector, test_timeout=0.05)
        owner = object()
        mgr.attach(owner, send)
        mgr.start(owner, "test", [1])
        run = mgr._run
        try:
            await wait_until(lambda: mgr.status()["state"] == "idle" and mgr.status()["reason"] == "test_timeout")
            await wait_until(lambda: run.timeout_task is None)
            assert run.timeout_task is None
        finally:
            await mgr.detach(owner)

    asyncio.run(check_timeout())


def test_mmpg_schedule_timeout_test_skips_when_stopping_and_handles_concurrent_slow_close():
    from services.stt.mmpg import MmpgManager

    async def check_race():
        class SlowCloseCloud(CloudSocket):
            async def close(self):
                await asyncio.sleep(0.08)
                await super().close()

        cloud = SlowCloseCloud()
        controls = []

        async def connector(_):
            cloud.push("session.created")
            return cloud

        async def send(msg):
            if isinstance(msg, dict):
                controls.append(msg)

        mgr = MmpgManager(lambda: CONFIG, str, connector=connector, test_timeout=0.04)
        owner = object()
        mgr.attach(owner, send)
        mgr.start(owner, "test", [1])
        run = mgr._run
        try:
            await wait_until(lambda: any(m["type"] == "rtc_join" for m in controls))
            await asyncio.sleep(0.045)
            await mgr.stop("user_stop")
            await wait_until(lambda: run.timeout_task is None)
            assert mgr.status()["state"] == "idle"
            assert run.timeout_task is None
        finally:
            await mgr.detach(owner)

        # Calling _schedule_timeout_test after stop_reason is set must be a no-op
        mgr2 = MmpgManager(lambda: CONFIG, str, connector=connector, test_timeout=10.0)
        mgr2.attach(owner, send)
        mgr2.start(owner, "test", [1])
        run2 = mgr2._run
        try:
            run2.stop_reason = "user_stop"
            mgr2._schedule_timeout_test(run2)
            assert run2.timeout_task is None
        finally:
            await mgr2.detach(owner)

    asyncio.run(check_race())


def test_mmpg_drain_input_monotonic_pacing_bounded_catchup_and_queue_limit(monkeypatch):
    from services.stt import mmpg
    from services.stt.mmpg import MAX_AUDIO_BYTES, MmpgManager

    async def check_jitter_and_stall():
        loop = asyncio.get_running_loop()
        fake_now = [100.0]
        monkeypatch.setattr(loop, "time", lambda: fake_now[0])

        # Case 1: alternating early/late timer wakes (Windows 15.625ms-style jitter) must
        # maintain ~40ms/chunk average without early-fire runaway or slow drift.
        mgr = MmpgManager(lambda: CONFIG, str)
        sent_times: list[float] = []
        wait_timeouts: list[float] = []
        done = asyncio.Event()

        def fake_offer(samples):
            assert len(samples) == 640 * 4
            sent_times.append(fake_now[0])
            if len(sent_times) >= 25:
                done.set()
            return True

        monkeypatch.setattr(mgr, "offer_audio", fake_offer)

        real_wait = asyncio.wait
        jitter_pattern = [-0.00875, 0.006875, -0.005, 0.009]
        wait_idx = [0]

        async def fake_wait(aws, *, timeout=None):
            assert timeout is not None and timeout > 0
            wait_timeouts.append(timeout)
            jitter = jitter_pattern[wait_idx[0] % len(jitter_pattern)]
            wait_idx[0] += 1
            fake_now[0] += max(0.015625, timeout + jitter)
            return await real_wait(aws, timeout=0)

        monkeypatch.setattr(mmpg.asyncio, "wait", fake_wait)

        async def finishing():
            await done.wait()
            return True

        assert await mgr._drain_input(finishing()) is True
        elapsed = sent_times[-1] - sent_times[0]
        expected = 24 * 0.04
        assert abs(elapsed - expected) < 0.025, f"drifted: elapsed={elapsed:.4f} expected={expected:.4f}"
        assert all(t > 0 for t in wait_timeouts)

        # Case 2: a sudden 2.0s stall must clamp catch-up so chunks cannot burst-overflow
        # the browser's 1s PcmQueue (25 chunks).
        mgr2 = MmpgManager(lambda: CONFIG, str)
        sent_at_stall: list[float] = []
        done2 = asyncio.Event()
        stalled = [False]

        def fake_offer2(_samples):
            sent_at_stall.append(fake_now[0])
            if len(sent_at_stall) >= 8:
                done2.set()
            return True

        monkeypatch.setattr(mgr2, "offer_audio", fake_offer2)

        async def stall_wait(aws, *, timeout=None):
            if not stalled[0]:
                stalled[0] = True
                fake_now[0] += 2.0  # 2-second event-loop stall
            else:
                fake_now[0] += timeout
            return await real_wait(aws, timeout=0)

        monkeypatch.setattr(mmpg.asyncio, "wait", stall_wait)

        async def finishing2():
            await done2.wait()
            return True

        assert await mgr2._drain_input(finishing2()) is True
        stall_timestamp = sent_at_stall[1]
        burst_at_stall = sum(1 for ts in sent_at_stall[1:] if ts == stall_timestamp)
        assert 1 <= burst_at_stall <= 4, f"unbounded burst after stall: {burst_at_stall} chunks"

        # Case 3: queue at MAX_AUDIO_BYTES skips offer_audio without failing until space frees up.
        mgr3 = MmpgManager(lambda: CONFIG, str)
        mgr3._queued_bytes = MAX_AUDIO_BYTES
        offered3: list[int] = []
        done3 = asyncio.Event()
        ticks = [0]

        def fake_offer3(samples):
            offered3.append(len(samples))
            if len(offered3) >= 2:
                done3.set()
            return True

        monkeypatch.setattr(mgr3, "offer_audio", fake_offer3)

        async def backpressure_wait(aws, *, timeout=None):
            ticks[0] += 1
            fake_now[0] += timeout
            if ticks[0] == 2:
                mgr3._queued_bytes = 0
            return await real_wait(aws, timeout=0)

        monkeypatch.setattr(mmpg.asyncio, "wait", backpressure_wait)

        async def finishing3():
            await done3.wait()
            return True

        assert await mgr3._drain_input(finishing3()) is True
        assert ticks[0] >= 3
        assert len(offered3) == 2
        assert mgr3._audio_failed is False

    asyncio.run(check_jitter_and_stall())


def test_mmpg_drain_transcripts_uses_dynamic_timeout_for_high_silence_ms():
    from core.config import Mmpg
    from services.stt import mmpg
    from services.stt.mmpg import MmpgManager

    # Verify recommended_drain_timeout() across config shapes and fallback paths.
    default_mgr = MmpgManager(lambda: None, str)
    assert default_mgr.recommended_drain_timeout() == 6.0

    high_model_mgr = MmpgManager(lambda: Mmpg(**{**CONFIG, "silence_ms": 2999}), str)
    expected_2999 = 2.999 + mmpg.ROLE_DRAIN_PLAYBACK_GRACE_SECONDS + 2.0
    assert high_model_mgr.recommended_drain_timeout() == pytest.approx(expected_2999)
    assert high_model_mgr.recommended_drain_timeout() > 6.0

    high_cfg = {**CONFIG, "silence_ms": 2500}

    async def check_high_silence_drain():
        cloud = CloudSocket()
        controls: list[dict] = []
        finals: list[str] = []
        sent_bytes = [0]
        done_pushed = [False]
        queued_pcm = np.ones(16_000 // 2, dtype=np.float32).tobytes()  # 0.5s spoken PCM
        required_bytes = len(queued_pcm) + 2500 * (mmpg.PCM_BYTES_PER_SECOND // 1000)

        async def connector(_):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            if isinstance(payload, bytes):
                sent_bytes[0] += len(payload)
                if not done_pushed[0] and sent_bytes[0] >= required_bytes:
                    done_pushed[0] = True
                    cloud.push("user.audio.transcription.done", conv_id="slow-turn", transcript="长静音末尾补全")
            else:
                controls.append(payload)

        mgr = MmpgManager(lambda: high_cfg, lambda text, _: finals.append(text), connector=connector)
        owner = object()
        assert mgr.attach(owner, send)
        mgr.start(owner, "start", [1])
        try:
            await wait_until(lambda: any(msg["type"] == "rtc_join" for msg in controls))
            gen = mgr.status()["generation"]
            await mgr.rtc_event(owner, {"type": "rtc_ready", "generation": gen})
            await wait_until(lambda: mgr.status()["state"] == "listening")

            assert mgr.recommended_drain_timeout() == max(
                6.0, 2.5 + mmpg.ROLE_DRAIN_PLAYBACK_GRACE_SECONDS + 2.0,
            )
            assert mgr.offer_audio(queued_pcm)
            cloud.push("user.conversation.created", conv_id="slow-turn")
            await wait_until(lambda: "slow-turn" in mgr._current.open_conversations)

            started = time.monotonic()
            # Default timeout=None must use recommended_drain_timeout() (>=6.0s) instead of old 3.0s.
            assert await mgr.drain_transcripts() is True
            elapsed = time.monotonic() - started
            assert elapsed > 3.0, f"expected real >3s drain with silence_ms=2500, got {elapsed:.2f}s"
            assert finals == ["长静音末尾补全"]
        finally:
            await mgr.detach(owner)

    asyncio.run(check_high_silence_drain())


def test_drain_dual_input_enforces_timeout_floor_and_preserves_reconnecting_tail_on_stop(
    monkeypatch, caplog,
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from api.assist import candidate_mmpg, mmpg_integration, pipeline
    from core import config
    from core.session import get_session, reset_session

    low_cfg = config.AppConfig(assist_interviewer_asr_drain_timeout_sec=1.0)
    monkeypatch.setattr(mmpg_integration, "get_config", lambda: low_cfg)
    monkeypatch.setattr(pipeline, "freeze_interview_inputs", Mock())
    monkeypatch.setattr(candidate_mmpg, "freeze_input", lambda: asyncio.sleep(0))

    async def check():
        session = reset_session()
        archive = Mock()
        session.archive = archive
        session.is_recording = True
        session.is_paused = False

        seen_timeouts: dict[str, float] = {}

        def fake_transport(role, state, drain_ok, recommended):
            async def drain_transcripts(*, timeout=None):
                seen_timeouts[role] = timeout
                return drain_ok

            return SimpleNamespace(
                active=True,
                single_mic_id=None,
                status=lambda: {"state": state, "mode": "start"},
                recommended_drain_timeout=lambda: recommended,
                drain_transcripts=drain_transcripts,
            )

        interviewer = fake_transport("interviewer", "listening", True, 6.24)
        candidate = fake_transport("candidate", "reconnecting", False, 6.0)
        monkeypatch.setattr(mmpg_integration, "manager", interviewer)
        monkeypatch.setattr(candidate_mmpg, "manager", candidate)
        monkeypatch.setattr(mmpg_integration, "_turns", Mock())

        logger = logging.getLogger("mmpg.diagnostics")
        attach_handler = not logger.propagate
        if attach_handler:
            logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING", logger="mmpg.diagnostics"):
                await mmpg_integration.drain_dual_input("stopped")
        finally:
            if attach_handler:
                logger.removeHandler(caplog.handler)

        assert seen_timeouts["interviewer"] == pytest.approx(6.24)
        assert seen_timeouts["candidate"] == pytest.approx(6.0)
        archive.seal_capture.assert_called_once()
        archive.stt_timeout.assert_called_once_with("candidate")
        mmpg_integration._turns.finish_role.assert_called_once()
        assert "ASR tail incomplete role=candidate reason=state=reconnecting" in caplog.text

        # Listening transport timing out logs reason=drain_timeout (distinct from non-listening state).
        caplog.clear()
        archive.reset_mock()
        interviewer_timeout = fake_transport("interviewer", "listening", False, 6.0)
        monkeypatch.setattr(mmpg_integration, "manager", interviewer_timeout)
        monkeypatch.setattr(candidate_mmpg, "manager", SimpleNamespace(active=False, status=lambda: {"state": "idle"}))
        if attach_handler:
            logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING", logger="mmpg.diagnostics"):
                await mmpg_integration.drain_dual_input("stopped")
        finally:
            if attach_handler:
                logger.removeHandler(caplog.handler)
        assert "ASR tail incomplete role=interviewer reason=drain_timeout" in caplog.text
        archive.stt_timeout.assert_called_once_with("interviewer")

        # Non-listening transport that recovers to "listening" mid-gather must still log its
        # initial non-listening state rather than misleadingly logging drain_timeout.
        caplog.clear()
        archive.reset_mock()
        cand_state = {"state": "reconnecting"}

        async def slow_interviewer_drain(*, timeout=None):
            cand_state["state"] = "listening"
            return True

        async def fast_candidate_fail(*, timeout=None):
            return False

        monkeypatch.setattr(
            mmpg_integration,
            "manager",
            SimpleNamespace(
                active=True,
                single_mic_id=None,
                status=lambda: {"state": "listening", "mode": "start"},
                recommended_drain_timeout=lambda: 6.0,
                drain_transcripts=slow_interviewer_drain,
            ),
        )
        monkeypatch.setattr(
            candidate_mmpg,
            "manager",
            SimpleNamespace(
                active=True,
                single_mic_id=None,
                status=lambda: {"state": cand_state["state"], "mode": "start"},
                recommended_drain_timeout=lambda: 6.0,
                drain_transcripts=fast_candidate_fail,
            ),
        )
        if attach_handler:
            logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING", logger="mmpg.diagnostics"):
                await mmpg_integration.drain_dual_input("stopped")
        finally:
            if attach_handler:
                logger.removeHandler(caplog.handler)
        assert "ASR tail incomplete role=candidate reason=state=reconnecting" in caplog.text
        archive.stt_timeout.assert_called_once_with("candidate")
        reset_session()

    asyncio.run(check())


def test_api_pause_drains_dual_channels_concurrently_without_sealing_archive_and_allows_resume(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from api.assist import candidate_mmpg, mmpg_integration, pipeline, routes
    from core import config
    from core.session import get_session, reset_session

    cfg = config.AppConfig(
        stt_provider="doubao_mmpg",
        candidate_stt_provider="doubao_mmpg",
        candidate_asr_enabled=True,
        mmpg=config.Mmpg(**CONFIG),
        assist_interviewer_asr_drain_timeout_sec=1.0,
    )
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "get_config", lambda: cfg)
    monkeypatch.setattr(mmpg_integration, "_session_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_session_settings", cfg.mmpg)
    monkeypatch.setattr(mmpg_integration, "_test_settings", None)
    monkeypatch.setattr(pipeline, "_interviewer_config", cfg)
    monkeypatch.setattr(pipeline, "_interviewer_streaming", True)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)
    monkeypatch.setattr(pipeline, "_start_candidate_capture", Mock())
    monkeypatch.setattr(pipeline, "broadcast", Mock())

    capture_events: list[tuple[str, str]] = []
    fake_main_capture = SimpleNamespace(
        is_running=True,
        start=lambda dev, **kw: capture_events.append(("start-main", f"{dev}:{kw.get('owner')}")),
        stop=lambda **kw: capture_events.append(("stop-main", str(kw.get("owner")))),
    )
    fake_cand_capture = SimpleNamespace(
        is_running=True,
        start=lambda dev, **kw: capture_events.append(("start-cand", f"{dev}:{kw.get('owner')}")),
        stop=lambda **kw: capture_events.append(("stop-cand", str(kw.get("owner")))),
    )
    monkeypatch.setattr(pipeline, "audio_capture", fake_main_capture)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", fake_cand_capture)
    monkeypatch.setattr(mmpg_integration, "audio_capture", fake_main_capture)

    async def check():
        monkeypatch.setattr(mmpg_integration, "http_lifecycle_lock", asyncio.Lock())
        session = reset_session()
        archive = Mock()
        session.archive = archive
        session.is_recording = True
        session.is_paused = False
        session.last_device_id = 10
        session.last_candidate_mic_device_id = 20
        pipeline._pause_event.clear()

        both_draining = asyncio.Event()
        draining: set[str] = set()
        stopped: list[str] = []
        turns_mock = Mock()
        monkeypatch.setattr(mmpg_integration, "_turns", turns_mock)

        def make_transport(role, drain_ok):
            obj = SimpleNamespace(
                active=True,
                single_mic_id=None,
                status=lambda: {"state": "listening", "mode": "start"},
                recommended_drain_timeout=lambda: 6.0,
                offer_audio=Mock(),
            )

            async def drain_transcripts(*, timeout=None):
                assert timeout >= 6.0
                # Both physical inputs must be frozen before draining starts.
                assert ("stop-main", "assist") in capture_events
                assert ("stop-cand", "assist-candidate-mmpg") in capture_events
                assert not stopped
                draining.add(role)
                if len(draining) == 2:
                    both_draining.set()
                await asyncio.wait_for(both_draining.wait(), timeout=1.0)
                return drain_ok

            async def stop(reason):
                assert reason == "paused"
                # _turns.finish_role() must have already run during drain before stop("paused").
                assert turns_mock.finish_role.call_count == 1
                stopped.append(role)
                obj.active = False

            obj.drain_transcripts = drain_transcripts
            obj.stop = stop
            return obj

        interviewer = make_transport("interviewer", True)
        candidate = make_transport("candidate", False)
        monkeypatch.setattr(mmpg_integration, "manager", interviewer)
        monkeypatch.setattr(candidate_mmpg, "manager", candidate)

        assert await routes.api_pause() == {"ok": True}
        assert draining == {"interviewer", "candidate"}
        assert set(stopped) == {"interviewer", "candidate"}
        turns_mock.finish_role.assert_called_once()
        # CRITICAL: pausing must NEVER seal archive or mark stt_timeout.
        archive.seal_capture.assert_not_called()
        archive.stt_timeout.assert_not_called()
        # Even though freeze_interview_inputs() already set _pause_event, api_pause must
        # transition session.is_paused=True so resume is accepted afterward.
        assert session.is_paused is True
        assert pipeline.is_paused() is True

        # Verify resume succeeds cleanly after dual-channel pause drain.
        await mmpg_integration.prepare("resume", [10], 20)
        await mmpg_integration.start_capture("resume", [10], 20)
        assert session.is_paused is False
        assert pipeline.is_paused() is False
        assert ("start-main", "10:assist") in capture_events

        # Edge case: is_streaming_interview() is True while mmpg.manager.active is False
        # must still pause cleanly.
        interviewer.active = False
        candidate.active = False
        assert await routes.api_pause() == {"ok": True}
        assert session.is_paused is True
        assert pipeline.is_paused() is True
        reset_session()
        pipeline._pause_event.clear()

    asyncio.run(check())


def test_api_pause_dual_channel_real_managers_and_archive_resume_then_stop(monkeypatch):
    from api.assist import candidate_mmpg, mmpg_integration, pipeline, routes
    from core import config
    from core.session import reset_session
    from services.interview_archive.audio import CapturedAudio
    from services.interview_archive.manager import InterviewArchive
    from services.stt import mmpg
    from services.stt.mmpg import MmpgManager

    monkeypatch.setattr(mmpg, "ROLE_DRAIN_PLAYBACK_GRACE_SECONDS", 0.05)
    cfg = config.AppConfig(
        stt_provider="doubao_mmpg",
        candidate_stt_provider="doubao_mmpg",
        candidate_asr_enabled=True,
        mmpg=config.Mmpg(**{**CONFIG, "silence_ms": 500}),
        auto_detect=False,
        assist_stop_answer_wait_sec=0,
    )
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "get_config", lambda: cfg)
    monkeypatch.setattr(mmpg_integration, "_session_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_session_settings", cfg.mmpg)
    monkeypatch.setattr(mmpg_integration, "_test_settings", None)
    monkeypatch.setattr(pipeline, "_interviewer_config", cfg)
    monkeypatch.setattr(pipeline, "_interviewer_streaming", True)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)
    monkeypatch.setattr(pipeline, "broadcast", lambda *_a, **_kw: None)
    monkeypatch.setattr(candidate_mmpg, "broadcast", lambda *_a, **_kw: None)
    monkeypatch.setattr(mmpg_integration, "broadcast", lambda *_a, **_kw: None)

    class _FakeCapture:
        def __init__(self):
            self.running = False
            self.callback = None

        def start(self, _dev, *, on_audio=None, owner=None, **_kw):
            self.running = True
            self.callback = on_audio

        def stop(self, *, owner=None, clear_queue=True):
            self.running = False
            self.callback = None

    ic = _FakeCapture()
    cc = _FakeCapture()
    monkeypatch.setattr(pipeline, "audio_capture", ic)
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", cc)
    monkeypatch.setattr(mmpg_integration, "audio_capture", ic)

    async def check():
        monkeypatch.setattr(mmpg_integration, "http_lifecycle_lock", asyncio.Lock())
        saved: list[dict] = []
        session = reset_session()
        archive = InterviewArchive(99, candidate_enabled=True, save=saved.append)
        try:
            session.archive = archive
            session.begin_transcript_capture()
            session.is_recording = True
            session.is_paused = False
            session.last_device_id = 10
            session.last_candidate_mic_device_id = 20
            pipeline._pause_event.clear()
            monkeypatch.setattr(mmpg_integration, "_capture_epoch", session.transcript_epoch)
            monkeypatch.setattr(candidate_mmpg, "_capture_epoch", session.transcript_epoch)
            monkeypatch.setattr(candidate_mmpg, "_pending", {})

            ic_cloud, cc_cloud = CloudSocket(), CloudSocket()

            async def iconn(_):
                ic_cloud.push("session.created")
                return ic_cloud

            async def cconn(_):
                cc_cloud.push("session.created")
                return cc_cloud

            imgr = MmpgManager(
                mmpg_integration._settings,
                mmpg_integration._final_transcript,
                on_turn=mmpg_integration._turn_event,
                on_capture_stop=mmpg_integration.stop_capture,
                connector=iconn,
            )
            cmgr = MmpgManager(
                lambda: cfg.mmpg,
                candidate_mmpg._on_transcript,
                on_turn=candidate_mmpg._on_turn,
                on_capture_stop=candidate_mmpg._stop_capture,
                connector=cconn,
            )
            monkeypatch.setattr(mmpg_integration, "manager", imgr)
            monkeypatch.setattr(candidate_mmpg, "manager", cmgr)
            monkeypatch.setattr(mmpg_integration, "_turns", mmpg_integration._new_turns())

            im_ctrl, cm_ctrl = [], []
            im_done_pushed = [False]
            cm_done_pushed = [False]

            async def isend(payload):
                if isinstance(payload, bytes):
                    if not ic.running and not any(payload) and not im_done_pushed[0]:
                        im_done_pushed[0] = True
                        ic_cloud.push("user.audio.transcription.done", conv_id="q1", transcript="暂停前问题？")
                else:
                    im_ctrl.append(payload)

            async def csend(payload):
                if isinstance(payload, bytes):
                    if not cc.running and not any(payload) and not cm_done_pushed[0]:
                        cm_done_pushed[0] = True
                        cc_cloud.push("user.audio.transcription.done", conv_id="a1", transcript="暂停前回答")
                else:
                    cm_ctrl.append(payload)

            o1, o2 = object(), object()
            imgr.attach(o1, isend)
            cmgr.attach(o2, csend)
            imgr.start(o1, "start", [10], 20)
            cmgr.start(o2, "start", [20])
            await wait_until(
                lambda: any(m["type"] == "rtc_join" for m in im_ctrl)
                and any(m["type"] == "rtc_join" for m in cm_ctrl)
            )
            await imgr.rtc_event(o1, {"type": "rtc_ready", "generation": imgr.status()["generation"]})
            await cmgr.rtc_event(o2, {"type": "rtc_ready", "generation": cmgr.status()["generation"]})
            await wait_until(
                lambda: imgr.status()["state"] == "listening" and cmgr.status()["state"] == "listening"
            )

            ic.running = True
            cc.running = True
            anchor = time.monotonic() - 1.0
            archive.capture(CapturedAudio(np.full(8000, 0.05, dtype=np.float32), "loop1", 0, anchor), "interviewer")
            archive.capture(CapturedAudio(np.full(8000, 0.05, dtype=np.float32), "mic1", 0, anchor), "candidate")
            assert imgr.offer_audio(np.full(640, 0.05, dtype=np.float32))
            assert cmgr.offer_audio(np.full(640, 0.05, dtype=np.float32))
            ic_cloud.push("user.conversation.created", conv_id="q1")
            cc_cloud.push("user.conversation.created", conv_id="a1")
            await wait_until(
                lambda: "q1" in imgr._current.open_conversations and "a1" in cmgr._current.open_conversations
            )

            # Dual-channel pause must drain both real cloud tails without sealing the real archive.
            assert await routes.api_pause() == {"ok": True}
            assert session.is_paused is True
            assert archive.audio._sealed is False
            assert not archive._stt_timeouts
            assert "暂停前问题？" in session.transcription_history
            assert "暂停前回答" in session.candidate_transcription_history

            # Resume into the same unsealed archive and capture another round cleanly.
            await mmpg_integration.prepare("resume", [10], 20)
            await mmpg_integration.start_capture("resume", [10], 20)
            assert session.is_paused is False
            assert ic.running is True
            anchor2 = time.monotonic()
            archive.capture(CapturedAudio(np.full(8000, 0.05, dtype=np.float32), "loop2", 0, anchor2), "interviewer")
            archive.transcript("interviewer", "恢复后问题？", "q2", anchor2, audio_end=anchor2 + 0.5)
            assert archive.stop()

            assert len(saved) == 2
            assert saved[0]["question_text"] == "暂停前问题？"
            assert saved[0]["candidate_answer_text"] == "暂停前回答"
            assert saved[1]["question_text"] == "恢复后问题？"
            assert saved[1]["audio_status"]["interviewer"]["state"] == "ready"
        finally:
            archive.stop()
            reset_session()
            pipeline._pause_event.clear()

    asyncio.run(check())
