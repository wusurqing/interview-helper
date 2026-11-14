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
