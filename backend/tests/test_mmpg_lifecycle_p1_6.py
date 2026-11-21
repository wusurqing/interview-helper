"""Unit tests for P1-6: MMPG error preservation, preflight mutex, and deadlock-free lifecycle epochs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from api.assist import candidate_mmpg, exam_test, mmpg_integration, pipeline, routes, sound_test
from api.assist.single_mic import SingleMicSession
from core import config
from core.session import Session, get_session, reset_session
from services.stt.mmpg import MmpgError, MmpgManager


_VALID_MMPG_KWARGS = {
    "volc_access_key": "test-ak",
    "volc_secret_key": "test-sk",
    "rtc_app_id": "a" * 24,
    "rtc_app_key": "test-rtc-key",
    "rtc_room_id": "room-1",
    "rtc_user_id": "user-1",
    "rtc_bot_id": "bot-1",
    "doubao_api_key": "test-asr-key",
}


class _FakeCloudSocket:
    def __init__(self):
        self._incoming: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    def push(self, event_type: str, **payload):
        import json

        self._incoming.put_nowait(json.dumps({"type": event_type, **payload}))

    async def send(self, _payload: str):
        return None

    async def recv(self) -> str:
        return await self._incoming.get()

    async def close(self):
        self.closed = True


def _mmpg_cfg(**overrides) -> config.AppConfig:
    return config.AppConfig(
        stt_provider="doubao_mmpg",
        candidate_stt_provider="doubao_mmpg",
        candidate_asr_enabled=True,
        mmpg=config.Mmpg(**_VALID_MMPG_KWARGS),
        assist_stop_answer_wait_sec=0,
        **overrides,
    )


def test_start_capture_preserves_explicit_mmpg_error_codes(monkeypatch: pytest.MonkeyPatch):
    cfg = _mmpg_cfg()
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_turns", None)
    monkeypatch.setattr(mmpg_integration, "_capture_epoch", None)
    monkeypatch.setattr(mmpg_integration, "_session_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_session_settings", cfg.mmpg)
    monkeypatch.setattr(mmpg_integration, "_prepared_lifecycle_epoch", None)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)
    monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: True)

    async def exercise():
        session = reset_session()
        single_mic = SingleMicSession("sm-1", device_id=1)
        session.single_mic = single_mic
        session.is_recording = True
        session.is_paused = True

        # Wrong device id on active single-mic must preserve "interview_active", not "device_unavailable".
        with pytest.raises(MmpgError) as wrong_dev:
            await mmpg_integration.start_capture("resume", [99], None, single_mic_id="sm-1")
        assert wrong_dev.value.code == "interview_active"

        # Mode "start" while paused single-mic exists must preserve "cannot_resume".
        with pytest.raises(MmpgError) as bad_mode:
            await mmpg_integration.start_capture("start", [1], None, single_mic_id="sm-1")
        assert bad_mode.value.code == "cannot_resume"

        # Different single_mic_id while recording must preserve "interview_active".
        with pytest.raises(MmpgError) as other_sm:
            await mmpg_integration.start_capture("start", [1], None, single_mic_id="sm-other")
        assert other_sm.value.code == "interview_active"

        reset_session()

    asyncio.run(exercise())


def test_start_capture_aborts_when_lifecycle_epoch_advances_after_prepare(monkeypatch: pytest.MonkeyPatch):
    cfg = _mmpg_cfg()
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_turns", "sentinel-turns")
    monkeypatch.setattr(mmpg_integration, "_capture_epoch", 77)
    monkeypatch.setattr(pipeline, "start_nonblocking", Mock(side_effect=AssertionError("must not start")))
    monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: False)

    async def exercise():
        reset_session()
        await mmpg_integration.prepare("start", [0], None)
        # Simulate api_stop / api_start / preflight advancing lifecycle after prepare released http_lifecycle_lock.
        mmpg_integration.bump_lifecycle_epoch()

        with pytest.raises(MmpgError) as exc:
            await mmpg_integration.start_capture("start", [0], None)
        assert exc.value.code == "interview_active"
        # Neither _turns nor _capture_epoch should have been clobbered.
        assert mmpg_integration._turns == "sentinel-turns"
        assert mmpg_integration._capture_epoch == 77
        pipeline.start_nonblocking.assert_not_called()

    asyncio.run(exercise())


def test_start_capture_and_api_stop_do_not_deadlock_under_http_lifecycle_lock(monkeypatch: pytest.MonkeyPatch):
    """Regression test for section 4.1 deadlock red line:
    api_stop holds http_lifecycle_lock while awaiting manager.stop(), while
    MmpgManager._run_transport shields and awaits start_capture().
    """
    cfg = _mmpg_cfg()
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_turns", None)
    monkeypatch.setattr(mmpg_integration, "_capture_epoch", None)
    monkeypatch.setattr(pipeline, "review_integration", Mock())
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)
    monkeypatch.setattr(pipeline, "stop_interview_loop", Mock())

    worker_entered = asyncio.Event()
    release_worker = asyncio.Event()
    paused_calls: list[str] = []

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        if fn is pipeline.start_nonblocking:
            worker_entered.set()
            await release_worker.wait()
            session = get_session()
            session.is_recording = True
            session.is_paused = False
            monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: True)
            return None
        if fn is pipeline.pause_interview:
            paused_calls.append("paused")
            get_session().is_paused = True
            return None
        return fn(*args, **kwargs)

    monkeypatch.setattr(mmpg_integration, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr(routes, "run_in_threadpool", fake_run_in_threadpool)

    async def exercise():
        reset_session()
        cloud = _FakeCloudSocket()
        controls: list[dict] = []

        async def connect(_cfg):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            if isinstance(payload, dict):
                controls.append(payload)

        mgr = MmpgManager(
            mmpg_integration._settings,
            lambda *_args: None,
            connector=connect,
            on_capture_start=mmpg_integration.start_capture,
            on_capture_stop=mmpg_integration.stop_capture,
            retry_delays=(),
        )
        monkeypatch.setattr(mmpg_integration, "manager", mgr)
        monkeypatch.setattr(candidate_mmpg, "stop", AsyncMock())

        owner = object()
        mgr.attach(owner, send)
        mgr.start(owner, "start", [1], preflight=mmpg_integration.prepare)
        try:
            for _ in range(100):
                if any(msg.get("type") == "rtc_join" for msg in controls):
                    break
                await asyncio.sleep(0.01)
            join = next(msg for msg in controls if msg.get("type") == "rtc_join")
            await mgr.rtc_event(owner, {"type": "rtc_ready", "generation": join["generation"]})
            await asyncio.wait_for(worker_entered.wait(), timeout=2.0)

            # Start api_stop while start_capture is suspended inside shielded worker!
            stop_task = asyncio.create_task(routes.api_stop())
            await asyncio.sleep(0.02)
            # Release start_capture's worker; api_stop holds http_lifecycle_lock and awaits mgr.stop().
            release_worker.set()
            result = await asyncio.wait_for(stop_task, timeout=2.0)
            assert result == {"ok": True}
            assert not mgr.active
            # Because lifecycle epoch bumped before start_capture committed, _turns was NOT initialized.
            assert mmpg_integration._turns is None
        finally:
            await mgr.detach(owner)
            reset_session()

    asyncio.run(exercise())


def test_candidate_mmpg_rejects_stale_epoch_and_rolls_back_mid_start_race(monkeypatch: pytest.MonkeyPatch):
    cfg = _mmpg_cfg()
    session = Session(is_recording=True, is_paused=False, last_device_id=10, last_candidate_mic_device_id=20)
    monkeypatch.setattr(candidate_mmpg.pipeline, "get_capture_config", lambda: cfg)
    monkeypatch.setattr(candidate_mmpg, "get_session", lambda: session)
    monkeypatch.setattr(candidate_mmpg, "_capture_epoch", None)
    monkeypatch.setattr(candidate_mmpg, "_pending", {})
    monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: True)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)

    mic_events: list[str] = []

    class FakeMic:
        def start(self, device_id, **kwargs):
            mic_events.append(f"start:{device_id}")
            # Simulate concurrent api_pause / api_stop advancing lifecycle epoch while mic opens.
            mmpg_integration.bump_lifecycle_epoch()

        def stop(self, *, owner=None):
            mic_events.append(f"stop:{owner}")

    monkeypatch.setattr(pipeline, "_candidate_audio_capture", FakeMic())

    async def exercise():
        await candidate_mmpg.prepare("start", [20], None)
        with pytest.raises(MmpgError) as exc:
            await candidate_mmpg._start_capture("start", [20], None)
        assert exc.value.code == "candidate_interview_inactive"
        # Capture opened then rolled back cleanly; _capture_epoch remained unset.
        assert mic_events == ["start:20", "stop:assist-candidate-mmpg"]
        assert candidate_mmpg._capture_epoch is None

        # Pre-start stale epoch: fails before touching mic at all.
        mic_events.clear()
        await candidate_mmpg.prepare("start", [20], None)
        mmpg_integration.bump_lifecycle_epoch()
        with pytest.raises(MmpgError) as exc2:
            await candidate_mmpg._start_capture("start", [20], None)
        assert exc2.value.code == "candidate_interview_inactive"
        assert mic_events == []

    asyncio.run(exercise())


def test_preflight_routes_serialize_with_http_lifecycle_lock_and_block_concurrent_starts(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = config.AppConfig(stt_provider="doubao", doubao_stt_api_key="test-key")
    monkeypatch.setattr(routes, "get_config", lambda: cfg)
    monkeypatch.setattr(routes, "get_session", lambda: SimpleNamespace(is_recording=False, is_paused=False, single_mic=None))
    monkeypatch.setattr(mmpg_integration, "get_config", lambda: cfg)
    monkeypatch.setattr(mmpg_integration, "get_session", lambda: SimpleNamespace(is_recording=False, is_paused=False, single_mic=None))
    monkeypatch.setattr(exam_test, "is_exam_preflight_running", lambda: False)

    replay_entered = asyncio.Event()
    release_replay = asyncio.Event()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        if fn is sound_test.replay_preflight_capture_stt:
            replay_entered.set()
            await release_replay.wait()
            return {"device_id": args[0], "success_count": 1, "failure_count": 0}
        return fn(*args, **kwargs)

    monkeypatch.setattr(routes, "run_in_threadpool", fake_run_in_threadpool)

    async def exercise():
        sound_test.end_preflight_replay()
        epoch_before = mmpg_integration.current_lifecycle_epoch()

        replay_task = asyncio.create_task(routes.api_preflight_replay({"device_id": 7, "repeats": 1}))
        await asyncio.wait_for(replay_entered.wait(), timeout=2.0)

        assert sound_test.is_preflight_replay_running() is True
        assert mmpg_integration.current_lifecycle_epoch() > epoch_before

        # Concurrent replay, preflight/run, exam-preflight/run, api_start, and mmpg.prepare must all reject.
        with pytest.raises(HTTPException) as dup_replay:
            await routes.api_preflight_replay({"device_id": 7, "repeats": 1})
        assert dup_replay.value.status_code == 409

        with pytest.raises(HTTPException) as dup_preflight:
            await routes.api_preflight_run({"device_id": 7})
        assert dup_preflight.value.status_code == 409

        with pytest.raises(HTTPException) as dup_exam:
            await routes.api_exam_preflight_run()
        assert dup_exam.value.status_code == 409

        with pytest.raises(HTTPException) as blocked_start:
            await routes.api_start({"device_id": 7})
        assert blocked_start.value.status_code == 409

        with pytest.raises(MmpgError) as blocked_mmpg:
            await mmpg_integration.prepare("test", [7], None)
        assert blocked_mmpg.value.code == "capture_busy"

        # http_lifecycle_lock itself must NOT be held during long replay execution, so api_stop is not blocked.
        assert not mmpg_integration.http_lifecycle_lock.locked()

        release_replay.set()
        res = await asyncio.wait_for(replay_task, timeout=2.0)
        assert res["success_count"] == 1
        assert sound_test.is_preflight_replay_running() is False

    asyncio.run(exercise())


def test_single_mic_start_capture_and_api_pause_race_pauses_session_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
):
    """If api_pause races with single-mic start_capture before session.single_mic is set,
    stale rollback must pause the newly started single-mic capture so api_pause does not
    return 200 OK while leaving recording unpaused and capture open.
    """
    cfg = _mmpg_cfg()
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(mmpg_integration, "_turns", None)
    monkeypatch.setattr(mmpg_integration, "_capture_epoch", None)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)

    worker_entered = asyncio.Event()
    release_worker = asyncio.Event()
    pause_calls: list[str] = []

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        if fn is pipeline.start_nonblocking:
            worker_entered.set()
            await release_worker.wait()
            session = get_session()
            session.single_mic = SingleMicSession("sm-race", device_id=1)
            session.is_recording = True
            session.is_paused = False
            monkeypatch.setattr(pipeline, "is_streaming_interview", lambda: True)
            monkeypatch.setattr(pipeline, "is_paused", lambda: session.is_paused)
            return None
        if fn is pipeline.pause_interview:
            pause_calls.append("paused")
            session = get_session()
            session.is_paused = True
            if session.single_mic is not None:
                session.single_mic.pause()
            return None
        return fn(*args, **kwargs)

    monkeypatch.setattr(mmpg_integration, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr(routes, "run_in_threadpool", fake_run_in_threadpool)

    async def exercise():
        reset_session()
        cloud = _FakeCloudSocket()
        controls: list[dict] = []

        async def connect(_cfg):
            cloud.push("session.created")
            return cloud

        async def send(payload):
            if isinstance(payload, dict):
                controls.append(payload)

        mgr = MmpgManager(
            mmpg_integration._settings,
            lambda *_args: None,
            connector=connect,
            on_capture_start=mmpg_integration.start_capture,
            on_capture_stop=mmpg_integration.stop_capture,
            retry_delays=(),
        )
        monkeypatch.setattr(mmpg_integration, "manager", mgr)
        monkeypatch.setattr(candidate_mmpg, "stop", AsyncMock())

        owner = object()
        mgr.attach(owner, send)
        mgr.start(owner, "start", [1], preflight=mmpg_integration.prepare, single_mic_id="sm-race")
        try:
            for _ in range(100):
                if any(msg.get("type") == "rtc_join" for msg in controls):
                    break
                await asyncio.sleep(0.01)
            join = next(msg for msg in controls if msg.get("type") == "rtc_join")
            await mgr.rtc_event(owner, {"type": "rtc_ready", "generation": join["generation"]})
            await asyncio.wait_for(worker_entered.wait(), timeout=2.0)

            pause_task = asyncio.create_task(routes.api_pause())
            await asyncio.sleep(0.02)
            release_worker.set()
            result = await asyncio.wait_for(pause_task, timeout=2.0)

            assert result == {"ok": True}
            assert not mgr.active
            assert pause_calls == ["paused"]
            assert get_session().is_paused is True
            assert mmpg_integration._turns is None
        finally:
            await mgr.detach(owner)
            reset_session()

    asyncio.run(exercise())
