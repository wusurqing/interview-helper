"""Connect the MMPG transport to the existing interview lifecycle."""

import asyncio
from contextlib import suppress

from fastapi.concurrency import run_in_threadpool

from api.realtime.ws import broadcast
from core.config import get_config
from core.logger import get_logger
from core.session import get_session
from services.audio import AudioBusyError, audio_capture
from services.stt.mmpg import MmpgError, MmpgManager
from . import pipeline
from .mmpg_turns import MmpgTurns

PROVIDER = "doubao_mmpg"
http_lifecycle_lock = asyncio.Lock()
_session_settings = None
_test_settings = None
_turns = None
_log = get_logger("mmpg.diagnostics")


def _enabled():
    session = get_session()
    return pipeline.is_streaming_interview() and not session.is_paused and get_config().auto_detect


def clear_turns():
    if _turns is not None:
        _turns.clear(cancel_running=True)


def answer_phase():
    return _turns.phase if _turns is not None else "idle"


def _turn_event(kind, trace):
    if _turns is None:
        return
    if kind == "reset":
        clear_turns()
    elif kind == "started":
        _turns.started(trace)
    elif kind == "cancelled":
        _turns.cancelled(trace)


def _final_transcript(text, trace):
    pipeline.publish_streaming_asr_turn(text)
    if _turns is not None:
        _turns.final(trace, text)


def _new_turns():
    loop = asyncio.get_running_loop()

    def submit(request):
        request.loop = loop
        request.on_started = lambda at: turns.request_started(request, at)
        request.on_finished = lambda: loop.call_soon_threadsafe(turns.request_finished, request)
        return pipeline.submit_mmpg_question(request)

    def log_event(event, **fields):
        _log.info("event=%s %s", event, " ".join(f"{name}={value}" for name, value in fields.items()))

    turns = MmpgTurns(
        submit=submit, cancel=pipeline.cancel_mmpg_answer,
        publish=lambda phase: broadcast({"type": "mmpg_answer_phase", "phase": phase}),
        enabled=_enabled, clock=loop.time, call_later=loop.call_later, log=log_event,
        cancel_on_clear=lambda request: pipeline.cancel_mmpg_answer(request, reason="stopped"),
        answer_delay_ms=_session_settings.answer_delay_ms,
        followup_window_ms=_session_settings.followup_window_ms,
    )
    return turns


def _settings():
    return _test_settings if _test_settings is not None else _session_settings


async def prepare(mode, device_ids, candidate_mic_id):
    """Validate the operation and capture its configuration before connecting."""
    from .exam_test import is_exam_preflight_running
    from .sound_test import get_preflight_status

    global _session_settings, _test_settings
    async with http_lifecycle_lock:
        cfg = get_config()
        session = get_session()
        if is_exam_preflight_running() or get_preflight_status()["running"]:
            raise MmpgError("capture_busy")
        if not device_ids or not pipeline._is_valid_device_id(device_ids[0]):
            raise MmpgError("device_required")
        if mode == "resume":
            if not pipeline.is_streaming_interview() or not session.is_paused:
                raise MmpgError("cannot_resume")
            _test_settings = None
            current = cfg.mmpg
            if (
                _session_settings.token_mode == current.token_mode == "manual"
                and _session_settings.rtc_app_id == current.rtc_app_id
                and _session_settings.rtc_room_id == current.rtc_room_id
                and _session_settings.rtc_user_id == current.rtc_user_id
            ):
                _session_settings = _session_settings.model_copy(update={
                    "rtc_token": current.rtc_token,
                    "rtc_token_expires_at": current.rtc_token_expires_at,
                })
        else:
            if session.is_recording:
                raise MmpgError("interview_active")
            if cfg.stt_provider != PROVIDER or (mode != "test" and cfg.written_exam_mode):
                raise MmpgError("provider_not_selected")
            snapshot = cfg.mmpg.model_copy(deep=True)
            if mode == "test":
                _test_settings = snapshot
            else:
                _session_settings = snapshot
                _test_settings = None


async def start_capture(mode, device_ids, candidate_mic_id):
    global _turns
    if get_session().is_recording and not pipeline.is_streaming_interview():
        raise MmpgError("interview_active")
    device = device_ids[0]
    try:
        if mode == "test":
            worker = run_in_threadpool(
                audio_capture.start, device, on_audio=manager.offer_audio,
                owner="mmpg-test", queue_audio=False,
            )
        else:
            _turns = _new_turns()
            action = pipeline.unpause_interview if pipeline.is_streaming_interview() else pipeline.start_nonblocking
            worker = run_in_threadpool(action, device, candidate_mic_id, stream_audio=manager.offer_audio)
        task = asyncio.create_task(worker)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Wait for the device operation before the manager closes capture.
            with suppress(Exception):
                await task
            raise
    except AudioBusyError:
        raise MmpgError("capture_busy") from None
    except Exception:
        raise MmpgError("device_unavailable") from None


async def stop_capture(mode, reason):
    if mode == "test":
        await run_in_threadpool(audio_capture.stop, owner="mmpg-test")
    elif pipeline.is_streaming_interview() and not pipeline.is_paused():
        await run_in_threadpool(pipeline.pause_interview)
    elif not get_session().is_recording:
        await run_in_threadpool(audio_capture.stop, owner="assist")


def _broadcast_state(status):
    if status.get("mode") != "test":
        broadcast({"type": "stt_status", **stt_status(status)})


def stt_status(status=None):
    state = dict(manager.status() if status is None else status)
    if state.get("mode") == "test":
        state = {"state": "idle", "error": None}
    phase = state.get("state", "idle")
    return {
        **state,
        "provider": PROVIDER,
        "model": PROVIDER,
        "loaded": phase == "listening",
        "loading": phase in ("connecting", "reconnecting"),
    }


manager = MmpgManager(
    _settings,
    _final_transcript,
    on_turn=_turn_event,
    on_state=_broadcast_state,
    on_capture_start=start_capture,
    on_capture_stop=stop_capture,
)
