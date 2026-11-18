"""Connect the MMPG transport to the existing interview lifecycle."""

import asyncio
import time
from contextlib import suppress

from fastapi.concurrency import run_in_threadpool

from api.realtime.ws import broadcast
from core.config import get_config
from core.logger import get_logger
from core.session import conversation_lock, get_session
from services.audio import AudioBusyError, audio_capture
from services.stt.mmpg import MmpgError, MmpgManager
from . import candidate_mmpg, pipeline
from .mmpg_turns import MmpgTurns
from .single_mic import Role, SingleMicConflict, SingleMicSession

PROVIDER = "doubao_mmpg"
http_lifecycle_lock = asyncio.Lock()
_session_settings = None
_test_settings = None
_turns = None
_capture_epoch: int | None = None
_speech_starts: dict[str, tuple[float, str | None]] = {}
_log = get_logger("mmpg.diagnostics")


def _enabled():
    session = get_session()
    return pipeline.is_streaming_interview() and not session.is_paused and session.transcript_epoch == _capture_epoch


def clear_turns():
    if _turns is not None:
        _turns.clear(cancel_running=True)
    session = get_session()
    for _, anchor in _speech_starts.values():
        session.discard_unfinished_transcript(anchor)
    _speech_starts.clear()


def answer_phase():
    return _turns.phase if _turns is not None else "idle"


def _turn_event(kind, trace):
    if _turns is None:
        return
    if kind == "reset":
        clear_turns()
    elif kind == "started" and _enabled():
        if trace not in _speech_starts:
            started_at = time.monotonic()
            anchor = get_session().record_transcript(
                "interviewer", "", started_at=started_at, utterance_key=trace,
                is_final=False, epoch=_capture_epoch,
            )
            _speech_starts[trace] = started_at, anchor
        _turns.started(trace)
    elif kind == "cancelled":
        _, anchor = _speech_starts.pop(trace, (0.0, None))
        get_session().discard_unfinished_transcript(anchor)
        _turns.cancelled(trace)


def _final_transcript(text, trace):
    if _turns is not None and _enabled():
        _turns.final(trace, text)


def _record_final(text, trace, group_id):
    started_at, _ = _speech_starts.pop(trace, (time.monotonic(), None))
    return pipeline.publish_streaming_asr_turn(
        text, trace=trace, started_at=started_at, question_group=group_id, transcript_epoch=_capture_epoch,
    )


def _new_turns():
    loop = asyncio.get_running_loop()

    def submit(request):
        if not _enabled():
            return False
        request.loop = loop
        request.on_started = lambda at: turns.request_started(request, at)
        request.on_finished = lambda: loop.call_soon_threadsafe(turns.request_finished, request)
        return pipeline.submit_mmpg_question(request, transcript_epoch=_capture_epoch)

    def log_event(event, **fields):
        _log.info("event=%s %s", event, " ".join(f"{name}={value}" for name, value in fields.items()))

    turns = MmpgTurns(
        submit=submit, cancel=pipeline.cancel_mmpg_answer,
        publish=lambda phase: broadcast({"type": "mmpg_answer_phase", "phase": phase}),
        enabled=_enabled, clock=loop.time, call_later=loop.call_later, log=log_event,
        cancel_on_clear=lambda request: pipeline.cancel_mmpg_answer(request, reason="stopped"),
        answer_delay_ms=_session_settings.answer_delay_ms,
        followup_window_ms=_session_settings.followup_window_ms,
        record_final=_record_final,
    )
    return turns


def _settings():
    return _test_settings if _test_settings is not None else _session_settings


async def prepare(mode, device_ids, candidate_mic_id, *, single_mic_id=None):
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
        if mode in ("start", "resume") and single_mic_id is not None:
            if not cfg.candidate_asr_enabled:
                raise MmpgError("candidate_provider_not_selected")
            if cfg.candidate_stt_provider == PROVIDER and cfg.mmpg.token_mode == "manual":
                raise MmpgError("candidate_mmpg_auto_token_required")
            if await run_in_threadpool(pipeline._device_is_loopback, device_ids[0]):
                raise MmpgError("candidate_mic_required")
        if mode == "resume":
            single_mic = session.single_mic
            if (single_mic is not None or single_mic_id is not None) and (
                single_mic is None or single_mic.id != single_mic_id or single_mic.device_id != device_ids[0]
            ):
                raise MmpgError("cannot_resume")
            if not pipeline.is_streaming_interview() or not session.is_paused:
                raise MmpgError("cannot_resume")
            if single_mic is not None:
                try:
                    await run_in_threadpool(pipeline.validate_single_mic_resume, single_mic, cfg)
                except SingleMicConflict:
                    raise MmpgError("single_mic_settings_changed") from None
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


async def start_capture(mode, device_ids, candidate_mic_id, *, single_mic_id=None):
    global _turns, _capture_epoch
    if get_session().is_recording and not pipeline.is_streaming_interview():
        raise MmpgError("interview_active")
    device = device_ids[0]
    try:
        if mode == "test":
            worker = run_in_threadpool(
                audio_capture.start, device, on_audio=manager.offer_audio,
                owner="mmpg-test", queue_audio=False,
            )
        elif single_mic_id is not None:
            session = get_session()
            single_mic = session.single_mic
            if single_mic is not None and single_mic.id == single_mic_id:
                if not session.is_recording or single_mic.device_id != device:
                    raise MmpgError("interview_active")
                _capture_epoch = session.transcript_epoch
                if not session.is_paused:
                    return  # Automatic retry reuses the current session and microphone.
                if mode != "resume":
                    raise MmpgError("cannot_resume")
                worker = run_in_threadpool(pipeline.unpause_interview, stream_audio=manager.offer_audio)
            elif session.is_recording or mode != "start":
                raise MmpgError("interview_active")
            else:
                worker = run_in_threadpool(
                    pipeline.start_nonblocking, device, stream_audio=manager.offer_audio,
                    single_mic_id=single_mic_id,
                )
            clear_turns()
            _turns = _new_turns()
        else:
            clear_turns()
            _turns = _new_turns()
            action = pipeline.unpause_interview if pipeline.is_streaming_interview() else pipeline.start_nonblocking
            worker = run_in_threadpool(action, device, candidate_mic_id, stream_audio=manager.offer_audio)
        task = asyncio.create_task(worker)
        try:
            await asyncio.shield(task)
            if mode != "test":
                with conversation_lock:
                    session = get_session()
                    _capture_epoch = session.transcript_epoch
                if single_mic_id is not None and mode == "start":
                    watch_single_mic(session.single_mic)
        except asyncio.CancelledError:
            # Wait for the device operation before the manager closes capture.
            with suppress(Exception):
                await task
            raise
    except AudioBusyError:
        raise MmpgError("capture_busy") from None
    except Exception:
        raise MmpgError("device_unavailable") from None


async def stop_capture(mode, reason, *, single_mic_id=None):
    if single_mic_id is not None:
        detach_single_mic_stream(single_mic_id, "interviewer")
        return
    if mode != "test" and reason == "owner_disconnected":
        await candidate_mmpg.stop(reason)
    if mode == "test":
        await run_in_threadpool(audio_capture.stop, owner="mmpg-test")
    elif pipeline.is_streaming_interview() and not pipeline.is_paused():
        await run_in_threadpool(pipeline.pause_interview)
    elif not get_session().is_recording:
        await run_in_threadpool(audio_capture.stop, owner="assist")


def detach_single_mic_stream(session_id: str, role: Role) -> None:
    single_mic = get_session().single_mic
    if single_mic is not None and single_mic.id == session_id:
        single_mic.set_stream(role, None)
        pipeline.broadcast_single_mic(single_mic)


def sync_single_mic_transport(role: Role, transport: MmpgManager, state: dict) -> bool:
    session_id = transport.single_mic_id
    if session_id is None:
        return False
    single_mic = get_session().single_mic
    if single_mic is None or single_mic.id != session_id:
        return True
    if state["state"] == "listening":
        single_mic.set_stream(role, transport.offer_audio)
    else:
        detach_single_mic_stream(session_id, role)
    if state["state"] == "error" or state.get("reason") in ("owner_disconnected", "settings_changed"):
        asyncio.create_task(_pause_after_loss(single_mic))
    return True


def single_mic_transports(single_mic: SingleMicSession):
    return [
        transport for transport in (manager, candidate_mmpg.manager)
        if transport.single_mic_id == single_mic.id
    ]


async def drain_single_mic_transports(single_mic: SingleMicSession, reason: str) -> None:
    single_mic.pause()
    pipeline.broadcast_single_mic(single_mic)
    role = single_mic.snapshot()["role"]
    active, idle = manager, candidate_mmpg.manager
    if role == "candidate":
        active, idle = idle, active

    async def drain_active():
        if active.single_mic_id == single_mic.id:
            timeout = max(6.0, get_config().assist_interviewer_asr_drain_timeout_sec)
            if await active.drain_single_mic_input(single_mic.id, timeout=timeout) and role == "interviewer":
                finish_interviewer_role(single_mic)

    async def stop_idle():
        if idle.single_mic_id == single_mic.id:
            await idle.stop(reason)

    await asyncio.gather(drain_active(), stop_idle())


async def _pause_after_loss(single_mic: SingleMicSession, *, message=None) -> None:
    async with http_lifecycle_lock:
        session = get_session()
        if session.single_mic is not single_mic or session.is_paused:
            return
        await run_in_threadpool(pipeline.pause_interview)
        await asyncio.gather(*(transport.stop("paused") for transport in single_mic_transports(single_mic)))
        if not session.is_recording:
            await run_in_threadpool(pipeline.stop_interview_loop)
        elif message:
            broadcast({"type": "error", "message": message})


def watch_single_mic(single_mic: SingleMicSession | None) -> None:
    if single_mic is None:
        return

    async def watch():
        session = get_session()
        not_ready_since = None
        while session.single_mic is single_mic:
            await asyncio.sleep(0.5)
            if session.single_mic is not single_mic:
                break
            if not session.is_recording:
                await _pause_after_loss(single_mic)
                break
            if session.is_paused:
                not_ready_since = None
                continue
            if single_mic.seconds_since_audio() > 2.5:
                await _pause_after_loss(single_mic, message="麦克风采集已中断，请检查设备后继续单人测试")
            elif single_mic.ready:
                not_ready_since = None
            elif not_ready_since is None:
                not_ready_since = time.monotonic()
            elif time.monotonic() - not_ready_since > 30:
                await _pause_after_loss(single_mic, message="语音识别连接超时，请检查配置后继续单人测试")

    asyncio.create_task(watch())


async def finish_single_mic_role(single_mic, operation, timeout: float) -> bool:
    if not operation.streaming:
        return await run_in_threadpool(pipeline.drain_single_mic_worker, single_mic, operation, timeout)
    transport = manager if operation.from_role == "interviewer" else candidate_mmpg.manager
    success = await transport.finish_role_input(single_mic.id, timeout=timeout)
    operation.complete(success)
    return success


def finish_interviewer_role(single_mic: SingleMicSession) -> None:
    if manager.single_mic_id == single_mic.id and _turns is not None:
        _turns.finish_role()


def _broadcast_state(status):
    if status.get("mode") != "test":
        broadcast({"type": "stt_status", **stt_status(status)})
        if sync_single_mic_transport("interviewer", manager, status):
            return
        if status["state"] == "error" and pipeline.is_streaming_interview() and candidate_mmpg.manager.active:
            asyncio.create_task(candidate_mmpg.stop("interviewer_disconnected"))


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
