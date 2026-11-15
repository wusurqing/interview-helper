"""Transmit the candidate microphone to its own MMPG session."""

from fastapi.concurrency import run_in_threadpool

from api.realtime.ws import broadcast
from core.config import MMPG_STT_PROVIDER, get_config
from core.session import conversation_lock, get_session
from services.audio import AudioBusyError
from services.stt.mmpg import MmpgError, MmpgManager
from . import pipeline

_CAPTURE_OWNER = "assist-candidate-mmpg"
_pending: dict[str, str] = {}


def enabled(cfg):
    return (
        getattr(cfg, "candidate_asr_enabled", False)
        and getattr(cfg, "candidate_stt_provider", None) == MMPG_STT_PROVIDER
    )


async def prepare(mode, device_ids, _candidate_mic_id):
    cfg = get_config()
    session = get_session()
    if mode not in ("start", "resume"):
        raise MmpgError("invalid_command")
    if not enabled(cfg):
        raise MmpgError("candidate_provider_not_selected")
    if not session.is_recording or session.is_paused:
        raise MmpgError("candidate_interview_inactive")
    if (len(device_ids) != 1 or not pipeline._is_valid_device_id(device_ids[0])
            or device_ids[0] != session.last_candidate_mic_device_id
            or device_ids[0] == session.last_device_id):
        raise MmpgError("device_required")
    if pipeline.is_streaming_interview() and cfg.mmpg.token_mode == "manual":
        raise MmpgError("candidate_mmpg_auto_token_required")


async def _start_capture(_mode, device_ids, _candidate_mic_id):
    cfg = get_config()
    session = get_session()
    device_id = device_ids[0]
    if not session.is_recording or session.last_candidate_mic_device_id != device_id or not enabled(cfg):
        raise MmpgError("candidate_interview_inactive")
    if await run_in_threadpool(pipeline._device_is_loopback, device_id):
        raise MmpgError("candidate_mic_required")
    try:
        await run_in_threadpool(
            pipeline._candidate_audio_capture.start,
            device_id, on_audio=manager.offer_audio, owner=_CAPTURE_OWNER,
            mic_compatibility_mode=cfg.candidate_mic_compatibility_mode, queue_audio=False,
        )
    except AudioBusyError:
        raise MmpgError("capture_busy") from None
    except Exception:
        raise MmpgError("device_unavailable") from None


async def _stop_capture(_mode, _reason):
    await run_in_threadpool(pipeline._candidate_audio_capture.stop, owner=_CAPTURE_OWNER)


async def stop(reason, *, drain=False):
    if drain and manager.status()["state"] == "listening":
        try:
            await _stop_capture(None, reason)
            await manager.drain_transcripts()
        finally:
            await manager.stop(reason)
    else:
        await manager.stop(reason)


def _on_turn(kind, trace):
    session = get_session()
    with conversation_lock:
        if kind == "started":
            if trace not in _pending:
                _pending[trace] = session.current_candidate_qa_id
        elif kind == "reset":
            _pending.clear()
        else:
            _pending.pop(trace, None)
        if _pending:
            session.mark_candidate_asr_busy(next(reversed(_pending.values())))
        else:
            session.mark_candidate_asr_idle()


def _on_transcript(text, trace):
    session = get_session()
    try:
        if session.is_recording and enabled(get_config()):
            pipeline._publish_candidate_transcription(
                session, text, MMPG_STT_PROVIDER, _pending.get(trace, ""),
                segment_id=f"candidate-mmpg-{trace}",
            )
    finally:
        _on_turn("cancelled", trace)


def status(state=None):
    state = manager.status() if state is None else state
    return {
        "type": "candidate_asr_status",
        "provider": MMPG_STT_PROVIDER,
        "loaded": state["state"] == "listening",
        "loading": state["state"] in ("connecting", "reconnecting"),
        "state": state["state"],
        "error": state["error"],
    }


manager = MmpgManager(
    lambda: get_config().mmpg, _on_transcript, on_turn=_on_turn,
    on_state=lambda state: broadcast(status(state)),
    on_capture_start=_start_capture, on_capture_stop=_stop_capture,
)
