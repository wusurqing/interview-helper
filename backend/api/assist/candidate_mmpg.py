"""Transmit the candidate microphone to its own MMPG session."""

import time

from fastapi.concurrency import run_in_threadpool

from api.realtime.ws import broadcast
from core.config import MMPG_STT_PROVIDER
from core.session import conversation_lock, get_session
from services.audio import AudioBusyError
from services.stt.mmpg import MmpgError, MmpgManager
from . import pipeline

_CAPTURE_OWNER = "assist-candidate-mmpg"
_pending: dict[str, tuple[str, float]] = {}
_capture_epoch: int | None = None
_prepared_lifecycle_epoch: int | None = None
_prepared_transcript_epoch: int | None = None


def enabled(cfg):
    return (
        getattr(cfg, "candidate_asr_enabled", False)
        and getattr(cfg, "candidate_stt_provider", None) == MMPG_STT_PROVIDER
    )


async def prepare(mode, device_ids, _candidate_mic_id, *, single_mic_id=None):
    global _prepared_lifecycle_epoch, _prepared_transcript_epoch
    from . import mmpg_integration

    async with mmpg_integration.http_lifecycle_lock:
        cfg = pipeline.get_capture_config()
        session = get_session()
        if mode not in ("start", "resume"):
            raise MmpgError("invalid_command")
        if not enabled(cfg):
            raise MmpgError("candidate_provider_not_selected")
        if not session.is_recording or session.is_paused:
            raise MmpgError("candidate_interview_inactive")
        single_mic = session.single_mic
        if single_mic is not None or single_mic_id is not None:
            if single_mic is None or single_mic.id != single_mic_id or device_ids != [single_mic.device_id]:
                raise MmpgError("candidate_interview_inactive")
        elif (len(device_ids) != 1 or not pipeline._is_valid_device_id(device_ids[0])
              or device_ids[0] != session.last_candidate_mic_device_id
              or device_ids[0] == session.last_device_id):
            raise MmpgError("device_required")
        if pipeline.is_streaming_interview() and cfg.mmpg.token_mode == "manual":
            raise MmpgError("candidate_mmpg_auto_token_required")
        with conversation_lock:
            _prepared_lifecycle_epoch = mmpg_integration.current_lifecycle_epoch()
            _prepared_transcript_epoch = session.transcript_epoch


async def _start_capture(_mode, device_ids, _candidate_mic_id, *, single_mic_id=None):
    # Never await http_lifecycle_lock here (see MmpgManager._run_transport shield).
    # Verify lifecycle and transcript epochs synchronously under conversation_lock.
    global _capture_epoch
    from . import mmpg_integration

    cfg = pipeline.get_capture_config()
    session = get_session()
    device_id = device_ids[0]

    def _validate_current_locked(expected_lifecycle: int, expected_transcript: int) -> int:
        if (
            mmpg_integration.current_lifecycle_epoch() != expected_lifecycle
            or session.transcript_epoch != expected_transcript
            or not session.is_recording
            or session.is_paused
            or session.last_candidate_mic_device_id != device_id
            or not enabled(pipeline.get_capture_config())
        ):
            raise MmpgError("candidate_interview_inactive")
        if single_mic_id is not None:
            single_mic = session.single_mic
            if single_mic is None or single_mic.id != single_mic_id:
                raise MmpgError("candidate_interview_inactive")
        return session.transcript_epoch

    with conversation_lock:
        expected_lifecycle = (
            _prepared_lifecycle_epoch
            if _prepared_lifecycle_epoch is not None
            else mmpg_integration.current_lifecycle_epoch()
        )
        expected_transcript = (
            _prepared_transcript_epoch
            if _prepared_transcript_epoch is not None
            else session.transcript_epoch
        )
        if not enabled(cfg):
            raise MmpgError("candidate_interview_inactive")
        epoch = _validate_current_locked(expected_lifecycle, expected_transcript)

    if await run_in_threadpool(pipeline._device_is_loopback, device_id):
        raise MmpgError("candidate_mic_required")

    if single_mic_id is not None:
        with conversation_lock:
            epoch = _validate_current_locked(expected_lifecycle, expected_transcript)
            _pending.clear()
            _capture_epoch = epoch
        return

    with conversation_lock:
        epoch = _validate_current_locked(expected_lifecycle, expected_transcript)
    try:
        await run_in_threadpool(
            pipeline._candidate_audio_capture.start,
            device_id, on_audio=manager.offer_audio, owner=_CAPTURE_OWNER,
            mic_compatibility_mode=cfg.candidate_mic_compatibility_mode, queue_audio=False,
        )
    except MmpgError:
        raise
    except AudioBusyError:
        raise MmpgError("capture_busy") from None
    except Exception:
        raise MmpgError("device_unavailable") from None

    stale = False
    with conversation_lock:
        try:
            epoch = _validate_current_locked(expected_lifecycle, expected_transcript)
        except MmpgError:
            stale = True
        else:
            _pending.clear()
            _capture_epoch = epoch
    if stale:
        try:
            await run_in_threadpool(pipeline._candidate_audio_capture.stop, owner=_CAPTURE_OWNER)
        except Exception:
            pass
        raise MmpgError("candidate_interview_inactive")


async def _stop_capture(_mode, _reason, *, single_mic_id=None):
    if single_mic_id is not None:
        from .mmpg_integration import detach_single_mic_stream

        detach_single_mic_stream(single_mic_id, "candidate")
        return
    await run_in_threadpool(pipeline._candidate_audio_capture.stop, owner=_CAPTURE_OWNER)


async def freeze_input():
    if manager.active and manager.single_mic_id is None:
        await _stop_capture(None, "stopped")


async def stop(reason, *, drain=False):
    if drain and manager.status()["state"] == "listening":
        try:
            await _stop_capture(None, reason, single_mic_id=manager.single_mic_id)
            await manager.drain_transcripts()
        finally:
            await manager.stop(reason)
    else:
        await manager.stop(reason)


def _on_turn(kind, trace):
    session = get_session()
    with conversation_lock:
        if session.transcript_epoch != _capture_epoch:
            return
        if kind == "started":
            if trace not in _pending:
                _pending[trace] = session.current_candidate_qa_id, time.monotonic()
                if session.archive:
                    session.record_transcript("candidate", "", started_at=_pending[trace][1],
                                              utterance_key=f"candidate-mmpg-{trace}", is_final=False,
                                              epoch=_capture_epoch, cloud=True)
        elif kind == "reset":
            if session.archive:
                session.archive.end_capture("candidate")
            for key in _pending:
                session.discard_unfinished_transcript(f"{_capture_epoch}:candidate:candidate-mmpg-{key}")
            _pending.clear()
        else:
            _pending.pop(trace, None)
            session.discard_unfinished_transcript(f"{_capture_epoch}:candidate:candidate-mmpg-{trace}")
        if _pending:
            qa_id, _ = next(reversed(_pending.values()))
            session.mark_candidate_asr_busy(qa_id)
        else:
            session.mark_candidate_asr_idle()


def _on_transcript(text, trace):
    session = get_session()
    try:
        with conversation_lock:
            epoch = _capture_epoch
            accepted = session.is_recording and session.transcript_epoch == epoch and enabled(pipeline.get_capture_config())
            qa_id, started_at = _pending.get(trace) or (session.current_candidate_qa_id, time.monotonic())
        if accepted:
            pipeline._publish_candidate_transcription(
                session, text, MMPG_STT_PROVIDER, qa_id,
                segment_id=f"candidate-mmpg-{trace}",
                started_at=started_at, transcript_epoch=epoch,
                cloud=True,
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


def _broadcast_state(state):
    from .mmpg_integration import sync_single_mic_transport

    broadcast(status(state))
    sync_single_mic_transport("candidate", manager, state)


manager = MmpgManager(
    lambda: pipeline.get_capture_config().mmpg, _on_transcript, on_turn=_on_turn,
    on_state=_broadcast_state,
    on_capture_start=_start_capture, on_capture_stop=_stop_capture,
)
