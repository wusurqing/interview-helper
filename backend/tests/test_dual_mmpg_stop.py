import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from api.assist import candidate_mmpg, mmpg_integration, pipeline, routes
from core.session import get_session
from services import review_integration
from services.storage import review


def test_stop_drains_both_cloud_tails_after_freezing_real_inputs(monkeypatch):
    session = get_session()
    archive = Mock()
    monkeypatch.setattr(session, "archive", archive)
    monkeypatch.setattr(session, "is_recording", True)
    monkeypatch.setattr(session, "single_mic", None)

    frozen = set()
    drained = set()
    closed = set()
    monkeypatch.setattr(pipeline.audio_capture, "stop", lambda **_: frozen.add("interviewer"))
    monkeypatch.setattr(pipeline._candidate_audio_capture, "stop", lambda **_: frozen.add("candidate"))
    monkeypatch.setattr(pipeline, "_pause_event", Mock())

    async def exercise():
        both_started = asyncio.Event()

        def transport(role, drain_ok):
            obj = SimpleNamespace(active=True, single_mic_id=None, status=lambda: {"state": "listening", "mode": "start"})

            async def drain(**_):
                assert frozen == {"interviewer", "candidate"}
                archive.seal_capture.assert_called_once()
                assert not closed
                drained.add(role)
                if len(drained) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=1)
                return drain_ok

            async def stop(_):
                closed.add(role)
                obj.active = False

            obj.drain_transcripts = drain
            obj.stop = stop
            return obj

        monkeypatch.setattr(mmpg_integration, "manager", transport("interviewer", True))
        monkeypatch.setattr(candidate_mmpg, "manager", transport("candidate", False))

        def save_tail():
            assert drained == closed == {"interviewer", "candidate"}
            archive.stop()
            session.archive = None
            session.is_recording = False

        monkeypatch.setattr(routes, "stop_interview_loop", save_tail)
        assert await routes.api_stop() == {"ok": True}
        assert await routes.api_stop() == {"ok": True}
        archive.stt_timeout.assert_called_once_with("candidate")
        archive.stop.assert_called_once()

    asyncio.run(exercise())


def test_repeat_stop_waits_for_the_detached_archive_to_commit(monkeypatch):
    session = get_session()
    monkeypatch.setattr(session, "archive", None)
    monkeypatch.setattr(session, "is_recording", True)
    monkeypatch.setattr(session, "single_mic", None)
    monkeypatch.setattr(review_integration, "_current_review_session_id", None)
    monkeypatch.setattr(review_integration, "get_config", lambda: SimpleNamespace(review_enabled=False))

    sid = review_integration.on_assist_start(1, None, False)
    archive = session.archive
    archive.transcript("interviewer", "这一题会保存一次吗？", "q1", time.monotonic())
    save = archive._save
    release = threading.Event()
    writing = threading.Event()
    commits = []

    def slow_save(record):
        writing.set()
        assert release.wait(5)
        commits.append(save(record))

    monkeypatch.setattr(archive, "_save", slow_save)
    stop = archive.stop
    monkeypatch.setattr(archive, "stop", lambda **_: stop(timeout=0))

    def stop_pipeline():
        review_integration.on_assist_stop(session)
        session.is_recording = False

    monkeypatch.setattr(routes, "stop_interview_loop", stop_pipeline)

    async def exercise():
        monkeypatch.setattr(mmpg_integration, "http_lifecycle_lock", asyncio.Lock())
        first = asyncio.create_task(routes.api_stop())
        assert await asyncio.to_thread(writing.wait, 2)
        assert session.archive is None
        close_window = asyncio.create_task(routes.finish_interview("shutdown"))
        await asyncio.sleep(0.05)
        assert not first.done() and not close_window.done()
        assert review.get_session_detail(sid)["archive_status"] == "draining"
        release.set()
        await asyncio.wait_for(asyncio.gather(first, close_window), timeout=2)

    try:
        asyncio.run(exercise())
        detail = review.get_session_detail(sid)
        assert detail["archive_status"] == "complete"
        assert detail["turn_count"] == len(commits) == 1
    finally:
        release.set()
        stop()
