"""Unit tests for Stage 2A P2-4 realtime pipeline, answer worker, and MMPG reliability fixes."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from api.assist import answer_worker, mmpg_integration, pipeline
from api.assist.asr_state import AssistAsrStateMachine
from api.assist.mmpg_turns import MmpgRequest, MmpgTurns
from api.assist.single_mic import SingleMicSession
from core.config import AppConfig, ModelConfig
from core.session import Session, conversation_lock, get_session, reset_session


def _reset_pipeline_state():
    pipeline.cancel_answer_work(reset_session_data=True)
    with pipeline._dispatch_lock:
        pipeline._next_submit_seq = 0
    with pipeline._commit_lock:
        pipeline._next_commit_seq = 0
        pipeline._commit_buffer.clear()
        pipeline._skipped_commit_seqs.clear()
        pipeline._commit_draining = False
    pipeline._stop_event.clear()
    pipeline._pause_event.clear()
    pipeline._candidate_flush_event.clear()
    pipeline._interview_thread = None
    pipeline._candidate_thread = None
    pipeline._interviewer_asr_thread = None
    pipeline._flush_thread = None
    pipeline._interviewer_runtime = None
    pipeline._interviewer_streaming = False
    pipeline._interviewer_config = None


# (a) _commit_lock is not held during apply_fn, while strict seq order is preserved.
def test_commit_queue_releases_commit_lock_during_apply_and_preserves_seq_order():
    _reset_pipeline_state()
    with pipeline._commit_lock:
        base_seq = pipeline._next_commit_seq

    commit0_entered = threading.Event()
    release_commit0 = threading.Event()
    commit1_enqueued_without_blocking = threading.Event()
    order: list[str] = []
    lock_was_held_during_apply = {"value": True}

    def commit0():
        lock_was_held_during_apply["value"] = pipeline._commit_lock.locked()
        order.append("start-0")
        commit0_entered.set()
        release_commit0.wait(timeout=2.0)
        order.append("end-0")

    def commit1():
        order.append("run-1")

    t0 = threading.Thread(target=pipeline._flush_commit, args=(base_seq, commit0), daemon=True)
    t0.start()
    assert commit0_entered.wait(timeout=2.0)
    assert lock_was_held_during_apply["value"] is False
    # Still busy while commit0 is executing.
    assert pipeline._answer_work_idle() is False

    def enqueue_commit1():
        pipeline._flush_commit(base_seq + 1, commit1)
        commit1_enqueued_without_blocking.set()

    t1 = threading.Thread(target=enqueue_commit1, daemon=True)
    t1.start()
    # Must NOT block waiting on commit0!
    assert commit1_enqueued_without_blocking.wait(timeout=1.0)
    assert order == ["start-0"]

    release_commit0.set()
    t0.join(timeout=2.0)
    t1.join(timeout=2.0)

    assert order == ["start-0", "end-0", "run-1"]
    assert pipeline._answer_work_idle() is True


# (b) _try_dispatch rolls back _in_flight_tasks and skips seq if Thread.start fails or model_idx unresolved.
def test_try_dispatch_rolls_back_on_thread_start_failure(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    cfg = AppConfig(
        models=[ModelConfig(id="m1", name="M1", api_key="sk-test", enabled=True)],
        assist_model_ids=["m1"],
    )
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "get_model_health", lambda _idx: "ok")
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    class BrokenThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread limit reached")

    monkeypatch.setattr(pipeline.threading, "Thread", BrokenThread)

    finished_called: list[bool] = []
    request = MmpgRequest("g1", 1, "测试问题？", ("t1",), 0.0, None)
    request.on_finished = lambda: finished_called.append(True)

    with pipeline._commit_lock:
        seq0 = pipeline._next_commit_seq

    committed_after: list[str] = []
    pipeline._flush_commit(seq0 + 1, lambda: committed_after.append("seq1-ok"))

    assert pipeline.submit_answer_task(
        ("测试问题？", None, False, "conversation_loopback", {"origin": "mmpg", "mmpg_request": request}),
        config_snapshot=cfg,
    )

    assert pipeline._in_flight_tasks == {}
    assert finished_called == [True]
    # seq0 must have been skipped, immediately draining seq1!
    assert committed_after == ["seq1-ok"]
    assert any(msg.get("type") == "error" for msg in broadcasts)


def test_try_dispatch_rolls_back_when_worker_model_idx_is_unresolved(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    cfg = AppConfig(
        models=[ModelConfig(id="m1", name="M1", api_key="sk-test", enabled=True, supports_vision=True)],
        written_exam_mode=True,
        written_exam_model_ids=["m1"],
    )
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "get_model_health", lambda _idx: "ok")
    # Force pipeline.eligible_exam_model_index to return None when resolving worker_model_idx inside _try_dispatch.
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)
    monkeypatch.setattr(pipeline, "eligible_exam_model_index", lambda _snapshot, _model_id: None)

    committed_after: list[str] = []
    pipeline._flush_commit(1, lambda: committed_after.append("seq1-unblocked"))

    assert pipeline.submit_answer_task(
        ("笔试题", "data:image/png;base64,fake", True, "server_screen_left", {"origin": "server_screen"}),
        config_snapshot=cfg,
    )

    assert pipeline._in_flight_tasks == {}
    assert committed_after == ["seq1-unblocked"]
    assert any(msg.get("type") == "error" for msg in broadcasts)


# (c) _reserve_answer_retry passes get_model_health so healthy backups win over degraded ones.
def test_reserve_answer_retry_prefers_healthy_backup_model(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    cfg = AppConfig(
        models=[
            ModelConfig(id="m0", name="Primary", api_key="sk-0", enabled=True),
            ModelConfig(id="m1", name="Unhealthy", api_key="sk-1", enabled=True),
            ModelConfig(id="m2", name="Healthy", api_key="sk-2", enabled=True),
        ],
        assist_model_ids=["m0", "m1", "m2"],
    )
    health = {0: "error", 1: "error", 2: "ok"}
    monkeypatch.setattr(pipeline, "get_model_health", lambda idx: health.get(idx))

    task = ("问题", None, True, "manual_text", {"origin": "manual"})
    with pipeline._dispatch_lock:
        pipeline._in_flight_tasks[0] = ("m0", task)

    try:
        picked = pipeline._reserve_answer_retry(task, 0, cfg, {0}, lambda: False)
        assert picked == 2
    finally:
        with pipeline._dispatch_lock:
            pipeline._in_flight_tasks.clear()


# (d) _interview_worker uses post-drain monotonic timestamp and skips redundant 50ms sleep.
def test_interview_worker_uses_post_drain_timestamp_and_skips_redundant_sleep(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    session = reset_session()
    session.is_recording = True
    cfg = AppConfig(
        stt_provider="doubao",
        doubao_stt_api_key="test-key",
        silence_threshold=0.01,
        silence_duration=0.05,
        assist_vad_min_speech_sec=0.1,
    )
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", lambda _msg: None)
    monkeypatch.setattr(pipeline, "_try_flush_asr_merge_buffer", lambda *_a, **_kw: None)
    monkeypatch.setattr(pipeline, "_try_flush_asr_question_group", lambda *_a, **_kw: None)
    monkeypatch.setattr(pipeline, "transcribe_segment", lambda *_a, **_kw: "")

    sleep_calls: list[float] = []
    real_sleep = time.sleep
    monkeypatch.setattr(pipeline.time, "sleep", lambda sec: sleep_calls.append(sec))

    clock = {"now": 100.0}
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock["now"])

    captured_segments: list[pipeline.InterviewerSegment] = []
    orig_emit = pipeline._emit_interviewer_segment

    def capture_emit(runtime, segment):
        captured_segments.append(segment)
        return orig_emit(runtime, segment)

    monkeypatch.setattr(pipeline, "_emit_interviewer_segment", capture_emit)

    class _DelayedAudio:
        is_running = True
        dropped_chunks_count = 0

        def __init__(self):
            self.calls = 0

        def drain_audio_chunks(self, timeout=0.1, max_chunks=None):
            self.calls += 1
            if self.calls == 1:
                # Empty batch should NOT trigger an extra 0.05s sleep!
                clock["now"] += 0.1
                return []
            if self.calls == 2:
                # Simulate drain blocking for 0.09s before delivering 0.2s speech + 0.1s silence.
                clock["now"] = 105.0
                speech = np.ones(3200, dtype=np.float32) * 0.2
                silence = np.zeros(1600, dtype=np.float32)
                return [np.concatenate([speech, silence])]
            pipeline._stop_event.set()
            return []

        def stop(self, owner=None, clear_queue=True):
            self.is_running = False

        def capture_stats_snapshot(self):
            return {"dropped_chunks_count": 0, "loopback_discontinuity_count": 0, "max_queue_chunk_samples": 0}

    monkeypatch.setattr(pipeline, "audio_capture", _DelayedAudio())
    pipeline._interview_worker(config_snapshot=cfg, transcript_epoch=session.transcript_epoch)

    assert 0.05 not in sleep_calls
    assert captured_segments
    # Speech (0.20s) + silence flush at 0.06s (0.04s before batch end 105.0) => 104.96,
    # proving post-drain clock (105.0) was used instead of pre-drain clock (100.1).
    assert captured_segments[0].ended_mono == pytest.approx(104.96, abs=0.01)
    real_sleep(0.01)


# (e1 + e2) Dual-mic stop (clear_queue=False) & pause drain remaining candidate tail chunks.
def test_candidate_worker_drains_dual_mic_queue_on_pause_and_stop(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    session = reset_session()
    session.is_recording = True
    cfg = AppConfig(
        candidate_asr_enabled=True,
        candidate_stt_provider="doubao",
        doubao_stt_api_key="test-key",
        silence_threshold=0.01,
        silence_duration=1.2,
    )
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "get_capture_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", lambda _msg: None)
    monkeypatch.setattr(pipeline, "review_integration", Mock())
    monkeypatch.setattr(pipeline, "_wait_for_answer_work_idle", lambda _t: True)
    monkeypatch.setattr(pipeline, "cancel_answer_work", lambda **_kw: None)
    monkeypatch.setattr(pipeline.audio_capture, "stop", lambda **_kw: None)

    published: list[str] = []
    texts = iter(["暂停前尾音", "停录前尾音"])
    monkeypatch.setattr(pipeline, "transcribe_segment", lambda *_a, **_kw: next(texts))
    monkeypatch.setattr(
        pipeline,
        "_publish_candidate_transcription",
        lambda _s, text, *_a, **_kw: published.append(text),
    )

    class _QueuedCandidateCapture:
        def __init__(self):
            self.is_running = True
            self.stop_clear_flags: list[bool] = []
            self.q: queue.Queue[np.ndarray] = queue.Queue()

        def push(self, arr: np.ndarray):
            self.q.put(arr)

        def drain_audio_chunks(self, timeout=0.1, max_chunks=None):
            items = []
            try:
                items.append(self.q.get(timeout=max(0.0, timeout)))
            except queue.Empty:
                return items
            while len(items) < (max_chunks or 6):
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            return items

        def stop(self, owner=None, *, clear_queue=True):
            self.is_running = False
            self.stop_clear_flags.append(clear_queue)
            if clear_queue:
                while not self.q.empty():
                    self.q.get_nowait()

    cap = _QueuedCandidateCapture()
    monkeypatch.setattr(pipeline, "_candidate_audio_capture", cap)

    worker = threading.Thread(
        target=pipeline._candidate_worker,
        args=(session.transcript_epoch, None, cfg),
        daemon=True,
    )
    pipeline._candidate_thread = worker
    worker.start()

    # Queue speech right as pause hits; pause stops capture with clear_queue=False and worker drains it.
    cap.push(np.ones(8000, dtype=np.float32) * 0.2)
    pipeline.pause_interview()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(published) < 1:
        time.sleep(0.02)
    assert published == ["暂停前尾音"]
    assert cap.stop_clear_flags[0] is False

    # Unpause capture flag, queue tail speech, then stop_interview_loop (must pass clear_queue=False!).
    cap.is_running = True
    pipeline._pause_event.clear()
    cap.push(np.ones(8000, dtype=np.float32) * 0.2)
    pipeline.stop_interview_loop()

    assert not worker.is_alive()
    assert cap.stop_clear_flags[1] is False
    assert published == ["暂停前尾音", "停录前尾音"]


# (e3) Candidate worker detects unexpected capture loss and exits cleanly.
def test_candidate_worker_exits_and_reports_error_on_unexpected_capture_stop(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    session = reset_session()
    session.is_recording = True
    cfg = AppConfig(candidate_asr_enabled=True, candidate_stt_provider="doubao", doubao_stt_api_key="test-key")
    monkeypatch.setattr(pipeline, "get_capture_config", lambda: cfg)
    broadcasts: list[dict] = []
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)

    class _DroppingCapture:
        is_running = False

        def drain_audio_chunks(self, timeout=0.1, max_chunks=None):
            return []

        def stop(self, owner=None, *, clear_queue=True):
            self.is_running = False

    monkeypatch.setattr(pipeline, "_candidate_audio_capture", _DroppingCapture())

    worker = threading.Thread(
        target=pipeline._candidate_worker,
        args=(session.transcript_epoch, None, cfg),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert any(
        msg.get("type") == "stt_error" and msg.get("role") == "candidate"
        for msg in broadcasts
    )


# (e4) Non-STTError during single-mic candidate role switch still completes barrier with False (no 504 hang).
def test_candidate_worker_completes_role_switch_barrier_on_non_stt_error(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    session = reset_session()
    session.is_recording = True
    cfg = AppConfig(candidate_asr_enabled=True, candidate_stt_provider="doubao", doubao_stt_api_key="test-key")
    monkeypatch.setattr(pipeline, "get_capture_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", lambda _msg: None)
    monkeypatch.setattr(
        pipeline,
        "transcribe_segment",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("unexpected decoder crash")),
    )

    single_mic = SingleMicSession("sm-err", device_id=0)
    # Switch interviewer -> candidate first (mark interviewer barrier done).
    op_to_cand = single_mic.begin_switch("candidate")
    assert op_to_cand is not None
    assert single_mic.interviewer_input.take_barrier() is op_to_cand
    op_to_cand.complete(True)
    assert single_mic.finish_switch(op_to_cand, True)

    worker = threading.Thread(
        target=pipeline._candidate_worker,
        args=(session.transcript_epoch, single_mic, cfg),
        daemon=True,
    )
    worker.start()
    try:
        # Offer enough speech to enter VAD buffer (> 0.3s), then begin switch candidate -> interviewer.
        single_mic.candidate_input.offer_audio(np.ones(8000, dtype=np.float32) * 0.2)
        time.sleep(0.05)
        op_back = single_mic.begin_switch("interviewer")
        assert op_back is not None
        # Must complete promptly (with False) instead of hanging until 504 timeout!
        assert op_back._done.wait(timeout=2.0) is True
        assert op_back.wait(0.0) is False
    finally:
        pipeline._stop_event.set()
        single_mic.close()
        worker.join(timeout=2.0)


# (f) _append_late_asr_constraint_tail groups the late tail anchor in LiveTranscript.
def test_append_late_asr_constraint_tail_groups_live_transcript_anchors(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    session = reset_session()
    session.begin_transcript_capture()
    cfg = AppConfig(
        auto_detect=True,
        assist_transcription_merge_gap_sec=0.0,
        assist_asr_late_constraint_grace_sec=2.0,
        models=[ModelConfig(id="m1", name="M1", api_key="sk-test", enabled=True)],
        assist_model_ids=["m1"],
    )
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "get_model_health", lambda _idx: "ok")
    monkeypatch.setattr(pipeline, "broadcast", lambda _msg: None)
    monkeypatch.setattr(pipeline, "_try_dispatch", lambda: None)

    now = time.monotonic()
    a1 = session.record_transcript("interviewer", "Redis 和 Memcached 区别是什么？", started_at=now, question_group="q1")
    a2 = session.record_transcript("interviewer", "不要结合项目", started_at=now + 0.5)
    assert a1 and a2

    state = AssistAsrStateMachine(
        broadcast=lambda _m: None,
        submit_answer_task=lambda task: pipeline.submit_answer_task(task, config_snapshot=cfg),
        begin_asr_turn=lambda: 1,
        append_late_constraint_tail=pipeline._append_late_asr_constraint_tail,
        clock=lambda: now + 0.2,
        logger=Mock(),
    )
    state.handle_auto_detect_asr_text(
        cfg, session, "Redis 和 Memcached 区别是什么？", "conversation_loopback", now,
        transcript_anchor=a1, transcript_anchors=(a1,),
    )
    state.flush_question_group_now(cfg, session)
    assert len(pipeline._pending) == 1

    merged = pipeline._append_late_asr_constraint_tail(
        "不要结合项目", "conversation_loopback", now + 0.6, transcript_anchor=a2,
    )
    assert merged is True

    # Both utterances must now belong to the same LiveTranscript question round!
    rounds = session.transcript_snapshot(a2)
    assert len(rounds) == 1
    assert [u.text for u in rounds[0]] == ["Redis 和 Memcached 区别是什么？", "不要结合项目"]


# (g) transcription_history updates/reads are synchronized with conversation_lock.
def test_transcription_history_reads_and_writes_are_thread_safe():
    session = reset_session()
    state = AssistAsrStateMachine(
        broadcast=lambda _m: None,
        submit_answer_task=lambda _t: False,
        begin_asr_turn=lambda: 1,
        logger=Mock(),
    )
    cfg = AppConfig(auto_detect=False)
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(100):
                state.accept_final_turn(cfg, session, f"面试官问题编号 {i} 请详细说明")
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for i in range(100):
                pipeline._publish_candidate_transcription(
                    session, f"候选人独立回答内容 {i} 架构细节", "doubao",
                )
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert errors == []


# (h) start_nonblocking rolls back audio capture and session state on mid-start failure.
def test_start_nonblocking_rolls_back_capture_and_recording_state_on_failure(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    session = reset_session()
    cfg = AppConfig(stt_provider="doubao", doubao_stt_api_key="test-key", assist_stop_answer_wait_sec=0)
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "_device_is_loopback", lambda _dev: False)
    monkeypatch.setattr(pipeline, "broadcast", lambda _msg: None)
    monkeypatch.setattr(pipeline, "_start_interview_worker_thread_if_needed", lambda: None)

    capture_events: list[str] = []

    class _FakeCapture:
        is_running = False

        def start(self, device_id, **kwargs):
            self.is_running = True
            capture_events.append(f"start:{device_id}")

        def stop(self, owner=None, **kwargs):
            if self.is_running:
                self.is_running = False
                capture_events.append(f"stop:{owner}")

    fake_cap = _FakeCapture()
    monkeypatch.setattr(pipeline, "audio_capture", fake_cap)
    monkeypatch.setattr(
        pipeline.review_integration,
        "on_assist_start",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("review db init failed")),
    )
    monkeypatch.setattr(pipeline.review_integration, "on_assist_stop", lambda _s: None)

    with pytest.raises(RuntimeError, match="review db init failed"):
        pipeline.start_nonblocking(3, config_snapshot=cfg)

    assert capture_events == ["start:3", "stop:assist"]
    assert session.is_recording is False
    assert session.single_mic is None
    assert pipeline._interviewer_config is None


# (i) _run_answer_worker finally tolerates closed event loop in request.on_finished and still calls _try_dispatch.
def test_run_answer_worker_finally_continues_dispatch_when_loop_is_closed(monkeypatch: pytest.MonkeyPatch):
    _reset_pipeline_state()
    dispatched: list[bool] = []
    monkeypatch.setattr(pipeline, "_process_question_parallel", lambda *_a, **_kw: None)
    monkeypatch.setattr(pipeline, "_try_dispatch", lambda: dispatched.append(True))

    request = MmpgRequest("g-closed", 1, "问题", ("t-closed",), 0.0, None)
    request.on_finished = lambda: (_ for _ in ()).throw(RuntimeError("Event loop is closed"))
    task = ("问题", None, False, "conversation_loopback", {"origin": "mmpg", "mmpg_request": request})

    with pipeline._dispatch_lock:
        pipeline._in_flight_tasks[0] = ("m0", task)

    pipeline._run_answer_worker(task, 0, 0, 0)

    assert 0 not in pipeline._in_flight_tasks
    assert dispatched == [True]

    # Also verify MmpgRequest.cancel() does not raise on a closed loop.
    loop = asyncio.new_event_loop()
    loop.close()
    req2 = MmpgRequest("g-closed-2", 1, "问题", ("t2",), 0.0, None, loop=loop)
    req2.stream_read = Mock()
    assert req2.cancel() is True


# (j) Streaming retry does NOT flush buffered residual fragments before answer_retry.
def test_streaming_retry_suppresses_buffered_residual_fragment_before_retry(monkeypatch: pytest.MonkeyPatch):
    reset_session()
    cfg = AppConfig(
        models=[
            ModelConfig(id="p", name="primary", api_key="sk-p", enabled=True),
            ModelConfig(id="b", name="backup", api_key="sk-b", enabled=True),
        ],
        assist_model_ids=["p", "b"],
    )
    broadcasts: list[dict] = []

    class PassthroughSanitizer:
        def push(self, chunk: str) -> str:
            return chunk

        def finish(self) -> str:
            return "-residual-tail"

    def fake_stream(model, _messages, **_kwargs):
        if model.name == "primary":
            # Only 2 chunks (< batch threshold 5): still buffered in pending_chunks when error hits!
            yield ("text", "残句1")
            yield ("text", "残句2")
            raise answer_worker.LLMError("upstream reset")
        yield ("text", "完整重试答案")

    monkeypatch.setattr(answer_worker, "create_answer_stream_sanitizer", lambda *_a, **_kw: PassthroughSanitizer())
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    deps = answer_worker.AnswerWorkerDeps(
        abort_check=lambda: False,
        is_session_current=lambda _v: True,
        flush_commit=lambda _seq, fn: fn(),
        mark_seq_skipped=lambda _seq: None,
        submit_knowledge_record=lambda *_a: True,
        broadcast=broadcasts.append,
        logger=Mock(),
        error_logger=Mock(),
        config_snapshot=cfg,
        reserve_retry=lambda attempted, _on_wait: 1 if 1 not in attempted else None,
    )

    # Use knowledge_practice=False + manual_image/structured (non-plain_text) so pending_chunks batching applies.
    answer_worker.process_question_parallel(
        ("结构化问题？", "data:image/png;base64,abc", True, "manual_image", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=deps,
    )

    attempt1_chunks = [e for e in broadcasts if e["type"] == "answer_chunk" and e["attempt"] == 1]
    assert attempt1_chunks == []
    assert any(e["type"] == "answer_retry" and e["attempt"] == 2 for e in broadcasts)
    done = next(e for e in broadcasts if e["type"] == "answer_done")
    assert done["attempt"] == 2
    assert "完整重试答案" in done["answer"]


# (k) mmpg_integration._spawn_background_task tracks references and cleans up on completion/error.
def test_mmpg_integration_background_task_registry_tracks_and_cleans_up():
    async def exercise():
        mmpg_integration._background_tasks.clear()
        gate = asyncio.Event()

        async def worker():
            await gate.wait()
            raise RuntimeError("expected background failure")

        task = mmpg_integration._spawn_background_task(worker(), name="test-bg-task")
        assert task in mmpg_integration._background_tasks
        gate.set()
        await asyncio.gather(task, return_exceptions=True)
        # Done callback runs on next loop turn.
        await asyncio.sleep(0)
        assert task not in mmpg_integration._background_tasks

    asyncio.run(exercise())


# (l) MmpgTurns._seen is bounded with FIFO eviction.
def test_mmpg_turns_seen_traces_is_bounded():
    turns = MmpgTurns(
        submit=lambda _req: True,
        cancel=lambda _req: None,
        publish=lambda _phase: None,
        enabled=lambda: True,
        clock=lambda: 0.0,
        call_later=lambda _delay, _cb: SimpleNamespace(cancel=lambda: None),
        answer_delay_ms=0,
        followup_window_ms=0,
        allow_answer=lambda: False,
        max_seen=3,
    )
    for idx in range(10):
        turns.final(f"trace-{idx}", f"text-{idx}")

    assert len(turns._seen) == 3
    assert list(turns._seen.keys()) == ["trace-7", "trace-8", "trace-9"]
    # Recent traces remain deduplicated; oldest evicted trace is no longer in _seen.
    assert "trace-9" in turns._seen
    assert "trace-0" not in turns._seen
