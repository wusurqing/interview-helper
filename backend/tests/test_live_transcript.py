import asyncio
from types import SimpleNamespace

from api.assist import candidate_mmpg, mmpg_integration, pipeline, routes
from api.assist.asr_state import AssistAsrStateMachine
from core.config import MMPG_STT_PROVIDER
from core.session import Session, reset_session
from services.audio import _SC_ID_BASE


def _round_text(session, anchor):
    return [[(item.speaker, item.text) for item in group]
            for group in session.transcript_snapshot(anchor)]


def test_live_transcript_orders_late_final_by_onset_and_cuts_at_next_question():
    session = Session()
    session.record_transcript("interviewer", "现在开始。", started_at=1)
    question = session.record_transcript("interviewer", "什么是 Redis？", started_at=3, question_group="first")
    next_question = session.record_transcript("interviewer", "什么是 Kafka？", started_at=8, question_group="next")
    session.record_transcript("candidate", "对。", started_at=6)
    session.record_transcript("interviewer", "可以。", started_at=7)
    session.record_transcript("candidate", "它是消息队列。", started_at=9)

    assert _round_text(session, question) == [
        [("interviewer", "现在开始。")],
        [("interviewer", "什么是 Redis？"), ("candidate", "对。"), ("interviewer", "可以。")],
    ]
    assert _round_text(session, next_question)[-1] == [
        ("interviewer", "什么是 Kafka？"), ("candidate", "它是消息队列。"),
    ]


def test_live_transcript_candidate_final_replaces_partial_in_place():
    session = Session()
    anchor = session.record_transcript("interviewer", "说说索引？", started_at=1, question_group="question")
    partial = session.record_transcript("candidate", "B+", started_at=2, utterance_key="candidate-1", is_final=False)
    session.record_transcript("candidate", "是的。", started_at=3)
    final = session.record_transcript("candidate", "B+ 树。", started_at=5, utterance_key="candidate-1")
    session.record_transcript("candidate", "旧听写", started_at=6, utterance_key="candidate-1", is_final=False)

    assert partial == final
    entries = session.transcript_snapshot(anchor)[-1]
    assert [(item.text, item.started_at, item.is_final) for item in entries[1:]] == [
        ("B+ 树。", 2, True), ("是的。", 3, True),
    ]


def test_live_transcript_capture_epoch_rejects_callbacks_after_clear_and_restart():
    session = Session()
    before_clear = session.transcript_epoch
    cleared = session.record_transcript("interviewer", "旧问题？", started_at=1)
    session.clear()
    before_restart = session.transcript_epoch
    session.begin_transcript_capture()
    assert session.record_transcript("candidate", "清空前音频", started_at=2, epoch=before_clear) is None
    assert session.record_transcript("candidate", "启动前音频", started_at=3, epoch=before_restart) is None
    current = session.record_transcript("interviewer", "新问题？", started_at=4)
    assert _round_text(session, cleared) == []
    assert _round_text(session, current) == [[("interviewer", "新问题？")]]


def test_queued_question_excludes_candidate_speech_after_unresolved_interviewer_onset():
    session = Session()
    current = session.record_transcript("interviewer", "解释缓存？", started_at=1, question_group="current")
    session.record_transcript("interviewer", "", utterance_key="next", started_at=3, is_final=False)
    session.record_transcript("candidate", "开始回答第二题。", started_at=4)

    assert _round_text(session, current) == [[("interviewer", "解释缓存？")]]


def test_stop_retains_completed_utterances_and_removes_unfinalized_audio():
    session = Session()
    current = session.record_transcript("interviewer", "解释缓存？", started_at=1, question_group="current")
    session.record_transcript("candidate", "用 TTL。", started_at=2)
    session.record_transcript("candidate", "还要", started_at=3, is_final=False)
    old_epoch = session.transcript_epoch
    session.stop_transcript_capture()

    assert session.record_transcript("candidate", "录音结束后返回。", started_at=4, epoch=old_epoch) is None
    assert _round_text(session, current) == [[("interviewer", "解释缓存？"), ("candidate", "用 TTL。")]]


def test_legacy_capture_creates_question_boundaries_with_auto_answer_off():
    session = Session()
    state = AssistAsrStateMachine(
        broadcast=lambda _payload: None,
        submit_answer_task=lambda _task: False,
        begin_asr_turn=lambda: 1,
        logger=SimpleNamespace(info=lambda *_args: None),
        clock=lambda: 22.0,
    )
    disabled = SimpleNamespace(auto_detect=False, assist_transcription_merge_gap_sec=0)
    state.append_transcription_fragment(disabled, session, "如何实现 Redis 缓存？", 20.0, started_at=3.0)
    session.record_transcript("candidate", "使用 TTL。", started_at=5.0)
    submitted = []
    state.submit_answer_task = lambda task: submitted.append(task) or False
    enabled = SimpleNamespace(auto_detect=True, assist_transcription_merge_gap_sec=0)
    state.append_transcription_fragment(enabled, session, "如何保证 Kafka 的顺序？", 21.0, started_at=8.0)
    state.flush_question_group_now(enabled, session)

    anchor = submitted[0][4]["transcript_anchor"]
    assert _round_text(session, anchor) == [
        [("interviewer", "如何实现 Redis 缓存？"), ("candidate", "使用 TTL。")],
        [("interviewer", "如何保证 Kafka 的顺序？")],
    ]


def test_legacy_task_keeps_all_question_fragments_in_one_round():
    session = Session()
    submitted = []
    state = AssistAsrStateMachine(
        broadcast=lambda _payload: None,
        submit_answer_task=lambda task: submitted.append(task) or False,
        begin_asr_turn=lambda: 1,
        logger=SimpleNamespace(info=lambda *_args: None),
        clock=lambda: 22.0,
    )
    cfg = SimpleNamespace(auto_detect=True, assist_transcription_merge_gap_sec=2.0)
    state.append_transcription_fragment(cfg, session, "什么是 Redis？", 20.0, started_at=3.0)
    state.append_transcription_fragment(cfg, session, "它如何持久化？", 20.3, started_at=5.0, force_flush_tail=True)
    state.flush_question_group_now(cfg, session)

    assert _round_text(session, submitted[0][4]["transcript_anchor"]) == [[
        ("interviewer", "什么是 Redis？"), ("interviewer", "它如何持久化？"),
    ]]


def test_mmpg_group_keeps_setup_with_question_and_splits_next_question_after_late_candidate_final():
    session = Session()
    setup = session.record_transcript("interviewer", "假设你在一家公司。", started_at=1)
    first = session.record_transcript("interviewer", "怎么设计 Redis 缓存？", started_at=2, question_group="group")
    next_question = session.record_transcript("interviewer", "怎么设计消息队列？", started_at=8, question_group="group")
    session.group_transcript_question((setup, first, next_question), split_after_reply=True)
    session.record_transcript("candidate", "我会使用 TTL。", started_at=5)

    assert _round_text(session, first) == [[
        ("interviewer", "假设你在一家公司。"),
        ("interviewer", "怎么设计 Redis 缓存？"),
        ("candidate", "我会使用 TTL。"),
    ]]
    assert _round_text(session, next_question)[-1] == [("interviewer", "怎么设计消息队列？")]


def test_candidate_mmpg_rejects_a_final_from_the_cleared_capture(monkeypatch):
    session = Session(is_recording=True)
    cfg = SimpleNamespace(candidate_asr_enabled=True, candidate_stt_provider=MMPG_STT_PROVIDER)
    monkeypatch.setattr(candidate_mmpg, "_pending", {})
    monkeypatch.setattr(candidate_mmpg, "_capture_epoch", session.transcript_epoch)
    monkeypatch.setattr(candidate_mmpg, "get_session", lambda: session)
    monkeypatch.setattr(candidate_mmpg.pipeline, "get_capture_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", lambda _payload: None)
    candidate_mmpg._on_turn("started", "old")

    session.clear()
    session.begin_transcript_capture()
    session.is_recording = True
    anchor = session.record_transcript("interviewer", "你确定吗？", started_at=1, question_group="current")
    candidate_mmpg._on_transcript("旧录音。", "old")

    monkeypatch.setattr(candidate_mmpg, "_capture_epoch", session.transcript_epoch)
    candidate_mmpg._on_turn("started", "new")
    candidate_mmpg._on_transcript("对。", "new")
    assert session.candidate_transcription_history == ["对。"]
    assert _round_text(session, anchor) == [[("interviewer", "你确定吗？"), ("candidate", "对。")]]


def test_mmpg_reuses_the_prior_interviewer_utterance_in_a_revision(monkeypatch):
    session = Session(is_recording=True)
    cfg = SimpleNamespace(
        auto_detect=True, mmpg=SimpleNamespace(answer_delay_ms=0, followup_window_ms=5000),
    )
    monkeypatch.setattr(pipeline, "_interviewer_streaming", True)
    monkeypatch.setattr(pipeline, "get_session", lambda: session)
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", lambda _payload: None)
    monkeypatch.setattr(mmpg_integration, "get_session", lambda: session)
    monkeypatch.setattr(mmpg_integration, "_capture_epoch", session.transcript_epoch)
    monkeypatch.setattr(mmpg_integration, "_speech_starts", {})
    monkeypatch.setattr(mmpg_integration, "_session_settings", cfg.mmpg)
    monkeypatch.setattr(pipeline, "cancel_mmpg_answer", lambda *_args, **_kwargs: None)
    submitted = []
    monkeypatch.setattr(pipeline, "submit_answer_task", lambda task: submitted.append(task) or True)

    async def run():
        monkeypatch.setattr(mmpg_integration, "_turns", mmpg_integration._new_turns())
        mmpg_integration._turn_event("started", "first")
        mmpg_integration._final_transcript("怎样实现缓存？", "first")
        mmpg_integration._turn_event("started", "tail")
        mmpg_integration._final_transcript("不要结合项目。", "tail")

    asyncio.run(run())

    first, revised = (item[4] for item in submitted)
    earlier = session.transcript_snapshot(first["transcript_anchor"])[0][0]
    revision = session.transcript_snapshot(revised["transcript_anchor"])[0]
    assert earlier.id == revision[0].id
    assert [item.text for item in revision] == ["怎样实现缓存？", "不要结合项目。"]
    assert revision[-1].id == revised["transcript_anchor"]


def test_native_loopback_keeps_its_speaker_when_current_enumeration_misses_it(monkeypatch):
    monkeypatch.setattr(pipeline.AudioCapture, "list_devices", lambda: [])
    assert pipeline._device_is_loopback(_SC_ID_BASE)


def test_short_segment_is_captured_below_publication_threshold(monkeypatch):
    session = Session()
    current = session.record_transcript("interviewer", "解释缓存？", started_at=1, question_group="current")
    runtime = pipeline._new_interviewer_runtime()
    runtime.transcript_epoch = session.transcript_epoch
    runtime.segment_queue.put(pipeline.InterviewerSegment(None, 16000, 3, 4, 1.0, "silence", False))
    runtime.drain_event.set()
    cfg = SimpleNamespace(stt_provider="doubao", position="", language="zh", transcription_min_sig_chars=2)
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "transcribe_segment", lambda *_args, **_kwargs: "好。")
    monkeypatch.setattr(pipeline, "broadcast", lambda _payload: None)

    pipeline._interviewer_asr_worker(runtime, session, cfg)

    assert session.transcription_history == []
    assert _round_text(session, current) == [[("interviewer", "解释缓存？"), ("interviewer", "好。")]]


def test_clear_stops_recording_and_a_new_capture_cannot_reuse_old_speech(monkeypatch):
    session = reset_session()
    session.begin_transcript_capture()
    session.is_recording = True
    old_epoch = session.transcript_epoch
    old_anchor = session.record_transcript("interviewer", "旧问题？", started_at=1)
    cfg = SimpleNamespace(written_exam_mode=True, stt_provider="", assist_stop_answer_wait_sec=0)
    broadcasts = []
    monkeypatch.setattr(routes, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "get_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "broadcast", broadcasts.append)
    monkeypatch.setattr(pipeline.review_integration, "on_assist_start", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline.review_integration, "on_assist_stop", lambda _session: None)

    async def clear_then_start():
        await routes.api_clear()
        assert not session.is_recording
        assert {"type": "recording", "value": False, "candidate_capture_provider": ""} in broadcasts
        await routes.api_start({})
        assert session.is_recording

    try:
        asyncio.run(clear_then_start())
        assert session.record_transcript("candidate", "旧录音。", started_at=2, epoch=old_epoch) is None
        assert session.transcript_snapshot(old_anchor) == ()
    finally:
        reset_session()
