import io
import time

import numpy as np
from services.interview_archive.policy import SimpleRoundBoundaryPolicy


def test_simple_policy_keeps_hints_but_not_new_questions():
    policy = SimpleRoundBoundaryPolicy()
    for text in ("哦对", "好的", "锁", "举个例子", "展开一点"):
        assert policy.decide(text, has_question=True, same_group=False).action == "append_context"
    assert policy.decide("Redis 为什么这么快？", has_question=True).action == "start_next_turn"
    assert policy.decide("下一题，请介绍线程池", has_question=True, same_group=True).action == "start_next_turn"
    assert policy.decide("如何实现？", has_question=True, same_group=True).action == "append_question"


def test_opus_preserves_source_length_and_does_not_write_files(tmp_path):
    import av
    from services.interview_archive.audio import encode_opus
    for size in (32000, 17808, 1):
        payload = encode_opus(np.zeros(size, dtype=np.int16))
        decoded = 0
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        with av.open(io.BytesIO(payload)) as source:
            for frame in source.decode(audio=0):
                decoded += sum(f.samples for f in resampler.resample(frame))
            decoded += sum(f.samples for f in resampler.resample(None))
        assert abs(decoded - size) <= 1
    assert not list(tmp_path.iterdir())


def test_rounds_save_when_closed_with_full_text_and_failed_ai():
    from services.interview_archive.manager import InterviewArchive
    saved = []
    archive = InterviewArchive(17, candidate_enabled=True, save=lambda s: saved.append(s))
    try:
        key = archive.transcript("interviewer", "介绍一下你的项目？", "q1", 1.0, True)
        archive.transcript("candidate", "嗯，我主要负责" + "架构" * 1800, "a1", 2.0, True)
        archive.transcript("interviewer", "锁", "hint", 3.0, True)
        archive.transcript("candidate", "对，还包括并发控制。", "a2", 4.0, True)
        run = archive.start_ai(key, "ai1", {"name": "test", "model": "model-1"})
        archive.transcript("interviewer", "Redis 为什么这么快？", "q2", 5.0, True)
        time.sleep(0.05)
        assert not saved  # closed, but the AI is still running
        archive.finish_ai(run, "failed", error="生成失败")
        assert archive.wait_saved(2)
        assert len(saved) == 1
        assert len(saved[0]["candidate_answer_text"]) > 2000
        assert saved[0]["ai_status"] == "failed"
        assert "锁" in saved[0]["question_text"]
        assert len(saved[0]["utterances"]) == 4
    finally:
        archive.stop(timeout=3)


def test_late_callbacks_keep_their_original_round_and_updates_are_idempotent():
    from services.interview_archive.manager import InterviewArchive
    saved = {}
    archive = InterviewArchive(18, candidate_enabled=True, save=lambda s: saved.update({s["round_id"]: s}))
    try:
        first = archive.transcript("interviewer", "什么是线程？", "q1", 1, True)
        archive.transcript("candidate", "", "answer", 2, False)
        archive.transcript("interviewer", "什么是进程？", "q2", 3, True)
        archive.transcript("candidate", "线程是执行单元", "answer", 2, True)
        assert archive.wait_saved(2)
        old = saved[first]
        assert old["candidate_answer_text"] == "线程是执行单元"
        archive.transcript("candidate", "线程是执行单元", "answer", 2, True)
        assert archive.wait_saved(2)
        assert len(saved) == 1
    finally:
        archive.stop(timeout=3)


def test_storage_upsert_and_media_ownership():
    from services.storage import review, review_archive
    sid = review.create_session(time.time(), True, True)
    record = dict(session_id=sid, round_id="round-test", seq=1, started_at=1, ended_at=2,
                  question_text="问题", candidate_answer_text="回答", reference_answer_text="参考",
                  ai_status="completed", candidate_status="recorded", policy_version="simple-v1",
                  utterances=[], generations=[], media={"interviewer": [], "candidate": []}, assets={})
    tid = review_archive.save_round(record)
    assert review_archive.save_round(record) == tid
    assert review.get_session_detail(sid)["turn_count"] == 1
    assert review_archive.get_archive(sid, tid)["round_id"] == "round-test"
    assert review_archive.get_archive(sid + 1, tid) is None
    assert review_archive.asset_path(sid, tid, "../../config.json") is None


def test_real_opus_playlists_use_sample_windows_and_keep_failed_asr_audio():
    from services.interview_archive.audio import CapturedAudio
    from services.interview_archive.manager import InterviewArchive

    saved = []
    archive = InterviewArchive(19, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic()
    try:
        for index in range(20):
            block = CapturedAudio(np.full(1024, 0.07, dtype=np.float32), "mic", index * 1024, anchor)
            archive.capture(block, "candidate")
        assert archive.audio.flush()
        archive.transcript("interviewer", "第一题？", "q1", anchor - 0.1, True)
        archive.transcript("candidate", "", "a1", anchor, False)
        archive.discard("a1", audio_end=anchor + 1.1)
        archive.transcript("interviewer", "第二题？", "q2", anchor + 1.2, True)
        assert archive.wait_saved(2)
        assert saved[0]["candidate_status"] == "transcription_failed"
        assert sum(c["sample_count"] for c in saved[0]["media"]["candidate"]) == 17600
        assert not saved[0]["media"]["interviewer"]
        assert saved[0]["assets"]
    finally:
        archive.stop()


def test_round_write_retries_and_does_not_report_saved_on_failure():
    from services.interview_archive.manager import InterviewArchive

    attempts = []
    def unavailable(snapshot):
        attempts.append(snapshot["round_id"])
        raise OSError("simulated full disk")

    archive = InterviewArchive(20, candidate_enabled=False, save=unavailable)
    archive.transcript("interviewer", "测试写入失败？", "q1", time.monotonic(), True)
    assert archive.stop()
    assert len(attempts) == 3 and len(set(attempts)) == 1
    status = archive.snapshot()
    assert status["persisted_turn_count"] == 0
    assert status["pending_round_count"] == 0
    assert "保存失败" in status["warning"]


def test_candidate_that_starts_before_question_final_is_assigned_by_capture_time():
    from services.interview_archive.manager import InterviewArchive

    saved = []
    archive = InterviewArchive(21, candidate_enabled=True, save=saved.append)
    archive.transcript("interviewer", "介绍一下线程？", "q1", 1)
    archive.transcript("candidate", "", "a1", 2, False)
    archive.transcript("interviewer", "", "q2", 3, False)
    archive.transcript("candidate", "", "a2", 4, False)
    archive.transcript("interviewer", "如何排查死锁？", "q2", 3)
    archive.transcript("candidate", "第一轮回答", "a1", 2)
    archive.transcript("candidate", "第二轮回答", "a2", 4)
    assert archive.stop()
    assert [r["candidate_answer_text"] for r in saved] == ["第一轮回答", "第二轮回答"]


def test_media_endpoint_requires_auth_and_verifies_session_ownership(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    from services.interview_archive.audio import encode_opus
    from services.storage import review, review_archive

    sid = review.create_session(time.time(), True, True)
    asset_id = "f" * 64
    payload = encode_opus(np.zeros(320, dtype=np.int16))
    tid = review_archive.save_round(dict(session_id=sid, round_id="audio-test", seq=1, started_at=1, ended_at=2,
                  question_text="问题", candidate_answer_text="回答", reference_answer_text="参考",
                  ai_status="completed", candidate_status="recorded", policy_version="simple-v1",
                  utterances=[], generations=[], media={"interviewer": [], "candidate": []}, assets={asset_id: payload}))
    monkeypatch.setattr(main, "is_auth_disabled", lambda: False)
    monkeypatch.setattr(main, "verify_token", lambda token: token == "test-token")
    client = TestClient(main.app)
    route = f"/api/review/sessions/{sid}/turns/{tid}/audio/{asset_id}"
    assert client.get(route).status_code == 401
    headers = {"Authorization": "Bearer test-token", "Range": "bytes=0-3"}
    response = client.get(route, headers=headers)
    assert response.status_code == 206
    assert response.content == b"OggS"
    assert client.get(route.replace(f"sessions/{sid}/", f"sessions/{sid+1}/"), headers=headers).status_code == 404
