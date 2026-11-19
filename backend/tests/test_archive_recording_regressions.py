import threading
import time

import numpy as np

from services.interview_archive.audio import RATE, CapturedAudio
from services.interview_archive.manager import InterviewArchive


def _offer(archive, speaker, capture_id, anchor, start, count, amplitude=0.05):
    archive.capture(CapturedAudio(np.full(count, amplitude, dtype=np.float32), capture_id, start, anchor), speaker)


def test_normal_stop_finalizes_complete_audio_once():
    saved = []
    completed = []
    archive = InterviewArchive(30, candidate_enabled=False, save=saved.append, on_complete=lambda: completed.append(True))
    anchor = time.monotonic() - 1
    try:
        _offer(archive, "interviewer", "loopback", anchor, 0, 3200)
        archive.transcript("interviewer", "第一道题？", "q1", anchor, audio_end=anchor + 0.2)
        assert archive.stop()
        assert archive.stop()
        assert len(completed) == len(saved) == 1
        assert sum(clip["sample_count"] for clip in saved[0]["media"]["interviewer"]) == 3200
        assert archive.snapshot()["archive_status"] == "complete"
        assert not archive.snapshot()["warning"]
    finally:
        archive.stop()


def test_late_encoding_keeps_closed_round_pending_until_media_is_ready(monkeypatch):
    from services.interview_archive import audio

    started = threading.Event()
    release = threading.Event()
    encode = audio.encode_opus

    def delayed_encode(pcm):
        started.set()
        assert release.wait(6)
        return encode(pcm)

    monkeypatch.setattr(audio, "encode_opus", delayed_encode)
    saved = []
    archive = InterviewArchive(31, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 3
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        _offer(archive, "candidate", "mic", anchor, RATE // 2, 3200)
        archive.transcript("candidate", "第一道题的回答", "a1", anchor + 0.5, audio_end=anchor + 0.7)
        archive.transcript("interviewer", "下一题，什么是线程？", "q2", anchor + 1)
        assert started.wait(2)
        assert not archive.wait_saved(2.2)
        assert not saved
        assert archive.snapshot()["pending_round_count"] == 1
        release.set()
        assert archive.wait_saved(3)
        assert sum(clip["sample_count"] for clip in saved[0]["media"]["candidate"]) == 3200
    finally:
        release.set()
        archive.stop()


def test_dual_input_quiet_answer_is_playable_from_disk_after_cloud_final(tmp_path, monkeypatch):
    import av
    from services.storage import review, review_archive

    monkeypatch.setattr(review_archive, "media_root", lambda: tmp_path)
    sid = review.create_session(time.time(), True, True)
    archive = InterviewArchive(sid, candidate_enabled=True)
    anchor = time.monotonic() - 3
    try:
        _offer(archive, "interviewer", "loopback", anchor, 0, RATE)
        archive.transcript("interviewer", "为什么？", "q1", anchor, audio_end=anchor + 1)
        archive.transcript("candidate", "", "a1", anchor + 0.5, is_final=False, cloud=True)
        archive.transcript("candidate", "有效但很轻的回答", "a1", anchor + 0.5, cloud=True)
        _offer(archive, "candidate", "mic", anchor, RATE // 2, RATE, amplitude=0.0059)
        assert archive.stop()
        turn = review.get_session_detail(sid)["turns"][0]
        clip, = turn["media"]["candidate"]
        assert clip["sample_count"] == RATE
        assert turn["audio_status"]["candidate"] == {"state": "approximate", "duration_ms": 1000}
        path = review_archive.asset_path(sid, turn["id"], clip["asset_id"])
        with av.open(str(path)) as source:
            decoded_duration = sum(frame.samples / frame.sample_rate for frame in source.decode(audio=0))
        assert abs(decoded_duration - 1.0) < 0.001
        assert turn["candidate_status"] == "recorded"
    finally:
        archive.stop()


def test_single_mic_sample_gap_cannot_extend_a_previous_candidate_utterance():
    saved = []
    archive = InterviewArchive(33, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 24
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        _offer(archive, "candidate", "shared-mic", anchor, RATE, RATE // 2)
        assert archive.audio.flush()
        archive.transcript("candidate", "我的第一段回答", "a1", anchor + 1.1, cloud=True)
        _offer(archive, "interviewer", "shared-mic", anchor, RATE * 2, RATE // 2)
        archive.transcript("interviewer", "继续", "hint", anchor + 2, audio_end=anchor + 2.5)
        _offer(archive, "candidate", "shared-mic", anchor, RATE * 21, RATE // 2)
        archive.transcript("candidate", "我补充第二段", "a2", anchor + 21.1, cloud=True)
        assert archive.stop()
        turn, = saved
        spans = [(clip["sample_start"], clip["sample_count"]) for clip in turn["media"]["candidate"]]
        assert spans == [(RATE, RATE // 2), (RATE * 21, RATE // 2)]
        assert sum(clip["sample_count"] for clip in turn["media"]["interviewer"]) == RATE // 2
    finally:
        archive.stop()


def test_dual_input_long_answer_crosses_next_question_without_losing_its_tail():
    saved = []
    archive = InterviewArchive(34, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 30
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        for second in range(1, 21):
            _offer(archive, "candidate", "mic", anchor, second * RATE, RATE,
                   amplitude=0 if second in (5, 10, 15) else 0.05)
            assert archive.audio.flush()
            if second == 1:
                archive.transcript("candidate", "", "a1", anchor + 1.2, is_final=False, cloud=True)
        _offer(archive, "interviewer", "loopback", anchor, RATE * 20, RATE // 2)
        archive.transcript("interviewer", "下一题，什么是线程？", "q2", anchor + 20, audio_end=anchor + 20.5)
        _offer(archive, "candidate", "mic", anchor, RATE * 21, RATE // 2)
        archive.transcript("candidate", "二十秒的完整回答", "a1", anchor + 1.2, cloud=True)
        _offer(archive, "candidate", "mic", anchor, RATE * 23, RATE // 2)
        archive.transcript("candidate", "第二题的回答", "a2", anchor + 23.1, cloud=True)
        assert archive.stop()
        first, second = sorted(saved, key=lambda turn: turn["seq"])
        first_audio, second_audio = first["media"]["candidate"], second["media"]["candidate"]
        assert first["candidate_answer_text"] == "二十秒的完整回答"
        assert second["candidate_answer_text"] == "第二题的回答"
        assert first_audio[0]["sample_start"] == RATE
        assert sum(clip["sample_count"] for clip in first_audio) == RATE * 20 + RATE // 2
        assert second_audio[0]["sample_start"] == RATE * 23
        assert sum(clip["sample_count"] for clip in second_audio) == RATE // 2
        assert sum(clip["sample_count"] for clip in second["media"]["interviewer"]) == RATE // 2
    finally:
        archive.stop()


def test_late_ai_update_keeps_saved_audio_and_coverage_after_media_memory_is_released():
    saved = []
    archive = InterviewArchive(35, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 10
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        _offer(archive, "candidate", "mic", anchor, RATE, RATE // 2)
        archive.transcript("candidate", "第一题的回答", "a1", anchor + 1.1, cloud=True)
        archive.transcript("interviewer", "下一题，什么是线程？", "q2", anchor + 5)
        assert archive.wait_saved()
        assert archive.audio.memory_bytes == 0
        archive.start_ai(saved[0]["round_id"], "late", {"name": "测试模型"})
        archive.ai_event("late", {"type": "answer_done", "answer": "迟到的参考答案"})
        assert archive.wait_saved()
        assert len(saved) == 2
        assert saved[1]["media"] == saved[0]["media"]
        assert saved[1]["audio_status"] == saved[0]["audio_status"]
        assert saved[1]["reference_answer_text"] == "迟到的参考答案"
    finally:
        archive.stop()


def test_encode_failure_is_reported_only_for_the_affected_role(monkeypatch):
    from services.interview_archive import audio

    encode = audio.encode_opus

    def fail_candidate(pcm):
        if pcm[0] < 0:
            raise RuntimeError("synthetic encoder failure")
        return encode(pcm)

    monkeypatch.setattr(audio, "encode_opus", fail_candidate)
    saved = []
    archive = InterviewArchive(36, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 2
    try:
        _offer(archive, "interviewer", "loopback", anchor, 0, RATE // 2)
        archive.transcript("interviewer", "第一道题？", "q1", anchor, audio_end=anchor + 0.5)
        _offer(archive, "candidate", "mic", anchor, RATE, RATE // 2, amplitude=-0.05)
        archive.transcript("candidate", "回答有文字", "a1", anchor + 1, audio_end=anchor + 1.5)
        assert archive.stop()
        turn, = saved
        assert turn["audio_status"]["interviewer"]["state"] == "ready"
        assert turn["audio_status"]["candidate"] == {"state": "missing", "reason": "encode_failed", "duration_ms": 0}
        assert turn["candidate_answer_text"] == "回答有文字"
        assert archive.snapshot()["warning"] == "第 1 题候选人录音缺失"
    finally:
        archive.stop()


def test_dual_input_timeout_keeps_audio_after_last_successful_transcript():
    saved = []
    archive = InterviewArchive(37, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 5
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        _offer(archive, "candidate", "mic", anchor, RATE, RATE // 2)
        archive.transcript("candidate", "已转写的话", "a1", anchor + 1, audio_end=anchor + 1.5)
        _offer(archive, "candidate", "mic", anchor, 3 * RATE, RATE // 2, amplitude=0.003)
        archive.stt_timeout("candidate")
        assert archive.stop()
        turn, = saved
        assert turn["candidate_answer_text"] == "已转写的话"
        assert turn["transcription_timeouts"] == ["candidate"]
        assert turn["audio_status"]["candidate"] == {"state": "approximate", "duration_ms": 1000}
        assert [clip["sample_start"] for clip in turn["media"]["candidate"]] == [RATE, 3 * RATE]
    finally:
        archive.stop()


def test_device_sample_loss_marks_turn_partial_without_discarding_audio():
    saved = []
    archive = InterviewArchive(38, candidate_enabled=False, save=saved.append)
    anchor = time.monotonic() - 2
    try:
        _offer(archive, "interviewer", "loopback", anchor, 0, RATE // 2)
        samples = CapturedAudio(np.full(RATE // 2, 0.05, dtype=np.float32), "loopback", RATE // 2, anchor,
                                discontinuity=True)
        archive.capture(samples, "interviewer")
        archive.transcript("interviewer", "回环采集有断点吗？", "q1", anchor, audio_end=anchor + 1)
        assert archive.stop()
        turn, = saved
        assert turn["audio_status"]["interviewer"] == {"state": "partial", "reason": "capture_gap", "duration_ms": 1000}
        assert sum(clip["sample_count"] for clip in turn["media"]["interviewer"]) == RATE
        assert archive.snapshot()["warning"] == "第 1 题面试官部分录音缺失"
    finally:
        archive.stop()


def test_quiet_answer_with_only_a_delayed_final_plays_its_entire_round_audio():
    saved = []
    archive = InterviewArchive(39, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 6
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        for second in range(1, 5):
            _offer(archive, "candidate", "mic", anchor, second * RATE, RATE, amplitude=0.003)
            assert archive.audio.flush()
        archive.transcript("candidate", "很轻声的长回答", "a1", anchor + 4.8, cloud=True)
        assert archive.stop()
        turn, = saved
        assert turn["audio_status"]["candidate"] == {"state": "approximate", "duration_ms": 4000}
        assert turn["media"]["candidate"][0]["sample_start"] == RATE
    finally:
        archive.stop()


def test_final_only_quiet_answer_after_next_question_updates_its_original_card():
    saved = {}
    archive = InterviewArchive(40, candidate_enabled=True, save=lambda turn: saved.update({turn["seq"]: turn}))
    anchor = time.monotonic() - 8
    try:
        first = archive.transcript("interviewer", "第一道题？", "q1", anchor)
        for second in range(1, 6):
            _offer(archive, "candidate", "mic", anchor, second * RATE, RATE, amplitude=0.003 if second < 4 else 0)
            assert archive.audio.flush()
        archive.transcript("interviewer", "下一题，什么是线程？", "q2", anchor + 5)
        assert archive.wait_saved()
        assert saved[1]["candidate_answer_text"] == ""
        _offer(archive, "candidate", "mic", anchor, RATE * 6, RATE, amplitude=0)
        assert archive.transcript("candidate", "第一题的迟到回答", "late", anchor + 6.9, cloud=True) == first
        assert archive.stop()
        assert saved[1]["candidate_answer_text"] == "第一题的迟到回答"
        assert saved[1]["audio_status"]["candidate"]["state"] == "approximate"
        assert saved[1]["audio_status"]["candidate"]["duration_ms"] >= 3000
        assert not saved[2]["candidate_answer_text"] and not saved[2]["media"]["candidate"]
    finally:
        archive.stop()


def test_cloud_started_does_not_snap_back_to_background_crossing_previous_turn():
    saved = {}
    archive = InterviewArchive(41, candidate_enabled=True, save=lambda turn: saved.update({turn["seq"]: turn}))
    anchor = time.monotonic() - 8
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        for second in range(7):
            _offer(archive, "candidate", "mic", anchor, second * RATE, RATE, amplitude=0.007)
            assert archive.audio.flush()
        second_turn = archive.transcript("interviewer", "下一题，什么是线程？", "q2", anchor + 5)
        assert archive.transcript("candidate", "", "a2", anchor + 6.5, is_final=False, cloud=True) == second_turn
        archive.transcript("candidate", "第二题的回答", "a2", anchor + 6.5, cloud=True)
        assert archive.stop()
        assert not saved[1]["candidate_answer_text"]
        assert saved[2]["candidate_answer_text"] == "第二题的回答"
        assert saved[2]["media"]["candidate"][0]["sample_start"] >= 5 * RATE
    finally:
        archive.stop()


def test_disconnected_candidate_keeps_an_untranscribed_tail_after_a_successful_final():
    saved = []
    archive = InterviewArchive(42, candidate_enabled=True, save=saved.append)
    anchor = time.monotonic() - 6
    try:
        archive.transcript("interviewer", "第一道题？", "q1", anchor)
        _offer(archive, "candidate", "mic", anchor, RATE, RATE // 2)
        assert archive.audio.flush()
        archive.transcript("candidate", "已转写的话", "a1", anchor + 1.2, cloud=True)
        _offer(archive, "candidate", "mic", anchor, RATE * 3, RATE // 2)
        archive.end_capture("candidate")
        assert archive.stop()
        turn, = saved
        assert turn["candidate_answer_text"] == "已转写的话"
        assert turn["audio_status"]["candidate"] == {"state": "approximate", "duration_ms": 1000}
        assert [clip["sample_start"] for clip in turn["media"]["candidate"]] == [RATE, 3 * RATE]
    finally:
        archive.stop()
