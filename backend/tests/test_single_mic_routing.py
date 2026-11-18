from __future__ import annotations

import numpy as np
import pytest

from api.assist import single_mic


def pcm(value: float, size: int = 3) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


def test_switch_drains_old_role_before_listening_to_new_role():
    session = single_mic.SingleMicSession("take-1", device_id=0)
    assert session.snapshot() == {
        "id": "take-1",
        "device_id": 0,
        "role": "interviewer",
        "switching": False,
        "revision": 0,
    }
    session.offer_audio(pcm(1))
    session.offer_audio(pcm(2))
    operation = session.begin_switch("candidate")

    assert session.snapshot()["switching"]
    assert session.begin_switch("candidate") is operation
    assert not operation.streaming
    assert not operation.wait(0)
    assert session.interviewer_input.take_barrier() is None
    with pytest.raises(single_mic.SingleMicConflict):
        session.begin_switch("interviewer")

    session.offer_audio(pcm(3))
    np.testing.assert_array_equal(
        session.interviewer_input.drain_audio_chunks(timeout=0, max_chunks=1), [pcm(1)],
    )
    assert session.interviewer_input.take_barrier() is None
    np.testing.assert_array_equal(
        session.interviewer_input.drain_audio_chunks(timeout=0), [pcm(2)],
    )
    assert session.interviewer_input.take_barrier() is operation
    assert session.interviewer_input.take_barrier() is None
    assert session.candidate_input.drain_audio_chunks(timeout=0) == []
    operation.complete()
    assert operation.wait(0)
    assert session.finish_switch(operation, success=True)

    session.offer_audio(pcm(4))
    assert session.begin_switch("candidate") is None
    assert session.snapshot()["role"] == "candidate"
    assert session.interviewer_input.drain_audio_chunks(timeout=0) == []
    np.testing.assert_array_equal(
        session.candidate_input.drain_audio_chunks(timeout=0), [pcm(4)],
    )


def test_rtc_receives_silence_during_handoff_and_in_inactive_role():
    interviewer_frames, candidate_frames = [], []
    session = single_mic.SingleMicSession(
        "take-2",
        device_id=0,
        interviewer_stream=interviewer_frames.append,
    )
    assert session.set_stream("candidate", candidate_frames.append)

    session.offer_audio(pcm(1, 2))
    operation = session.begin_switch("candidate")
    assert operation.streaming
    assert operation.from_role == "interviewer"
    assert operation.to_role == "candidate"
    assert session.interviewer_input.take_barrier() is None
    session.offer_audio(pcm(2, 1))
    operation.complete()
    assert session.finish_switch(operation, success=True)
    session.offer_audio(pcm(3, 4))

    for actual, expected in zip(interviewer_frames, [pcm(1, 2), pcm(0, 1), pcm(0, 4)], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(candidate_frames, [pcm(0, 2), pcm(0, 1), pcm(3, 4)], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert session.interviewer_input.drain_audio_chunks(timeout=0) == []
    assert session.candidate_input.drain_audio_chunks(timeout=0) == []


def test_timed_out_switch_keeps_current_role_and_late_completion_cannot_apply():
    session = single_mic.SingleMicSession("take-3", device_id=0)
    operation = session.begin_switch("candidate")
    assert not operation.wait(0)
    assert session.finish_switch(operation, success=False)

    next_attempt = session.begin_switch("candidate")
    assert next_attempt is not operation
    operation.complete()
    assert not operation.wait(0)
    assert not session.finish_switch(operation, success=True)
    assert session.snapshot()["role"] == "interviewer"
    assert session.interviewer_input.take_barrier() is next_attempt


def test_pause_and_close_wake_pending_switch_and_block_capture():
    session = single_mic.SingleMicSession("take-4", device_id=0)
    session.offer_audio(pcm(1))
    operation = session.begin_switch("candidate")
    session.pause()
    session.offer_audio(pcm(2))

    assert not operation.wait(0)
    assert not session.finish_switch(operation, success=True)
    assert not session.snapshot()["switching"]
    assert not session.interviewer_input.is_running
    assert session.interviewer_input.take_barrier() is None
    np.testing.assert_array_equal(session.interviewer_input.drain_audio_chunks(timeout=0), [pcm(1)])
    with pytest.raises(single_mic.SingleMicConflict):
        session.begin_switch("candidate")

    assert session.resume()
    operation = session.begin_switch("candidate")
    session.close()
    assert not operation.wait(0)
    assert not session.finish_switch(operation, success=True)
    assert not session.resume()
    assert not session.set_stream("candidate", lambda _: None)


def test_input_caps_buffer_and_drops_oldest_audio():
    session = single_mic.SingleMicSession("take-5", device_id=0)
    for index in range(941):
        session.offer_audio(pcm(index))
    assert session.interviewer_input.dropped_chunks_count == 1
    np.testing.assert_array_equal(
        session.interviewer_input.drain_audio_chunks(timeout=0, max_chunks=1),
        [pcm(1)],
    )


def test_required_rtc_disconnect_gates_both_roles_until_stream_reattaches():
    candidate_frames = []
    session = single_mic.SingleMicSession(
        "take-6", device_id=0, required_streams=frozenset({"candidate"}),
    )
    assert not session.ready
    session.offer_audio(pcm(1))
    assert session.interviewer_input.drain_audio_chunks(timeout=0) == []
    with pytest.raises(single_mic.SingleMicConflict, match="连接"):
        session.begin_switch("candidate")

    assert session.set_stream("candidate", candidate_frames.append)
    assert session.ready
    session.offer_audio(pcm(2))
    np.testing.assert_array_equal(session.interviewer_input.drain_audio_chunks(timeout=0), [pcm(2)])
    np.testing.assert_array_equal(candidate_frames, [pcm(0)])

    operation = session.begin_switch("candidate")
    assert session.interviewer_input.take_barrier() is operation
    session.set_stream("candidate", None)
    assert not operation.wait(0)
    assert not session.finish_switch(operation, success=True)
    assert session.snapshot()["role"] == "interviewer"
    assert not session.ready
    session.offer_audio(pcm(3))
    assert session.interviewer_input.drain_audio_chunks(timeout=0) == []

    assert session.set_stream("candidate", candidate_frames.append)
    operation = session.begin_switch("candidate")
    assert session.interviewer_input.take_barrier() is operation
    operation.complete()
    assert session.finish_switch(operation, success=True)
    session.set_stream("candidate", None)
    session.offer_audio(pcm(4))
    assert session.candidate_input.drain_audio_chunks(timeout=0) == []
    assert session.set_stream("candidate", candidate_frames.append)
    session.offer_audio(pcm(5))
    np.testing.assert_array_equal(candidate_frames, [pcm(0), pcm(5)])
    assert session.snapshot()["role"] == "candidate"


def test_required_rtc_connecting_leg_keeps_attached_leg_fed_with_silence():
    interviewer_frames, candidate_frames = [], []
    session = single_mic.SingleMicSession(
        "take-7",
        device_id=0,
        interviewer_stream=interviewer_frames.append,
        required_streams=frozenset({"interviewer", "candidate"}),
    )
    session.offer_audio(pcm(1))
    session.set_stream("candidate", candidate_frames.append)
    session.offer_audio(pcm(2))
    np.testing.assert_array_equal(interviewer_frames, [pcm(0), pcm(2)])
    np.testing.assert_array_equal(candidate_frames, [pcm(0)])
