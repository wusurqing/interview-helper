from types import SimpleNamespace
from unittest.mock import Mock

from api.assist.asr_state import AssistAsrStateMachine


def make_state():
    return AssistAsrStateMachine(
        broadcast=Mock(), submit_answer_task=Mock(), begin_asr_turn=lambda: 7,
        record_asr_turn=Mock(), is_high_churn_submission=lambda *_: False,
        logger=Mock(),
    )


def test_mmpg_final_submits_exact_complete_question_immediately():
    state = make_state()
    session = SimpleNamespace(add_transcription=Mock(), capture_is_loopback=True)
    question = "请解释 child process。也请给出几个例子，不要使用线程。"
    state.accept_final_turn(SimpleNamespace(auto_detect=True), session, question)
    session.add_transcription.assert_called_once_with(question)
    task = state.submit_answer_task.call_args.args[0]
    assert task[0] == question
    assert task[3] == "conversation_loopback"
    assert "dispatch_after_mono" not in task[4]
    assert state.pending_group is None


def test_mmpg_records_finals_but_does_not_answer_backchannels_or_when_disabled():
    state = make_state()
    session = SimpleNamespace(add_transcription=Mock(), capture_is_loopback=False)
    state.accept_final_turn(SimpleNamespace(auto_detect=True), session, "嗯。")
    state.accept_final_turn(SimpleNamespace(auto_detect=False), session, "怎么设计缓存？")
    state.accept_final_turn(SimpleNamespace(auto_detect=True), session, "  ")
    assert session.add_transcription.call_count == 2
    state.submit_answer_task.assert_not_called()
