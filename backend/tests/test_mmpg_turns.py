from types import SimpleNamespace
from unittest.mock import Mock

from api.assist.mmpg_turns import MmpgTurns


class Clock:
    now = 0.0

    def __init__(self):
        self.timers = []

    def call_later(self, delay, callback):
        timer = SimpleNamespace(when=self.now + delay, callback=callback, cancelled=False)
        timer.cancel = lambda: setattr(timer, "cancelled", True)
        self.timers.append(timer)
        return timer

    def at(self, when):
        while self.timers and min(t.when for t in self.timers) <= when:
            timer = min(self.timers, key=lambda item: item.when)
            self.timers.remove(timer)
            self.now = timer.when
            if not timer.cancelled:
                timer.callback()
        self.now = when


def setup(delay_ms=1500, followup_ms=1500):
    clock = Clock()
    submit = Mock(return_value=True)
    cancel = Mock(side_effect=lambda request: request.completed_qa_id)
    turns = MmpgTurns(
        submit=submit, cancel=cancel, publish=Mock(),
        enabled=lambda: True, clock=lambda: clock.now, call_later=clock.call_later,
        answer_delay_ms=delay_ms, followup_window_ms=followup_ms,
    )
    return turns, clock, submit, cancel


def test_cplusplus_followup_at_63_ms_submits_one_complete_prompt():
    turns, clock, submit, _ = setup()
    turns.final("g/1/1", "你熟悉 C++ 吗？")
    clock.at(0.063)
    turns.started("g/1/2")
    clock.at(4)
    submit.assert_not_called()
    turns.final("g/1/2", "讲讲编译过程。")
    clock.at(5.5)
    request = submit.call_args.args[0]
    assert request.text == "你熟悉 C++ 吗？\n讲讲编译过程。"
    assert request.traces == ("g/1/1", "g/1/2")
    assert submit.call_count == 1


def test_three_revisions_restart_wait_and_request_window():
    turns, clock, submit, cancel = setup()
    turns.final("1", "第一段")
    clock.at(1.5)
    first = submit.call_args.args[0]
    turns.request_started(first, clock.now)
    clock.at(2)
    turns.started("2")
    cancel.assert_called_once_with(first)
    clock.at(5)
    turns.final("2", "第二段")
    clock.at(6.5)
    second = submit.call_args.args[0]
    assert second.group_id == first.group_id and second.revision == 2
    turns.request_started(second, clock.now)
    clock.at(7)
    turns.started("3")
    assert cancel.call_args.args == (second,)
    turns.final("3", "第三段")
    clock.at(8.5)
    third = submit.call_args.args[0]
    assert third.revision == 3
    assert third.text == "第一段\n第二段\n第三段"


def test_completed_answer_is_retained_when_followup_arrives_before_window_end():
    turns, clock, submit, _ = setup(delay_ms=0)
    turns.final("1", "第一问")
    first = submit.call_args.args[0]
    turns.request_started(first, clock.now)
    first.completed_qa_id = "qa-1"
    clock.at(1)
    turns.started("2")
    turns.final("2", "追加条件")
    assert submit.call_count == 2
    assert submit.call_args.args[0].updates_qa_id == "qa-1"


def test_queued_task_is_cancelled_and_empty_followup_rearms_original():
    turns, clock, submit, cancel = setup()
    turns.final("1", "题目")
    clock.at(1.5)
    request = submit.call_args.args[0]
    clock.at(8)
    turns.started("2")
    turns.started("2")
    cancel.assert_called_once_with(request)
    turns.cancelled("2")
    clock.at(9.49)
    assert submit.call_count == 1
    clock.at(9.5)
    assert submit.call_args.args[0].text == "题目"
    assert submit.call_count == 2


def test_closed_window_can_start_new_group_and_clear_drops_timer():
    turns, clock, submit, cancel = setup(delay_ms=0, followup_ms=0)
    turns.final("1", "题目一")
    first = submit.call_args.args[0]
    turns.request_started(first, clock.now)
    turns.final("2", "题目二")
    assert first.group_id != submit.call_args.args[0].group_id
    cancel.assert_not_called()

    turns, clock, submit, _ = setup()
    turns.final("1", "题目")
    turns.clear()
    clock.at(4)
    submit.assert_not_called()


def test_window_includes_deadline_and_deduplicates_final_events():
    turns, clock, submit, cancel = setup()
    turns.final("1", "第一句")
    clock.call_later(1.5, lambda: turns.started("2"))
    clock.at(1.5)
    submit.assert_not_called()
    turns.final("2", "第二句")
    turns.final("2", "重复最终事件")
    clock.at(3)
    request = submit.call_args.args[0]
    turns.request_started(request, clock.now)
    clock.call_later(1.5, lambda: turns.started("3"))
    clock.at(4.5)
    cancel.assert_called_once_with(request)
    turns.final("3", "第三句")
    clock.at(6)
    assert submit.call_args.args[0].text == "第一句\n第二句\n第三句"


def test_empty_followup_keeps_completed_answer_and_reset_rejects_late_final():
    turns, clock, submit, cancel = setup()
    turns.final("1", "第一问")
    clock.at(1.5)
    request = submit.call_args.args[0]
    turns.request_started(request, clock.now)
    request.completed_qa_id = "completed"
    turns.request_finished(request)
    clock.at(2)
    turns.started("2")
    turns.final("2", "")
    clock.at(5)
    assert submit.call_count == 1
    turns.started("3")
    turns.clear(cancel_running=True)
    turns.final("3", "迟到结果")
    clock.at(10)
    assert submit.call_count == 1
