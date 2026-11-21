"""Collect continuous MMPG utterances before and during answer generation."""

import asyncio
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable
from uuid import uuid4

_MAX_SEEN_TRACES = 2048


@dataclass
class MmpgRequest:
    group_id: str
    revision: int
    text: str
    traces: tuple[str, ...]
    final_at: float
    updates_qa_id: str | None
    transcript_anchor: str | None = field(default=None, kw_only=True)
    transcript_anchors: tuple[str, ...] = field(default=(), kw_only=True)
    completed_qa_id: str | None = None
    cancelled: bool = False
    on_started: Callable[[float], None] | None = None
    on_finished: Callable[[], None] | None = None
    shown_qa_id: str | None = None
    seq: int | None = None
    started_at: float | None = None
    finished: bool = False
    loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)
    stream_read: asyncio.Task | None = field(default=None, repr=False)

    def cancel(self) -> bool:
        with self.lock:
            if self.cancelled or self.completed_qa_id:
                return False
            self.cancelled = True
            if self.stream_read is not None and self.loop is not None and not self.loop.is_closed():
                with suppress(RuntimeError):
                    self.loop.call_soon_threadsafe(self.stream_read.cancel)
            return True


@dataclass
class _Question:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    utterances: dict[str, tuple[int, str]] = field(default_factory=dict)
    transcript_anchors: dict[str, str] = field(default_factory=dict)
    speaking: dict[str, int] = field(default_factory=dict)
    final_at: float = 0
    revision: int = 0
    request: MmpgRequest | None = None
    followup_until: float | None = None
    updates_qa_id: str | None = None
    waiting_for_new_text: bool = False


class MmpgTurns:
    def __init__(self, *, submit, cancel, publish, enabled, clock, call_later,
                 answer_delay_ms, followup_window_ms, log=None, cancel_on_clear=None, record_final=None,
                 allow_answer=lambda: True, max_seen: int = _MAX_SEEN_TRACES):
        self._submit = submit
        self._cancel = cancel
        self._cancel_on_clear = cancel_on_clear or cancel
        self._publish = publish
        self._enabled = enabled
        self._allow_answer = allow_answer
        self._record_final = record_final
        self._clock = clock
        self._call_later = call_later
        self._delay = answer_delay_ms / 1000
        self._followup = followup_window_ms / 1000
        self._log = log or (lambda *_args, **_fields: None)
        self._group = None
        self._timer = None
        self._timer_version = 0
        self._order = 0
        self._max_seen = max(1, int(max_seen))
        self._seen: OrderedDict[str, None] = OrderedDict()
        self.phase = "idle"

    def _mark_seen(self, trace: str) -> None:
        if trace in self._seen:
            self._seen.move_to_end(trace)
            return
        self._seen[trace] = None
        while len(self._seen) > self._max_seen:
            self._seen.popitem(last=False)

    def _set_phase(self, phase):
        if self.phase != phase:
            self.phase = phase
            self._publish(phase)

    def _cancel_timer(self):
        self._timer_version += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule(self, delay, callback, *, allow_opening_at_deadline=False):
        self._cancel_timer()
        version = self._timer_version

        def fire():
            if version != self._timer_version:
                return
            self._timer = None
            callback()

        # Yield one event-loop turn at the deadline so a ready speech event
        # can invalidate the timer before it submits an answer.
        if allow_opening_at_deadline:
            def due():
                if version == self._timer_version:
                    self._timer = self._call_later(0, fire)

            self._timer = self._call_later(delay, due)
        else:
            self._timer = self._call_later(delay, fire)

    def clear(self, *, cancel_running=False):
        group, self._group = self._group, None
        self._cancel_timer()
        if group:
            for trace in group.speaking:
                self._mark_seen(trace)
            if group.request and (cancel_running or group.followup_until is None):
                self._cancel_on_clear(group.request)
            self._log("group_cleared", group=group.id)
        self._set_phase("idle")

    def disable_automatic_answers(self):
        # Detach the pending group without cancelling submitted work or marking active speech as seen.
        self._group = None
        self._cancel_timer()
        self._set_phase("idle")

    def finish_role(self):
        group = self._group
        if group is None or group.speaking:
            return
        self._cancel_timer()
        if group.request is None and group.utterances and not group.waiting_for_new_text:
            self._answer(group)
        if self._group is group:
            self._group = None
            self._set_phase("idle")

    def request_finished(self, request):
        request.finished = True
        group = self._group
        if group and group.request is request:
            if group.followup_until is None or self._clock() > group.followup_until:
                group.request = None
                self.clear()
            else:
                self._set_phase("idle")

    def started(self, trace):
        if not self._enabled():
            self.clear()
            self._mark_seen(trace)
            return
        if trace in self._seen:
            return
        if not self._allow_answer():
            return
        group = self._group
        if group and trace in group.speaking:
            return
        now = self._clock()
        if group and group.request and group.request.started_at is not None and group.followup_until is None:
            self.request_started(group.request, group.request.started_at)
            group = self._group
        if group and group.followup_until is not None and now > group.followup_until:
            self.clear()
            group = None
        if group is None:
            group = self._group = _Question()
        elif group.utterances:
            self._log("continuation_accepted", group=group.id, turn=trace, revision=group.revision)
        self._cancel_timer()
        self._order += 1
        group.speaking[trace] = self._order
        if group.request is not None:
            completed = self._cancel(group.request)
            if completed:
                group.updates_qa_id = completed
                group.waiting_for_new_text = True
                self._log("previous_answer_retained", group=group.id, revision=group.revision)
            group.request = None
            group.followup_until = None
        elif self.phase == "waiting":
            self._log("answer_wait_cancelled", group=group.id, turn=trace)
        self._set_phase("continuing" if group.utterances else "idle")

    def final(self, trace, text):
        if trace in self._seen:
            return
        if not self._allow_answer():
            self._mark_seen(trace)
            if text and self._record_final is not None and self._enabled():
                self._record_final(text, trace, None)
            return
        group = self._group
        if group is None or trace not in group.speaking:
            self.started(trace)
            group = self._group
        if group is None or trace not in group.speaking:
            return
        order = group.speaking.pop(trace)
        self._mark_seen(trace)
        if text:
            group.utterances[trace] = (order, text)
            if self._record_final is not None:
                if anchor := self._record_final(text, trace, group.id):
                    group.transcript_anchors[trace] = anchor
            group.final_at = self._clock()
            group.waiting_for_new_text = False
        if group.speaking:
            return
        if not group.utterances or group.waiting_for_new_text:
            self.clear()
            return
        self._set_phase("waiting")
        self._log("answer_wait_started", group=group.id, turn=trace, delay_ms=round(self._delay * 1000))
        if self._delay == 0:
            self._answer(group)
        else:
            self._schedule(self._delay, lambda: self._answer(group), allow_opening_at_deadline=True)

    def cancelled(self, trace):
        if self._group and trace in self._group.speaking:
            self.final(trace, "")
        else:
            self._mark_seen(trace)

    def _answer(self, group):
        if self._group is not group or group.speaking:
            return
        if not self._enabled():
            self.clear()
            return
        if not self._allow_answer():
            self.disable_automatic_answers()
            return
        group.revision += 1
        traces = tuple(sorted(group.utterances, key=lambda key: group.utterances[key][0]))
        request = MmpgRequest(
            group.id, group.revision,
            "\n".join(group.utterances[key][1] for key in traces),
            traces, group.final_at, group.updates_qa_id,
            transcript_anchor=group.transcript_anchors.get(traces[-1]),
            transcript_anchors=tuple(group.transcript_anchors[trace] for trace in traces if trace in group.transcript_anchors),
        )
        group.request = request
        self._log("answer_wait_elapsed", group=group.id, revision=group.revision, turns=",".join(traces))
        if self._submit(request):
            self._set_phase("generating")
        else:
            group.request = None
            self.clear()

    def request_started(self, request, started_at):
        group = self._group
        if not group or group.request is not request or group.followup_until is not None:
            return
        if self._followup == 0:
            group.followup_until = started_at
            self.clear()
            return
        group.followup_until = started_at + self._followup
        self._log("followup_window_started", group=group.id, revision=request.revision,
                  window_ms=round(self._followup * 1000))

        def expire():
            self._log("followup_window_elapsed", group=group.id, revision=request.revision)
            if request.finished:
                self.clear()

        self._schedule(max(0, group.followup_until - self._clock()), expire,
                       allow_opening_at_deadline=True)
