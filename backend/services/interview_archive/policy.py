"""Replaceable business boundaries; ASR microsegments are not numbered turns."""
from dataclasses import dataclass
import re
from typing import Protocol

from services.stt import classify_asr_question_candidate


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


class RoundBoundaryPolicy(Protocol):
    version: str

    def decide(self, text: str, *, has_question: bool, same_group: bool) -> Decision: ...


class SimpleRoundBoundaryPolicy:
    version = "simple-v1"
    _hints = re.compile(
        r"^(?:哦+|噢+|嗯+|对+|哦对|没错|是的|好的?|可以|行|继续(?:说|吧)?|"
        r"展开(?:一点|讲讲)?|举个例子|比如呢|还有呢|然后呢|想想|提示一下)[。！!？?，,\s]*$",
        re.I,
    )
    _next = re.compile(r"(?:下一[个道]?题|下一个问题|换个问题|另一个问题|接下来问)")

    def decide(self, text: str, *, has_question: bool, same_group: bool = False) -> Decision:
        if not has_question:
            return Decision("append_question", "opening")
        if self._next.search(text):
            return Decision("start_next_turn", "explicit_next_question")
        if self._hints.fullmatch(text.strip()):
            return Decision("append_context", "hint_or_backchannel")
        if same_group:
            return Decision("append_question", "same_question_group")
        kind, _ = classify_asr_question_candidate(text)
        if kind == "promote":
            return Decision("start_next_turn", "explicit_question")
        return Decision("append_context", "conservative_continuation")
