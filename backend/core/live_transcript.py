"""Accepted speech for realtime interview prompts."""

from dataclasses import dataclass, replace
from collections.abc import Iterable
from threading import RLock
from typing import Literal
from uuid import uuid4

Speaker = Literal["interviewer", "candidate"]


@dataclass(frozen=True)
class TranscriptUtterance:
    id: str
    speaker: Speaker
    text: str
    started_at: float
    is_final: bool
    epoch: int
    question_group: str | None = None
    question_like: bool = False
    split_after_reply: bool = False


class LiveTranscript:
    def __init__(self):
        self._lock = RLock()
        self._epoch = 0
        self._utterances: dict[str, TranscriptUtterance] = {}

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def reset(self) -> None:
        with self._lock:
            self._epoch += 1
            self._utterances.clear()

    def stop(self) -> None:
        with self._lock:
            self._epoch += 1
            self._utterances = {
                key: item for key, item in self._utterances.items() if item.is_final
            }

    def record(
        self,
        speaker: Speaker,
        text: str,
        *,
        started_at: float,
        utterance_key: str = "",
        is_final: bool = True,
        epoch: int | None = None,
        question_group: str | None = None,
    ) -> str | None:
        cleaned = text.strip()
        if not cleaned and (not utterance_key or is_final):
            return None
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                return None
            utterance_id = f"{self._epoch}:{speaker}:{utterance_key or uuid4().hex}"
            previous = self._utterances.get(utterance_id)
            if previous:
                if not previous.is_final and cleaned:
                    self._utterances[utterance_id] = replace(
                        previous, text=cleaned, is_final=is_final,
                        question_group=question_group or previous.question_group,
                        question_like=bool(question_group) or previous.question_like,
                    )
            else:
                self._utterances[utterance_id] = TranscriptUtterance(
                    utterance_id, speaker, cleaned, started_at, is_final,
                    self._epoch, question_group, bool(question_group),
                )
            return utterance_id

    def discard_unfinished(self, utterance_id: str | None) -> None:
        with self._lock:
            current = self._utterances.get(utterance_id)
            if current is not None and not current.is_final:
                del self._utterances[current.id]

    def group_question(self, anchors: Iterable[str | None], *, split_after_reply: bool = False) -> None:
        with self._lock:
            items = [self._utterances[anchor] for anchor in anchors if anchor in self._utterances]
            if not items:
                return
            group_id = min(items, key=lambda item: item.started_at).id
            for item in items:
                self._utterances[item.id] = replace(
                    item, question_group=group_id, split_after_reply=split_after_reply,
                )

    def snapshot(self, anchor: str | None) -> tuple[tuple[TranscriptUtterance, ...], ...]:
        with self._lock:
            focus = self._utterances.get(anchor)
            if not focus or focus.speaker != "interviewer" or not focus.text:
                return ()
            utterances = sorted(self._utterances.values(), key=lambda item: item.started_at)

        rounds: list[list[TranscriptUtterance]] = [[]]
        current_group = None
        has_question = False
        has_candidate_reply = False
        found_focus = False
        for item in utterances:
            if found_focus and item.speaker == "interviewer" and not item.text:
                break
            starts_round = item.question_group and (
                item.question_group != current_group
                or (item.split_after_reply and item.question_like and has_question and has_candidate_reply)
            )
            if starts_round:
                if found_focus:
                    break
                current_group = item.question_group
                has_question = False
                has_candidate_reply = False
                if rounds[-1]:
                    rounds.append([])
            if item.text and (not found_focus or not item.question_like):
                rounds[-1].append(item)
            has_question = has_question or item.question_like
            if item.speaker == "candidate" and item.text and has_question:
                has_candidate_reply = True
            if item.id == anchor:
                found_focus = True
        return tuple(tuple(group) for group in rounds if group)
