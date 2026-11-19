"""Route one physical microphone to the selected interview ASR path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Literal

import numpy as np


Role = Literal["interviewer", "candidate"]
Stream = Callable[[np.ndarray], object]


class SingleMicConflict(RuntimeError):
    """The session cannot start the requested role transition."""


@dataclass(eq=False)
class RoleSwitch:
    from_role: Role
    to_role: Role
    streaming: bool
    _done: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _success: bool = field(default=False, init=False, repr=False)

    def complete(self, success: bool = True) -> None:
        with self._lock:
            if not self._done.is_set():
                self._success = success
                self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout) and self._success


class RoleAudioInput:
    """A bounded FIFO for one role's existing audio consumer."""

    _MAX_BUFFERED_CHUNKS = 940
    _DEFAULT_DRAIN_CHUNKS = 32

    def __init__(self, running: bool):
        self._condition = threading.Condition()
        self._chunks: deque[np.ndarray] = deque(maxlen=self._MAX_BUFFERED_CHUNKS)
        self._barrier: RoleSwitch | None = None
        self._running = running
        self._dropped_chunks = 0
        self._max_chunk_samples = 0
        self._last_chunk_samples = 0

    def offer_audio(self, samples: np.ndarray) -> None:
        with self._condition:
            if len(self._chunks) == self._MAX_BUFFERED_CHUNKS:
                self._dropped_chunks += 1
            self._last_chunk_samples = len(samples)
            self._max_chunk_samples = max(self._max_chunk_samples, len(samples))
            self._chunks.append(samples)
            self._condition.notify()

    def drain_audio_chunks(
        self, timeout: float = 0.1, max_chunks: int | None = None,
    ) -> list[np.ndarray]:
        limit = max_chunks if max_chunks and max_chunks > 0 else self._DEFAULT_DRAIN_CHUNKS
        with self._condition:
            self._condition.wait_for(
                lambda: self._chunks or self._barrier or not self._running,
                timeout=max(0.0, timeout),
            )
            return [self._chunks.popleft() for _ in range(min(limit, len(self._chunks)))]

    def set_barrier(self, barrier: RoleSwitch) -> None:
        with self._condition:
            self._barrier = barrier
            self._condition.notify()

    def take_barrier(self) -> RoleSwitch | None:
        """Claim the boundary after the consumer has processed the preceding PCM."""
        with self._condition:
            if self._chunks:
                return None
            barrier, self._barrier = self._barrier, None
            return barrier

    def cancel_barrier(self, barrier: RoleSwitch) -> None:
        with self._condition:
            if self._barrier is barrier:
                self._barrier = None

    def set_running(self, running: bool) -> None:
        with self._condition:
            self._running = running
            self._condition.notify_all()

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._running

    @property
    def dropped_chunks_count(self) -> int:
        with self._condition:
            return self._dropped_chunks

    def capture_stats_snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "dropped_chunks_count": self._dropped_chunks,
                "loopback_discontinuity_count": 0,
                "max_queue_chunk_samples": self._max_chunk_samples,
                "last_queue_chunk_samples": self._last_chunk_samples,
            }


class SingleMicSession:
    def __init__(
        self,
        session_id: str,
        *,
        device_id: int,
        interviewer_stream: Stream | None = None,
        candidate_stream: Stream | None = None,
        paused: bool = False,
        required_streams: frozenset[Role] = frozenset(),
    ):
        self.id = session_id
        self.device_id = device_id
        self.interviewer_input = RoleAudioInput(not paused)
        self.candidate_input = RoleAudioInput(not paused)
        self._inputs: dict[Role, RoleAudioInput] = {
            "interviewer": self.interviewer_input,
            "candidate": self.candidate_input,
        }
        self._streams: dict[Role, Stream | None] = {
            "interviewer": interviewer_stream,
            "candidate": candidate_stream,
        }
        self.required_streams = required_streams
        self._lock = threading.Lock()
        self._role: Role = "interviewer"
        self._switch: RoleSwitch | None = None
        self._paused = paused
        self._closed = False
        self._last_audio_at = time.monotonic()
        self._revision = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "device_id": self.device_id,
                "role": self._role,
                "switching": self._switch is not None,
                "revision": self._revision,
            }

    def _streams_ready(self) -> bool:
        return all(self._streams[role] is not None for role in self.required_streams)

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._closed and self._streams_ready()

    def seconds_since_audio(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_audio_at

    def set_stream(self, role: Role, callback: Stream | None) -> bool:
        with self._lock:
            if self._closed:
                return False
            if self._streams[role] != callback:
                self._cancel_switch()
                self._streams[role] = callback
            return True

    def offer_audio(self, samples: np.ndarray) -> None:
        """Called on the capture thread with a stable 16 kHz mono PCM chunk."""
        if not len(samples):
            return
        with self._lock:
            if self._paused or self._closed:
                return
            self._last_audio_at = time.monotonic()
            active_role = self._role if self._switch is None and self._streams_ready() else None
            sink = getattr(self, "archive_sink", None)
            if active_role and sink:
                sink(samples, active_role)
            silence = None
            for role, stream in self._streams.items():
                if stream is None:
                    if role == active_role:
                        self._inputs[role].offer_audio(samples)
                elif role == active_role:
                    stream(samples)
                else:
                    if silence is None:
                        silence = np.zeros_like(samples)
                    stream(silence)

    def begin_switch(self, target: Role) -> RoleSwitch | None:
        with self._lock:
            if self._paused or self._closed:
                raise SingleMicConflict("单人测试未在收音")
            if not self._streams_ready():
                raise SingleMicConflict("语音识别正在连接，请稍后再切换角色")
            if self._switch is not None:
                if self._switch.to_role == target:
                    return self._switch
                raise SingleMicConflict("正在切换角色")
            if self._role == target:
                return None
            operation = RoleSwitch(self._role, target, streaming=self._streams[self._role] is not None)
            self._switch = operation
            self._revision += 1
            if not operation.streaming:
                self._inputs[self._role].set_barrier(operation)
            return operation

    def owns_switch(self, operation: RoleSwitch) -> bool:
        with self._lock:
            return self._switch is operation

    def finish_switch(self, operation: RoleSwitch, success: bool) -> bool:
        with self._lock:
            if operation is not self._switch:
                return False
            if success:
                if not operation.wait(0):
                    return False
                self._role = operation.to_role
                self._switch = None
                self._revision += 1
            else:
                self._cancel_switch()
            return True

    def _cancel_switch(self) -> None:
        if self._switch is None:
            return
        operation = self._switch
        self._switch = None
        self._revision += 1
        self._inputs[operation.from_role].cancel_barrier(operation)
        operation.complete(False)

    def pause(self) -> None:
        with self._lock:
            self._pause()

    def _pause(self) -> None:
        self._paused = True
        self._cancel_switch()
        for role_input in self._inputs.values():
            role_input.set_running(False)

    def resume(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._paused = False
            self._last_audio_at = time.monotonic()
            for role_input in self._inputs.values():
                role_input.set_running(True)
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pause()
            for role in self._streams:
                self._streams[role] = None
