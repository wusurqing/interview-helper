from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

from core.logger import get_logger


TaskArgs = tuple[object, ...]


class BoundedTaskWorker:
    """Single-worker queue that drains accepted tasks on stop and can then restart."""

    def __init__(
        self,
        name: str,
        handler: Callable[..., None],
        *,
        maxsize: int = 128,
    ):
        self._name = name
        self._handler = handler
        self._maxsize = maxsize
        self._pending: deque[TaskArgs] = deque()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Condition()
        self._stopping = False
        self._log = get_logger(name)
        self.dropped_count = 0

    def start(self) -> None:
        with self._ready:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def submit(self, *args: object) -> bool:
        """Accept a task unless the queue is full or the previous worker is stopping."""
        with self._ready:
            self._start_locked()
            if self._stopping:
                return False
            if self._maxsize <= 0 or len(self._pending) < self._maxsize:
                self._pending.append(args)
                self._ready.notify()
                return True
            self.dropped_count += 1
            dropped = self.dropped_count
        if dropped == 1 or dropped % 25 == 0:
            self._log.warning("background queue full; dropped=%d", dropped)
        return False

    def stop(self, timeout: float = 2.0) -> None:
        with self._ready:
            thread = self._thread
            if not thread:
                return
            self._stopping = True
            self._ready.notify()
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._ready:
                self._ready.wait_for(lambda: self._stopping or bool(self._pending))
                if not self._pending:
                    return
                item = self._pending.popleft()
            try:
                self._handler(*item)
            except Exception:
                self._log.exception("background task failed")
