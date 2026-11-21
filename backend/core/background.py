from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from core.logger import get_logger


TaskArgs = tuple[object, ...]


class BoundedTaskWorker:
    """Single-worker bounded queue for best-effort background tasks."""

    def __init__(
        self,
        name: str,
        handler: Callable[..., None],
        *,
        maxsize: int = 128,
    ):
        self._name = name
        self._handler = handler
        self._queue: queue.Queue[Optional[TaskArgs]] = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._log = get_logger(name)
        self.dropped_count = 0

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def submit(self, *args: object) -> bool:
        self.start()
        try:
            self._queue.put_nowait(args)
            return True
        except queue.Full:
            with self._lock:
                self.dropped_count += 1
                dropped = self.dropped_count
            if dropped == 1 or dropped % 25 == 0:
                self._log.warning("background queue full; dropped=%d", dropped)
            return False

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            thread = self._thread
            if not thread:
                return
            self._thread = None
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    self._log.warning("failed to enqueue shutdown sentinel")
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._handler(*item)
            except Exception:
                self._log.exception("background task failed")
            finally:
                self._queue.task_done()
