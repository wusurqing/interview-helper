from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional, TypeVar

from core.env import env_int
from core.logger import get_logger

T = TypeVar("T")

_log = get_logger(__name__)


class ResourceLaneBusyError(RuntimeError):
    """Raised when a bounded resource lane cannot accept more work."""


class ResourceLane:
    """Bounded executor for non-critical work that must not starve the main flow."""

    def __init__(
        self,
        name: str,
        *,
        max_workers: int,
        max_pending: int,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_pending < 0:
            raise ValueError("max_pending must be >= 0")
        self.name = name
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=name,
        )
        self._slots = threading.BoundedSemaphore(max_workers + max_pending)
        self._shutdown = False
        self._lock = threading.Lock()
        self.rejected_count = 0

    def submit(self, fn: Callable[..., T], *args: object, **kwargs: object) -> Future[T]:
        with self._lock:
            if self._shutdown:
                raise ResourceLaneBusyError(f"{self.name} lane is shut down")
            if not self._slots.acquire(blocking=False):
                self.rejected_count += 1
                if self.rejected_count == 1 or self.rejected_count % 25 == 0:
                    _log.warning("%s lane full; rejected=%d", self.name, self.rejected_count)
                raise ResourceLaneBusyError(f"{self.name} lane is busy")
            try:
                future = self._executor.submit(fn, *args, **kwargs)
            except Exception:
                self._slots.release()
                raise
        future.add_done_callback(lambda _future: self._slots.release())
        return future

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=wait)


_lanes_lock = threading.Lock()
# 长耗时后台 LLM / 探测泳道（如简历优化 30-60s、模型探测），独立于 KB 短查询泳道
_low_priority_lane: Optional[ResourceLane] = None
# 短耗时 KB 检索/状态/文档列表泳道
_kb_query_lane: Optional[ResourceLane] = None
# KB 文件解析/重建索引泳道（避免大 PDF 索引阻塞快速 KB 查询）
_kb_index_lane: Optional[ResourceLane] = None


def _get_low_priority_lane() -> ResourceLane:
    global _low_priority_lane
    with _lanes_lock:
        if _low_priority_lane is None or _low_priority_lane._shutdown:
            _low_priority_lane = ResourceLane(
                "ia-low-priority",
                max_workers=env_int("IA_LOW_PRIORITY_WORKERS", 1, minimum=1),
                max_pending=env_int("IA_LOW_PRIORITY_MAX_PENDING", 8, minimum=0),
            )
        return _low_priority_lane


def _get_kb_query_lane() -> ResourceLane:
    global _kb_query_lane
    with _lanes_lock:
        if _kb_query_lane is None or _kb_query_lane._shutdown:
            _kb_query_lane = ResourceLane(
                "ia-kb-query",
                max_workers=env_int("IA_KB_QUERY_WORKERS", 2, minimum=1),
                max_pending=env_int("IA_KB_QUERY_MAX_PENDING", 16, minimum=0),
            )
        return _kb_query_lane


def _get_kb_index_lane() -> ResourceLane:
    global _kb_index_lane
    with _lanes_lock:
        if _kb_index_lane is None or _kb_index_lane._shutdown:
            _kb_index_lane = ResourceLane(
                "ia-kb-index",
                max_workers=env_int("IA_KB_INDEX_WORKERS", 1, minimum=1),
                max_pending=env_int("IA_KB_INDEX_MAX_PENDING", 8, minimum=0),
            )
        return _kb_index_lane


def submit_low_priority_background(fn: Callable[..., T], *args: object, **kwargs: object) -> bool:
    """提交长耗时低优先级后台任务（简历优化、后台模型健康探测）。"""
    try:
        future = _get_low_priority_lane().submit(fn, *args, **kwargs)
    except ResourceLaneBusyError:
        return False
    future.add_done_callback(_log_background_failure)
    return True


async def run_kb_query(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """在短耗时 KB 查询泳道执行，避免被简历优化等长耗时 LLM 后台任务饿死。"""
    future = _get_kb_query_lane().submit(fn, *args, **kwargs)
    return await asyncio.wrap_future(future)


async def run_kb_index(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """在独立 KB 索引泳道执行文件解析与重建，避免阻塞快速检索或被 LLM 后台阻塞。"""
    future = _get_kb_index_lane().submit(fn, *args, **kwargs)
    return await asyncio.wrap_future(future)


async def run_low_priority(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """兼容入口：短耗时交互式低优先级查询默认走独立查询泳道，不再与后台 LLM 争抢单 worker。"""
    return await run_kb_query(fn, *args, **kwargs)


def shutdown_resource_lanes(*, wait: bool = False) -> None:
    """停机时关闭全部泳道线程池；若后续同进程重启 lifespan 会按需懒重建。"""
    global _low_priority_lane, _kb_query_lane, _kb_index_lane
    with _lanes_lock:
        lanes = [
            lane
            for lane in (_low_priority_lane, _kb_query_lane, _kb_index_lane)
            if lane is not None
        ]
        _low_priority_lane = None
        _kb_query_lane = None
        _kb_index_lane = None
    for lane in lanes:
        try:
            lane.shutdown(wait=wait)
        except Exception:
            _log.exception("Failed to shutdown resource lane %s", lane.name)


def _log_background_failure(future: Future[object]) -> None:
    try:
        future.result()
    except Exception:
        _log.exception("low-priority background task failed")
