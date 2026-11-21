"""Tests for BoundedTaskWorker thread-safe drop counting and lock-free stop join."""

from __future__ import annotations

import threading

from core.background import BoundedTaskWorker


def test_bounded_task_worker_stop_does_not_hold_lock_during_join():
    entered = threading.Event()
    release = threading.Event()
    reentered_ok = threading.Event()
    worker_holder: list[BoundedTaskWorker] = []

    def handler(_payload):
        entered.set()
        release.wait(timeout=3.0)
        # Re-acquiring worker._lock from handler while stop() is joining must not deadlock.
        with worker_holder[0]._lock:
            reentered_ok.set()

    worker = BoundedTaskWorker("test-bg-stop", handler, maxsize=4)
    worker_holder.append(worker)
    assert worker.submit("job-1") is True
    assert entered.wait(timeout=2.0)

    stopper = threading.Thread(target=worker.stop, kwargs={"timeout": 3.0})
    stopper.start()
    # Allow stopper to enqueue sentinel and enter thread.join()
    release.set()
    assert reentered_ok.wait(timeout=2.0)
    stopper.join(timeout=3.0)
    assert not stopper.is_alive()


def test_bounded_task_worker_counts_concurrent_drops_accurately():
    entered = threading.Event()
    release = threading.Event()

    def handler(_payload):
        entered.set()
        release.wait(timeout=3.0)

    worker = BoundedTaskWorker("test-bg-drops", handler, maxsize=1)
    assert worker.submit("hold-worker") is True
    assert entered.wait(timeout=2.0)
    assert worker.submit("fill-queue") is True

    dropped_results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def flood():
        barrier.wait()
        local_drops = 0
        for _ in range(25):
            if not worker.submit("overflow"):
                local_drops += 1
        with results_lock:
            dropped_results.append(local_drops)

    threads = [threading.Thread(target=flood) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    expected_drops = sum(dropped_results)
    assert expected_drops == 200
    assert worker.dropped_count == expected_drops

    release.set()
    worker.stop(timeout=3.0)
