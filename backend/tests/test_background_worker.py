"""Exercise the bounded worker's capacity, graceful shutdown, and restart."""

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
        # Re-entering the public API while stop() joins must not deadlock.
        worker_holder[0].start()
        reentered_ok.set()

    worker = BoundedTaskWorker("test-bg-stop", handler, maxsize=4)
    worker_holder.append(worker)
    assert worker.submit("job-1") is True
    assert entered.wait(timeout=2.0)

    stopper = threading.Thread(target=worker.stop, kwargs={"timeout": 3.0})
    stopper.start()
    # Allow the handler to finish while the stopper joins.
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


def test_timed_out_stop_rejects_submissions_until_the_worker_can_restart():
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def handler(job):
        if job == "running":
            entered.set()
            release.wait(timeout=3)
        processed.append(job)

    worker = BoundedTaskWorker("test-bg-restart", handler, maxsize=4)
    assert worker.submit("running")
    assert entered.wait(timeout=2)
    try:
        worker.stop(timeout=0)
        worker.start()
        assert not worker.submit("during shutdown")
    finally:
        release.set()
        worker.stop(timeout=2)

    assert worker.submit("after restart")
    worker.stop(timeout=2)
    assert processed == ["running", "after restart"]


def test_concurrent_stop_drains_a_full_queue_before_restart():
    entered = threading.Event()
    release = threading.Event()
    drained = threading.Event()
    processed = []

    def handler(job):
        if job == "running":
            entered.set()
            release.wait(timeout=3)
        processed.append(job)
        if job == "queued":
            drained.set()

    worker = BoundedTaskWorker("test-bg-drain", handler, maxsize=1)
    assert worker.submit("running")
    assert entered.wait(timeout=2)
    assert worker.submit("queued")

    barrier = threading.Barrier(3)

    def stop():
        barrier.wait(timeout=2)
        worker.stop(timeout=0)

    stoppers = [threading.Thread(target=stop) for _ in range(2)]
    try:
        for thread in stoppers:
            thread.start()
        barrier.wait(timeout=2)
        for thread in stoppers:
            thread.join(timeout=2)
            assert not thread.is_alive()
    finally:
        release.set()
        worker.stop(timeout=2)

    assert drained.wait(timeout=1)
    assert worker.submit("after restart")
    worker.stop(timeout=2)
    assert processed == ["running", "queued", "after restart"]
