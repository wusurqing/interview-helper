"""P1-12 回归测试：复盘分析原子防重入、有界线程池、lifespan 启动恢复遗留状态与 review_integration 加锁快照。"""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

review_router = importlib.import_module("api.review.router")
from core.session import QAPair, Session, conversation_lock
from services import review_async_analysis, review_integration
from services.storage import review


@pytest.fixture
def isolated_review_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(review, "DB_PATH", str(tmp_path / "review.db"))
    review.init_db()
    with review_integration._state_lock:
        review_integration._current_review_session_id = None
        review_integration._existing_qa_ids = frozenset()
    yield
    with review_integration._state_lock:
        review_integration._current_review_session_id = None
        review_integration._existing_qa_ids = frozenset()
    review_async_analysis.shutdown_executor(wait=True)


def test_generate_concurrent_reentry_runs_analysis_exactly_once(
    isolated_review_db, monkeypatch: pytest.MonkeyPatch
):
    session_id = review.create_session(time.time(), True, True)
    review.add_turn(session_id, "qa-1", 1, "Q1", "A1")
    review.end_session(session_id, status="recorded")

    started_calls: list[int] = []
    started_lock = threading.Lock()
    barrier = threading.Barrier(6)

    def fake_analyze(sid: int):
        with started_lock:
            started_calls.append(sid)

    monkeypatch.setattr(review_router.review_async_analysis, "analyze_session_async", fake_analyze)

    results: list[dict] = []
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait(timeout=5.0)
            res = asyncio.run(review_router.trigger_review_analysis(session_id))
            with started_lock:
                results.append(res)
        except Exception as exc:
            with started_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    assert len(results) == 6
    assert started_calls == [session_id]
    statuses = [r["status"] for r in results]
    assert statuses.count("started") == 1
    assert statuses.count("pending") == 5
    assert review.get_session_detail(session_id)["status"] == "analyzing"


def test_claim_session_for_analysis_rejects_completed_with_analysis(isolated_review_db):
    sid_done = review.create_session(time.time(), True, True)
    review.end_session(sid_done, status="completed", summary_markdown="## 已完成")
    assert review.claim_session_for_analysis(sid_done) is False

    sid_empty_completed = review.create_session(time.time(), True, True)
    review.add_turn(sid_empty_completed, "qa-1", 1, "Q", "A", analysis_status="pending")
    review.end_session(sid_empty_completed, status="completed")
    assert review.claim_session_for_analysis(sid_empty_completed) is True
    # 第二次已被占位为 analyzing，必然返回 False
    assert review.claim_session_for_analysis(sid_empty_completed) is False


def test_generate_rolls_back_status_and_sanitizes_500_on_submit_error(
    isolated_review_db, monkeypatch: pytest.MonkeyPatch
):
    session_id = review.create_session(time.time(), True, True)
    review.add_turn(session_id, "qa-1", 1, "Q1", "A1")
    review.end_session(session_id, status="recorded")

    def boom(_sid: int):
        raise RuntimeError("secret-internal-executor-stacktrace")

    monkeypatch.setattr(review_router.review_async_analysis, "analyze_session_async", boom)

    with pytest.raises(review_router.HTTPException) as exc_info:
        asyncio.run(review_router.trigger_review_analysis(session_id))

    assert exc_info.value.status_code == 500
    assert "secret-internal" not in str(exc_info.value.detail)
    # 状态已回滚回 recorded，不卡死在 analyzing
    assert review.get_session_detail(session_id)["status"] == "recorded"


def test_bounded_thread_pool_caps_concurrent_review_workers(
    isolated_review_db, monkeypatch: pytest.MonkeyPatch
):
    review_async_analysis.shutdown_executor(wait=True)
    monkeypatch.setattr(review_async_analysis, "_MAX_ANALYSIS_WORKERS", 2)

    active = 0
    max_active = 0
    lock = threading.Lock()
    entered_two = threading.Event()
    release = threading.Event()
    completed = 0
    all_done = threading.Event()

    def fake_worker(session_id: int, system_prompts=None):
        nonlocal active, max_active, completed
        with lock:
            active += 1
            if active > max_active:
                max_active = active
            if active >= 2:
                entered_two.set()
        release.wait(timeout=5.0)
        with lock:
            active -= 1
            completed += 1
            if completed == 5:
                all_done.set()

    monkeypatch.setattr(review_async_analysis, "_analyze_session_worker", fake_worker)

    sids = []
    for _ in range(5):
        sid = review.create_session(time.time(), True, True)
        review.end_session(sid, status="recorded")
        sids.append(sid)

    for sid in sids:
        review_async_analysis.analyze_session_async(sid)

    assert entered_two.wait(timeout=3.0)
    time.sleep(0.05)
    with lock:
        assert max_active <= 2
    release.set()
    assert all_done.wait(timeout=5.0)
    assert max_active <= 2


def test_lifespan_recovers_stale_analyzing_and_recording_sessions(
    isolated_review_db, monkeypatch: pytest.MonkeyPatch
):
    import main as main_mod

    sid_analyzing = review.create_session(time.time() - 300, True, True)
    review.update_session_status(sid_analyzing, "analyzing")

    sid_recording = review.create_session(time.time() - 200, True, True)
    # 保持 status='recording', ended_at=None

    sid_recorded = review.create_session(time.time() - 100, True, True)
    review.end_session(sid_recorded, status="recorded")

    sid_completed = review.create_session(time.time() - 50, True, True)
    review.end_session(sid_completed, status="completed", summary_markdown="## OK")

    # 避免测试里碰真实硬件或历史简历恢复
    monkeypatch.setattr("services.storage.resume_history.restore_active_resume", lambda: False)

    with TestClient(main_mod.app) as client:
        assert review.get_session_detail(sid_analyzing)["status"] == "analysis_failed"
        rec_detail = review.get_session_detail(sid_recording)
        assert rec_detail["status"] == "partial_capture"
        assert rec_detail["ended_at"] is not None
        assert review.get_session_detail(sid_recorded)["status"] == "recorded"
        assert review.get_session_detail(sid_completed)["status"] == "completed"

        # 恢复后的 analysis_failed 会话可重新触发生成
        started: list[int] = []
        monkeypatch.setattr(
            review_router.review_async_analysis,
            "analyze_session_async",
            lambda sid: started.append(sid),
        )
        resp = client.post(f"/api/review/sessions/{sid_analyzing}/generate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        assert started == [sid_analyzing]


def test_review_integration_locks_state_and_snapshots_under_conversation_lock(
    isolated_review_db, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        review_integration,
        "get_config",
        lambda: type("_Cfg", (), {"review_enabled": False})(),
    )
    sid = review_integration.on_assist_start(1, 2, True)
    assert sid is not None

    session = Session()
    session.qa_pairs = [QAPair(id="qa-1", question="Q1", answer="A1")]

    # 验证 on_assist_stop 必须在 conversation_lock 内快照 qa_pairs
    stop_done = threading.Event()
    stop_result: list[int | None] = []

    with conversation_lock:
        t = threading.Thread(
            target=lambda: (stop_result.append(review_integration.on_assist_stop(session)), stop_done.set())
        )
        t.start()
        # 持锁期间 on_assist_stop 应阻塞在 conversation_lock 上，不能读未完成状态
        assert not stop_done.wait(timeout=0.08)
        session.qa_pairs.append(QAPair(id="qa-2", question="Q2", answer="A2"))

    assert stop_done.wait(timeout=3.0)
    t.join(timeout=3.0)
    assert stop_result == [sid]
    detail = review.get_session_detail(sid)
    assert detail["turn_count"] == 2
    assert [turn["qa_id"] for turn in detail["turns"]] == ["qa-1", "qa-2"]

    # 并发二次停止应幂等返回 None
    assert review_integration.on_assist_stop(session) is None
