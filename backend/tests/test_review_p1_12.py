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
    from core.session import get_session
    active_archive = get_session().archive
    if active_archive is not None:
        get_session().archive = None
        active_archive.stop(timeout=2)
    with review_integration._archive_lock:
        pending = list(review_integration._draining_archives)
        review_integration._draining_archives.clear()
        review_integration._archive_status = None
    for archive in pending:
        archive.stop(timeout=2)
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


def test_lifespan_recovers_interrupted_archives_before_legacy_recovery_and_prunes_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import hashlib
    import main as main_mod
    from services.storage import review_archive

    monkeypatch.setenv("IA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(review, "DB_PATH", str(tmp_path / "review.db"))
    review.init_db()

    # 1) 中断中的 recording 归档会话 + 已提交轮次（含受引用音频与孤儿临时/未引用音频）
    sid_rec = review.create_session(time.time() - 300, True, True)
    asset_bytes = b"OggS-valid-referenced-audio"
    asset_id = hashlib.sha256(asset_bytes).hexdigest()
    tid = review_archive.save_round({
        "session_id": sid_rec,
        "round_id": "round-1",
        "seq": 1,
        "started_at": time.time() - 280,
        "ended_at": time.time() - 260,
        "question_text": "什么是事务隔离级别？",
        "candidate_answer_text": "包含读未提交到串行化四种。",
        "reference_answer_text": "参考答案",
        "candidate_status": "recorded",
        "ai_status": "completed",
        "policy_version": "simple-v1",
        "utterances": [],
        "generations": [],
        "media": {"interviewer": [], "candidate": []},
        "assets": {asset_id: asset_bytes},
    })
    assert tid > 0
    review_archive.set_state(sid_rec, "recording")

    referenced_file = tmp_path / "review_media" / str(sid_rec) / "round-1" / f"{asset_id}.ogg"
    orphan_tmp = tmp_path / "review_media" / str(sid_rec) / "round-uncommitted" / "temp.tmp"
    orphan_ogg = tmp_path / "review_media" / str(sid_rec) / "round-uncommitted" / ("0" * 64 + ".ogg")
    orphan_tmp.parent.mkdir(parents=True, exist_ok=True)
    orphan_tmp.write_bytes(b"incomplete")
    orphan_ogg.write_bytes(b"unreferenced")

    # 2) 中断中的 draining 归档会话
    sid_drain = review.create_session(time.time() - 250, True, False)
    review_archive.set_state(sid_drain, "draining")

    # 3) 传统非归档遗留会话（analyzing + recording）
    sid_legacy_analyzing = review.create_session(time.time() - 200, True, True)
    review.update_session_status(sid_legacy_analyzing, "analyzing")
    sid_legacy_recording = review.create_session(time.time() - 150, True, True)

    monkeypatch.setattr("services.storage.resume_history.restore_active_resume", lambda: False)

    with TestClient(main_mod.app):
        rec_detail = review.get_session_detail(sid_rec)
        assert rec_detail["status"] == "partial_capture"
        assert rec_detail["archive_status"] == "partial"
        assert rec_detail["ended_at"] is not None
        assert rec_detail["archive_warning"]

        drain_detail = review.get_session_detail(sid_drain)
        assert drain_detail["status"] == "partial_capture"
        assert drain_detail["archive_status"] == "partial"
        assert drain_detail["ended_at"] is not None

        assert referenced_file.is_file()
        assert not orphan_tmp.exists()
        assert not orphan_ogg.exists()

        assert review.get_session_detail(sid_legacy_analyzing)["status"] == "analysis_failed"
        legacy_rec = review.get_session_detail(sid_legacy_recording)
        assert legacy_rec["status"] == "partial_capture"
        assert legacy_rec["ended_at"] is not None

    review_async_analysis.shutdown_executor(wait=True)


def test_pre_archive_sqlite_schema_upgrades_preserving_legacy_rows_and_saving_new_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import hashlib
    import sqlite3
    from services.storage import review_archive

    db_path = tmp_path / "review.db"
    monkeypatch.setenv("IA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(review, "DB_PATH", str(db_path))

    # Build exact pre-archive `main` schema (no archive_* columns, no round_id, no archive tables).
    now = time.time() - 500
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE review_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL DEFAULT 'recording',
                started_at REAL NOT NULL,
                ended_at REAL,
                source TEXT NOT NULL DEFAULT 'assist',
                title TEXT,
                company TEXT,
                role TEXT,
                application_id INTEGER,
                jd_snapshot TEXT,
                resume_snapshot TEXT,
                interviewer_capture_enabled INTEGER NOT NULL DEFAULT 0,
                candidate_capture_enabled INTEGER NOT NULL DEFAULT 0,
                turn_count INTEGER NOT NULL DEFAULT 0,
                avg_score REAL,
                summary_markdown TEXT,
                strong_points_json TEXT,
                weak_points_json TEXT,
                behavior_traits_json TEXT,
                domain_summary_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE review_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                qa_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                candidate_answer_text TEXT,
                original_candidate_answer_text TEXT,
                reference_answer_text TEXT,
                code_text TEXT,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                is_partial INTEGER NOT NULL DEFAULT 0,
                analysis_status TEXT NOT NULL DEFAULT 'pending',
                strengths_json TEXT,
                risks_json TEXT,
                evidence_json TEXT,
                behavior_signals_json TEXT,
                tags_json TEXT,
                scorecard_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES review_sessions(id)
            );
            CREATE TABLE review_profile_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at REAL NOT NULL,
                session_count INTEGER NOT NULL DEFAULT 0,
                summary_json TEXT,
                strengths_json TEXT,
                weaknesses_json TEXT,
                behavior_traits_json TEXT,
                domain_mastery_json TEXT,
                trend_json TEXT
            );
            CREATE INDEX idx_review_turns_session_id ON review_turns(session_id);
            CREATE INDEX idx_review_turns_qa_id ON review_turns(qa_id);
            CREATE INDEX idx_review_sessions_application_id ON review_sessions(application_id);
            """
        )
        cur = conn.execute(
            """
            INSERT INTO review_sessions (
                status, started_at, ended_at, source, title, company, role,
                interviewer_capture_enabled, candidate_capture_enabled,
                turn_count, summary_markdown, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("recorded", now, now + 60, "assist", "历史面试", "旧公司", "后端", 1, 1, 2, "## 历史复盘", now, now + 60),
        )
        legacy_recorded_sid = int(cur.lastrowid)
        # Multiple legacy turns in the same session will all receive round_id=NULL after upgrade.
        conn.execute(
            """
            INSERT INTO review_turns (
                session_id, qa_id, seq, question_text, candidate_answer_text,
                original_candidate_answer_text, reference_answer_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (legacy_recorded_sid, "qa-legacy-1", 1, "历史问题1", "历史回答1", "原始回答1", "参考1", now + 10, now + 10),
        )
        conn.execute(
            """
            INSERT INTO review_turns (
                session_id, qa_id, seq, question_text, candidate_answer_text,
                original_candidate_answer_text, reference_answer_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (legacy_recorded_sid, "qa-legacy-2", 2, "历史问题2", "历史回答2", "原始回答2", "参考2", now + 20, now + 20),
        )
        cur2 = conn.execute(
            """
            INSERT INTO review_sessions (
                status, started_at, source, interviewer_capture_enabled, candidate_capture_enabled,
                turn_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("recording", now + 100, "assist", 1, 0, 1, now + 100, now + 100),
        )
        legacy_recording_sid = int(cur2.lastrowid)
        conn.execute(
            """
            INSERT INTO review_turns (
                session_id, qa_id, seq, question_text, candidate_answer_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (legacy_recording_sid, "qa-legacy-3", 1, "中断历史问题", "", now + 110, now + 110),
        )

    # Upgrade schema in-place and run archive startup recovery.
    review.init_db()
    review_archive.recover_interrupted()

    # Verify all new columns, tables, and unique index exist.
    with sqlite3.connect(str(db_path)) as conn:
        session_cols = {row[1] for row in conn.execute("PRAGMA table_info(review_sessions)").fetchall()}
        turn_cols = {row[1] for row in conn.execute("PRAGMA table_info(review_turns)").fetchall()}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        raw_legacy_turns = conn.execute(
            "SELECT qa_id, round_id, archive_version FROM review_turns WHERE session_id=? ORDER BY seq",
            (legacy_recorded_sid,),
        ).fetchall()

    assert {"archive_status", "archive_warning", "archive_version"} <= session_cols
    assert {
        "round_id", "archive_version", "candidate_status", "ai_status",
        "started_at", "ended_at", "media_json", "model_summary_json", "audio_status_json",
    } <= turn_cols
    assert {"review_archive_payloads", "review_audio_assets"} <= tables
    assert "idx_review_round" in indexes
    assert raw_legacy_turns == [("qa-legacy-1", None, 0), ("qa-legacy-2", None, 0)]

    # Verify legacy sessions and turns remain intact and readable.
    legacy_detail = review.get_session_detail(legacy_recorded_sid)
    assert legacy_detail["status"] == "recorded"
    assert legacy_detail["summary_markdown"] == "## 历史复盘"
    assert legacy_detail["turn_count"] == 2
    assert [t["qa_id"] for t in legacy_detail["turns"]] == ["qa-legacy-1", "qa-legacy-2"]
    assert legacy_detail["turns"][0]["original_candidate_answer_text"] == "原始回答1"
    # archive_version=0 legacy recording session must not be falsely rewritten by archive recovery.
    assert review.get_session_detail(legacy_recording_sid)["status"] == "recording"

    # Verify new archive rounds persist cleanly on the upgraded database (including idempotent update).
    new_sid = review.create_session(time.time(), True, True)
    asset_bytes = b"OggS-post-upgrade-audio"
    asset_id = hashlib.sha256(asset_bytes).hexdigest()
    round_payload = {
        "session_id": new_sid,
        "round_id": "round-upgrade-1",
        "seq": 1,
        "started_at": time.time() - 20,
        "ended_at": time.time() - 5,
        "question_text": "升级后的新问题？",
        "candidate_answer_text": "升级后的新回答。",
        "reference_answer_text": "升级后的参考答案。",
        "candidate_status": "recorded",
        "ai_status": "completed",
        "policy_version": "simple-v1",
        "utterances": [{"speaker": "interviewer", "text": "升级后的新问题？"}],
        "generations": [{"id": "g1", "status": "completed"}],
        "media": {"interviewer": [{"asset_id": asset_id}], "candidate": []},
        "assets": {asset_id: asset_bytes},
    }
    tid = review_archive.save_round(round_payload)
    assert tid > 0
    assert review_archive.save_round({**round_payload, "candidate_answer_text": "更新后的新回答。"}) == tid

    upgraded_detail = review.get_session_detail(new_sid)
    assert upgraded_detail["turn_count"] == 1
    assert upgraded_detail["turns"][0]["candidate_answer_text"] == "更新后的新回答。"
    assert review_archive.get_archive(new_sid, tid)["round_id"] == "round-upgrade-1"
    saved_asset = review_archive.asset_path(new_sid, tid, asset_id)
    assert saved_asset is not None and saved_asset.read_bytes() == asset_bytes

    # Re-verify legacy rows were completely unaffected after new archive writes.
    assert [t["qa_id"] for t in review.get_session_detail(legacy_recorded_sid)["turns"]] == [
        "qa-legacy-1",
        "qa-legacy-2",
    ]
