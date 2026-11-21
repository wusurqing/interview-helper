"""P1-11 回归测试：SQLite 连接关闭/事务管理、WAL 一次性初始化、retriever 关库竞态、job_tracker TOCTOU 与 review 外键/N+1。"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.kb.store import KBStore
from services.kb.types import Chunk
from services.storage import job_tracker as jt
from services.storage import knowledge
from services.storage import resume_history as rh
from services.storage import review
from services.storage import sqlite_utils


class _TrackedConnection:
    """包装 sqlite3.Connection，追踪 close 次数与执行的 PRAGMA 语句。"""

    def __init__(self, inner: sqlite3.Connection, log: list[str], state: dict[str, Any]):
        self._inner = inner
        self._log = log
        self._state = state
        self._closed = False
        self._state["opened"] += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def row_factory(self) -> Any:
        return self._inner.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._inner.row_factory = value

    def execute(self, sql: str, *args: Any, **kwargs: Any):
        self._log.append(sql.strip())
        return self._inner.execute(sql, *args, **kwargs)

    def executescript(self, sql_script: str):
        self._log.append(sql_script.strip())
        return self._inner.executescript(sql_script)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._inner.__exit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._state["closed"] += 1
        self._inner.close()


def _install_conn_tracker(monkeypatch):
    real_connect = sqlite3.connect
    log: list[str] = []
    state = {"opened": 0, "closed": 0}

    def tracked_connect(*args, **kwargs):
        return _TrackedConnection(real_connect(*args, **kwargs), log, state)

    monkeypatch.setattr(sqlite_utils.sqlite3, "connect", tracked_connect)
    return log, state


def test_kb_store_closes_all_connections_and_runs_wal_only_in_init(tmp_path: Path, monkeypatch):
    log, state = _install_conn_tracker(monkeypatch)
    db_file = tmp_path / "kb.sqlite"
    store = KBStore(str(db_file))
    store.init_schema()
    assert any("journal_mode=WAL" in s for s in log)
    assert state["opened"] == state["closed"]

    log.clear()
    doc_id = store.upsert_doc("a.md", 1.0, 10, "markdown", "Title")
    store.replace_chunks(doc_id, [Chunk("Title", "Redis snapshot persistence", 0)])
    store.set_status("a.md", "ok", None)
    assert store.get_doc("a.md") is not None
    assert len(store.list_docs(limit=5)) == 1
    assert store.stats()["total_docs"] == 1
    assert len(store.fts_search("Redis", limit=5)) == 1
    store.delete_doc("a.md")
    assert store.list_tables()

    assert not any("journal_mode=WAL" in s for s in log)
    assert state["opened"] == state["closed"]


def test_kb_store_list_docs_rejects_non_positive_limit(tmp_path: Path):
    store = KBStore(str(tmp_path / "kb.sqlite"))
    store.init_schema()
    store.upsert_doc("a.md", 1.0, 10, "markdown", "A")
    with pytest.raises(ValueError, match="limit"):
        store.list_docs(limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.list_docs(limit=-1)
    assert store.fts_search("A", limit=0) == []


def test_retriever_timeout_closes_connection_only_in_worker_thread(tmp_path: Path, monkeypatch):
    from core import config as cfg_mod
    from services.kb import indexer, retriever, store as store_mod

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "doc.md").write_text("# Redis\n\nRDB snapshot.", encoding="utf-8")

    cfg = cfg_mod.get_config()
    monkeypatch.setattr(cfg, "kb_dir", str(kb_dir), raising=False)
    monkeypatch.setattr(cfg, "kb_db_path", str(tmp_path / "kb.sqlite"), raising=False)
    monkeypatch.setattr(cfg, "kb_enabled", True, raising=False)
    monkeypatch.setattr(cfg, "kb_trigger_modes", ["manual_text"], raising=False)

    indexer.reset()
    indexer.reindex()

    events: dict[str, Any] = {
        "runner_tid": None,
        "close_tids": [],
        "interrupt_called": 0,
        "close_while_running": False,
    }
    runner_running = threading.Event()
    allow_runner_exit = threading.Event()
    worker_closed = threading.Event()

    real_open = store_mod.KBStore.open_connection

    class _SpyConn:
        def __init__(self, inner: sqlite3.Connection):
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def interrupt(self) -> None:
            events["interrupt_called"] += 1
            self._inner.interrupt()

        def close(self) -> None:
            events["close_tids"].append(threading.get_ident())
            if runner_running.is_set() and not allow_runner_exit.is_set():
                events["close_while_running"] = True
            self._inner.close()
            worker_closed.set()

    def spy_open(self):
        return _SpyConn(real_open(self))

    def slow_fts_search(self, *args, **kwargs):
        events["runner_tid"] = threading.get_ident()
        runner_running.set()
        allow_runner_exit.wait(timeout=2.0)
        return []

    monkeypatch.setattr(store_mod.KBStore, "open_connection", spy_open)
    monkeypatch.setattr(store_mod.KBStore, "fts_search", slow_fts_search)

    hits = retriever.retrieve("Redis", k=3, deadline_ms=30, mode="manual_text")
    assert hits == []
    assert events["interrupt_called"] == 1
    # 主线程返回时，_runner 仍在运行，绝对不能在主线程关连接
    assert events["close_tids"] == []
    assert events["close_while_running"] is False

    allow_runner_exit.set()
    assert worker_closed.wait(timeout=2.0)
    assert events["close_tids"] == [events["runner_tid"]]
    indexer.reset()


def test_job_tracker_patch_application_concurrent_no_lost_update(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(jt, "DB_PATH", str(tmp_path / "job_tracker.db"))
    monkeypatch.setattr(review, "DB_PATH", str(tmp_path / "review.db"))
    jt.init_db()
    review.init_db()

    app = jt.create_application({"company": "InitCo", "position": "InitPos", "notes": "init"})
    barrier = threading.Barrier(3)
    errors: list[Exception] = []

    def worker(payload: dict[str, Any]):
        try:
            barrier.wait(timeout=5.0)
            for _ in range(10):
                jt.patch_application(app["id"], payload)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=({"city": "Shanghai"},)),
        threading.Thread(target=worker, args=({"notes": "updated-notes"},)),
        threading.Thread(target=worker, args=({"feedback": "good-feedback"},)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    final = jt.get_application(app["id"])
    assert final is not None
    assert final["city"] == "Shanghai"
    assert final["notes"] == "updated-notes"
    assert final["feedback"] == "good-feedback"


def test_job_tracker_offer_concurrent_upsert_and_patch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(jt, "DB_PATH", str(tmp_path / "job_tracker.db"))
    jt.init_db()

    app = jt.create_application({"company": "OfferCo", "position": "SWE"})
    barrier = threading.Barrier(4)
    errors: list[Exception] = []

    def upsert_worker(idx: int):
        try:
            barrier.wait(timeout=5.0)
            jt.create_or_update_offer(
                {"application_id": app["id"], "base_salary": f"{30 + idx}k"}
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=upsert_worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    offer = jt.get_offer_by_application(app["id"])
    assert offer is not None
    assert len(jt.list_offers()) == 1

    # 并发 patch_offer 不同字段不互相覆盖
    barrier2 = threading.Barrier(3)

    def patch_worker(payload: dict[str, Any]):
        try:
            barrier2.wait(timeout=5.0)
            for _ in range(10):
                jt.patch_offer(offer["id"], payload)
        except Exception as exc:
            errors.append(exc)

    pthreads = [
        threading.Thread(target=patch_worker, args=({"bonus": "4 months"},)),
        threading.Thread(target=patch_worker, args=({"equity": "100k RSU"},)),
        threading.Thread(target=patch_worker, args=({"pros": "great team"},)),
    ]
    for t in pthreads:
        t.start()
    for t in pthreads:
        t.join(timeout=10.0)

    assert not errors
    final_offer = jt.get_offer(offer["id"])
    assert final_offer is not None
    assert final_offer["bonus"] == "4 months"
    assert final_offer["equity"] == "100k RSU"
    assert final_offer["pros"] == "great team"


def test_job_tracker_reorder_checks_bijection_atomically(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(jt, "DB_PATH", str(tmp_path / "job_tracker.db"))
    jt.init_db()

    a1 = jt.create_application({"company": "A", "stage": "applied"})
    a2 = jt.create_application({"company": "B", "stage": "applied"})

    # 子集或重复必须返回 0
    assert jt.reorder_stage_applications("applied", [a1["id"]]) == 0
    assert jt.reorder_stage_applications("applied", [a1["id"], a1["id"]]) == 0
    # 完整双射成功
    assert jt.reorder_stage_applications("applied", [a2["id"], a1["id"]]) == 2
    rows = jt.list_applications(stage="applied", sort_by="sort_order", sort_dir="asc")
    assert [r["id"] for r in rows] == [a2["id"], a1["id"]]


def test_job_tracker_delete_application_unlinks_review_sessions(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(jt, "DB_PATH", str(tmp_path / "job_tracker.db"))
    monkeypatch.setattr(review, "DB_PATH", str(tmp_path / "review.db"))
    jt.init_db()
    review.init_db()

    app = jt.create_application({"company": "DeleteCo", "position": "Backend"})
    sid = review.create_session(
        started_at=time.time(),
        interviewer_enabled=True,
        candidate_enabled=True,
        application_id=app["id"],
    )
    assert review.get_session_detail(sid)["application_id"] == app["id"]

    assert jt.delete_application(app["id"]) is True
    detail = review.get_session_detail(sid)
    assert detail is not None
    assert detail["application_id"] is None
    assert detail["application"] is None


def test_review_enforces_foreign_keys_and_batches_application_briefs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(jt, "DB_PATH", str(tmp_path / "job_tracker.db"))
    monkeypatch.setattr(review, "DB_PATH", str(tmp_path / "review.db"))
    jt.init_db()
    review.init_db()

    # PRAGMA foreign_keys=ON 生效：向不存在的 session_id 写 turn 应触发 IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        review.add_turn(
            session_id=999999,
            qa_id="qa-orphan",
            seq=1,
            question_text="orphan",
        )

    app1 = jt.create_application({"company": "Alpha", "position": "SWE"})
    app2 = jt.create_application({"company": "Beta", "position": "Infra"})
    review.create_session(time.time() - 30, True, True, application_id=app1["id"])
    review.create_session(time.time() - 20, True, True, application_id=app2["id"])
    review.create_session(time.time() - 10, True, True, application_id=app1["id"])
    review.create_session(time.time(), True, True, application_id=None)

    single_calls = 0
    batch_calls = 0
    real_get = jt.get_application
    real_batch = jt.get_applications_by_ids

    def spy_get(app_id: int):
        nonlocal single_calls
        single_calls += 1
        return real_get(app_id)

    def spy_batch(app_ids: list[int]):
        nonlocal batch_calls
        batch_calls += 1
        return real_batch(app_ids)

    monkeypatch.setattr(jt, "get_application", spy_get)
    monkeypatch.setattr(jt, "get_applications_by_ids", spy_batch)

    page = review.list_sessions(page=1, page_size=20)
    assert page["total"] == 4
    assert single_calls == 0
    assert batch_calls == 1

    by_app_id = {item["application_id"]: item["application"] for item in page["items"]}
    assert by_app_id[app1["id"]]["company"] == "Alpha"
    assert by_app_id[app2["id"]]["company"] == "Beta"
    assert by_app_id[None] is None


def test_resume_history_and_knowledge_close_conns_and_skip_per_conn_pragmas(tmp_path: Path, monkeypatch):
    # 模拟缺 resume_summary 列的旧库，验证 init_db 一次性迁移成功
    old_rh_db = tmp_path / "resume_history.db"
    raw = sqlite3.connect(str(old_rh_db))
    raw.execute(
        """
        CREATE TABLE resume_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            stored_basename TEXT NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            created_at REAL NOT NULL,
            last_used_at REAL NOT NULL,
            parsed_ok INTEGER NOT NULL DEFAULT 0,
            preview TEXT,
            parse_error TEXT
        )
        """
    )
    raw.commit()
    raw.close()

    monkeypatch.setattr(rh, "DB_PATH", str(old_rh_db))
    monkeypatch.setattr(knowledge, "DB_PATH", str(tmp_path / "knowledge.db"))

    log, state = _install_conn_tracker(monkeypatch)
    rh.init_db()
    knowledge.init_db()
    assert any("journal_mode=WAL" in s for s in log)
    assert any("PRAGMA table_info(resume_entries)" in s for s in log)
    assert state["opened"] == state["closed"]

    log.clear()
    assert rh.list_entries() == []
    assert rh.get_filename_for_id(999) is None
    knowledge.save_record("assist", "Q1", "A1", tags=["Redis"])
    assert knowledge.get_history(page=1, page_size=10)["total"] == 1
    assert knowledge.get_summary()
    knowledge.reset_all()

    assert not any("journal_mode=WAL" in s for s in log)
    assert not any("PRAGMA table_info" in s for s in log)
    assert state["opened"] == state["closed"]
