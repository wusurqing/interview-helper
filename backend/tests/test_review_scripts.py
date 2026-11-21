from __future__ import annotations

import importlib
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.storage import job_tracker as jt
from services.storage import review

create_demo_reviews = importlib.import_module("scripts.create_demo_reviews")
import_review_from_logs = importlib.import_module("scripts.import_review_from_logs")
seed_job_tracker_demo = importlib.import_module("scripts.seed_job_tracker_demo")


def _isolate_dbs(monkeypatch, tmp_path: Path):
    review_db = tmp_path / "review.db"
    jt_db = tmp_path / "job_tracker.db"
    monkeypatch.setattr(review, "DB_PATH", review_db)
    monkeypatch.setattr(jt, "DB_PATH", jt_db)
    review.init_db()
    jt.init_db()
    return review_db, jt_db


def test_create_demo_reviews_uses_storage_api_sets_recorded_and_deduplicates(monkeypatch, tmp_path: Path):
    _isolate_dbs(monkeypatch, tmp_path)

    create_demo_reviews.main()
    first_list = review.list_sessions(page=1, page_size=20)
    assert first_list["total"] == len(create_demo_reviews.DEMO_SESSIONS)
    for item in first_list["items"]:
        assert item["status"] == "recorded"
        assert item["source"] == "demo"
        assert item["company"]
        assert item["role"]

    # Running a second time must not duplicate sessions
    create_demo_reviews.main()
    second_list = review.list_sessions(page=1, page_size=20)
    assert second_list["total"] == len(create_demo_reviews.DEMO_SESSIONS)


def test_import_review_from_logs_extracts_repr_deduplicates_and_sets_recorded(monkeypatch, tmp_path: Path):
    _isolate_dbs(monkeypatch, tmp_path)

    followup_repr = repr(
        "以下内容来自同一轮实时面试中的连续追问，请合并理解。\n\n"
        "主问题：Redis 持久化\n\n"
        "连续追问：\n1. RDB 和 AOF 区别\n2. AOF 重写会阻塞主线程吗"
    )
    log_path = tmp_path / "interview.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-25 09:30:00 | INFO | interview | INTERVIEW_START device=20000",
                '2026-06-25 09:30:10 | INFO | interview | ASR_QUESTION turn=1 text="What\'s Redis?"',
                "2026-06-25 09:30:11 | INFO | interview | ANSWER_START id=qa-1 source=conversation_loopback",
                "2026-06-25 09:30:15 | INFO | interview | ANSWER_DONE id=qa-1 answer_len=50",
                f"2026-06-25 09:31:00 | INFO | interview | ASR_QUESTION turn=2 text={followup_repr}",
                "2026-06-25 09:31:05 | INFO | interview | ANSWER_DONE id=qa-2 answer_len=60",
                "2026-06-25 09:35:00 | INFO | interview | INTERVIEW_STOP reason=manual",
            ]
        ),
        encoding="utf-8",
    )

    sessions = import_review_from_logs.parse_log_file(str(log_path))
    assert len(sessions) == 1
    turns = sessions[0]["turns"]
    assert [t["question"] for t in turns] == [
        "What's Redis?",
        "AOF 重写会阻塞主线程吗",
    ]

    # Passing duplicated sessions in the same call imports only once
    imported_first = import_review_from_logs.import_sessions_to_db([sessions[0], dict(sessions[0])])
    assert imported_first == 1
    # Importing again deduplicates against DB by started_at
    imported_second = import_review_from_logs.import_sessions_to_db(sessions)
    assert imported_second == 0

    listed = review.list_sessions(page=1, page_size=10)
    assert listed["total"] == 1
    assert listed["items"][0]["status"] == "recorded"
    assert listed["items"][0]["source"] == "log_import"


def test_import_review_from_logs_preserves_last_turn_of_crashed_session_before_next_start(tmp_path: Path):
    log_path = tmp_path / "crashed.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-25 09:00:00 | INFO | interview | INTERVIEW_START device=20000",
                "2026-06-25 09:00:10 | INFO | interview | ASR_QUESTION turn=1 text='问题1'",
                "2026-06-25 09:00:11 | INFO | interview | ANSWER_START id=qa-1 source=conversation_loopback",
                "2026-06-25 09:00:15 | INFO | interview | ANSWER_DONE id=qa-1 answer_len=50",
                "2026-06-25 09:01:00 | INFO | interview | ASR_QUESTION turn=2 text='问题2'",
                "2026-06-25 09:01:05 | INFO | interview | ANSWER_DONE id=qa-2 answer_len=60",
                # Crash: next INTERVIEW_START without INTERVIEW_STOP
                "2026-06-25 10:00:00 | INFO | interview | INTERVIEW_START device=20000",
                "2026-06-25 10:00:10 | INFO | interview | ASR_QUESTION turn=1 text='新会话问题'",
                "2026-06-25 10:00:11 | INFO | interview | ANSWER_START id=qa-3 source=conversation_loopback",
                "2026-06-25 10:00:20 | INFO | interview | INTERVIEW_STOP qa_count=1",
                # Stray line after stop must not leak into the closed session
                "2026-06-25 10:05:00 | INFO | interview | ASR_QUESTION turn=2 text='停录后噪音'",
            ]
        ),
        encoding="utf-8",
    )

    sessions = import_review_from_logs.parse_log_file(str(log_path))
    assert len(sessions) == 2
    assert [t["question"] for t in sessions[0]["turns"]] == ["问题1", "问题2"]
    assert [t["question"] for t in sessions[1]["turns"]] == ["新会话问题"]


def test_seed_job_tracker_demo_clears_only_demo_records_via_storage_connections(monkeypatch, tmp_path: Path):
    _isolate_dbs(monkeypatch, tmp_path)

    # Create a real (non-demo) application and review session that must survive cleanup
    real_app = jt.create_application({"company": "RealCorp", "position": "Backend", "notes": "real interview"})
    real_sess = review.create_session(
        started_at=1_700_000_000.0,
        interviewer_enabled=True,
        candidate_enabled=True,
        company="RealCorp",
        role="Backend",
    )

    summary = seed_job_tracker_demo.seed_demo_data()
    assert len(summary["applications"]) > 0
    assert summary["review_count"] > 0

    removed_apps, removed_sessions = seed_job_tracker_demo.clear_demo_data()
    assert removed_apps == len(summary["applications"])
    assert removed_sessions == summary["review_count"]

    remaining_apps = jt.list_applications()
    assert [app["id"] for app in remaining_apps] == [real_app["id"]]
    remaining_reviews = review.list_sessions(page=1, page_size=50)
    assert [s["id"] for s in remaining_reviews["items"]] == [real_sess]
