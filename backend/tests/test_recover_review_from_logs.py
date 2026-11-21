from __future__ import annotations

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.recover_review_from_logs import (
    backfill_turns_from_knowledge_db,
    parse_log_sessions,
    recover_session,
)
from services.storage import knowledge, review


def test_parse_log_sessions_recovers_question_and_candidate_answer(tmp_path: Path):
    log_path = tmp_path / "interview.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-24 10:48:35 | INFO  | interview | INTERVIEW_START device=20000 loopback=False",
                "2026-06-24 10:49:01 | INFO  | interview | ASR_QUESTION turn=1 utterances=1 churn=False text='请介绍一下你的项目'",
                "2026-06-24 10:49:01 | INFO  | interview | ANSWER_START id=qa-1-100 model=demo source=conversation_loopback followup=False q='请介绍一下你的项目'",
                "2026-06-24 10:49:12 | INFO  | interview | CANDIDATE_ASR_PARTIAL segment=cand-1 qa_id=qa-1-100 provider=whisper raw=3.0s stt=800ms chars=4 text='我做了风控'",
                "2026-06-24 10:49:18 | INFO  | interview | CANDIDATE_ASR_FINAL segment=cand-1 qa_id=qa-1-100 provider=whisper raw=6.0s stt=900ms chars=15 replaced_partial=True text='我做了风控规则引擎'",
                "2026-06-24 10:49:20 | INFO  | interview | ANSWER_DONE id=qa-1-100 model=demo first_token=100ms total=1000ms answer_len=42 think_len=0",
            ]
        ),
        encoding="utf-8",
    )

    sessions = parse_log_sessions(log_path)

    assert len(sessions) == 1
    turns = list(sessions[0].turns.values())
    assert len(turns) == 1
    assert turns[0].qa_id == "qa-1-100"
    assert turns[0].question == "请介绍一下你的项目"
    assert turns[0].candidate_answer == "我做了风控规则引擎"
    assert turns[0].answer_done is True


def test_parse_redacted_logs_and_backfill_from_knowledge_db(tmp_path: Path, monkeypatch):
    kb_db = tmp_path / "knowledge.db"
    review_db = tmp_path / "review.db"
    monkeypatch.setattr(knowledge, "DB_PATH", kb_db)
    monkeypatch.setattr(review, "DB_PATH", review_db)
    knowledge.init_db()
    review.init_db()

    knowledge.save_record(
        session_type="assist",
        qa_id="qa-1-100",
        question="请介绍一下你的项目",
        answer="完整的系统参考答案内容",
        candidate_answer="从 knowledge.db 回填的候选人真实回答",
    )

    log_path = tmp_path / "interview.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-24 10:48:35 | INFO  | interview | INTERVIEW_START device=20000 loopback=False",
                "2026-06-24 10:49:01 | INFO  | interview | ASR_QUESTION turn=1 utterances=1 churn=False text='请介绍一下你的项目'",
                "2026-06-24 10:49:01 | INFO  | interview | ANSWER_START id=qa-1-100 model=demo source=conversation_loopback followup=False q='请介绍一下你的项目'",
                "2026-06-24 10:49:18 | INFO  | interview | CANDIDATE_ASR_PUBLISH segment=cand-1 qa_id=qa-1-100 final=True provider=doubao chars=21",
                "2026-06-24 10:49:20 | INFO  | interview | ANSWER_DONE id=qa-1-100 model=demo first_token=100ms total=1000ms answer_len=42 think_len=0",
                "2026-06-24 10:50:01 | INFO  | interview | ANSWER_START id=qa-2-200 model=demo source=conversation_loopback followup=False q='第二题未入库'",
                "2026-06-24 10:50:12 | INFO  | interview | CANDIDATE_ASR_FINAL segment=cand-2 qa_id=qa-2-200 provider=doubao raw=4.0s stt=500ms chars=35",
                "2026-06-24 10:50:15 | INFO  | interview | ANSWER_DONE id=qa-2-200 model=demo first_token=100ms total=900ms answer_len=30 think_len=0",
                "2026-06-24 10:51:00 | INFO  | interview | INTERVIEW_STOP reason=manual",
            ]
        ),
        encoding="utf-8",
    )

    sessions = parse_log_sessions(log_path)
    assert len(sessions) == 1
    sess = sessions[0]
    assert sess.turns["qa-1-100"].candidate_answer == ""
    assert sess.turns["qa-1-100"].redacted_candidate_chars == 21
    assert sess.turns["qa-2-200"].redacted_candidate_chars == 35

    stats = backfill_turns_from_knowledge_db(sess, knowledge_db_path=kb_db)
    assert stats == {"matched_turns": 1, "candidate_filled": 1, "reference_filled": 1}
    assert sess.turns["qa-1-100"].candidate_answer == "从 knowledge.db 回填的候选人真实回答"
    assert sess.turns["qa-1-100"].reference_answer == "完整的系统参考答案内容"
    assert sess.turns["qa-1-100"].candidate_backfilled is True
    assert sess.turns["qa-2-200"].candidate_answer == ""

    session_id = recover_session(sess, session_id=None, overwrite=False, dry_run=False)
    detail = review.get_session_detail(session_id)
    assert detail is not None
    # qa-2-200 had no knowledge.db row -> partial_capture
    assert detail["status"] == "partial_capture"
    assert len(detail["turns"]) == 2
    assert detail["turns"][0]["candidate_answer_text"] == "从 knowledge.db 回填的候选人真实回答"
    assert detail["turns"][0]["is_partial"] == 0
    assert detail["turns"][1]["candidate_answer_text"] == ""
    assert detail["turns"][1]["is_partial"] == 1

    # When all turns are filled, overwriting succeeds via storage lock/connection and marks recorded
    sess.turns.pop("qa-2-200")
    recover_session(sess, session_id=session_id, overwrite=True, dry_run=False)
    updated = review.get_session_detail(session_id)
    assert updated is not None
    assert updated["status"] == "recorded"
    assert len(updated["turns"]) == 1
