"""Recover a review session from interview.log events (with optional knowledge.db backfill).

This is intended for crash/force-close cases where review_sessions contains a
recording session but /api/assist/stop never ran, so review_turns were not
written.

Log redaction boundary:
    Current interview.log redacts CANDIDATE_ASR_* transcript text and only logs
    ``chars=<n>``. When ``text=`` is absent from logs, this script can backfill
    candidate answers (and full reference answers) from ``knowledge.db``
    (``question_records``) matched by ``qa_id``. Any turn whose ``qa_id`` was
    never persisted to ``knowledge.db`` cannot recover redacted candidate text
    from logs alone and will be stored with ``is_partial=True``.

Usage:
    python -m scripts.recover_review_from_logs --session-id 872
    python -m scripts.recover_review_from_logs --since "2026-06-24 10:48:35"
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.storage import knowledge, review  # noqa: E402
from services.storage.sqlite_utils import sqlite_connection  # noqa: E402

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
QA_ID_RE = re.compile(r"\bid=([^\s]+)")
SEGMENT_RE = re.compile(r"\bsegment=([^\s]+)")
CANDIDATE_QA_RE = re.compile(r"\bqa_id=([^\s]+)")
CHARS_RE = re.compile(r"\bchars=(\d+)")
ANSWER_LEN_RE = re.compile(r"\banswer_len=(\d+)")
MODEL_RE = re.compile(r"\bmodel=([^\s]+)")


@dataclass
class CandidateSegment:
    text: str = ""
    is_final: bool = False


@dataclass
class LogTurn:
    qa_id: str
    seq: int
    question: str = ""
    reference_answer: str = ""
    answer_done: bool = False
    candidate_segments: "OrderedDict[str, CandidateSegment]" = field(default_factory=OrderedDict)
    redacted_candidate_chars: int = 0
    candidate_backfilled: bool = False

    @property
    def candidate_answer(self) -> str:
        return "\n".join(
            segment.text.strip()
            for segment in self.candidate_segments.values()
            if segment.text.strip()
        ).strip()


@dataclass
class LogSession:
    started_at: float
    ended_at: Optional[float] = None
    turns: "OrderedDict[str, LogTurn]" = field(default_factory=OrderedDict)
    last_event_at: Optional[float] = None


def _parse_ts(line: str) -> Optional[float]:
    match = TS_RE.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), TS_FORMAT).timestamp()


def _parse_datetime(value: str) -> float:
    return datetime.strptime(value, TS_FORMAT).timestamp()


def _extract_repr(line: str, key: str) -> str:
    match = re.search(rf"\b{re.escape(key)}=('(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")", line)
    if not match:
        return ""
    try:
        value = ast.literal_eval(match.group(1))
    except Exception:
        return ""
    return str(value or "").strip()


def _clean_question(text: str) -> str:
    return " ".join((text or "").split()).strip()


def parse_log_sessions(log_path: Path, *, until: Optional[float] = None) -> list[LogSession]:
    sessions: list[LogSession] = []
    current: Optional[LogSession] = None
    pending_question = ""

    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            ts = _parse_ts(line)
            if until is not None and ts is not None and ts > until:
                if current is not None:
                    current.ended_at = current.last_event_at or until
                    sessions.append(current)
                break
            if ts is not None and current is not None:
                current.last_event_at = ts

            if "INTERVIEW_START" in line:
                if current is not None:
                    sessions.append(current)
                current = LogSession(started_at=ts or time.time(), last_event_at=ts)
                pending_question = ""
                continue

            if current is None:
                continue

            if "INTERVIEW_STOP" in line:
                current.ended_at = ts or current.last_event_at
                sessions.append(current)
                current = None
                pending_question = ""
                continue

            if "ASR_QUESTION" in line and "text=" in line:
                pending_question = _clean_question(_extract_repr(line, "text"))
                continue

            if "ANSWER_START" in line:
                qa_match = QA_ID_RE.search(line)
                if not qa_match:
                    continue
                qa_id = qa_match.group(1)
                turn = current.turns.get(qa_id)
                if turn is None:
                    turn = LogTurn(qa_id=qa_id, seq=len(current.turns) + 1)
                    current.turns[qa_id] = turn
                question = _clean_question(_extract_repr(line, "q"))
                turn.question = question or pending_question or turn.question
                pending_question = ""
                continue

            if "ANSWER_DONE" in line and "EXAM_PREFLIGHT_ANSWER_DONE" not in line:
                qa_match = QA_ID_RE.search(line)
                if not qa_match:
                    continue
                qa_id = qa_match.group(1)
                if qa_id in ("", "None", "null"):
                    continue
                turn = current.turns.get(qa_id)
                if turn is None:
                    turn = LogTurn(qa_id=qa_id, seq=len(current.turns) + 1)
                    current.turns[qa_id] = turn
                model = MODEL_RE.search(line)
                answer_len = ANSWER_LEN_RE.search(line)
                model_name = model.group(1) if model else "unknown"
                chars = answer_len.group(1) if answer_len else "unknown"
                turn.answer_done = True
                turn.reference_answer = (
                    f"系统参考答案已生成（模型 {model_name}，日志仅保留长度 {chars} 字，未保存全文）。"
                )
                continue

            if (
                "CANDIDATE_ASR_" in line
                and "CANDIDATE_ASR_SEGMENT_START" not in line
                and "qa_id=" in line
            ):
                qa_match = CANDIDATE_QA_RE.search(line)
                seg_match = SEGMENT_RE.search(line)
                if not qa_match:
                    continue
                qa_id = qa_match.group(1)
                if qa_id in ("", "None", "null"):
                    continue
                has_text = "text=" in line
                chars_match = None if has_text else CHARS_RE.search(line)
                if not has_text and chars_match is None:
                    continue
                turn = current.turns.get(qa_id)
                if turn is None:
                    turn = LogTurn(qa_id=qa_id, seq=len(current.turns) + 1)
                    current.turns[qa_id] = turn
                if has_text:
                    segment_id = seg_match.group(1) if seg_match else f"{qa_id}-segment-{len(current.turns)}"
                    text = _extract_repr(line, "text")
                    if not text:
                        continue
                    existing = turn.candidate_segments.get(segment_id)
                    is_final = "CANDIDATE_ASR_FINAL" in line or "final=True" in line
                    if existing is None or is_final or not existing.is_final:
                        turn.candidate_segments[segment_id] = CandidateSegment(
                            text=text,
                            is_final=is_final,
                        )
                elif chars_match is not None:
                    turn.redacted_candidate_chars = max(
                        turn.redacted_candidate_chars,
                        int(chars_match.group(1)),
                    )

    if current is not None:
        sessions.append(current)
    return sessions


def backfill_turns_from_knowledge_db(
    log_session: LogSession,
    knowledge_db_path: Optional[Path] = None,
) -> dict[str, int]:
    """Backfill redacted candidate answers (and full reference answers) from knowledge.db by qa_id."""
    stats = {"matched_turns": 0, "candidate_filled": 0, "reference_filled": 0}
    db_path = Path(knowledge_db_path) if knowledge_db_path is not None else Path(knowledge.DB_PATH)
    if not db_path.exists() or not log_session.turns:
        return stats

    with sqlite_connection(db_path) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'question_records'"
        ).fetchone()
        if not table_exists:
            return stats
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(question_records)").fetchall()
        }
        if "qa_id" not in columns:
            return stats
        has_candidate_col = "candidate_answer" in columns
        select_cols = "question, answer, candidate_answer" if has_candidate_col else "question, answer"
        order_col = "created_at DESC, id DESC" if "created_at" in columns else "id DESC"
        for qa_id, turn in log_session.turns.items():
            if not qa_id or qa_id in ("None", "null"):
                continue
            rows = conn.execute(
                f"""
                SELECT {select_cols}
                FROM question_records
                WHERE qa_id = ?
                ORDER BY {order_col}
                """,
                (qa_id,),
            ).fetchall()
            if not rows:
                continue
            stats["matched_turns"] += 1
            db_question = next(
                (q for row in rows if (q := _clean_question(str(row["question"] or "")))),
                "",
            )
            db_answer = next(
                (a for row in rows if (a := str(row["answer"] or "").strip())),
                "",
            )
            db_candidate = (
                next(
                    (c for row in rows if (c := str(row["candidate_answer"] or "").strip())),
                    "",
                )
                if has_candidate_col
                else ""
            )
            if not turn.question and db_question:
                turn.question = db_question
            if not turn.candidate_answer and db_candidate:
                turn.candidate_segments["knowledge-db"] = CandidateSegment(
                    text=db_candidate,
                    is_final=True,
                )
                turn.candidate_backfilled = True
                stats["candidate_filled"] += 1
            if db_answer and (
                not turn.reference_answer
                or turn.reference_answer.startswith("系统参考答案已生成（")
            ):
                turn.reference_answer = db_answer
                turn.answer_done = True
                stats["reference_filled"] += 1
    return stats


def select_log_session(
    sessions: list[LogSession],
    *,
    started_at: Optional[float],
    since: Optional[float],
    until: Optional[float],
) -> Optional[LogSession]:
    candidates = sessions
    if since is not None:
        candidates = [
            sess for sess in candidates
            if (sess.last_event_at or sess.started_at) >= since and sess.started_at <= (until or time.time())
        ]
    if until is not None:
        candidates = [sess for sess in candidates if sess.started_at <= until]
    if started_at is not None:
        candidates = sorted(candidates, key=lambda sess: abs(sess.started_at - started_at))
    else:
        candidates = sorted(candidates, key=lambda sess: sess.started_at, reverse=True)
    for sess in candidates:
        if any(turn.question for turn in sess.turns.values()):
            return sess
    return None


def _delete_existing_turns(session_id: int) -> None:
    with review._db_lock, review._connect(transactional=True) as conn:
        conn.execute("DELETE FROM review_turns WHERE session_id = ?", (session_id,))
        conn.execute(
            "UPDATE review_sessions SET turn_count = 0, updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )


def recover_session(
    log_session: LogSession,
    *,
    session_id: Optional[int],
    overwrite: bool,
    dry_run: bool,
    knowledge_db_path: Optional[Path] = None,
) -> int:
    if knowledge_db_path is not None:
        backfill_turns_from_knowledge_db(log_session, knowledge_db_path=knowledge_db_path)

    turns = [turn for turn in log_session.turns.values() if turn.question.strip()]
    if not turns:
        raise RuntimeError("日志会话里没有可恢复的问题")

    if session_id is None:
        if dry_run:
            return 0
        session_id = review.create_session(
            started_at=log_session.started_at,
            interviewer_enabled=True,
            candidate_enabled=True,
            source="log_recovery",
            title="日志恢复的面试复盘",
        )
    else:
        detail = review.get_session_detail(session_id)
        if detail is None:
            raise RuntimeError(f"review session 不存在: {session_id}")
        if detail.get("turn_count", 0) > 0 and not overwrite:
            raise RuntimeError(f"session {session_id} 已有 turns；如需重建请加 --overwrite")
        if not dry_run:
            _delete_existing_turns(session_id)

    for idx, turn in enumerate(turns, start=1):
        candidate_answer = turn.candidate_answer
        is_partial = not candidate_answer or not turn.answer_done
        if dry_run:
            print(
                f"[DRY] #{idx} {turn.qa_id} partial={is_partial} "
                f"question={turn.question[:80]!r} candidate_chars={len(candidate_answer)} "
                f"redacted_chars={turn.redacted_candidate_chars} backfilled={turn.candidate_backfilled}"
            )
            continue
        review.add_turn(
            session_id=session_id,
            qa_id=turn.qa_id,
            seq=idx,
            question_text=turn.question,
            candidate_answer_text=candidate_answer,
            reference_answer_text=turn.reference_answer,
            duration_ms=0,
            is_partial=is_partial,
            analysis_status="pending",
        )

    if not dry_run:
        ended_at = log_session.ended_at or log_session.last_event_at or time.time()
        status = "partial_capture" if any(
            (not turn.candidate_answer) or (not turn.answer_done)
            for turn in turns
        ) else "recorded"
        review.end_session(session_id=session_id, status=status, ended_at=ended_at)
    return int(session_id or 0)


def main(argv: Optional[list[str]] = None) -> None:
    backend_dir = Path(__file__).resolve().parent.parent
    default_log = backend_dir.parent / "log" / "interview.log"
    parser = argparse.ArgumentParser(description="Recover review turns from interview.log")
    parser.add_argument("--log", default=str(default_log), help="interview.log path")
    parser.add_argument(
        "--knowledge-db",
        default=str(knowledge.DB_PATH),
        help="knowledge.db path for backfilling redacted candidate/reference answers by qa_id",
    )
    parser.add_argument(
        "--no-knowledge-db",
        action="store_true",
        help="disable knowledge.db backfill (redacted candidate answers will remain empty)",
    )
    parser.add_argument("--session-id", type=int, default=None, help="existing review session id")
    parser.add_argument("--since", default="", help='start time, e.g. "2026-06-24 10:48:35"')
    parser.add_argument("--until", default="", help='end time, e.g. "2026-06-24 11:40:00"')
    parser.add_argument("--overwrite", action="store_true", help="replace existing turns for session")
    parser.add_argument("--dry-run", action="store_true", help="show recovered turns without writing")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"日志不存在: {log_path}")

    target_session = None
    if args.session_id is not None:
        target_session = review.get_session_detail(args.session_id)
        if target_session is None:
            raise SystemExit(f"review session 不存在: {args.session_id}")
    elif not args.since:
        target_session = review.get_current_session()

    since = _parse_datetime(args.since) if args.since else None
    until = _parse_datetime(args.until) if args.until else None
    started_at = float(target_session["started_at"]) if target_session else since

    sessions = parse_log_sessions(log_path, until=until)
    selected = select_log_session(
        sessions,
        started_at=started_at,
        since=since,
        until=until,
    )
    if selected is None:
        raise SystemExit("没有找到匹配的日志会话")

    kb_path = None if args.no_knowledge_db or not args.knowledge_db else Path(args.knowledge_db)
    if kb_path is not None:
        stats = backfill_turns_from_knowledge_db(selected, knowledge_db_path=kb_path)
        print(
            "knowledge.db 回填统计: "
            f"matched={stats['matched_turns']} "
            f"candidate_filled={stats['candidate_filled']} "
            f"reference_filled={stats['reference_filled']}"
        )

    unresolved_redacted = sum(
        1
        for turn in selected.turns.values()
        if turn.redacted_candidate_chars > 0 and not turn.candidate_answer
    )
    if unresolved_redacted:
        print(
            f"[NOTE] 有 {unresolved_redacted} 轮候选人回答在日志中已脱敏（仅保留 chars）"
            "且在 knowledge.db 中未找到可回填记录，将标记为 is_partial=True。"
        )

    target_id = args.session_id or (int(target_session["id"]) if target_session else None)
    print(
        "选中日志会话: "
        f"start={datetime.fromtimestamp(selected.started_at).strftime(TS_FORMAT)} "
        f"last={datetime.fromtimestamp(selected.last_event_at or selected.started_at).strftime(TS_FORMAT)} "
        f"turns={len(selected.turns)} target_session={target_id or 'new'}"
    )
    recovered_id = recover_session(
        selected,
        session_id=target_id,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[DRY] 未写入数据库")
    else:
        print(f"[OK] 已恢复到 review session {recovered_id}")


if __name__ == "__main__":
    main()
