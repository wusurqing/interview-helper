"""Atomic, idempotent round writes; protected media is never a static mount."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time

from services.storage import review
from services.storage.paths import data_dir


def media_root():
    return Path(data_dir()) / "review_media"


def set_state(session_id, state, warning=None):
    with review._db_lock, review._connect(transactional=True) as conn:
        conn.execute(
            "UPDATE review_sessions SET archive_status=?, archive_warning=?, archive_version=1 WHERE id=?",
            (state, warning, session_id),
        )


def set_candidate_capture(session_id, enabled):
    with review._db_lock, review._connect(transactional=True) as conn:
        conn.execute("UPDATE review_sessions SET candidate_capture_enabled=? WHERE id=?", (int(enabled), session_id))


def finish_session(session_id, warning):
    now = time.time()
    with review._db_lock, review._connect(transactional=True) as conn:
        conn.execute(
            """UPDATE review_sessions SET status=?, archive_status=?, archive_warning=?,
                ended_at=?, updated_at=? WHERE id=?""",
            (
                "partial_capture" if warning else "recorded",
                "partial" if warning else "complete",
                warning or None,
                now,
                now,
                session_id,
            ),
        )


def save_round(record):
    sid, rid = record["session_id"], record["round_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
        raise ValueError("invalid round id")
    paths = {}
    for asset_id, payload in record.get("assets", {}).items():
        if not re.fullmatch(r"[a-f0-9]{64}", asset_id):
            raise ValueError("invalid asset id")
        relative = Path(str(sid)) / rid / (asset_id + ".ogg")
        path = media_root() / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        paths[asset_id] = relative.as_posix()
    details = {k: record[k] for k in ("round_id", "policy_version", "utterances", "generations")}
    if record.get("transcription_timeouts"):
        details["transcription_timeouts"] = record["transcription_timeouts"]
    now = time.time()
    with review._db_lock, review._connect(transactional=True) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO review_turns (
                session_id, qa_id, seq, question_text, candidate_answer_text, reference_answer_text,
                created_at, updated_at, round_id) VALUES (?,?,?,?,?,?,?,?,?)""",
            (sid, rid, record["seq"], "", "", "", now, now, rid),
        )
        tid = conn.execute("SELECT id FROM review_turns WHERE session_id=? AND round_id=?", (sid, rid)).fetchone()["id"]
        conn.execute(
            """UPDATE review_turns SET question_text=?, candidate_answer_text=?,
                reference_answer_text=?, candidate_status=?, ai_status=?, archive_version=1,
                started_at=?, ended_at=?, media_json=?, model_summary_json=?, audio_status_json=?, updated_at=? WHERE id=?""",
            (
                record["question_text"],
                record["candidate_answer_text"],
                record["reference_answer_text"],
                record["candidate_status"],
                record["ai_status"],
                record["started_at"],
                record["ended_at"],
                json.dumps(record["media"], ensure_ascii=False),
                json.dumps(record.get("model_summary"), ensure_ascii=False),
                json.dumps(record.get("audio_status"), ensure_ascii=False),
                now,
                tid,
            ),
        )
        conn.execute("INSERT OR REPLACE INTO review_archive_payloads VALUES (?,?)", (tid, json.dumps(details, ensure_ascii=False)))
        for asset_id, relative in paths.items():
            conn.execute("INSERT OR REPLACE INTO review_audio_assets VALUES (?,?,?,?)", (sid, tid, asset_id, relative))
        conn.execute(
            """UPDATE review_sessions SET turn_count=(SELECT COUNT(*) FROM review_turns WHERE session_id=?),
                updated_at=? WHERE id=?""",
            (sid, now, sid),
        )
        return tid


def get_archive(session_id, turn_id):
    with review._db_lock, review._connect() as conn:
        row = conn.execute(
            """SELECT p.details_json FROM review_archive_payloads p
                JOIN review_turns t ON t.id=p.turn_id WHERE t.session_id=? AND t.id=?""",
            (session_id, turn_id),
        ).fetchone()
        return json.loads(row["details_json"]) if row else None


def asset_path(session_id, turn_id, asset_id):
    if not re.fullmatch(r"[a-f0-9]{64}", asset_id):
        return None
    with review._db_lock, review._connect() as conn:
        row = conn.execute(
            "SELECT relative_path FROM review_audio_assets WHERE session_id=? AND turn_id=? AND asset_id=?",
            (session_id, turn_id, asset_id),
        ).fetchone()
    if not row:
        return None
    root = media_root().resolve()
    path = (root / row["relative_path"]).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def recover_interrupted():
    """Keep committed rounds; discard only unreferenced temporary archive files."""
    now = time.time()
    with review._db_lock, review._connect(transactional=True) as conn:
        conn.execute(
            """UPDATE review_sessions SET status='partial_capture', archive_status='partial',
                archive_warning=COALESCE(archive_warning, '面试未正常结束，已保留写入完成的轮次'),
                ended_at=COALESCE(ended_at, ?), updated_at=?
                WHERE archive_version=1 AND (archive_status IN ('recording','draining') OR status='recording')""",
            (now, now),
        )
        referenced = {row["relative_path"] for row in conn.execute("SELECT relative_path FROM review_audio_assets")}
    root = media_root()
    if root.exists():
        for path in root.glob("*/*/*"):
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                continue
            if path.suffix in (".tmp", ".ogg") and path.relative_to(root).as_posix() not in referenced:
                path.unlink(missing_ok=True)
