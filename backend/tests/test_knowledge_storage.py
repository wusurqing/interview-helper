from __future__ import annotations

from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.assist import pipeline  # noqa: E402
from core import config  # noqa: E402
from services.llm import streaming  # noqa: E402
from services.storage import knowledge  # noqa: E402


@pytest.fixture
def isolated_knowledge_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(knowledge, "DB_PATH", str(tmp_path / "knowledge.db"))
    knowledge.init_db()
    knowledge.reset_all()
    return knowledge


def test_assist_history_includes_optional_candidate_answer(isolated_knowledge_db):
    store = isolated_knowledge_db

    store.save_record(
        "assist",
        "Redis 怎么持久化？",
        "助手建议：RDB 和 AOF。",
        qa_id="qa-1",
        tags=["Redis"],
    )
    assert store.update_candidate_answer_for_qa(
        "qa-1",
        "我实际回答了 RDB 快照、AOF 追加日志，以及重写机制。",
    )

    history = store.get_history(1, 20, ("assist",))
    assert history["records"][0]["qa_id"] == "qa-1"
    assert history["records"][0]["candidate_answer"] == "我实际回答了 RDB 快照、AOF 追加日志，以及重写机制。"


def test_records_without_candidate_voice_stay_compatible(isolated_knowledge_db):
    store = isolated_knowledge_db

    store.save_record("assist", "MySQL 索引？", "B+ 树。", tags=["MySQL"])

    history = store.get_history(1, 20, ("assist",))
    assert history["records"][0]["qa_id"] == ""
    assert history["records"][0]["candidate_answer"] == ""


def test_candidate_answer_update_keeps_existing_tags(isolated_knowledge_db):
    knowledge.save_record("assist", "Redis 如何限流？", "助手建议：令牌桶。", qa_id="qa-3", tags=["Redis", "限流"])

    assert knowledge.update_candidate_answer_for_qa("qa-3", "我实际回答了令牌桶和漏桶。")

    history = knowledge.get_history(1, 20, ("assist",))
    assert history["records"][0]["candidate_answer"] == "我实际回答了令牌桶和漏桶。"
    assert history["records"][0]["tags"] == ["Redis", "限流"]


def test_live_answer_and_candidate_storage_make_no_model_calls(isolated_knowledge_db, monkeypatch):
    monkeypatch.setattr(config, "_config", config.AppConfig())
    calls = 0

    def record_model_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("model fixture")

    monkeypatch.setattr(streaming, "chat_completion_create", record_model_call)
    pipeline._save_knowledge_record("save", "Redis 如何限流？", "助手建议：令牌桶。", "qa-3", "")
    pipeline._save_knowledge_record("candidate_update", "qa-3", "我实际回答了令牌桶和漏桶。")

    history = knowledge.get_history(1, 20, ("assist",))
    assert history["records"][0]["answer"] == "助手建议：令牌桶。"
    assert history["records"][0]["candidate_answer"] == "我实际回答了令牌桶和漏桶。"
    assert history["records"][0]["tags"] == []
    assert calls == 0
