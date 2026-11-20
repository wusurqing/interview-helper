import json
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.analytics.router import router as knowledge_router
from api.assist import pipeline
from api.common.router import router as common_router
from core import config
from services import knowledge_analysis
from services.llm import protocols, streaming, trace
from services.storage import knowledge


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_analysis, "_job", None)
    monkeypatch.setattr(knowledge, "DB_PATH", str(tmp_path / "knowledge.db"))
    knowledge.init_db()
    monkeypatch.setattr(config, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(config, "_config", config.AppConfig(models=[
        config.ModelConfig(name="Assist", model="assist-model", api_key="test-assist"),
        config.ModelConfig(name="Analysis", model="analysis-model", api_key="test-analysis", supports_think=True, supports_vision=True),
    ]))
    app = FastAPI()
    app.include_router(knowledge_router, prefix="/api")
    app.include_router(common_router, prefix="/api")
    return TestClient(app)


def test_model_options_include_vision_capability(client):
    models = client.get("/api/knowledge/settings").json()["models"]
    assert [model["supports_vision"] for model in models] == [False, True]


def _configure(client, **overrides):
    response = client.get("/api/knowledge/settings")
    assert response.status_code == 200
    payload = response.json()
    settings = {
        **payload["settings"],
        "model_index": 1,
        "model_fingerprint": payload["models"][1]["fingerprint"],
        **overrides,
    }
    saved = client.put("/api/knowledge/settings", json=settings)
    assert saved.status_code == 200, saved.text
    return saved.json()


def _finish(client):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        result = client.get("/api/knowledge/analysis").json()
        if result["job"]["status"] != "running":
            return result
        sleep(0.01)
    pytest.fail("analysis job did not finish")


def _completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_manual_analysis_processes_pending_records_and_retries_failures(client, monkeypatch, caplog):
    caplog.set_level("INFO", logger="knowledge.tags")
    calls = []

    def completion(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        if "会失败" in messages[1]["content"] and len(calls) == 1:
            return _completion("")
        return _completion('["Redis", "限流", "令牌桶"]')

    monkeypatch.setattr(streaming, "chat_completion_create", completion)
    knowledge.save_record("assist", "旧记录", "", tags=["已有"], score=8)
    knowledge.save_record("assist", "会失败", "参考答案")
    knowledge.save_record("assist", "新问题", "参考答案", candidate_answer="口述答案")
    _configure(client, system_prompt="自定义系统提示词", temperature=0.7, max_tokens=2048, think_effort="low")
    assert client.get("/api/knowledge/analysis").json()["pending_count"] == 2
    assert calls == []
    assert "KNOWLEDGE_TAGS_REQUEST" not in caplog.text
    assert client.post("/api/knowledge/analysis").status_code == 202
    result = _finish(client)
    assert result["job"]["succeeded"] == result["job"]["failed"] == result["pending_count"] == 1
    assert result["job"]["skipped"] == 0
    assert len(calls) == 2
    model, messages, kwargs = calls[1]
    assert (model.model, model.api_key) == ("analysis-model", "test-analysis")
    assert messages[0]["role"] == "system"
    assert "自定义系统提示词" in messages[0]["content"] and "JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert all(text in messages[1]["content"] for text in ["新问题", "口述答案", "参考答案"])
    assert (kwargs["temperature"], kwargs["max_tokens"]) == (0.7, 2048)
    assert kwargs["think_params"]
    assert caplog.text.count("KNOWLEDGE_TAGS_REQUEST") == 2

    assert client.post("/api/knowledge/analysis").status_code == 202
    assert _finish(client)["pending_count"] == 0
    assert len(calls) == 3
    assert client.post("/api/knowledge/analysis").status_code == 200
    assert len(calls) == 3
    history = client.get("/api/knowledge/history").json()["records"]
    old = next(record for record in history if record["question"] == "旧记录")
    assert (old["tags"], old["score"]) == (["已有"], 8)
    summary = client.get("/api/knowledge/summary").json()["tags"]
    assert any(row["tag"] == "Redis" and row["count"] == 2 for row in summary)


def test_live_qa_is_traced_only_after_manual_analysis(client, monkeypatch, tmp_path):
    trace_file = tmp_path / "llm.jsonl"
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    model = config.get_config().models[1]
    model.api_type = "anthropic"
    model.api_base_url = "https://analysis.example.test"
    tags = ["Redis", "限流", "令牌桶"]

    def post(*_args, **_kwargs):
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": json.dumps(tags, ensure_ascii=False)}],
        })

    monkeypatch.setattr(protocols._http, "post", post)
    _configure(client, system_prompt="提取口述知识点", temperature=0.7, max_tokens=2048)
    pipeline._save_knowledge_record("save", "Redis 如何限流？", "可以用令牌桶。", "qa-live")
    pipeline._save_knowledge_record("candidate_update", "qa-live", "我介绍了令牌桶。")

    record = client.get("/api/knowledge/history").json()["records"][0]
    assert record["candidate_answer"] == "我介绍了令牌桶。"
    assert record["tags"] == []
    assert client.get("/api/knowledge/analysis").json()["pending_count"] == 1
    assert not trace_file.exists()

    started = client.post("/api/knowledge/analysis")
    assert started.status_code == 202
    job_id = started.json()["job"]["job_id"]
    assert _finish(client)["job"]["succeeded"] == 1
    request, result = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert (request["event"], result["event"]) == ("request", "result")
    assert request["context"] == result["context"] == {
        "scene": "knowledge_tags", "job_id": job_id, "record_id": record["id"],
    }
    assert request["call_id"] == result["call_id"]
    assert result["status"] == "completed"
    payload = request["request"]
    assert payload["model"] == "analysis-model"
    assert "提取口述知识点" in payload["system"]
    assert (payload["temperature"], payload["max_tokens"]) == (0.7, 2048)
    assert "我介绍了令牌桶。" in payload["messages"][0]["content"]
    assert client.get("/api/knowledge/history").json()["records"][0]["tags"] == tags


@pytest.mark.parametrize("change", ["candidate", "reset"])
def test_inflight_analysis_skips_changed_or_deleted_rows(client, monkeypatch, change):
    entered, release = Event(), Event()

    def completion(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return _completion('["Redis", "限流", "令牌桶"]')

    monkeypatch.setattr(streaming, "chat_completion_create", completion)
    knowledge.save_record("assist", "题目", "", qa_id="qa")
    _configure(client)
    first = client.post("/api/knowledge/analysis").json()["job"]
    try:
        assert entered.wait(5)
        again = client.post("/api/knowledge/analysis").json()["job"]
        assert again["job_id"] == first["job_id"]
        if change == "candidate":
            assert knowledge.update_candidate_answer_for_qa("qa", "补充回答")
        else:
            assert client.delete("/api/knowledge/reset").status_code == 200
        knowledge.save_record("assist", "新问题", "")
    finally:
        release.set()
    result = _finish(client)
    assert result["job"]["succeeded"] == 0
    assert result["job"]["skipped"] == 1
    assert result["pending_count"] == (2 if change == "candidate" else 1)
    assert client.get("/api/knowledge/summary").json() == {"tags": []}


def test_settings_survive_reorder_and_clear_if_selected_model_is_deleted(client):
    initial = _configure(client)
    assert initial["settings"]["max_tokens"] == 1024
    assert "test-analysis" not in str(initial)
    assert client.post("/api/config/models-layout", json={"order": [1, 0]}).status_code == 200
    assert client.get("/api/knowledge/settings").json()["settings"]["model_index"] == 0
    assert client.post("/api/config", json={"models": [
        {"name": "Assist", "model": "assist-model", "api_key": "test-assist", "model_original_index": 1},
    ]}).status_code == 409
    layout = client.get("/api/model-providers").json()
    analysis, assist = layout["models"]
    layout["models"] = [assist]
    layout["confirmed_remove_ids"] = [analysis["id"]]
    assert client.put("/api/model-providers", json=layout).status_code == 200
    assert client.get("/api/knowledge/settings").json()["settings"]["model_index"] is None
    assert config._load_config().knowledge_analysis.model_index is None


def test_unchecking_model_lets_inflight_analysis_finish_and_skips_queued_records(client, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def completion(model, *_args, **_kwargs):
        calls.append(model.id)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return _completion('["Redis", "限流", "令牌桶"]')

    monkeypatch.setattr(streaming, "chat_completion_create", completion)
    knowledge.save_record("assist", "当前记录", "")
    knowledge.save_record("assist", "排队记录", "")
    _configure(client)
    assert client.post("/api/knowledge/analysis").status_code == 202
    try:
        assert entered.wait(5)
        layout = client.get("/api/model-providers").json()
        selected = next(model for model in layout["models"] if model["id"] == calls[0])
        selected["enabled"] = False
        layout["confirmed_remove_ids"] = [selected["id"]]
        assert client.put("/api/model-providers", json=layout).status_code == 200
    finally:
        release.set()
    result = _finish(client)
    assert len(calls) == 1
    assert result["job"]["succeeded"] == 1
    assert result["job"]["skipped"] == 1
    assert result["pending_count"] == 1


def test_model_list_save_does_not_rebind_analysis_to_a_stale_origin(client):
    _configure(client)
    assert client.post("/api/config/models-layout", json={"order": [1, 0]}).status_code == 200
    assert client.post("/api/config", json={"models": [
        {"name": "Assist", "model": "assist-model", "api_key": "test-assist", "model_original_index": 0},
        {"name": "Analysis", "model": "analysis-model", "api_key": "test-analysis", "model_original_index": 1},
    ]}).status_code == 200
    saved = client.get("/api/knowledge/settings").json()
    assert saved["settings"]["model_index"] == 1
