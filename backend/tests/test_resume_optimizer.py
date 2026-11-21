from __future__ import annotations

import importlib
import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

resume_optimizer = importlib.import_module("services.resume_optimizer")
resume_router = importlib.import_module("api.resume.router")


class _FakeCfg:
    resume_text = "负责接口优化\n建设缓存治理"

    def get_active_model(self):
        return SimpleNamespace(model="fake-model")


def test_resume_optimizer_marks_rewrite_suggestions_without_line_references(monkeypatch):
    calls = {"stream": 0}

    monkeypatch.setattr(resume_optimizer, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(resume_optimizer, "get_client", lambda: object())
    monkeypatch.setattr(
        resume_optimizer,
        "_non_stream_chat",
        lambda *args, **kwargs: '{"must_have":["Redis"],"nice_to_have":[],"soft_skills":[]}',
    )

    def fake_stream(*args, **kwargs):
        calls["stream"] += 1
        if calls["stream"] == 1:
            yield '### ✅ 命中能力\n- **Redis** — L2 / "建设缓存治理"'
            return
        yield "#### 建议 1 — 强化量化结果\n- 改写: 主导接口优化并降低延迟。"

    monkeypatch.setattr(resume_optimizer, "_stream_chat", fake_stream)

    output = "".join(resume_optimizer.optimize_resume_stream("需要 Redis 经验"))

    assert "需人工确认" in output
    assert "缺少行号引用" in output


def test_resume_optimizer_accepts_rewrite_suggestions_with_line_references(monkeypatch):
    calls = {"stream": 0}

    monkeypatch.setattr(resume_optimizer, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(resume_optimizer, "get_client", lambda: object())
    monkeypatch.setattr(
        resume_optimizer,
        "_non_stream_chat",
        lambda *args, **kwargs: '{"must_have":["Redis"],"nice_to_have":[],"soft_skills":[]}',
    )

    def fake_stream(*args, **kwargs):
        calls["stream"] += 1
        if calls["stream"] == 1:
            yield '### ✅ 命中能力\n- **Redis** — L2 / "建设缓存治理"'
            return
        yield '#### 建议 1 — 强化量化结果\n- 引用: **L1** / "负责接口优化"\n- 改写: 主导接口优化。'

    monkeypatch.setattr(resume_optimizer, "_stream_chat", fake_stream)

    output = "".join(resume_optimizer.optimize_resume_stream("需要 Redis 经验"))

    assert "缺少行号引用" not in output


def test_resume_opt_router_stops_broadcasting_stale_job_chunks(monkeypatch):
    events: list[dict] = []

    monkeypatch.setattr(resume_router, "broadcast", lambda payload: events.append(payload))

    def fake_stream(_jd: str, *, system_prompts):
        yield "old-visible"
        resume_router._resume_opt_current_job_id = "job-new"
        yield "old-stale"

    monkeypatch.setattr(resume_router, "optimize_resume_stream", fake_stream)
    monkeypatch.setattr(resume_router, "_resume_opt_current_job_id", "job-old")

    resume_router._run_optimize("需要 Redis 经验", "job-old", {})

    assert [event["type"] for event in events] == ["resume_opt_start", "resume_opt_chunk"]
    assert all(event["job_id"] == "job-old" for event in events)
    assert events[1]["chunk"] == "old-visible"


def test_resume_job_snapshots_prompts_across_queue_and_stream_stages(monkeypatch):
    config = _FakeCfg()
    config.system_prompts = {
        "resume_jd": "旧抽取 {字段}",
        "resume_match": "旧匹配 {字段}",
        "resume_rewrite": "旧改写 {字段}",
    }
    monkeypatch.setattr(resume_optimizer, "get_config", lambda: config)
    monkeypatch.setattr(resume_router, "get_config", lambda: config)
    monkeypatch.setattr(resume_router, "_resume_opt_current_job_id", None)
    monkeypatch.setattr(resume_optimizer, "get_client", lambda: object())
    events, jobs, calls = [], [], []
    monkeypatch.setattr(resume_router, "broadcast", events.append)
    monkeypatch.setattr(
        resume_router, "submit_low_priority_background",
        lambda func, *args: jobs.append(lambda: func(*args)) or True,
    )

    def extract(_client, _model, messages, **kwargs):
        calls.append(messages)
        config.system_prompts["resume_rewrite"] = "执行中改写"
        return '{"must_have":["Redis"],"nice_to_have":[],"soft_skills":[]}'

    def stream(_client, _model, messages, **kwargs):
        calls.append(messages)
        yield "匹配结果 L2" if len(calls) == 2 else "改写结果"

    monkeypatch.setattr(resume_optimizer, "_non_stream_chat", extract)
    monkeypatch.setattr(resume_optimizer, "_stream_chat", stream)
    asyncio.run(resume_router.api_resume_optimize(resume_router.OptimizeRequest(jd="输入 JD {Redis}")))
    config.system_prompts.update({key: "排队时修改" for key in config.system_prompts})
    jobs.pop()()

    assert len(calls) == 3
    for messages, prompt, data in zip(
        calls, ("旧抽取", "旧匹配", "旧改写"),
        ("输入 JD {Redis}", "Redis", "匹配结果 L2"),
    ):
        system, user = messages
        assert system["role"] == "system"
        assert system["content"].startswith(f"{prompt} {{字段}}")
        assert data not in system["content"]
        assert user["role"] == "user"
        assert data in user["content"]
    assert "must_have" in calls[0][0]["content"]
    assert calls[1][0]["content"] == "旧匹配 {字段}"
    assert calls[2][0]["content"] == "旧改写 {字段}"
    assert "改写结果" in events[-2]["text"]
    assert "缺少行号引用" not in events[-2]["text"]


@pytest.mark.parametrize("invalid", [
    '{"must_have":["Redis", {"name":"HTTP"}],"nice_to_have":[],"soft_skills":[]}',
    '{"must_have":"Redis","nice_to_have":[],"soft_skills":[]}',
    '{"must_have":["Redis"],"nice_to_have":[]}',
    '额外说明',
])
def test_invalid_jd_extraction_uses_custom_fallback(monkeypatch, invalid):
    config = _FakeCfg()
    config.system_prompts = {"resume_fallback": "自定义综合分析 {字面量}"}
    monkeypatch.setattr(resume_optimizer, "get_config", lambda: config)
    monkeypatch.setattr(resume_optimizer, "get_client", lambda: object())
    monkeypatch.setattr(resume_optimizer, "_non_stream_chat", lambda *args, **kwargs: invalid)
    calls = []

    def stream(_client, _model, messages, **kwargs):
        calls.append(messages)
        yield "兜底输出"

    monkeypatch.setattr(resume_optimizer, "_stream_chat", stream)
    output = "".join(resume_optimizer.optimize_resume_stream("输入 JD"))

    assert "抽取步骤失败" in output
    assert "兜底输出" in output
    assert len(calls) == 1
    system, user = calls[0]
    assert system["role"] == "system"
    assert system["content"] == "自定义综合分析 {字面量}"
    assert user["role"] == "user"
    assert "输入 JD" in user["content"] and config.resume_text in user["content"]


@pytest.mark.parametrize("api_type", ["openai", "anthropic", "responses"])
def test_resume_transports_keep_system_prompts_and_input_separate(monkeypatch, api_type):
    from core.config import ModelConfig

    model = ModelConfig(model="test-model", api_key="test-key", api_type=api_type)
    config = _FakeCfg()
    config.system_prompts = {
        "resume_jd": "抽取规则",
        "resume_match": "匹配规则",
        "resume_rewrite": "改写规则",
    }
    monkeypatch.setattr(config, "get_active_model", lambda: model)
    monkeypatch.setattr(resume_optimizer, "get_config", lambda: config)
    payloads = []
    jd_result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"must_have":["Redis"],"nice_to_have":[],"soft_skills":[]}',
    ))])
    stream_result = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="建议 L2"))])]

    def openai_create(**payload):
        payloads.append(payload)
        return iter(stream_result) if payload.get("stream") else jd_result

    def post_blocking(_url, protocol, _key, payload, **kwargs):
        assert protocol == api_type
        payloads.append(payload)
        return jd_result

    def post_stream(_url, protocol, _key, payload):
        assert protocol == api_type
        payloads.append(payload)
        return iter(stream_result)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=openai_create)))
    monkeypatch.setattr(resume_optimizer, "get_client", lambda: client)
    monkeypatch.setattr("services.llm.streaming.get_client_for_model", lambda _: client)
    monkeypatch.setattr("services.llm.streaming.post_chat_blocking", post_blocking)
    monkeypatch.setattr("services.llm.protocols.post_chat_stream", post_stream)
    output = "".join(resume_optimizer.optimize_resume_stream("JD 输入"))

    assert "建议 L2" in output
    assert len(payloads) == 3
    for payload, expected in zip(payloads, ("抽取规则", "匹配规则", "改写规则")):
        if api_type == "anthropic":
            system, messages = payload["system"], payload["messages"]
        elif api_type == "responses":
            system, messages = payload["instructions"], payload["input"]
        else:
            system_message, *messages = payload["messages"]
            assert system_message["role"] == "system"
            system = system_message["content"]
        assert system.startswith(expected)
        assert "JD 输入" not in system
        assert [message["role"] for message in messages] == ["user"]
        assert "JD" in messages[0]["content"]


def test_resume_stream_chat_passes_timeout_completion_tokens_and_think_params():
    from core.config import ModelConfig

    captured = {}
    closed = {"called": False}

    class FakeStream:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="chunk"))], usage=None)

        def close(self):
            closed["called"] = True

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: captured.update(kwargs) or FakeStream(),
            )
        )
    )
    o3_model = ModelConfig(model="o3-mini", api_key="sk-test", api_type="openai")
    assert list(
        resume_optimizer._stream_chat(
            client,
            o3_model,
            [{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_tokens=500,
        )
    ) == ["chunk"]
    assert captured["max_completion_tokens"] == 500
    assert "max_tokens" not in captured
    assert captured["timeout"] == 90.0
    assert closed["called"] is True

    captured.clear()
    r1_model = ModelConfig(model="deepseek-r1", api_key="sk-test", api_type="openai")
    list(
        resume_optimizer._stream_chat(
            client,
            r1_model,
            [{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_tokens=500,
        )
    )
    assert captured["extra_body"] == {
        "thinking": {"type": "disabled"},
        "think_mode": False,
        "enable_thinking": False,
    }


def test_resume_router_closes_stream_when_job_becomes_stale(monkeypatch):
    events: list[dict] = []
    closed = {"called": False}
    monkeypatch.setattr(resume_router, "broadcast", events.append)

    def fake_stream(_jd: str, *, system_prompts):
        try:
            yield "first-chunk"
            resume_router._resume_opt_current_job_id = "job-newer"
            yield "second-chunk"
        finally:
            closed["called"] = True

    monkeypatch.setattr(resume_router, "optimize_resume_stream", fake_stream)
    monkeypatch.setattr(resume_router, "_resume_opt_current_job_id", "job-old")

    resume_router._run_optimize("JD", "job-old", {})
    assert closed["called"] is True
