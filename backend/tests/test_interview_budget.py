import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from api.assist.mmpg_turns import MmpgRequest
from services.llm import protocols, streaming, trace
from services.llm.streaming import LLMContextExceeded, LLMError


def _model(*, window=1000, api_type="openai"):
    return SimpleNamespace(
        context_window_tokens=window, api_type=api_type, api_key="sk-test",
        api_base_url="https://example.com/v1", model="fixture", name="fixture",
        supports_vision=False, supports_think=False,
    )


def test_budget_keeps_newest_contiguous_whole_rounds_and_counts_rendered_utf8():
    from services.llm.interview_budget import prepare_interview_prompt

    # After reserving output, 156 units fit. Overhead, Chinese text and
    # separators leave space for two rounds, but not the third whole round.
    rounds = [["old " + "x" * 34, "old answer"], ["提问", "回答"], ["当前问题"]]
    prompt = prepare_interview_prompt(
        _model(window=200), rounds, system_prompt="系统", context_prefix="资料",
        override_max_tokens=44,
    )
    assert prompt.user_text == "资料\n\n提问\n\n回答\n\n当前问题"
    assert prompt.retained_rounds == 2


def test_budget_cannot_skip_an_oversized_recent_round_to_use_an_older_round():
    from services.llm.interview_budget import prepare_interview_prompt

    prompt = prepare_interview_prompt(
        _model(window=200), [["short old"], ["long " + "x" * 100], ["current"]],
        override_max_tokens=44,
    )
    assert prompt.user_text == "current"


@pytest.mark.parametrize(("output_limit", "current_bytes"), [(10, 116), (50, 86)])
def test_budget_reserves_the_larger_of_window_headroom_and_output(output_limit, current_bytes):
    from services.llm.interview_budget import prepare_interview_prompt

    model = _model(window=200)
    prompt = prepare_interview_prompt(model, [["x" * current_bytes]], override_max_tokens=output_limit)
    assert len(prompt.user_text) == current_bytes
    with pytest.raises(LLMContextExceeded):
        prepare_interview_prompt(model, [["x" * (current_bytes + 1)]], override_max_tokens=output_limit)


def test_unfit_protected_round_never_starts_transport():
    from services.llm.interview_budget import stream_interview_answer

    def stream(*_args, **_kwargs):
        pytest.fail("mandatory context could not fit")

    with pytest.raises(LLMContextExceeded, match="must fit"):
        list(stream_interview_answer(
            _model(window=150, api_type="responses"), [["题" * 21 + "aa"]],
            system_prompt="系统", override_max_tokens=1, stream_fn=stream,
        ))


@pytest.mark.parametrize(("limit", "retry_user_text"), [
    (None, "资料\n\n最新问"),
    (235, "资料\n\n中间问\n\n中间答\n\n最新问"),
])
def test_context_retry_reduces_history_and_preserves_logical_request(limit, retry_user_text):
    from services.llm.interview_budget import stream_interview_answer

    sent = []
    mmpg_request = SimpleNamespace(cancelled=False)
    cfg = SimpleNamespace(max_tokens=101)

    def stream(_model_cfg, messages, **kwargs):
        sent.append((messages, kwargs))
        if len(sent) == 1:
            raise LLMContextExceeded("fixture failure", context_window_tokens=limit)
        yield "text", "answer"

    chunks = list(stream_interview_answer(
        _model(), [["最老 " + "x" * 30], ["中间问", "中间答"], ["最新问"]],
        system_prompt="system", context_prefix="资料", config_snapshot=cfg,
        mmpg_request=mmpg_request, trace_context={"qa_id": "same"}, stream_fn=stream,
    ))
    assert chunks == [("text", "answer")]
    assert len(sent) == 2
    assert sent[0][0][0]["content"].startswith("资料\n\n最老 ")
    assert sent[1][0] == [{"role": "user", "content": retry_user_text}]
    for _, kwargs in sent:
        assert kwargs["system_prompt"] == "system"
        assert kwargs["override_max_tokens"] == 101
        assert kwargs["mmpg_request"] is mmpg_request
        assert kwargs["config_snapshot"] is cfg
        assert kwargs["trace_context"]["qa_id"] == "same"
    assert sent[0][1]["trace_context"]["call_id"] == sent[1][1]["trace_context"]["call_id"]


@pytest.mark.parametrize(("reason", "expected_attempts"), [
    ("think", 1), ("text", 1), ("cancelled", 1), ("latest_only", 1),
    ("protected_round_exceeds_provider_limit", 1), ("twice", 2),
])
def test_retry_stops_after_output_cancellation_or_unfit_reduction(reason, expected_attempts):
    from services.llm.interview_budget import stream_interview_answer

    sent = []
    mmpg_request = SimpleNamespace(cancelled=False)

    def stream(_model_cfg, messages, **_kwargs):
        sent.append(messages)
        if reason in ("think", "text"):
            yield reason, "visible"
        elif reason == "cancelled":
            mmpg_request.cancelled = True
        limit = 100 if reason == "protected_round_exceeds_provider_limit" else None
        raise LLMContextExceeded("fixture failure", context_window_tokens=limit)

    rounds = [["latest"]] if reason == "latest_only" else [["old"], ["latest"]]
    output = stream_interview_answer(
        _model(), rounds, override_max_tokens=50, mmpg_request=mmpg_request, stream_fn=stream,
    )
    if reason == "cancelled":
        assert list(output) == []
    else:
        if reason in ("think", "text"):
            assert next(output) == (reason, "visible")
        with pytest.raises(LLMContextExceeded):
            list(output)
    assert len(sent) == expected_attempts


@pytest.mark.parametrize(("body", "expected_limit"), [
    ({"error": {"code": "context_length_exceeded", "message": "Maximum context length is 8,192 tokens."}}, 8192),
    ({"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 300 tokens > 128 maximum"}}, 128),
    ({"error": {"code": "context_length_exceeded", "maximum_context_length": 1234}}, 1234),
    ({"error": {"code": "context_length_exceeded",
                "message": "max_tokens plus prompt length exceeds the model context window of 4096 tokens."}}, 4096),
])
def test_raw_http_context_error_keeps_explicit_provider_limit(body, expected_limit):
    with pytest.raises(LLMContextExceeded) as error:
        protocols.raise_for_llm_status(400, json.dumps(body))
    assert error.value.context_window_tokens == expected_limit


@pytest.mark.parametrize("body", [
    {"error": {"message": "Invalid API token"}},
    {"error": {"message": "max_tokens must be less than the maximum context length of 8192 tokens"}},
    {"error": {"message": "Invalid authorization token", "code": "invalid_api_key"}},
    {"error": {"message": "max_output_tokens value must be at most 8192", "param": "max_output_tokens"}},
    {"error": {"code": "invalid_value", "param": "max_tokens",
               "message": "Invalid parameter: 'max_tokens' exceeds the maximum context window of 4096 tokens."}},
])
def test_raw_bad_request_only_classifies_input_context_overflow(body):
    with pytest.raises(LLMError) as error:
        protocols.raise_for_llm_status(400, json.dumps(body))
    assert not isinstance(error.value, LLMContextExceeded)


@pytest.mark.parametrize(("protocol", "error_event"), [
    ("openai", {"error": {"code": "context_length_exceeded", "message": "too many input tokens"}}),
    ("anthropic", {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}),
    ("responses", {"type": "response.failed", "response": {"error": {"code": "context_length_exceeded"}}}),
    ("responses", {"type": "error", "code": "context_length_exceeded", "message": "Context window exceeded"}),
])
def test_stream_error_events_classify_context_overflow(protocol, error_event):
    with pytest.raises(LLMContextExceeded):
        protocols._parse_sse_data(
            "data: " + json.dumps(error_event), protocol, protocols._CHUNK_PARSERS[protocol],
        )


@pytest.mark.parametrize(("provider_error", "status", "retried"), [
    ({"code": "context_length_exceeded", "message": "Context window exceeded"}, 400, True),
    ({"code": "invalid_api_key", "message": "Invalid API token"}, 401, False),
    ({"code": "invalid_value", "message": "max_tokens exceeds allowed maximum of 4096 tokens"}, 400, False),
])
def test_sdk_retry_and_trace_use_one_call_with_distinct_attempts(tmp_path, monkeypatch, provider_error, status, retried):
    from services.llm.interview_budget import stream_interview_answer

    received = []

    def handle(request):
        received.append(json.loads(request.content)["messages"])
        if len(received) == 1:
            return httpx.Response(status, json={"error": provider_error})
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=(
            'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'
        ))

    transport = trace.TracingHttpxClient()
    transport._transport.close()
    transport._transport = httpx.MockTransport(handle)
    model = _model()
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    with OpenAI(api_key=model.api_key, base_url=model.api_base_url, http_client=transport) as client:
        monkeypatch.setattr(streaming, "get_client_for_model", lambda _model_cfg: client)
        result = stream_interview_answer(model, [["old"], ["latest"]], override_max_tokens=50)
        if retried:
            assert list(result) == [("text", "answer")]
        else:
            with pytest.raises(LLMError) as error:
                list(result)
            assert not isinstance(error.value, LLMContextExceeded)

    assert received[0][-1]["content"] == "old\n\nlatest"
    assert [messages[-1]["content"] for messages in received[1:]] == (["latest"] if retried else [])
    rows = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
    results = [row for row in rows if row["event"] == "result"]
    assert [row["status"] for row in results] == (["failed", "completed"] if retried else ["failed"])
    assert len({row["call_id"] for row in rows}) == 1
    assert len({row["attempt_id"] for row in rows}) == len(received)


def test_sdk_budgeted_output_is_authoritative_over_custom_thinking_fields(tmp_path, monkeypatch):
    from services.llm.interview_budget import stream_interview_answer

    received = []

    def handle(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text="data: [DONE]\n\n")

    model = _model()
    model.think_disabled_params = {"max_tokens": 2000, "max_completion_tokens": 2001, "enable_thinking": False}
    cfg = SimpleNamespace(max_tokens=300, think_mode=False, think_effort="off", temperature=0.4)
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    with OpenAI(api_key=model.api_key, base_url=model.api_base_url,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        monkeypatch.setattr(streaming, "get_client_for_model", lambda _model_cfg: client)
        list(stream_interview_answer(model, [["latest"]], override_max_tokens=50, config_snapshot=cfg))

    assert received[0]["max_tokens"] == 50
    assert "max_completion_tokens" not in received[0]
    assert received[0]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_mmpg_overflow_retry_keeps_first_transport_start_and_cleans_up(tmp_path, monkeypatch):
    from services.llm.interview_budget import stream_interview_answer

    starts = []
    received = []
    first_start = None
    mmpg_request = MmpgRequest(
        "group", 1, "latest", ("asr/1",), 0, None,
        on_started=starts.append, loop=asyncio.get_running_loop(),
    )

    async def handle(request):
        nonlocal first_start
        payload = json.loads(request.content)
        received.append(payload["messages"])
        if len(received) == 1:
            first_start = mmpg_request.started_at
            return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=(
            'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'
        ))

    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    async_client = httpx.AsyncClient
    monkeypatch.setattr(protocols.httpx, "AsyncClient", lambda **kwargs: async_client(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    result = await asyncio.to_thread(lambda: list(stream_interview_answer(
        _model(), [["old"], ["latest"]], override_max_tokens=50, mmpg_request=mmpg_request,
        trace_context={"mmpg_group_id": "group", "mmpg_revision": 1},
    )))

    assert result == [("text", "answer")]
    assert received == [[{"role": "user", "content": "old\n\nlatest"}], [{"role": "user", "content": "latest"}]]
    assert first_start is not None
    assert starts == [first_start]
    assert mmpg_request.started_at == first_start
    assert mmpg_request.stream_read is None

    rows = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({row["call_id"] for row in rows}) == 1
    assert [row["context"]["mmpg_group_id"] for row in rows] == ["group"] * 4
