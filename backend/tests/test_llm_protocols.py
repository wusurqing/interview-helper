"""LLM 协议层单测：URL 规范化 / 消息转换 / payload 构建 / SSE 解析。"""

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.llm import protocols
from services.llm.streaming import LLMError


class _Model:
    def __init__(self, api_type="openai", base="https://api.example.com"):
        self.name = "t"
        self.api_base_url = base
        self.api_key = "sk-test"
        self.model = "m-1"
        self.api_type = api_type


# ---------------------------------------------------------------------------
# URL normalization & candidates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url,api_type,expected",
    [
        ("https://api.openai.com/v1", "openai", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1/", "openai", "https://api.openai.com/v1"),
        ("https://api.deepseek.com", "openai", "https://api.deepseek.com/v1"),
        ("https://ark.cn-beijing.volces.com/api/v3", "openai", "https://ark.cn-beijing.volces.com/api/v3"),
        ("https://api.anthropic.com", "anthropic", "https://api.anthropic.com"),
        ("https://api.anthropic.com/v1", "anthropic", "https://api.anthropic.com"),
        ("https://api.anthropic.com/v1/", "anthropic", "https://api.anthropic.com"),
        ("https://gateway/api/anthropic", "anthropic", "https://gateway/api/anthropic"),
        ("https://api.openai.com", "responses", "https://api.openai.com/v1"),
    ],
)
def test_normalize_api_base_url(url, api_type, expected):
    assert protocols.normalize_api_base_url(url, api_type) == expected


@pytest.mark.parametrize(
    "url,api_type,expected",
    [
        (
            "https://api.openai.com/v1",
            "openai",
            ["https://api.openai.com/v1/chat/completions"],
        ),
        (
            "https://api.deepseek.com",
            "openai",
            [
                "https://api.deepseek.com/v1/chat/completions",
                "https://api.deepseek.com/chat/completions",
            ],
        ),
        (
            "https://api.anthropic.com",
            "anthropic",
            ["https://api.anthropic.com/v1/messages", "https://api.anthropic.com/messages"],
        ),
        (
            "https://api.anthropic.com/v1",
            "anthropic",
            ["https://api.anthropic.com/v1/messages"],
        ),
        (
            "https://api.openai.com/v1",
            "responses",
            ["https://api.openai.com/v1/responses"],
        ),
        (
            "https://api.openai.com",
            "responses",
            ["https://api.openai.com/v1/responses", "https://api.openai.com/responses"],
        ),
    ],
)
def test_chat_url_candidates(url, api_type, expected):
    assert protocols.chat_url_candidates(url, api_type) == expected


def test_build_llm_headers_anthropic():
    # 默认只发 x-api-key（双头同发会被严格网关 401）；Bearer 由 401 重试路径按需使用
    headers = protocols.build_llm_headers("anthropic", "sk-ant")
    assert headers["x-api-key"] == "sk-ant"
    assert headers["anthropic-version"] == protocols.ANTHROPIC_VERSION
    assert "Authorization" not in headers

    bearer = protocols.build_llm_headers("anthropic", "sk-ant", bearer=True)
    assert bearer["Authorization"] == "Bearer sk-ant"
    assert bearer["anthropic-version"] == protocols.ANTHROPIC_VERSION
    assert "x-api-key" not in bearer


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def test_convert_messages_anthropic_extracts_system_and_images():
    messages = [
        {"role": "system", "content": "你是面试助手"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ],
        },
    ]
    system_text, converted = protocols.convert_messages(messages, "anthropic")
    assert system_text == "你是面试助手"
    assert len(converted) == 1
    content = converted[0]["content"]
    assert content[0] == {"type": "text", "text": "看图"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_convert_messages_responses_maps_image_parts():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/1.png"}}]},
    ]
    system_text, converted = protocols.convert_messages(messages, "responses")
    assert system_text == "sys"
    assert converted[0]["content"] == [{"type": "input_image", "image_url": "https://x/1.png"}]


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def test_build_chat_payload_anthropic_thinking_forces_temperature_1():
    payload = protocols.build_chat_payload(
        "anthropic",
        "m",
        [{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=500,
        stream=True,
        think_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    assert payload["max_tokens"] == 500
    assert payload["stream"] is True
    assert payload["thinking"]["budget_tokens"] == 1024
    assert payload["temperature"] == 1
    assert "messages" in payload


def test_build_chat_payload_responses_uses_max_output_tokens():
    payload = protocols.build_chat_payload(
        "responses",
        "m",
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=300,
        stream=False,
        think_params={"reasoning": {"effort": "low"}},
    )
    assert payload["max_output_tokens"] == 300
    assert payload["instructions"] == "s"
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["input"] == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Reply extraction
# ---------------------------------------------------------------------------

def test_extract_reply_text_anthropic():
    body = {
        "content": [
            {"type": "thinking", "thinking": "让我想想"},
            {"type": "text", "text": "答案是 42"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    text, reasoning = protocols.extract_reply_text(body, "anthropic")
    assert text == "答案是 42"
    assert reasoning == "让我想想"


def test_extract_reply_text_responses():
    body = {
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "推理中"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "OK"}]},
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2},
    }
    text, reasoning = protocols.extract_reply_text(body, "responses")
    assert text == "OK"
    assert reasoning == "推理中"


# ---------------------------------------------------------------------------
# SSE parsers
# ---------------------------------------------------------------------------

def test_anthropic_chunk_parsing():
    chunk = protocols._anthropic_chunk({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "你好"}})
    assert chunk.choices[0].delta.content == "你好"
    chunk = protocols._anthropic_chunk({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "思考"}})
    assert chunk.choices[0].delta.reasoning_content == "思考"
    chunk = protocols._anthropic_chunk({"type": "message_delta", "usage": {"output_tokens": 9}})
    assert chunk.usage.completion_tokens == 9
    empty = protocols._anthropic_chunk({"type": "content_block_start"})
    assert not empty.choices and empty.usage is None


def test_responses_chunk_parsing():
    chunk = protocols._responses_chunk({"type": "response.output_text.delta", "delta": "OK"})
    assert chunk.choices[0].delta.content == "OK"
    chunk = protocols._responses_chunk({
        "type": "response.completed",
        "response": {"usage": {"input_tokens": 3, "output_tokens": 4}},
    })
    assert chunk.usage.prompt_tokens == 3 and chunk.usage.completion_tokens == 4
    with pytest.raises(LLMError):
        protocols._responses_chunk({
            "type": "response.failed",
            "response": {"error": {"message": "boom"}},
        })


# ---------------------------------------------------------------------------
# Raw HTTP paths
# ---------------------------------------------------------------------------

def _fake_response(status_code=200, lines=None, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = ""
    resp.iter_lines.return_value = [
        (line.encode("utf-8") if isinstance(line, str) else line) for line in (lines or [])
    ]
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_post_chat_stream_tries_url_candidates_on_404():
    responses = [
        _fake_response(status_code=404),
        _fake_response(lines=[
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "嗨"}}),
            "data: " + json.dumps({"type": "message_stop"}),
        ]),
    ]
    with patch.object(protocols, "_http") as mock_http:
        mock_http.post.side_effect = responses
        chunks = list(
            protocols.post_chat_stream(
                "https://gw.example.com", "anthropic", "sk-x",
                {"model": "m", "messages": [], "stream": True},
            )
        )
    assert mock_http.post.call_count == 2
    assert mock_http.post.call_args_list[0].args[0] == "https://gw.example.com/v1/messages"
    assert mock_http.post.call_args_list[1].args[0] == "https://gw.example.com/messages"
    assert any(c.choices and c.choices[0].delta.content == "嗨" for c in chunks)


def test_post_chat_stream_openai_done_marker():
    resp = _fake_response(lines=[
        "data: " + json.dumps({"choices": [{"delta": {"content": "A"}}]}),
        "data: [DONE]",
    ])
    with patch.object(protocols, "_http") as mock_http:
        mock_http.post.return_value = resp
        chunks = list(
            protocols.post_chat_stream(
                "https://api.example.com/v1", "openai", "sk-x",
                {"model": "m", "messages": [], "stream": True},
            )
        )
    assert chunks and chunks[0].choices[0].delta.content == "A"


def test_post_chat_blocking_maps_auth_error():
    resp = _fake_response(status_code=401)
    with patch.object(protocols, "_http") as mock_http:
        mock_http.post.return_value = resp
        with pytest.raises(Exception) as exc_info:
            protocols.post_chat_blocking(
                "https://api.example.com/v1", "openai", "sk-x", {"model": "m"}
            )
    assert "401" in str(exc_info.value)


def test_get_api_type_invalid_falls_back_to_openai():
    assert protocols.get_api_type(_Model(api_type="bogus")) == "openai"
    assert protocols.get_api_type(_Model(api_type="")) == "openai"
    assert protocols.get_api_type(_Model(api_type="anthropic")) == "anthropic"


def test_post_chat_stream_retries_bearer_variant_on_401():
    seen_headers = []
    responses = [
        _fake_response(status_code=401),
        _fake_response(lines=[
            "data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}}),
            "data: " + json.dumps({"type": "message_stop"}),
        ]),
    ]

    def fake_post(url, headers=None, **_kwargs):
        seen_headers.append(headers)
        resp = responses[len(seen_headers) - 1]
        resp.__enter__ = lambda self=resp: self
        return resp

    with patch.object(protocols, "_http") as mock_http:
        mock_http.post.side_effect = fake_post
        chunks = list(
            protocols.post_chat_stream(
                "https://gw.example.com", "anthropic", "sk-ant",
                {"model": "m", "messages": [], "stream": True},
            )
        )
    assert len(seen_headers) == 2
    assert "x-api-key" in seen_headers[0] and "Authorization" not in seen_headers[0]
    assert "Authorization" in seen_headers[1] and "x-api-key" not in seen_headers[1]
    assert any(c.choices and c.choices[0].delta.content == "ok" for c in chunks)


def test_get_client_for_model_sets_no_retry_timeout_and_bounds_lru_cache(monkeypatch):
    from services.llm import streaming

    created = []
    closed = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def close(self):
            closed.append(self)

    monkeypatch.setattr(streaming, "OpenAI", FakeClient)
    monkeypatch.setattr(streaming, "_CLIENT_CACHE_MAX_SIZE", 2)
    streaming.close_cached_clients()

    try:
        c1 = streaming.get_client_for_model(_Model(base="https://api1.example.com"))
        c2 = streaming.get_client_for_model(_Model(base="https://api2.example.com"))
        assert c1.kwargs["max_retries"] == 0
        assert c1.kwargs["timeout"] is not None
        # Touch c1 so c2 becomes LRU
        assert streaming.get_client_for_model(_Model(base="https://api1.example.com")) is c1
        c3 = streaming.get_client_for_model(_Model(base="https://api3.example.com"))
        assert len(streaming._client_cache) == 2
        assert closed == [c2]

        streaming.close_cached_clients()
        assert len(streaming._client_cache) == 0
        assert closed == [c2, c1, c3]
    finally:
        streaming.close_cached_clients()


def test_chat_stream_passes_sdk_timeout_and_closes_response_on_early_abort(monkeypatch):
    from types import SimpleNamespace
    from services.llm import streaming

    captured_kwargs = {}
    closed = {"called": False}

    class FakeResponse:
        def __iter__(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="first", reasoning_content=None, reasoning=None))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="second", reasoning_content=None, reasoning=None))],
                usage=None,
            )

        def close(self):
            closed["called"] = True

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResponse()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    active = SimpleNamespace(
        name="m1",
        api_base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        api_type="openai",
        enabled=True,
        supports_vision=False,
        supports_think=False,
    )
    cfg = SimpleNamespace(
        active_model=0,
        models=[active],
        get_active_model=lambda: active,
        temperature=0.3,
        max_tokens=256,
        think_mode=False,
        think_effort="off",
    )
    monkeypatch.setattr(streaming, "get_config", lambda: cfg)
    monkeypatch.setattr(streaming, "get_client_for_model", lambda _m: fake_client)

    aborted = False
    out = []
    for item in streaming.chat_stream([{"role": "user", "content": "hi"}], abort_check=lambda: aborted):
        out.append(item)
        aborted = True

    assert out == [("text", "first")]
    assert captured_kwargs["timeout"] == 90.0
    assert closed["called"] is True


def test_mmpg_stream_aclose_tolerates_closed_loop_runtime_error(monkeypatch):
    from types import SimpleNamespace
    from services.llm import streaming

    class FakeLoop:
        def time(self):
            return 1.0

        def is_running(self):
            return True

    request = SimpleNamespace(
        loop=FakeLoop(),
        lock=streaming.threading.Lock(),
        started_at=None,
        cancelled=False,
        traces=("t1",),
        group_id="g1",
        revision=1,
        seq=1,
        final_at=0.5,
        on_started=None,
        stream_read=None,
    )
    model = SimpleNamespace(
        name="m1",
        api_base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        api_type="openai",
        supports_think=False,
    )
    cfg = SimpleNamespace(temperature=0.3, max_tokens=128, think_mode=False, think_effort="off")

    async def fake_stream(*_args, **_kwargs):
        yield SimpleNamespace(choices=[], usage=None)

    calls = {"count": 0}

    class FakeFuture:
        def result(self, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                return None
            raise RuntimeError("Event loop is closed")

    def fake_run_coro(coro, _loop):
        coro.close()
        return FakeFuture()

    monkeypatch.setattr(streaming, "post_cancellable_chat_stream", fake_stream)
    monkeypatch.setattr(streaming.asyncio, "run_coroutine_threadsafe", fake_run_coro)

    assert list(streaming._mmpg_stream(model, [{"role": "user", "content": "hi"}], cfg, request, lambda: False)) == []
    assert calls["count"] == 2


def test_stateless_http_client_isolates_threads_and_rejects_cookies():
    import threading

    client = protocols._StatelessHttpClient()
    sessions = []

    def grab():
        s = client._get_session()
        s.cookies.set("sid", "secret", domain="example.com")
        sessions.append((s, dict(s.cookies)))

    t1 = threading.Thread(target=grab)
    t2 = threading.Thread(target=grab)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(sessions) == 2
    assert sessions[0][0] is not sessions[1][0]
    assert sessions[0][1] == {}
    assert sessions[1][1] == {}
    client.close()


def test_stateless_http_client_releases_exited_thread_sessions():
    import gc
    import threading

    client = protocols._StatelessHttpClient()

    def worker():
        s = client._get_session()
        assert s is not None

    for _ in range(5):
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    gc.collect()
    assert len(client._sessions) == 0


def test_chat_stream_single_model_closes_aborted_stream(monkeypatch):
    from types import SimpleNamespace
    from services.llm import streaming

    closed = {"count": 0}

    class FakeStream:
        def __init__(self):
            self._items = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="part1", reasoning_content=None, reasoning=None))],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="part2", reasoning_content=None, reasoning=None))],
                    usage=None,
                ),
            ]
            self._idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx >= len(self._items):
                raise StopIteration
            item = self._items[self._idx]
            self._idx += 1
            return item

        def close(self):
            closed["count"] += 1

    model = SimpleNamespace(
        name="m1",
        api_base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        api_type="openai",
        supports_vision=False,
        supports_think=False,
    )
    cfg = SimpleNamespace(temperature=0.3, max_tokens=128, think_mode=False, think_effort="off")

    monkeypatch.setattr(streaming, "_try_stream_with_model", lambda *_a, **_kw: FakeStream())

    aborted = False
    out = []
    for item in streaming.chat_stream_single_model(
        model,
        [{"role": "user", "content": "hi"}],
        abort_check=lambda: aborted,
        config_snapshot=cfg,
    ):
        out.append(item)
        aborted = True

    assert out == [("text", "part1")]
    assert closed["count"] == 1


def test_try_stream_with_model_forwards_custom_timeout_for_non_openai(monkeypatch):
    from types import SimpleNamespace
    from services.llm import streaming

    captured = {}

    def fake_post_chat_stream(base_url, api_type, api_key, payload, *, timeout=(10, 90)):
        captured["timeout"] = timeout
        return iter([])

    model = SimpleNamespace(
        name="claude",
        api_base_url="https://api.anthropic.com",
        api_key="sk-ant",
        model="claude-3-7-sonnet",
        api_type="anthropic",
        supports_vision=False,
        supports_think=False,
    )
    cfg = SimpleNamespace(temperature=0.3, max_tokens=128, think_mode=False, think_effort="off")
    monkeypatch.setattr(streaming, "post_chat_stream", fake_post_chat_stream)

    streaming._try_stream_with_model(model, [{"role": "user", "content": "hi"}], cfg, timeout=(5, 25))
    assert captured["timeout"] == (5, 25)


@pytest.mark.asyncio
async def test_async_client_reused_per_loop_with_redirects_disabled():
    await protocols.aclose_async_client()
    try:
        c1 = protocols._get_async_client()
        c2 = protocols._get_async_client()
        assert c1 is c2
        assert c1.follow_redirects is False
    finally:
        await protocols.aclose_async_client()
