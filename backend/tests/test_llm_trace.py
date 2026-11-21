import json
import gzip
from pathlib import Path
from urllib.parse import quote

import httpx
from openai import APIError, OpenAI, RateLimitError
import pytest
from unittest.mock import MagicMock, patch

from services.llm import trace
from services.llm import protocols


def test_trace_preserves_payload_usage_and_image_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    image = "data:image/png;base64,aGVsbG8="
    with trace.scope("vision", qa_id="qa-1"):
        attempt = trace.begin("https://user:pass@example.com/v1?api_key=secret", "openai", {
            "model": "m", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "你好"}, {"type": "image_url", "image_url": {"url": image, "detail": "high"}},
            ]}], "temperature": 0.3,
        }, api_key="secret")
        attempt.headers(200)
        attempt.response({"choices": [{"message": {"content": "答复"}}], "usage": {
            "prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0},
            "vendor_extra": 7,
        }})
        attempt.finish()
    lines = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["request", "result"]
    assert lines[0]["call_id"] == lines[1]["call_id"]
    assert lines[0]["attempt_id"] == lines[1]["attempt_id"]
    assert lines[0]["context"] == {"scene": "vision", "qa_id": "qa-1"}
    assert lines[0]["request"]["messages"][0]["content"][0]["text"] == "你好"
    asset = lines[0]["request"]["messages"][0]["content"][1]["image_url"]["url"]
    assert (tmp_path / asset["path"]).read_bytes() == b"hello"
    assert asset["mime"] == "image/png"
    assert lines[1]["usage_raw"]["vendor_extra"] == 7
    assert lines[1]["tokens"]["cached_read"] == 0
    assert lines[1]["response"]["choices"][0]["message"]["content"] == "答复"
    assert "secret" not in (tmp_path / "llm.jsonl").read_text(encoding="utf-8")


def test_trace_stream_merges_usage_and_keeps_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    attempt = trace.begin("https://example.com/v1", "anthropic", {"model": "m", "messages": []})
    attempt.headers(200)
    attempt.stream_event({"type": "message_start", "message": {"usage": {
        "input_tokens": 5, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 0,
    }}})
    attempt.stream_event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "部分"}})
    attempt.stream_event({"type": "message_delta", "usage": {"output_tokens": 3}})
    attempt.finish("cancelled")
    result = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["status"] == "cancelled"
    assert result["output_text"] == "部分"
    assert result["tokens"]["input"] == 25
    assert result["tokens"]["output"] == 3
    assert result["tokens"]["cached_read"] == 20
    assert len(result["stream_events"]) == 3


def test_sdk_client_observes_decoded_stream_and_connection_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ConnectError("offline")
        data = b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0\xe5\xa5\xbd"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
                              stream=httpx.ByteStream(gzip.compress(data)))

    client = trace.TracingHttpxClient()
    client._transport.close()
    client._transport = httpx.MockTransport(handler)
    with trace.scope("sdk", qa_id="q"):
        request = client.build_request("POST", "https://example.com/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "问"}], "stream": True,
        }, headers={"authorization": "Bearer sk-test"})
        try:
            client.send(request, stream=True)
        except httpx.ConnectError:
            pass
        response = client.send(request, stream=True)
        assert "你好" in response.read().decode()
    client.close()
    results = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in results] == ["request", "result", "request", "result"]
    assert results[1]["status"] == "failed"
    assert results[3]["output_text"] == "你好"
    assert results[3]["status"] == "completed"
    assert "sk-test" not in (tmp_path / "llm.jsonl").read_text(encoding="utf-8")


def test_anthropic_image_source_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    attempt = trace.begin("https://example.com/messages", "anthropic", {
        "model": "m", "messages": [{"content": [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "aGVsbG8=",
        }}]}],
    })
    attempt.finish()
    request = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    source = request["request"]["messages"][0]["content"][0]["source"]
    assert source["type"] == "base64"
    assert source["data"]["mime"] == "image/png"
    assert (tmp_path / source["data"]["path"]).read_bytes() == b"hello"


def test_asset_cleanup_keeps_only_referenced_images(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    first = trace.begin("https://example.com", "openai", {"image": "data:image/png;base64,aGVsbG8="})
    first.finish()
    keep = next((tmp_path / "llm_assets").iterdir())
    remove = keep.with_name("orphan.png")
    remove.write_bytes(b"orphan")
    trace.prune_unused_assets()
    assert keep.exists()
    assert not remove.exists()


def test_raw_http_stream_logs_retry_and_partial_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    missing = MagicMock(status_code=404, text="missing")
    missing.__enter__.return_value = missing
    missing.__exit__.return_value = False
    good = MagicMock(status_code=200)
    good.__enter__.return_value = good
    good.__exit__.return_value = False
    good.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}',
        b'data: {"usage":{"prompt_tokens":8,"completion_tokens":1,"prompt_cache_hit_tokens":4}}',
    ]
    with patch.object(protocols, "_http") as transport:
        transport.post.side_effect = [missing, good]
        with trace.scope("retry", qa_id="q"):
            stream = protocols.post_chat_stream("https://example.com", "openai", "sk-key", {
                "model": "m", "messages": [{"role": "user", "content": "question"}], "stream": True,
            })
            assert next(stream).choices[0].delta.content == "partial"
            stream.close()
    records = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert records[0]["call_id"] == records[2]["call_id"]
    assert records[1]["status"] == "failed"
    assert records[3]["status"] == "cancelled"
    assert records[3]["output_text"] == "partial"


@pytest.mark.parametrize("error_status", [401, 403, 404, 405])
@pytest.mark.parametrize(("error_content", "expected_body"), [
    (
        b'{"error":{"message":"rejected fixture-dummy-token","api_key":"fixture-dummy-token"}}',
        {"error": {"message": "rejected [REDACTED]", "api_key": "[REDACTED]"}},
    ),
    (
        b"rejected fixture-dummy-token",
        {"unparsed_body": "rejected [REDACTED]"},
    ),
])
def test_non_stream_retry_logs_redacted_error_body(tmp_path, monkeypatch, error_status, error_content, expected_body):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    failed = httpx.Response(error_status, content=error_content)
    accepted = httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})
    with patch.object(protocols._http, "post", side_effect=[failed, accepted]), trace.scope("blocking_retry"):
        reply = protocols.post_chat_blocking(
            "https://example.com", "anthropic", "fixture-dummy-token",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert reply.choices[0].message.content == "ok"
    logged = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    request, failure, retry, success = map(json.loads, logged.splitlines())
    assert [row["event"] for row in (request, failure, retry, success)] == ["request", "result", "request", "result"]
    assert failure["http_status"] == error_status
    assert failure["status"] == "failed"
    assert success["status"] == "completed"
    assert request["call_id"] == failure["call_id"] == retry["call_id"] == success["call_id"]
    assert request["attempt_id"] == failure["attempt_id"]
    assert retry["attempt_id"] == success["attempt_id"]
    assert failure["attempt_id"] != success["attempt_id"]
    assert (request["url"] == retry["url"]) == (error_status in (401, 403))
    assert failure["response"] == expected_body
    assert "fixture-dummy-token" not in logged


def test_openai_sdk_retry_has_two_attempts_with_one_call_id(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    count = 0

    def handler(_request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"id": "r", "object": "chat.completion", "model": "m",
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                                                   "prompt_cache_hit_tokens": 3}})

    transport = trace.TracingHttpxClient()
    transport._transport.close()
    transport._transport = httpx.MockTransport(handler)
    sdk = OpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=transport, max_retries=1)
    with trace.scope("sdk_retry"):
        result = sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    sdk.close()
    assert result.choices[0].message.content == "ok"
    rows = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows if row["event"] == "result"] == ["failed", "completed"]
    assert rows[0]["call_id"] == rows[2]["call_id"]
    assert rows[3]["usage_raw"]["prompt_cache_hit_tokens"] == 3


def test_log_write_failure_does_not_interrupt_model_path(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    monkeypatch.setattr(trace, "LOG_DIR", str(blocked))
    attempt = trace.begin("https://user:pass@example.com/?token=secret", "openai", {"model": "m"})
    attempt.headers(200)
    attempt.response({"choices": [{"message": {"content": "ok"}}]})
    attempt.finish()


def test_responses_body_keeps_reasoning_and_cache_details(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    attempt = trace.begin("https://example.com/responses", "responses", {"model": "m", "input": []})
    attempt.response({"status": "completed", "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "推理"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "答案"}]},
    ], "usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 5}}})
    attempt.finish()
    result = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["output_text"] == "答案"
    assert result["reasoning_text"] == "推理"
    assert result["tokens"]["cached_read"] == 5

def test_trace_redacts_url_context_and_reply_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    key = "credential-dummy+/and space"
    quoted_key = quote(key, safe="")
    delegated_key = "delegated-dummy-token"
    url = (
        f"https://dummy-user:dummy-password@example.com/{quoted_key}/messages"
        f"?route={quoted_key}&client_secret={delegated_key}&view=summary"
    )
    attempt = trace.begin(url, "openai", {"metadata": {key: "value", "api-key": delegated_key}},
                          api_key=key, context={"qa_id": key})
    attempt.response({"choices": [{"message": {
        "content": f"Before {key} after.",
        "reasoning_content": f"Authorization: Bearer {delegated_key}",
    }, "finish_reason": "stop"}]})
    attempt.finish()

    logged = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    request, result = map(json.loads, logged.splitlines())
    assert not any(secret in logged for secret in
                   [key, quoted_key, delegated_key, "dummy-user", "dummy-password"])
    assert request["url"].startswith("https://example.com/")
    assert "view=summary" in request["url"]
    assert request["context"]["qa_id"] == "[REDACTED]"
    assert result["output_text"] == "Before [REDACTED] after."
    assert result["reasoning_text"] == "Authorization: Bearer [REDACTED]"


def test_trace_redacts_credentials_split_across_stream_events(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    first, second = "prefix-secret-", "tail-secret"
    attempt = trace.begin("https://example.com/v1", "openai", {}, api_key=first + second)
    attempt.stream_event({"choices": [{"delta": {"content": "Answer " + first,
                                                "reasoning_content": "Reason " + first}}]})
    attempt.stream_event({"choices": [{"delta": {"content": second + " done.",
                                                "reasoning_content": second}}]})
    attempt.finish()

    logged = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    result = json.loads(logged.splitlines()[-1])
    assert first not in logged and second not in logged
    assert result["output_text"] == "Answer [REDACTED] done."
    assert result["reasoning_text"] == "Reason [REDACTED]"


def _fixture_sdk(handler):
    transport = trace.TracingHttpxClient()
    transport._transport.close()
    transport._transport = httpx.MockTransport(handler)
    return OpenAI(api_key="fixture-dummy-token", base_url="https://example.com/v1",
                  http_client=transport, max_retries=0)


@pytest.mark.parametrize(("end", "expected_status"), [("read_error", "failed"), ("close", "cancelled")])
def test_sdk_partial_stream_distinguishes_read_failure_from_close(tmp_path, monkeypatch, end, expected_status):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))

    class PartialStream(httpx.SyncByteStream):
        def __iter__(self):
            yield (b'data: {"id":"fixture","created":0,"model":"m","object":"chat.completion.chunk",'
                   b'"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n')
            if end == "read_error":
                raise httpx.ReadError("fixture connection dropped: fixture-dummy-token")
            yield b"data: [DONE]\n\n"

    def handler(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=PartialStream())

    with _fixture_sdk(handler) as sdk:
        stream = sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
        assert next(stream).choices[0].delta.content == "partial"
        if end == "read_error":
            with pytest.raises(httpx.ReadError, match="fixture connection dropped"):
                next(stream)
        else:
            stream.close()

        logged = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
        request, result = map(json.loads, logged.splitlines())
        assert result["event"] == "result"
        assert result["attempt_id"] == request["attempt_id"]
        assert result["status"] == expected_status
        assert result["output_text"] == "partial"
        assert "fixture-dummy-token" not in logged
        assert result["error"] == ("fixture connection dropped: [REDACTED]" if end == "read_error" else None)
        stream.close()
        assert len((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_sdk_records_json_error_when_stream_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))

    def handler(_request):
        return httpx.Response(429, json={"error": {"message": "busy", "api_key": "fixture-dummy-token"}})

    with _fixture_sdk(handler) as sdk, pytest.raises(RateLimitError):
        sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)

    request, result = map(json.loads, (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines())
    assert result["attempt_id"] == request["attempt_id"]
    assert result["status"] == "failed"
    assert result["response"] == {"error": {"message": "busy", "api_key": "[REDACTED]"}}

@pytest.mark.parametrize(("protocol", "body"), [
    ("openai", {"choices": [{"message": {"content": [
        {"text": "Before fixture-prefix-"}, {"text": "tail-fixture after."},
    ]}}]}),
    ("anthropic", {"content": [
        {"type": "text", "text": "Before fixture-prefix-"},
        {"type": "text", "text": "tail-fixture after."},
    ]}),
    ("responses", {"output": [{"content": [
        {"type": "output_text", "text": "Before fixture-prefix-"},
        {"type": "output_text", "text": "tail-fixture after."},
    ]}]}),
])
def test_trace_redacts_credentials_split_across_blocking_reply_parts(tmp_path, monkeypatch, protocol, body):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    attempt = trace.begin("https://example.com/v1", protocol, {}, api_key="fixture-prefix-tail-fixture")
    attempt.response(body)
    attempt.finish()

    logged = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    result = json.loads(logged.splitlines()[-1])
    assert "fixture-prefix-" not in logged and "tail-fixture" not in logged
    assert result["output_text"] == "Before [REDACTED] after."


def test_trace_redacts_lowercase_percent_escapes_in_error_text(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    attempt = trace.begin("https://example.com/v1", "openai", {}, api_key="fixture/key+suffix")
    attempt.finish("failed", "provider request failed: https://example.com/fixture%2fkey%2bsuffix")

    result = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["error"] == "provider request failed: https://example.com/[REDACTED]"


def test_sdk_records_provider_sse_error_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))

    def handler(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
            b'data: {"error":{"message":"fixture provider rejected fixture-dummy-token"}}\n\n'
        ))

    with _fixture_sdk(handler) as sdk, pytest.raises(APIError, match="fixture provider rejected"):
        with sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                         stream=True) as stream:
            next(stream)

    request, result = map(json.loads, (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines())
    assert result["attempt_id"] == request["attempt_id"]
    assert result["status"] == "failed"
    assert result["error"] == "fixture provider rejected [REDACTED]"


def test_trace_disabled_via_env_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "0")
    image = "data:image/png;base64,aGVsbG8="
    attempt = trace.begin("https://example.com/v1", "openai", {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "private resume"}, {"type": "image_url", "image_url": {"url": image}}]}],
    })
    assert isinstance(attempt, trace._NullAttempt)
    attempt.headers(200)
    attempt.response({"choices": [{"message": {"content": "private reply"}}]})
    attempt.finish()
    assert not (tmp_path / "llm.jsonl").exists()
    assert not (tmp_path / "llm_assets").exists()


def test_trace_metadata_mode_omits_prompts_replies_and_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "metadata")
    image = "data:image/png;base64,aGVsbG8="
    with trace.scope("resume_review", qa_id="qa-9"):
        attempt = trace.begin("https://example.com/v1", "openai", {
            "model": "gpt-4o",
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "SUPER_PRIVATE_RESUME_TEXT"},
                {"type": "image_url", "image_url": {"url": image}},
            ]}],
        })
        attempt.headers(200)
        attempt.stream_event({"choices": [{"delta": {"reasoning_content": "PRIVATE_THINK"}}]})
        attempt.stream_event({"choices": [{"delta": {"content": "PRIVATE_ANSWER"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})
        attempt.finish()

    raw_log = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    assert "SUPER_PRIVATE_RESUME_TEXT" not in raw_log
    assert "PRIVATE_THINK" not in raw_log
    assert "PRIVATE_ANSWER" not in raw_log
    assert "aGVsbG8=" not in raw_log
    assert not (tmp_path / "llm_assets").exists()

    req, res = [json.loads(line) for line in raw_log.splitlines()]
    assert req["trace_mode"] == "metadata"
    assert req["request"]["redacted"] is True
    assert req["request"]["model"] == "gpt-4o"
    assert req["request"]["message_count"] == 1
    assert req["request"]["image_count"] == 1
    assert res["trace_mode"] == "metadata"
    assert res["output_text"] is None
    assert res["reasoning_text"] is None
    assert res["response"] is None
    assert res["stream_events"] is None
    assert res["output_chars"] == len("PRIVATE_ANSWER")
    assert res["reasoning_chars"] == len("PRIVATE_THINK")
    assert res["stream_event_count"] == 2
    assert res["tokens"]["input"] == 120
    assert res["tokens"]["output"] == 30


def test_trace_assets_quota_evicts_oldest_and_respects_zero_cap(tmp_path, monkeypatch):
    import base64
    import os

    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "full")

    # Zero cap: image is never written to disk
    monkeypatch.setenv("IA_LLM_TRACE_MAX_ASSETS_MB", "0")
    img_zero = "data:image/png;base64," + base64.b64encode(b"0123456789").decode("ascii")
    a0 = trace.begin("https://example.com/v1", "openai", {"image": img_zero})
    a0.finish()
    req0 = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert req0["request"]["image"]["path"] is None
    assert req0["request"]["image"]["omitted"] == "asset_quota_exceeded"
    assert not (tmp_path / "llm_assets").exists()

    # Tight quota: keep at most ~25 bytes total so 3rd 12-byte image evicts the oldest
    monkeypatch.setenv("IA_LLM_TRACE_MAX_ASSETS_MB", str(25 / (1024 * 1024)))
    saved_paths = []
    for idx, payload_bytes in enumerate((b"aaaaaaaaaaaa", b"bbbbbbbbbbbb", b"cccccccccccc")):
        data_url = "data:image/png;base64," + base64.b64encode(payload_bytes).decode("ascii")
        attempt = trace.begin("https://example.com/v1", "openai", {"image": data_url})
        attempt.finish()
        req = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-2])
        p = tmp_path / req["request"]["image"]["path"]
        saved_paths.append(p)
        if p.exists():
            os.utime(p, (1_700_000_000 + idx * 10, 1_700_000_000 + idx * 10))

    remaining = list((tmp_path / "llm_assets").iterdir())
    assert sum(f.stat().st_size for f in remaining) <= 25
    assert not saved_paths[0].exists()
    assert saved_paths[2].exists()


def test_rollover_prunes_assets_asynchronously_without_blocking_emit(tmp_path, monkeypatch):
    import logging
    import threading
    import time

    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "full")
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_prune(log_dir=None, *, min_age_seconds=0.0):
        calls.append((Path(log_dir), min_age_seconds, threading.current_thread().name))
        started.set()
        release.wait(timeout=3.0)

    monkeypatch.setattr(trace, "prune_unused_assets", slow_prune)

    handler = trace._TraceFileHandler(str(tmp_path / "llm.jsonl"), when="midnight", backupCount=5, encoding="utf-8")
    try:
        handler.rolloverAt = int(time.time()) - 10
        record = logging.LogRecord("llm.trace", logging.INFO, __file__, 1, '{"event":"ping"}', (), None)
        t0 = time.monotonic()
        handler.emit(record)
        elapsed = time.monotonic() - t0
        # emit() must return immediately instead of waiting on slow_prune
        assert elapsed < 1.0
        assert started.wait(timeout=2.0)
        assert calls[0][0] == tmp_path
        assert calls[0][1] > 0
        assert calls[0][2] == "llm-trace-asset-pruner"
    finally:
        release.set()
        assert trace.wait_for_pending_prune(timeout=3.0)
        handler.close()


def test_prune_during_concurrent_attempt_never_deletes_newly_logged_asset(tmp_path, monkeypatch):
    import base64
    import threading

    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "full")

    # Seed llm_assets/ with one unrelated orphan and an initial log file
    assets_dir = tmp_path / "llm_assets"
    assets_dir.mkdir(parents=True)
    orphan = assets_dir / "orphan.png"
    orphan.write_bytes(b"orphan-bytes")
    (tmp_path / "llm.jsonl").write_text("", encoding="utf-8")

    scanned = threading.Event()
    resume_prune = threading.Event()
    orig_glob = Path.glob

    def paused_glob(self, pattern):
        items = list(orig_glob(self, pattern))
        if self == tmp_path and pattern == "llm.jsonl*":
            scanned.set()
            resume_prune.wait(timeout=3.0)
        return iter(items)

    monkeypatch.setattr(Path, "glob", paused_glob)

    prune_thread = threading.Thread(target=trace.prune_unused_assets, kwargs={"log_dir": tmp_path, "min_age_seconds": 0.0})
    prune_thread.start()
    assert scanned.wait(timeout=2.0)

    # While pruner has already collected the pre-write log snapshot and is about to unlink,
    # a concurrent Attempt writes a brand-new asset, logs it, and exits __init__ (clearing _in_flight_assets).
    live_url = "data:image/png;base64," + base64.b64encode(b"live-concurrent-image").decode("ascii")
    attempt = trace.begin("https://example.com/v1", "openai", {"image": live_url})
    attempt.finish()
    req = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    live_asset = tmp_path / req["request"]["image"]["path"]
    assert live_asset.exists()

    resume_prune.set()
    prune_thread.join(timeout=3.0)
    assert not prune_thread.is_alive()

    assert live_asset.exists(), "asset logged during active prune scan must never be deleted"
    assert not orphan.exists(), "unrelated orphan asset should still be pruned"


def test_trace_asset_reuse_refreshes_mtime_for_lru_quota_eviction(tmp_path, monkeypatch):
    import base64
    import os

    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "full")
    monkeypatch.setenv("IA_LLM_TRACE_MAX_ASSETS_MB", str(25 / (1024 * 1024)))

    img_a = "data:image/png;base64," + base64.b64encode(b"aaaaaaaaaaaa").decode("ascii")
    img_b = "data:image/png;base64," + base64.b64encode(b"bbbbbbbbbbbb").decode("ascii")
    img_c = "data:image/png;base64," + base64.b64encode(b"cccccccccccc").decode("ascii")

    a1 = trace.begin("https://example.com/v1", "openai", {"image": img_a})
    a1.finish()
    path_a = tmp_path / json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-2])["request"]["image"]["path"]
    os.utime(path_a, (1_700_000_000, 1_700_000_000))

    b1 = trace.begin("https://example.com/v1", "openai", {"image": img_b})
    b1.finish()
    path_b = tmp_path / json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-2])["request"]["image"]["path"]
    os.utime(path_b, (1_700_000_100, 1_700_000_100))

    # Reuse A -> should refresh A's mtime so B becomes the oldest asset
    a2 = trace.begin("https://example.com/v1", "openai", {"image": img_a})
    a2.finish()

    c1 = trace.begin("https://example.com/v1", "openai", {"image": img_c})
    c1.finish()
    path_c = tmp_path / json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()[-2])["request"]["image"]["path"]

    assert path_a.exists()
    assert not path_b.exists()
    assert path_c.exists()


def test_trace_metadata_mode_anthropic_responses_and_off_skips_sdk_json_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    monkeypatch.setenv("IA_LLM_TRACE", "metadata")

    anth = trace.begin("https://example.com/v1/messages", "anthropic", {
        "model": "claude-sonnet",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "TOP_SECRET_ANTHROPIC_PROMPT"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
        ]}],
    })
    anth.headers(200)
    anth.response({
        "content": [
            {"type": "thinking", "thinking": "SECRET_CLAUDE_THINK"},
            {"type": "text", "text": "SECRET_CLAUDE_REPLY"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    anth.finish()

    resp_api = trace.begin("https://example.com/v1/responses", "responses", {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "TOP_SECRET_RESPONSES_PROMPT"}],
    })
    resp_api.headers(200)
    resp_api.stream_event({"type": "response.output_text.delta", "delta": "SECRET_RESP_DELTA"})
    resp_api.stream_event({"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 7, "output_tokens": 3}}})
    resp_api.finish()

    raw = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    for secret in ("TOP_SECRET_ANTHROPIC_PROMPT", "SECRET_CLAUDE_THINK", "SECRET_CLAUDE_REPLY",
                   "TOP_SECRET_RESPONSES_PROMPT", "SECRET_RESP_DELTA", "aGVsbG8="):
        assert secret not in raw
    assert not (tmp_path / "llm_assets").exists()

    # IA_LLM_TRACE=0 short-circuits TracingHttpxClient._request without calling json.loads
    monkeypatch.setenv("IA_LLM_TRACE", "0")
    calls = []
    monkeypatch.setattr(trace.json, "loads", lambda *a, **k: calls.append(True))
    req = httpx.Request("POST", "https://example.com/v1/chat/completions", content=b'{"model":"m"}')
    trace.TracingHttpxClient._request(req)
    assert isinstance(req.extensions["llm_trace"], trace._NullAttempt)
    assert calls == []
