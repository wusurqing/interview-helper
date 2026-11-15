import asyncio
import json
from types import SimpleNamespace

import pytest

from api.assist.mmpg_turns import MmpgRequest
from services.llm.streaming import chat_stream_single_model


@pytest.mark.parametrize("stage", ["headers", "first_token", "streaming"])
@pytest.mark.asyncio
async def test_cancellation_closes_the_actual_http_connection(stage, tmp_path, monkeypatch):
    from services.llm import trace
    monkeypatch.setattr(trace, "LOG_DIR", str(tmp_path))
    received = asyncio.Event()
    closed = asyncio.Event()
    chunks = []

    async def serve(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = next(line for line in headers.lower().splitlines()
                                  if line.startswith(b"content-length:"))
            await reader.readexactly(int(content_length.split(b":")[1]))
            if stage != "headers":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            if stage == "streaming":
                data = json.dumps({"choices": [{"delta": {"content": "first"}}]})
                writer.write(f"data: {data}\n\n".encode())
            await writer.drain()
            received.set()
            await reader.read()
            closed.set()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        request = MmpgRequest("group", 1, "question", ("asr/1/1",), 0, None)
        request.loop = asyncio.get_running_loop()
        model = SimpleNamespace(
            api_base_url=f"http://127.0.0.1:{port}/v1", api_type="openai",
            api_key="test-key", model="fixture", name="fixture", supports_vision=False,
            supports_think=False,
        )
        started = []
        request.on_started = started.append

        def consume():
            for chunk in chat_stream_single_model(
                model, [{"role": "user", "content": "fixture"}], mmpg_request=request,
            ):
                chunks.append(chunk)

        consumer = asyncio.create_task(asyncio.to_thread(consume))
        try:
            await asyncio.wait_for(received.wait(), 4)
            assert len(started) == 1
            if stage == "streaming":
                async with asyncio.timeout(2):
                    while not chunks:
                        await asyncio.sleep(0.002)
                assert chunks == [("text", "first")]
            request.cancel()
            await asyncio.wait_for(closed.wait(), 2)
            await asyncio.wait_for(consumer, 2)
        finally:
            request.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        assert chunks == ([("text", "first")] if stage == "streaming" else [])
        results = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text(encoding="utf-8").splitlines()]
        terminal = [row for row in results if row["event"] == "result"]
        assert len(terminal) == 1
        assert terminal[0]["status"] == "cancelled"
        assert terminal[0]["output_text"] == ("first" if stage == "streaming" else "")


@pytest.mark.parametrize(("api_type", "suffix", "event"), [
    ("openai", "/v1/chat/completions", {"choices": [{"delta": {"content": "ok"}}]}),
    ("anthropic", "/v1/messages", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}}),
    ("responses", "/v1/responses", {"type": "response.output_text.delta", "delta": "ok"}),
])
@pytest.mark.asyncio
async def test_cancellable_transport_preserves_protocol_streams(api_type, suffix, event):
    async def serve(reader, writer):
        header = await reader.readuntil(b"\r\n\r\n")
        assert header.startswith(f"POST {suffix} ".encode())
        length = next(line for line in header.lower().splitlines() if line.startswith(b"content-length:"))
        await reader.readexactly(int(length.split(b":")[1]))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
        writer.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        request = MmpgRequest("group", 1, "fixture", ("trace",), 0, None, loop=asyncio.get_running_loop())
        model = SimpleNamespace(api_base_url=f"http://127.0.0.1:{port}/v1", api_type=api_type,
                                api_key="test-key", model="fixture", name="fixture", supports_vision=False,
                                supports_think=False)
        chunks = await asyncio.to_thread(lambda: list(chat_stream_single_model(
            model, [{"role": "user", "content": "fixture"}], mmpg_request=request,
        )))
        assert chunks == [("text", "ok")]
