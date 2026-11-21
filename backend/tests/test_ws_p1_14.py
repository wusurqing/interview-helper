"""P1-14 后端回归测试：每客户端有界队列 + 单发送协程（无队头阻塞/无并发 send_json）、握手缓冲不丢消息、停机安全清理。"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import status

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.realtime import ws as ws_mod


def test_slow_client_does_not_block_fast_client():
    async def exercise():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        ws_mod.init_broadcast(loop, q)

        slow_unblock = asyncio.Event()
        slow_entered = asyncio.Event()
        fast_received: list[dict] = []
        fast_done = asyncio.Event()

        class _SlowWS:
            closed_code: int | None = None

            async def send_json(self, data):
                slow_entered.set()
                await slow_unblock.wait()

            async def close(self, code: int = 1000):
                self.closed_code = code

        class _FastWS:
            async def send_json(self, data):
                fast_received.append(data)
                if len(fast_received) >= 5:
                    fast_done.set()

            async def close(self, code: int = 1000):
                pass

        now = time.monotonic()
        slow_ws = _SlowWS()
        fast_ws = _FastWS()
        slow_client = ws_mod._ClientConn(slow_ws, asyncio.Queue(maxsize=32), now, now, handshaking=False)
        fast_client = ws_mod._ClientConn(fast_ws, asyncio.Queue(maxsize=32), now, now, handshaking=False)
        ws_mod._clients[slow_ws] = slow_client
        ws_mod._clients[fast_ws] = fast_client
        ws_mod.ws_clients.add(slow_ws)
        ws_mod.ws_clients.add(fast_ws)

        slow_client.sender_task = asyncio.create_task(ws_mod._client_sender(slow_client))
        fast_client.sender_task = asyncio.create_task(ws_mod._client_sender(fast_client))
        dispatcher = asyncio.create_task(ws_mod.ws_dispatcher())

        try:
            t0 = time.monotonic()
            for idx in range(5):
                ws_mod.broadcast({"type": "answer_chunk", "seq": idx})
            await asyncio.wait_for(fast_done.wait(), timeout=1.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5
            assert [m["seq"] for m in fast_received] == [0, 1, 2, 3, 4]
            await asyncio.wait_for(slow_entered.wait(), timeout=1.0)
        finally:
            slow_unblock.set()
            ws_mod.shutdown_broadcast()
            await ws_mod.close_all_clients()
            dispatcher.cancel()
            await asyncio.gather(dispatcher, return_exceptions=True)

    asyncio.run(exercise())


def test_single_sender_serializes_broadcast_heartbeat_and_pong(monkeypatch: pytest.MonkeyPatch):
    async def exercise():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        ws_mod.init_broadcast(loop, q)

        in_flight = 0
        max_in_flight = 0
        sent: list[dict] = []
        incoming: asyncio.Queue[str | None] = asyncio.Queue()

        class _SerializedWS:
            client = SimpleNamespace(host="127.0.0.1")
            query_params: dict[str, str] = {}
            headers: dict[str, str] = {}
            url = SimpleNamespace(hostname="127.0.0.1", scheme="ws", port=18080)

            async def accept(self):
                return None

            async def send_json(self, data):
                nonlocal in_flight, max_in_flight
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.005)
                sent.append(data)
                in_flight -= 1

            async def receive_text(self):
                item = await incoming.get()
                if item is None:
                    raise ws_mod.WebSocketDisconnect()
                return item

            async def close(self, code: int = 1000):
                await incoming.put(None)

        monkeypatch.setattr(ws_mod, "HEARTBEAT_INTERVAL", 0.01)
        ws = _SerializedWS()
        dispatcher = asyncio.create_task(ws_mod.ws_dispatcher())
        heartbeat = asyncio.create_task(ws_mod.ws_heartbeat())
        endpoint_task = asyncio.create_task(ws_mod.websocket_endpoint(ws))

        try:
            # 等待握手完成
            for _ in range(50):
                if any(m.get("type") == "stt_status" for m in sent):
                    break
                await asyncio.sleep(0.01)

            # 并发压入广播、客户端 ping 与服务端心跳
            for i in range(6):
                ws_mod.broadcast({"type": "answer_chunk", "i": i})
                await incoming.put(json.dumps({"type": "ping"}))
            await asyncio.sleep(0.15)
            await incoming.put(None)
            await asyncio.wait_for(endpoint_task, timeout=2.0)
        finally:
            ws_mod.shutdown_broadcast()
            await ws_mod.close_all_clients()
            dispatcher.cancel()
            heartbeat.cancel()
            await asyncio.gather(dispatcher, heartbeat, return_exceptions=True)

        assert max_in_flight == 1
        types = [m.get("type") for m in sent]
        assert "init" in types
        assert "pong" in types
        assert "ping" in types
        assert types.count("answer_chunk") == 6

    asyncio.run(exercise())


def test_handshake_window_buffers_broadcasts_without_loss():
    async def exercise():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        ws_mod.init_broadcast(loop, q)

        sent: list[dict] = []
        init_in_progress = asyncio.Event()
        release_init = asyncio.Event()
        buffered_delivered = asyncio.Event()
        incoming: asyncio.Queue[str | None] = asyncio.Queue()

        class _HandshakeWS:
            client = SimpleNamespace(host="127.0.0.1")
            query_params: dict[str, str] = {}
            headers: dict[str, str] = {}
            url = SimpleNamespace(hostname="127.0.0.1", scheme="ws", port=18080)

            async def accept(self):
                return None

            async def send_json(self, data):
                if data.get("type") == "init":
                    init_in_progress.set()
                    await release_init.wait()
                sent.append(data)
                if data.get("type") == "transcription":
                    buffered_delivered.set()

            async def receive_text(self):
                item = await incoming.get()
                if item is None:
                    raise ws_mod.WebSocketDisconnect()
                return item

            async def close(self, code: int = 1000):
                await incoming.put(None)

        dispatcher = asyncio.create_task(ws_mod.ws_dispatcher())
        endpoint_task = asyncio.create_task(ws_mod.websocket_endpoint(_HandshakeWS()))

        try:
            await asyncio.wait_for(init_in_progress.wait(), timeout=2.0)
            # 在 init 尚未发完的握手窗口期内投递广播
            ws_mod.broadcast({"type": "transcription", "text": "during-handshake"})
            await asyncio.sleep(0.03)
            release_init.set()
            await asyncio.wait_for(buffered_delivered.wait(), timeout=2.0)
            await incoming.put(None)
            await asyncio.wait_for(endpoint_task, timeout=2.0)
        finally:
            ws_mod.shutdown_broadcast()
            await ws_mod.close_all_clients()
            dispatcher.cancel()
            await asyncio.gather(dispatcher, return_exceptions=True)

        types = [m.get("type") for m in sent]
        assert types[0] == "init"
        assert "stt_status" in types
        assert "transcription" in types
        assert types.index("init") < types.index("transcription")
        assert any(m.get("text") == "during-handshake" for m in sent)

    asyncio.run(exercise())


def test_slow_client_queue_overflow_drops_only_slow_client_and_drops_audio_level_safely():
    async def exercise():
        now = time.monotonic()

        class _DummyWS:
            closed_code: int | None = None

            async def send_json(self, _data):
                await asyncio.sleep(10)

            async def close(self, code: int = 1000):
                self.closed_code = code

        ws = _DummyWS()
        client = ws_mod._ClientConn(ws, asyncio.Queue(maxsize=2), now, now, handshaking=False)
        ws_mod._clients[ws] = client
        ws_mod.ws_clients.add(ws)

        try:
            assert client.enqueue({"type": "answer_chunk", "n": 1}) is True
            assert client.enqueue({"type": "answer_chunk", "n": 2}) is True
            # 队列满时瞬态 audio_level 被安全丢弃，不断连
            assert client.enqueue({"type": "audio_level", "value": 0.5}) is True
            # 非瞬态消息在满队列时返回 False 触发断连清理
            assert client.enqueue({"type": "answer_chunk", "n": 3}) is False
            await ws_mod._drop_client(client, code=status.WS_1011_INTERNAL_ERROR)
            assert ws.closed_code == status.WS_1011_INTERNAL_ERROR
            assert ws not in ws_mod.ws_clients
            assert ws not in ws_mod._clients
        finally:
            await ws_mod.close_all_clients()

    asyncio.run(exercise())


def test_broadcast_is_safe_after_shutdown_and_on_closed_loop():
    ws_mod.shutdown_broadcast()
    assert ws_mod._loop is None
    assert ws_mod._msg_queue is None
    ws_mod.broadcast({"type": "noop"})

    class _ClosedLoop:
        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, *_args, **_kwargs):
            raise RuntimeError("Event loop is closed")

    q: asyncio.Queue = asyncio.Queue()
    ws_mod.init_broadcast(_ClosedLoop(), q)  # type: ignore[arg-type]
    try:
        # 即使在 is_closed 检查和 call_soon_threadsafe 之间关闭，也不得抛出 RuntimeError
        ws_mod.broadcast({"type": "late"})
    finally:
        ws_mod.shutdown_broadcast()

    # 仅实现 call_soon_threadsafe 的最小鸭子类型 loop 也能正常工作
    class _InlineLoop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    q2: asyncio.Queue = asyncio.Queue(maxsize=2)
    ws_mod.init_broadcast(_InlineLoop(), q2)  # type: ignore[arg-type]
    try:
        ws_mod.broadcast({"type": "ping"})
        assert q2.qsize() == 1
    finally:
        ws_mod.shutdown_broadcast()
