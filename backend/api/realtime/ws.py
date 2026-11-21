import asyncio
from dataclasses import dataclass, field
import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from core.auth import is_auth_disabled, loopback_bypass_allowed, verify_token
from core.env import env_int
from core.logger import get_logger
from core.config import get_config
from core.session import snapshot_session
from services.stt import stt_configuration_error

_log = get_logger("ws")

router = APIRouter()

# 心跳/超时配置（秒）。25s 间隔可穿透多数代理 60s 空闲断连阈值。
HEARTBEAT_INTERVAL = 25
HEARTBEAT_TIMEOUT = 60
# 单条 send 的硬超时，仅作用于该客户端自己的独立发送协程，不阻塞其他客户端。
SEND_TIMEOUT = 5.0
# 每客户端有界发送队列容量，防止慢客户端无限堆积内存。
CLIENT_QUEUE_SIZE = env_int("IA_WS_CLIENT_QUEUE_SIZE", 256, minimum=16)


@dataclass
class _ClientConn:
    ws: WebSocket
    queue: asyncio.Queue[Optional[dict]]
    connected_at: float
    last_pong: float
    handshaking: bool = True
    handshake_buffer: list[dict] = field(default_factory=list)
    closed: bool = False
    dropped_count: int = 0
    sender_task: Optional[asyncio.Task] = None

    def enqueue(self, data: dict) -> bool:
        """非阻塞向该客户端入队；若慢客户端队列满且消息不可丢弃，返回 False 触发断连清理。"""
        if self.closed:
            return False
        if self.handshaking:
            if len(self.handshake_buffer) >= self.queue.maxsize:
                if data.get("type") == "audio_level":
                    self.dropped_count += 1
                    return True
                return False
            self.handshake_buffer.append(data)
            return True
        try:
            self.queue.put_nowait(data)
            return True
        except asyncio.QueueFull:
            # 瞬态音量电平可安全丢弃，避免突发峰值误杀正常客户端
            if data.get("type") == "audio_level":
                self.dropped_count += 1
                return True
            self.dropped_count += 1
            return False


ws_clients: set[WebSocket] = set()
_clients: dict[WebSocket, _ClientConn] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None
_msg_queue: Optional[asyncio.Queue] = None
_broadcast_drop_count = 0


def init_broadcast(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    global _loop, _msg_queue, _broadcast_drop_count
    _loop = loop
    _msg_queue = queue
    _broadcast_drop_count = 0


def shutdown_broadcast() -> None:
    """停机时清空事件循环与广播队列引用，防止后台线程向已关闭 loop 投递任务。"""
    global _loop, _msg_queue
    _loop = None
    _msg_queue = None


async def _safe_send(ws: WebSocket, data: dict) -> bool:
    try:
        await asyncio.wait_for(ws.send_json(data), timeout=SEND_TIMEOUT)
        return True
    except asyncio.TimeoutError:
        _log.warning("WS send timeout, dropping client")
        return False
    except Exception:
        return False


def _unregister_client(client: _ClientConn) -> None:
    client.closed = True
    ws_clients.discard(client.ws)
    _clients.pop(client.ws, None)


async def _close_ws_quietly(ws: WebSocket, code: int) -> None:
    close_fn = getattr(ws, "close", None)
    if close_fn is None:
        return
    try:
        await close_fn(code=code)
    except Exception:
        pass


async def _drop_client(client: _ClientConn, *, code: int) -> None:
    if client.closed and client.ws not in _clients:
        return
    _unregister_client(client)
    # 唤醒发送协程尽快退出
    try:
        client.queue.put_nowait(None)
    except asyncio.QueueFull:
        pass
    await _close_ws_quietly(client.ws, code)


async def _client_sender(client: _ClientConn) -> None:
    """每个客户端唯一的发送协程：串行写 socket，彻底消除同 socket 并发 send_json 与队头阻塞。"""
    try:
        while not client.closed:
            msg = await client.queue.get()
            if msg is None or client.closed:
                break
            ok = await _safe_send(client.ws, msg)
            if not ok:
                _unregister_client(client)
                await _close_ws_quietly(client.ws, status.WS_1011_INTERNAL_ERROR)
                break
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _log.warning("WS client sender failed: %s", e)
        _unregister_client(client)
        await _close_ws_quietly(client.ws, status.WS_1011_INTERNAL_ERROR)


async def close_all_clients() -> None:
    """停机时关闭所有活跃客户端并等待其发送协程退出。"""
    clients = list(_clients.values())
    tasks: list[asyncio.Task] = []
    for client in clients:
        _unregister_client(client)
        if client.sender_task is not None:
            client.sender_task.cancel()
            tasks.append(client.sender_task)
        await _close_ws_quietly(client.ws, status.WS_1001_GOING_AWAY)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def ws_dispatcher():
    if _msg_queue is None:
        return
    _crash_backoff = 1
    while True:
        try:
            data = await _msg_queue.get()
            clients = list(_clients.values())
            if not clients:
                continue
            overflowed: list[_ClientConn] = []
            for client in clients:
                if not client.enqueue(data):
                    overflowed.append(client)
            for client in overflowed:
                _log.warning("WS client send queue full (dropped=%d), closing slow client", client.dropped_count)
                await _drop_client(client, code=status.WS_1011_INTERNAL_ERROR)
            _crash_backoff = 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.error("ws_dispatcher crashed (retry in %ds): %s", _crash_backoff, e, exc_info=True)
            await asyncio.sleep(_crash_backoff)
            _crash_backoff = min(_crash_backoff * 2, 60)


async def ws_heartbeat():
    """服务端主动心跳：通过每客户端发送队列投递 ping，长时间无 pong 视为僵尸连接并关闭。"""
    _crash_backoff = 1
    _crash_sleep = 0.0
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            now = time.monotonic()
            clients = list(_clients.values())
            if not clients:
                _crash_sleep = 0.0
                continue
            ping_msg = {"type": "ping", "ts": now}
            effective_timeout = HEARTBEAT_TIMEOUT + _crash_sleep
            _crash_sleep = 0.0
            for client in clients:
                last = getattr(client.ws, "_ia_last_pong", None) or client.last_pong or client.connected_at
                if now - last > effective_timeout:
                    _log.info("WS heartbeat timeout, closing stale client")
                    await _drop_client(client, code=status.WS_1001_GOING_AWAY)
                    continue
                if not client.enqueue(ping_msg):
                    _log.warning("WS client send queue full during heartbeat, closing slow client")
                    await _drop_client(client, code=status.WS_1011_INTERNAL_ERROR)
            _crash_backoff = 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.error("ws_heartbeat crashed (retry in %ds): %s", _crash_backoff, e, exc_info=True)
            _crash_sleep = _crash_backoff
            await asyncio.sleep(_crash_backoff)
            _crash_backoff = min(_crash_backoff * 2, 60)


def broadcast(data: dict):
    """Thread-safe broadcast to all connected WebSocket clients."""
    loop = _loop
    queue = _msg_queue
    if loop is None or queue is None:
        return
    try:
        is_closed = getattr(loop, "is_closed", None)
        if callable(is_closed) and is_closed():
            return
        loop.call_soon_threadsafe(_enqueue_broadcast, data, queue)
    except RuntimeError:
        # 停机窗口事件循环已关闭，忽略迟到广播
        return


def _enqueue_broadcast(data: dict, queue: Optional[asyncio.Queue] = None):
    global _broadcast_drop_count
    target_queue = queue if queue is not None else _msg_queue
    if target_queue is None:
        return
    try:
        target_queue.put_nowait(data)
    except asyncio.QueueFull:
        _broadcast_drop_count += 1
        if _broadcast_drop_count == 1 or _broadcast_drop_count % 25 == 0:
            _log.warning(
                "WS broadcast queue full; dropped=%d type=%s",
                _broadcast_drop_count,
                data.get("type"),
            )


def _ws_authorized(ws: WebSocket) -> bool:
    if is_auth_disabled():
        return True
    _client = getattr(ws, "client", None)
    client_host = _client.host if _client else None
    request_url = getattr(ws, "url", None)
    if loopback_bypass_allowed(
        client_host,
        ws.headers.get("origin"),
        getattr(request_url, "hostname", None),
        getattr(request_url, "scheme", "ws"),
        getattr(request_url, "port", None),
    ):
        return True
    token = ws.query_params.get("token")
    if not token:
        auth = ws.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth.split(None, 1)[1].strip()
    return verify_token(token)


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not _ws_authorized(ws):
        await _close_ws_quietly(ws, status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    now = time.monotonic()
    ws._ia_connected_at = now  # type: ignore[attr-defined]
    ws._ia_last_pong = now  # type: ignore[attr-defined]
    client = _ClientConn(
        ws=ws,
        queue=asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE),
        connected_at=now,
        last_pong=now,
        handshaking=True,
    )
    # 先注册客户端（处于握手缓冲态）再拍快照并下发 init，避免握手窗口期漏广播
    _clients[ws] = client
    ws_clients.add(ws)
    try:
        from api.assist import candidate_mmpg, mmpg_integration
        cfg = mmpg_integration.pipeline.get_capture_config()
        provider = cfg.stt_provider
        streaming = mmpg_integration.pipeline.is_streaming_interview()
        if provider == mmpg_integration.PROVIDER:
            current_stt = mmpg_integration.stt_status()
        else:
            detail = stt_configuration_error(cfg, provider)
            current_stt = {"loaded": not detail, "loading": False, "provider": provider, "detail": detail}
        snapshot = snapshot_session()
        written_exam_mode = get_config().written_exam_mode
        initial_frames: list[dict] = [
            {
                "type": "init",
                "written_exam_mode": written_exam_mode,
                "is_recording": snapshot["is_recording"],
                "is_paused": snapshot["is_paused"],
                "single_mic": snapshot.get("single_mic"),
                "interviewer_streaming": streaming,
                "candidate_capture_provider": mmpg_integration.pipeline.candidate_capture_provider(),
                "mmpg_answer_phase": mmpg_integration.answer_phase(),
                "stt_loaded": current_stt["loaded"],
                "transcriptions": snapshot["transcriptions"],
                "candidate_transcriptions": snapshot.get("candidate_transcriptions", []),
                "candidate_answer_segments": snapshot.get("candidate_answer_segments", []),
                "qa_pairs": snapshot["qa_pairs"],
            },
            {"type": "stt_status", **current_stt},
        ]
        if candidate_mmpg.enabled(cfg) or candidate_mmpg.manager.active:
            initial_frames.append(candidate_mmpg.status())

        # 握手阶段由当前协程独占发送初始帧（sender_task 尚未启动，广播写入 handshake_buffer）
        for frame in initial_frames:
            if not await _safe_send(ws, frame):
                return

        if client.closed:
            return

        # 若握手期间模式已切换且缓冲里没有 workspace_mode，补发最新状态
        current_mode = get_config().written_exam_mode
        if current_mode != written_exam_mode and not any(
            m.get("type") == "workspace_mode" for m in client.handshake_buffer
        ):
            client.handshake_buffer.append({"type": "workspace_mode", "written_exam_mode": current_mode})

        # 将握手窗口期内缓存的广播有序排入该客户端独立发送队列，再启动单发送协程
        buffered = client.handshake_buffer
        client.handshake_buffer = []
        client.handshaking = False
        for buffered_msg in buffered:
            if not client.enqueue(buffered_msg):
                _log.warning("WS client buffer overflowed after handshake, closing")
                await _drop_client(client, code=status.WS_1011_INTERNAL_ERROR)
                return

        client.sender_task = asyncio.create_task(_client_sender(client))
        _log.info("WS connect clients=%d", len(ws_clients))

        while not client.closed:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(msg, dict):
                mtype = msg.get("type")
                if mtype == "ping":
                    ts = time.monotonic()
                    client.last_pong = ts
                    ws._ia_last_pong = ts  # type: ignore[attr-defined]
                    if not client.enqueue({"type": "pong"}):
                        await _drop_client(client, code=status.WS_1011_INTERNAL_ERROR)
                        return
                elif mtype == "pong":
                    ts = time.monotonic()
                    client.last_pong = ts
                    ws._ia_last_pong = ts  # type: ignore[attr-defined]
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _unregister_client(client)
        if client.sender_task is not None:
            client.sender_task.cancel()
            try:
                await client.sender_task
            except (asyncio.CancelledError, Exception):
                pass
        _log.info("WS disconnect clients=%d", len(ws_clients))
