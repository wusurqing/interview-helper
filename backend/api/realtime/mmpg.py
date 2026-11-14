"""Private browser publisher channel for the desktop MMPG recognizer."""

import asyncio
import json
import sys
from contextlib import suppress
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect

from api.realtime.ws import _ws_authorized
from core.auth import is_loopback_host
from services.stt.mmpg import MmpgError


_START_COMMANDS = {"start", "resume", "test"}
_RTC_MESSAGES = {"rtc_ready", "rtc_fault", "rtc_reconnecting", "rtc_recovered", "renew", "audio_status"}
_MAX_CONTROL_BYTES = 2048


def _local_owner(ws):
    if sys.platform != "win32" or not ws.client or not is_loopback_host(ws.client.host):
        return False
    try:
        source = urlsplit(ws.headers.get("origin", ""))
        target = ws.url
        scheme = "https" if target.scheme == "wss" else "http"
        default_port = 443 if scheme == "https" else 80
        return (
            is_loopback_host(target.hostname)
            and source.scheme == scheme
            and source.hostname == target.hostname
            and (source.port or default_port) == (target.port or default_port)
            and source.username is None
            and not (source.path or source.query or source.fragment)
            and _ws_authorized(ws)
        )
    except (ValueError, IndexError):
        return False


def _parse_control(payload):
    if not isinstance(payload, str) or len(payload) > _MAX_CONTROL_BYTES:
        raise MmpgError("invalid_command")
    try:
        message = json.loads(payload)
    except (ValueError, RecursionError):
        raise MmpgError("invalid_command") from None
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise MmpgError("invalid_command")
    return message


def _devices(message):
    ids = message.get("device_ids")
    mic = message.get("candidate_mic_id")
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
            or any(type(value) is not int or not -2**31 <= value < 2**31 for value in ids)
            or len(set(ids)) != len(ids)
            or (mic is not None and (type(mic) is not int or not -2**31 <= mic < 2**31))):
        raise MmpgError("invalid_command")
    return ids, mic


def bind_mmpg_routes(app, manager, *, on_control):
    @app.websocket("/ws/stt/mmpg")
    async def publisher_endpoint(ws: WebSocket):
        if not _local_owner(ws):
            await ws.close(code=1008)
            return

        send_lock = asyncio.Lock()

        async def send(message):
            async with send_lock:
                operation = ws.send_bytes if isinstance(message, bytes) else ws.send_json
                await asyncio.wait_for(operation(message), timeout=2)

        if not manager.attach(ws, send):
            with suppress(WebSocketDisconnect, OSError, RuntimeError, asyncio.TimeoutError):
                await ws.accept()
                await send({"type": "error", "code": "mmpg_busy"})
                await ws.close(code=1008, reason="mmpg_busy")
            return
        try:
            await ws.accept()
            await send({"type": "state", "initial": True, **manager.status()})
            while True:
                payload = await ws.receive()
                if payload["type"] == "websocket.disconnect":
                    break
                try:
                    message = _parse_control(payload.get("text"))
                    kind = message["type"]
                    if kind in _START_COMMANDS:
                        ids, mic = _devices(message)
                        manager.start(ws, kind, ids, mic, preflight=on_control)
                    elif kind == "stop":
                        await manager.stop()
                    elif kind in _RTC_MESSAGES:
                        await manager.rtc_event(ws, message)
                    elif kind == "ping":
                        await send({"type": "pong"})
                    else:
                        raise MmpgError("invalid_command")
                except MmpgError as exc:
                    if exc.code == "owner_unavailable":
                        break
                    await send({"type": "error", "code": exc.code})
        except (WebSocketDisconnect, OSError, RuntimeError, asyncio.TimeoutError):
            pass
        finally:
            await manager.detach(ws)
            with suppress(WebSocketDisconnect, RuntimeError, OSError):
                await ws.close()
