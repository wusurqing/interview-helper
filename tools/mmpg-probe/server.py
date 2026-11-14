"""Local, manually started MMPG observation tool; never imports the main app.

Cloud calls occur only after an accepted /ws/probe start command. JSONL logs
contain transcripts and redacted protocol events, never recorded audio.
"""

import asyncio
import contextlib
import importlib.util
import ipaddress
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.local.json"
LOGS_DIR = BASE_DIR / "logs"
MMPG_URL = "wss://interactai.volcvideo.com/api/v2/mmpg"
MMPG_PATH = "/api/v2/mmpg"
PORT = 18765
DEFAULTS = {
    "rtc_app_id": "", "rtc_room_id": "mmpg-probe", "rtc_user_id": "tester",
    "rtc_bot_id": "mmpg-agent", "rtc_token": "", "volc_access_key": "",
    "volc_secret_key": "", "doubao_api_key": "",
    "doubao_resource_id": "volc.seedasr.sauc.duration",
    "silence_ms": 800, "aivad": True, "twopass": True,
    "enable_prefetch": True, "expire_time_ms": None, "force_end": False,
    "max_session_seconds": 1800,
}
REQUIRED = (
    "rtc_app_id", "rtc_room_id", "rtc_user_id", "rtc_bot_id", "rtc_token",
    "volc_access_key", "volc_secret_key", "doubao_api_key",
)
SECRET_FIELDS = ("volc_access_key", "volc_secret_key", "doubao_api_key", "rtc_token")
ALLOWED_ORIGINS = {
    f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}", f"http://[::1]:{PORT}",
}
ERROR_MESSAGES = {
    "config_invalid": "本地配置无效；请检查字段类型和 JSON 格式。",
    "config_missing": "本地配置缺少必填字段；请填写配置后重试。",
    "options_invalid": "测试选项无效：aivad/twopass 必须为布尔值，silence_ms 为 500–2999 的整数。",
    "dependencies_missing": "缺少后端依赖；请按 README 安装 requirements.txt。",
    "cloud_connection_failed": "MMPG 连接或鉴权失败；请核对开通状态、AK/SK、网络及本机时间。",
    "cloud_protocol_error": "MMPG 返回了无法解析的事件，已停止本次测试。",
    "cloud_error": "MMPG 返回错误；请查看脱敏事件和官方文档核对配置。",
    "session_timeout": "本次测试达到本地时长上限，已请求关闭云端会话。",
    "session_create_timeout": "未在本地等待期限内收到 session.created，已请求关闭会话。",
    "probe_busy": "已有一场测试运行中；请先停止它。",
    "not_running": "当前没有运行中的测试。",
    "invalid_command": "不支持此命令或消息格式。",
    "local_io_error": "本地日志或连接发生错误，已请求关闭云端会话。",
}


class ProbeError(ValueError):
    def __init__(self, code, fields=None):
        self.code = code
        self.fields = list(fields or [])
        super().__init__(ERROR_MESSAGES.get(code, "测试失败。"))


def load_config(path=CONFIG_PATH):
    """Read only the dedicated tool config; absent files leave the UI usable."""
    config = dict(DEFAULTS)
    path = Path(path)
    if not path.exists():
        return config
    try:
        if path.stat().st_size > 64 * 1024:
            raise ProbeError("config_invalid")
        supplied = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict):
            raise ProbeError("config_invalid")
        # Unknown keys cannot change network endpoints, paths, or Python options.
        config.update({key: supplied[key] for key in DEFAULTS if key in supplied})
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError("config_invalid") from None
    return config


def missing_fields(config):
    result = []
    for key in REQUIRED:
        value = config.get(key)
        if (not isinstance(value, str) or not value.strip()
                or value.strip().lower().startswith(("your_", "paste_", "replace_"))
                or "请填写" in value):
            result.append(key)
    return result


def validate_options(config, options=None):
    """Browser overrides are deliberately limited to three nonsecret values."""
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - {"aivad", "silence_ms", "twopass"}:
        raise ProbeError("options_invalid")
    values = {key: config.get(key, DEFAULTS[key]) for key in (
        "aivad", "silence_ms", "twopass", "enable_prefetch", "expire_time_ms",
        "force_end", "max_session_seconds",
    )}
    values.update(options)
    for key in ("aivad", "twopass", "enable_prefetch", "force_end"):
        if type(values[key]) is not bool:
            raise ProbeError("options_invalid")
    silence = values["silence_ms"]
    if type(silence) is not int or not 500 <= silence < 3000:
        raise ProbeError("options_invalid")
    for key in ("expire_time_ms", "max_session_seconds"):
        value = values[key]
        if key == "expire_time_ms" and value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ProbeError("config_invalid", [key])
    # Vendor examples say Prefetch can also enable AIVAD. Keep the off arm
    # a true control and show/log the effective value, not the requested one.
    if not values["aivad"]:
        values["enable_prefetch"] = False
    return values


def public_config(config):
    """Strict whitelist. rtcToken is the one browser-required credential."""
    config = {**DEFAULTS, **config}
    output = {}
    for target, source in (
        ("appId", "rtc_app_id"), ("roomId", "rtc_room_id"),
        ("userId", "rtc_user_id"), ("botId", "rtc_bot_id"), ("rtcToken", "rtc_token"),
    ):
        output[target] = config[source] if isinstance(config[source], str) else ""
    missing = missing_fields(config)
    try:
        opts = validate_options(config)
    except ProbeError:
        opts = validate_options(DEFAULTS)
        missing.append("test_options")
    if output["userId"] and output["userId"] == output["botId"]:
        missing.append("rtc_bot_id_must_differ")
    output.update({
        "aivad": opts["aivad"], "silenceMs": opts["silence_ms"],
        "twopass": opts["twopass"], "prefetch": opts["enable_prefetch"],
        "expireTimeMs": opts["expire_time_ms"], "forceEnd": opts["force_end"],
        "maxSessionSeconds": opts["max_session_seconds"],
        "configured": not missing, "missing": missing,
    })
    return output


def build_session_create(config, options=None):
    config = {**DEFAULTS, **config}
    missing = missing_fields(config)
    if missing:
        raise ProbeError("config_missing", missing)
    if config["rtc_user_id"] == config["rtc_bot_id"]:
        raise ProbeError("config_invalid", ["rtc_bot_id"])
    resource = config["doubao_resource_id"]
    if resource not in {"volc.seedasr.sauc.duration", "volc.seedasr.sauc.concurrent"}:
        raise ProbeError("config_invalid", ["doubao_resource_id"])
    opts = validate_options(config, options)
    vad = {
        "SilenceTime": opts["silence_ms"], "AIVAD": opts["aivad"],
        "ForceEnd": opts["force_end"], "EnablePrefetch": opts["enable_prefetch"],
    }
    if opts["expire_time_ms"] is not None:
        vad["ExpireTime"] = opts["expire_time_ms"]
    # MMPG field hierarchy, not a StartVoiceChat request. No LLM/TTS calls.
    return {
        "event_id": uuid.uuid4().hex, "type": "session.create",
        "session": {
            "AgentConfig": {
                "TransmitConfig": {
                    "Type": "rtc", "AppId": config["rtc_app_id"],
                    "RoomId": config["rtc_room_id"], "UserId": config["rtc_bot_id"],
                    "TargetUserId": [config["rtc_user_id"]],
                },
                "IdleTimeout": 180, "TurnDetectionMode": 0,
                "SubtitleConfig": {"DisableRTSSubtitle": True, "UserTranscriptMode": 1},
            },
            "AudioConfig": {
                "AudioFormat": {"UserAudioFormat": "pcm16", "AssistantAudioFormat": "pcm24", "AnsMode": 0},
                "UserAudioCallbackMode": 0, "VADConfig": vad,
            },
            "ASRConfig": {"Provider": "volcano", "ProviderParams": {
                "Mode": "bigmodel", "StreamMode": 2,
                "Credential": {"ApiKey": config["doubao_api_key"], "ApiResourceId": resource},
                "VolcanoASRParameters": json.dumps({"request": {"enable_nonstream": opts["twopass"]}}, separators=(",", ":")),
            }},
        },
    }


def redact(value, secrets=()):
    """Remove credential fields and literal secrets even inside error text."""
    secret_values = sorted({s for s in secrets if isinstance(s, str) and s}, key=len, reverse=True)
    sensitive = {"authorization", "credential", "credentials", "apikey", "accesskey", "secretkey",
                 "accesskeyid", "secretaccesskey", "accesstoken", "rtctoken", "token", "password",
                 "volcaccesskey", "volcsecretkey", "doubaoapikey", "xsecuritytoken", "xapiaccesskey"}

    def clean(item):
        if isinstance(item, dict):
            result = {}
            media = item.get("type") in {"user.audio.delta", "user.audio.stream", "user.image.delta"}
            for key, child in item.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if normalized in sensitive or normalized.endswith(("apikey", "secretkey", "accesstoken")):
                    result[key] = "[REDACTED]"
                elif media and key == "payload":
                    result[key] = "[OMITTED_MEDIA]"
                else:
                    result[key] = clean(child)
            return result
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, str):
            for secret in secret_values:
                item = item.replace(secret, "[REDACTED]")
            item = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", item)
            item = re.sub(r"(?i)(Authorization\s*[:=]\s*)[^\r\n]+", r"\1[REDACTED]", item)
            return item
        return item

    return clean(value)


def process_cloud_event(event, seen_done):
    """Return one final transcript per conv_id; callers still log every event."""
    if event.get("type") != "user.audio.transcription.done":
        return None
    conv_id, transcript = event.get("conv_id"), event.get("transcript")
    if not isinstance(conv_id, str) or not conv_id or not isinstance(transcript, str):
        return None
    if conv_id in seen_done:
        return None
    seen_done.add(conv_id)
    return {"conv_id": conv_id, "transcript": transcript}


async def connect_mmpg(config):
    """The only cloud connection entry point. Importing this module is offline."""
    try:
        from volcenginesdkcore.signv4 import SignerV4
        from websockets.asyncio.client import connect
    except ImportError:
        raise ProbeError("dependencies_missing") from None
    # Match the vendor's MMPG GET signing example; do not log these headers.
    headers = {}
    SignerV4.sign(MMPG_PATH, "GET", headers, "", None, {},
                  config["volc_access_key"], config["volc_secret_key"], "cn-north-1", "rtc")
    return await connect(MMPG_URL, additional_headers=headers, open_timeout=12,
                         close_timeout=3, ping_interval=20, ping_timeout=20, max_size=1024 * 1024)


class ProbeRun:
    def __init__(self, config, options, send, logs_dir=LOGS_DIR, connector=None):
        self.config = {**DEFAULTS, **config}
        self.options = validate_options(self.config, options)
        self.request = build_session_create(self.config, options)
        self.send = send
        self.connector = connector or connect_mmpg
        self.logs_dir = Path(logs_dir)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.started = time.monotonic()
        self.secrets = [self.config[key] for key in SECRET_FIELDS]
        self.cloud = None
        self.task = None
        self.log_file = None
        self.seen_done = set()
        self.stopping = False
        self.ready = False
        self.reason = "stopped"
        self.close_sent = False
        self.current_conv_id = None
        self.closed_conv_ids = set()
        self.marks_by_conv = {}
        self.mark_number = 0
        self.done_count = 0
        self.cleanup_started = False
        self.stop_lock = asyncio.Lock()

    def stamp(self):
        return {"run_id": self.run_id, "elapsed_ms": round((time.monotonic() - self.started) * 1000, 3),
                "received_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}

    async def emit(self, kind, **fields):
        record = redact({"type": kind, **self.stamp(), **fields}, self.secrets)
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.log_file.flush()
        await self.send(record)

    async def error(self, code):
        with contextlib.suppress(Exception):
            await self.emit("error", code=code, message=ERROR_MESSAGES.get(code, "测试失败。"))

    async def mark_end(self):
        self.mark_number += 1
        stamp = self.stamp()
        conv_id = self.current_conv_id
        if conv_id is not None:
            self.marks_by_conv[conv_id] = (self.mark_number, stamp["elapsed_ms"])
        await self.emit("mark_end_ack", **stamp, mark_id=self.mark_number,
                        mark_label="manual_button", conv_id=conv_id,
                        associated=conv_id is not None)

    async def close_cloud(self):
        if self.cloud is None or self.close_sent:
            return
        self.close_sent = True
        message = {"event_id": uuid.uuid4().hex, "type": "session.close"}
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.cloud.send(json.dumps(message)), 2)
            await self.emit("event", direction="server_to_cloud", event=message)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.cloud.close(), 4)

    async def run(self):
        self.task = asyncio.current_task()
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = (self.logs_dir / f"{self.run_id}.jsonl").open("x", encoding="utf-8", newline="\n")
            await self.emit("status", status="starting", options=self.options,
                            log_file=f"logs/{self.run_id}.jsonl", timing_basis="server_probe_start")
            if self.stopping:
                return
            async with asyncio.timeout(self.options["max_session_seconds"]):
                try:
                    self.cloud = await self.connector(self.config)
                except ProbeError:
                    raise
                except Exception:
                    raise ProbeError("cloud_connection_failed") from None
                if self.stopping:
                    return
                await self.emit("status", status="cloud_connected")
                await self.cloud.send(json.dumps(self.request, ensure_ascii=False))
                await self.emit("event", direction="server_to_cloud", event=self.request)
                creation_deadline = time.monotonic() + 20
                while not self.stopping:
                    remaining = None if self.ready else max(0.001, creation_deadline - time.monotonic())
                    try:
                        payload = await asyncio.wait_for(self.cloud.recv(), remaining)
                    except TimeoutError:
                        raise ProbeError("session_create_timeout") from None
                    stamp = self.stamp()
                    try:
                        event = json.loads(payload)
                        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                            raise ValueError()
                    except (ValueError, TypeError, UnicodeError):
                        raise ProbeError("cloud_protocol_error") from None
                    event = redact(event, self.secrets)
                    event_type, conv_id = event["type"], event.get("conv_id")
                    valid_conv = isinstance(conv_id, str) and bool(conv_id)
                    if valid_conv:
                        if event_type in {"user.conversation.created", "user.audio.transcription.started",
                                          "user.audio.transcription.segment"}:
                            if conv_id not in self.closed_conv_ids:
                                self.current_conv_id = conv_id
                        elif event_type in {"user.audio.transcription.done", "user.conversation.canceled"}:
                            self.closed_conv_ids.add(conv_id)
                            if self.current_conv_id == conv_id:
                                self.current_conv_id = None
                            if event_type == "user.conversation.canceled":
                                self.marks_by_conv.pop(conv_id, None)
                    duplicate = event_type == "user.audio.transcription.done" and valid_conv and conv_id in self.seen_done
                    # Settle association before yielding to the browser sender:
                    # a click received after this final must not attach to it.
                    final = process_cloud_event(event, self.seen_done)
                    bound_mark = self.marks_by_conv.pop(final["conv_id"], None) if final else None
                    await self.emit("event", **stamp, direction="cloud_to_server", event=event, duplicate_done=duplicate)
                    if event["type"] == "session.created":
                        self.ready = True
                        await self.emit("status", **stamp, status="session_created")
                    if event["type"] == "error":
                        raise ProbeError("cloud_error")
                    if final is not None:
                        self.done_count += 1
                        mark_id, delay = None, None
                        if bound_mark is not None:
                            mark_id, mark_ms = bound_mark
                            delay = round(stamp["elapsed_ms"] - mark_ms, 3)
                        await self.emit("done", **stamp, **final, duplicate=False,
                                        since_mark_ms=delay, mark_id=mark_id,
                                        mark_label="manual_button", done_count=self.done_count)
        except asyncio.CancelledError:
            pass
        except ProbeError as exc:
            self.reason = exc.code
            await self.error(exc.code)
        except TimeoutError:
            self.reason = "session_timeout"
            await self.error(self.reason)
        except Exception:
            self.reason = "local_io_error" if self.cloud is None else "cloud_connection_failed"
            await self.error(self.reason)
        finally:
            self.stopping = True
            self.cleanup_started = True
            await self.close_cloud()
            with contextlib.suppress(Exception):
                await self.emit("status", status="stopped", reason=self.reason, done_count=self.done_count)
            if self.log_file is not None:
                self.log_file.close()

    async def stop(self, reason="user_stop"):
        async with self.stop_lock:
            already_stopping = self.stopping
            if not already_stopping:
                self.reason = reason
            self.stopping = True
            # Let a just-created run enter its try/finally before cancellation.
            await asyncio.sleep(0)
            if self.task and self.task is not asyncio.current_task() and not self.task.done():
                # Never cancel a second time while session.close is in flight.
                if not already_stopping and not self.cleanup_started:
                    self.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(asyncio.shield(self.task), 8)


def local_request_allowed(host, origin=None, client_host=None, require_origin=False):
    try:
        parsed = urlsplit("http://" + (host or ""))
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.port != PORT:
            return False
        if client_host and not ipaddress.ip_address(client_host).is_loopback:
            return False
    except ValueError:
        return False
    return (origin in ALLOWED_ORIGINS) if (origin or require_origin) else True


def create_app(config_path=CONFIG_PATH, logs_dir=LOGS_DIR, connector=None):
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    @contextlib.asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            active = application.state.active_probe
            if active is not None:
                await active.stop("server_shutdown")

    app = FastAPI(title="MMPG manual probe", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.active_probe = None

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        if (not local_request_allowed(request.headers.get("host"), request.headers.get("origin"),
                                      request.client.host if request.client else None)
                or request.headers.get("sec-fetch-site") == "cross-site"):
            return JSONResponse({"error": "local_only"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/config")
    async def config_endpoint():
        try:
            return public_config(load_config(config_path))
        except ProbeError:
            response = public_config(DEFAULTS)
            response["missing"] = ["config.local.json_invalid"]
            return response

    @app.get("/api/health")
    async def health_endpoint():
        active = app.state.active_probe
        return {"ok": True, "cloudConnected": bool(active and active.ready and not active.stopping),
                "running": bool(active and not active.stopping),
                "missingDependencies": [module for module in ("websockets", "volcenginesdkcore")
                                        if importlib.util.find_spec(module) is None]}

    @app.get("/")
    @app.get("/index.html")
    async def index():
        if not (BASE_DIR / "index.html").is_file():
            return JSONResponse({"error": "frontend_not_built"}, status_code=503)
        return FileResponse(BASE_DIR / "index.html")

    @app.get("/style.css")
    async def stylesheet():
        if not (BASE_DIR / "style.css").is_file():
            return JSONResponse({"error": "frontend_not_built"}, status_code=404)
        return FileResponse(BASE_DIR / "style.css", media_type="text/css")

    # Never expose the tool root: it contains config.local.json and transcripts.
    app.mount("/dist", StaticFiles(directory=BASE_DIR / "dist", check_dir=False), name="dist")

    @app.websocket("/ws/probe")
    async def probe_socket(websocket: WebSocket):
        if not local_request_allowed(websocket.headers.get("host"), websocket.headers.get("origin"),
                                     websocket.client.host if websocket.client else None, require_origin=True):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        send_lock = asyncio.Lock()
        owned = None

        async def send(message):
            async with send_lock:
                await websocket.send_json(message)

        async def reject(code, fields=None):
            await send({"type": "error", "code": code, "message": ERROR_MESSAGES[code],
                        "missing": fields or []})

        async def worker(run):
            try:
                await run.run()
            finally:
                if app.state.active_probe is run:
                    app.state.active_probe = None

        try:
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 4096:
                    await reject("invalid_command")
                    continue
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError()
                except ValueError:
                    await reject("invalid_command")
                    continue
                kind = message.get("type")
                if kind == "start":
                    if app.state.active_probe is not None:
                        await reject("probe_busy")
                        continue
                    try:
                        config = load_config(config_path)
                        if "options" in message:
                            if set(message) - {"type", "options"}:
                                raise ProbeError("options_invalid")
                            options = message["options"]
                        else:
                            options = {key: value for key, value in message.items() if key != "type"}
                        owned = ProbeRun(config, options, send, logs_dir=logs_dir, connector=connector)
                    except ProbeError as exc:
                        await reject(exc.code, exc.fields)
                        continue
                    app.state.active_probe = owned
                    owned.task = asyncio.create_task(worker(owned))
                elif kind == "stop":
                    if owned is not None and not owned.stopping:
                        await owned.stop()
                    else:
                        await send({"type": "status", "status": "stopped", "reason": "not_running"})
                elif kind == "mark_end":
                    if owned is None or owned.stopping or not owned.ready:
                        await reject("not_running")
                    else:
                        await owned.mark_end()
                else:
                    await reject("invalid_command")
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if owned is not None:
                await owned.stop("frontend_disconnected")

    return app


if __name__ == "__main__":
    try:
        import uvicorn
        logging.getLogger("websockets").setLevel(logging.WARNING)
        uvicorn.run(create_app(), host="127.0.0.1", port=PORT, access_log=False, log_level="warning")
    except ImportError:
        raise SystemExit("缺少依赖，请先按 README 安装 tools/mmpg-probe/requirements.txt。") from None
