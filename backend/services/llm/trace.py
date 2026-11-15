"""Private, lossless-on-observed-data LLM request journal.

The JSONL file contains prompts and replies. It is intentionally separate from
application and console logging and must never include authentication material.
"""

import base64
import contextlib
import contextvars
import codecs
import functools
import inspect
from datetime import datetime
import hashlib
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from openai import DefaultHttpxClient

from core.logger import LOG_DIR as _DEFAULT_LOG_DIR

LOG_DIR = _DEFAULT_LOG_DIR
_scope = contextvars.ContextVar("llm_trace_scope", default=None)
_lock = threading.RLock()
_handler_path = None
_handler = None
_log = logging.getLogger("llm.trace")
_log.propagate = False
_REDACTED = "[REDACTED]"
_SECRET_KEYS = {
    "authorization", "proxy-authorization", "x-api-key", "api-key", "api_key", "apikey",
    "x-auth-token", "access_token", "refresh_token", "client_secret", "token", "secret", "password",
}
_DATA_URL = re.compile(r"^data:([^;,]+);base64,([A-Za-z0-9+/=]+)$", re.I)
_URL_CREDENTIALS = re.compile(r"(https?://)[^/?#@\s]+@", re.I)
_QUERY_SECRET = re.compile(r"([?&](?:" + "|".join(sorted(_SECRET_KEYS)) + r")=)[^&#\s]+", re.I)
_AUTH_HEADER = re.compile(r"(\b(?:proxy-)?authorization\s*[:=]\s*(?:(?:bearer|basic)\s+)?)[^\s,;]+", re.I)


def prune_unused_assets():
    """Remove only assets no longer named by the live log or retained archives."""
    directory = Path(LOG_DIR)
    assets = directory / "llm_assets"
    if not assets.exists():
        return
    referenced = set()

    def visit(value):
        if isinstance(value, dict):
            path = value.get("path")
            if isinstance(path, str) and path.startswith("llm_assets/"):
                referenced.add(path)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    try:
        for logfile in directory.glob("llm.jsonl*"):
            with logfile.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        visit(json.loads(line))
                    except ValueError:
                        continue
        for asset in assets.iterdir():
            if asset.is_file() and f"llm_assets/{asset.name}" not in referenced:
                asset.unlink()
    except OSError:
        logging.getLogger("llm.trace.failure").exception("Unable to prune LLM assets")


class _TraceFileHandler(TimedRotatingFileHandler):
    def emit(self, record):
        old_rollover_at = self.rolloverAt
        super().emit(record)
        if self.rolloverAt != old_rollover_at:
            prune_unused_assets()


def _now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _redact_text(text, api_key):
    if api_key:
        text = text.replace(api_key, _REDACTED)
        for encoded in {quote(api_key, safe=""), quote_plus(api_key)} - {api_key}:
            # Fold case only in escapes, preserving the key's case-sensitive text.
            pattern = re.sub(r"%[0-9A-F]{2}", lambda match: f"(?i:{match.group()})", re.escape(encoded))
            text = re.sub(pattern, _REDACTED, text)
    text = _URL_CREDENTIALS.sub(r"\g<1>[REDACTED]@", text)
    text = _QUERY_SECRET.sub(r"\g<1>[REDACTED]", text)
    return _AUTH_HEADER.sub(r"\g<1>[REDACTED]", text)


def _safe_url(url, api_key):
    try:
        parts = urlsplit(str(url))
        authority = _redact_text(parts.netloc.rsplit("@", 1)[-1], api_key)
        path = parts.path
        if api_key and api_key in unquote(path):
            path = quote(unquote(path).replace(api_key, _REDACTED), safe="/")
        query = urlencode([
            (_redact_text(key, api_key),
             _REDACTED if key.lower() in _SECRET_KEYS else _redact_text(value, api_key))
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ])
        return urlunsplit((parts.scheme, authority, path, query, ""))
    except (ValueError, TypeError):
        return "[invalid URL]"


def _asset(data_url):
    match = _DATA_URL.fullmatch(data_url)
    if not match:
        return None
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    mime = match.group(1).lower()
    if not mime.startswith("image/"):
        return None
    digest = hashlib.sha256(data).hexdigest()
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}.get(mime, "img")
    rel = Path("llm_assets") / f"{digest}.{ext}"
    path = Path(LOG_DIR) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        pass
    return {"path": rel.as_posix(), "mime": mime, "bytes": len(data), "sha256": digest}


def _safe(value, api_key=""):
    if isinstance(value, dict):
        if value.get("type") == "base64" and str(value.get("media_type", "")).startswith("image/"):
            try:
                asset = _asset(f"data:{value['media_type']};base64,{value.get('data', '')}")
            except OSError:
                asset = None
            if asset is not None:
                value = {**value, "data": asset}
        return {
            _redact_text(str(key), api_key): (
                _REDACTED if str(key).lower() in _SECRET_KEYS else _safe(item, api_key)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(v, api_key) for v in value]
    if isinstance(value, str):
        try:
            asset = _asset(value)
        except OSError:
            asset = None
        if asset is not None:
            return asset
        return _redact_text(value, api_key)
    return value


def _text_parts(value):
    if isinstance(value, list):
        return [
            piece
            for part in value
            for piece in _text_parts(part.get("text") if isinstance(part, dict) else part)
        ]
    return [] if value is None else [str(value)]


def _text(value):
    return "".join(_text_parts(value))


def _redact_joined_text(parts, api_key):
    text = "".join(parts)
    redacted = _redact_text(text, api_key)
    split_secret = redacted != text and redacted != "".join(_redact_text(part, api_key) for part in parts)
    return redacted, split_secret


def _decode_body(body):
    try:
        return json.loads(body)
    except ValueError:
        return {"unparsed_body": body.decode("utf-8", "replace")}


def _writer():
    global _handler, _handler_path
    path = str(Path(LOG_DIR) / "llm.jsonl")
    with _lock:
        if _handler_path != path:
            if _handler is not None:
                _log.removeHandler(_handler)
                _handler.close()
            Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
            _handler = _TraceFileHandler(path, when="midnight", backupCount=30, encoding="utf-8")
            _handler.setFormatter(logging.Formatter("%(message)s"))
            _log.addHandler(_handler)
            _log.setLevel(logging.INFO)
            _handler_path = path
        return _log


def _write(value):
    try:
        with _lock:
            _writer().info(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))
    except Exception:
        logging.getLogger("llm.trace.failure").exception("Unable to write LLM trace")


@contextlib.contextmanager
def scope(scene, **ids):
    parent = _scope.get() or {}
    state = {**parent, "scene": scene, **{k: v for k, v in ids.items() if v is not None}}
    state.setdefault("call_id", uuid4().hex)
    token = _scope.set(state)
    try:
        yield state
    finally:
        _scope.reset(token)


def current_context():
    return dict(_scope.get() or {})


def traced(scene):
    """Scope a complete logical call, including generators consumed later."""
    def decorate(func):
        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def generator(*args, **kwargs):
                context = dict(kwargs.pop("trace_context", None) or {})
                with scope(context.pop("scene", scene), call_id=uuid4().hex, **context):
                    yield from func(*args, **kwargs)
            return generator
        @functools.wraps(func)
        def normal(*args, **kwargs):
            context = dict(kwargs.pop("trace_context", None) or {})
            with scope(context.pop("scene", scene), call_id=uuid4().hex, **context):
                return func(*args, **kwargs)
        return normal
    return decorate


def begin(url, api_type, payload, *, api_key="", context=None):
    try:
        return Attempt(url, api_type, payload, api_key=api_key, context=context)
    except Exception:
        logging.getLogger("llm.trace.failure").exception("Unable to begin LLM trace")
        return _NullAttempt()


class _NullAttempt:
    status_code = None

    def headers(self, *_args): pass
    def response(self, *_args): pass
    def stream_event(self, *_args): pass
    def finish(self, *_args): pass


class TracingHttpxClient(DefaultHttpxClient):
    """Observe SDK attempts after HTTPX decoding without changing its transport."""

    def __init__(self):
        super().__init__(event_hooks={"request": [self._request], "response": [self._response]})

    @staticmethod
    def _request(request):
        try:
            payload = json.loads(request.content)
        except (ValueError, RuntimeError):
            payload = {}
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        api_key = token.strip() if scheme.lower() in {"bearer", "basic"} else authorization
        api_key = api_key or request.headers.get("x-api-key", "")
        request.extensions["llm_trace_stream"] = isinstance(payload, dict) and bool(payload.get("stream"))
        request.extensions["llm_trace"] = begin(str(request.url), "openai", payload, api_key=api_key)

    @staticmethod
    def _response(response):
        attempt = response.request.extensions.get("llm_trace")
        if attempt is None:
            return
        attempt.headers(response.status_code)
        old_iter = response.iter_bytes
        old_close = response.close
        failed_http = response.status_code >= 400
        stream = response.request.extensions.get("llm_trace_stream", False) and not failed_http
        if hasattr(response, "_content") and not stream:
            attempt.response(_decode_body(response.content))
            attempt.finish("failed" if failed_http else "completed")
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        body = bytearray()
        reading = False
        provider_error = None

        def finish(status, error=None):
            if not stream and body:
                attempt.response(_decode_body(body))
            if provider_error is not None:
                attempt.finish("failed", error if error is not None else provider_error)
            else:
                attempt.finish(status, error)

        def observed_iter(chunk_size=None):
            nonlocal pending, reading, provider_error
            reading = True
            status = "failed" if failed_http else "completed"
            error = None
            try:
                for chunk in old_iter(chunk_size=chunk_size):
                    reading = False
                    if stream:
                        pending += decoder.decode(chunk)
                        while match := re.search(r"\r?\n\r?\n", pending):
                            frame, pending = pending[:match.start()], pending[match.end():]
                            data = "\n".join(line[5:].lstrip() for line in frame.splitlines()
                                             if line.startswith("data:"))
                            if data and not data.startswith("[DONE]"):
                                try:
                                    event = json.loads(data)
                                except ValueError:
                                    event = {"unparsed_data": data}
                                    provider_error = "Malformed JSON in provider stream"
                                attempt.stream_event(event)
                                if isinstance(event, dict) and (details := event.get("error")):
                                    message = details.get("message") if isinstance(details, dict) else None
                                    provider_error = message if isinstance(message, str) and message else (
                                        "Provider reported a streaming error"
                                    )
                    else:
                        body.extend(chunk)
                    yield chunk
                    reading = True
            except GeneratorExit:
                status = "failed" if failed_http else "cancelled"
                raise
            except Exception as exc:
                status = "failed"
                error = exc
                raise
            finally:
                reading = False
                finish(status, error)

        def observed_close():
            try:
                old_close()
            except Exception as exc:
                finish("failed", exc)
                raise
            else:
                # HTTPX closes on EOF while reading. A close between chunks is explicit.
                if not reading:
                    finish("failed" if failed_http else "cancelled")

        response.iter_bytes = observed_iter
        response.close = observed_close

    def send(self, request, **kwargs):
        try:
            return super().send(request, **kwargs)
        except Exception as exc:
            attempt = request.extensions.get("llm_trace")
            if attempt is not None:
                attempt.finish("failed", exc)
            raise


class Attempt:
    def __init__(self, url, api_type, payload, *, api_key="", context=None):
        self.api_key = api_key
        context = dict(context if context is not None else current_context())
        self.call_id = _safe(context.pop("call_id", uuid4().hex), api_key)
        self.context = _safe(context, api_key)
        self.attempt_id = uuid4().hex
        self.api_type = api_type
        self.request_at = _now()
        self.t0 = time.monotonic()
        self.response_headers_at = None
        self.first_output_at = None
        self.headers_latency_ms = None
        self.first_output_latency_ms = None
        self.status_code = None
        self.events = []
        self.body = None
        self.usage = None
        self.output_text = []
        self.reasoning = []
        self.finish_reason = None
        self._finished = False
        _write({"schema_version": 1, "event": "request", "call_id": self.call_id,
                "attempt_id": self.attempt_id, "context": self.context, "at": self.request_at,
                "protocol": api_type, "url": _safe_url(url, api_key), "request": _safe(payload, api_key)})

    def headers(self, status_code):
        self.status_code = status_code
        self.response_headers_at = _now()
        self.headers_latency_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def response(self, body):
        self.body = _safe(body, self.api_key)
        if isinstance(body, dict):
            self._usage(body.get("usage"))
            if self.api_type == "anthropic":
                self.output_text = [_text(b.get("text")) for b in body.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
                self.reasoning = [_text(b.get("thinking")) for b in body.get("content", []) if isinstance(b, dict) and b.get("type") == "thinking"]
                self.finish_reason = body.get("stop_reason")
            elif self.api_type == "responses":
                self.output_text = [_text(part.get("text")) for item in body.get("output", []) if isinstance(item, dict)
                                    for part in item.get("content", []) if isinstance(part, dict) and part.get("type") == "output_text"]
                self.reasoning = [_text(part.get("text")) for item in body.get("output", [])
                                  if isinstance(item, dict) and item.get("type") == "reasoning"
                                  for part in item.get("summary", []) if isinstance(part, dict)]
                self.finish_reason = body.get("status")
            else:
                choices = body.get("choices") or []
                for choice in choices:
                    if isinstance(choice, dict):
                        message = choice.get("message") or {}
                        if isinstance(message, dict):
                            self.output_text.extend(_text_parts(message.get("content")))
                            self.reasoning.extend(_text_parts(message.get("reasoning_content") or message.get("reasoning")))
                        self.finish_reason = choice.get("finish_reason") or self.finish_reason
        if (any(self.output_text) or any(self.reasoning)) and self.first_output_at is None:
            self.first_output_at = _now()
            self.first_output_latency_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def _usage(self, usage):
        if isinstance(usage, dict):
            if self.usage is None:
                self.usage = {}
            for key, value in usage.items():
                if value is not None:
                    self.usage[key] = value

    def stream_event(self, obj):
        self.events.append({"at": _now(), "data": _safe(obj, self.api_key)})
        if not isinstance(obj, dict):
            return
        event = obj.get("type")
        if self.api_type == "anthropic":
            if event == "message_start":
                self._usage((obj.get("message") or {}).get("usage"))
            elif event == "message_delta":
                self._usage(obj.get("usage"))
                self.finish_reason = (obj.get("delta") or {}).get("stop_reason") or self.finish_reason
            elif event == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta":
                    self.output_text.append(_text(delta.get("text")))
                elif delta.get("type") == "thinking_delta":
                    self.reasoning.append(_text(delta.get("thinking")))
        elif self.api_type == "responses":
            if event == "response.output_text.delta":
                self.output_text.append(_text(obj.get("delta")))
            elif event in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                self.reasoning.append(_text(obj.get("delta")))
            elif event in ("response.completed", "response.incomplete", "response.failed"):
                response = obj.get("response") or {}
                self._usage(response.get("usage"))
                self.finish_reason = response.get("status") or event
        else:
            self._usage(obj.get("usage"))
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    self.output_text.extend(_text_parts(delta.get("content")))
                    self.reasoning.extend(_text_parts(delta.get("reasoning_content") or delta.get("reasoning")))
                self.finish_reason = choice.get("finish_reason") or self.finish_reason
        if (any(self.output_text) or any(self.reasoning)) and self.first_output_at is None:
            self.first_output_at = _now()
            self.first_output_latency_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def finish(self, status="completed", error=None):
        if self._finished:
            return
        self._finished = True
        usage = _safe(self.usage, self.api_key)
        output_text, split_output_secret = _redact_joined_text(self.output_text, self.api_key)
        reasoning_text, split_reasoning_secret = _redact_joined_text(self.reasoning, self.api_key)
        # Individually safe fragments can still expose a joined credential.
        keep_raw_output = not (split_output_secret or split_reasoning_secret)
        input_raw = (usage.get("prompt_tokens", usage.get("input_tokens")) if usage else None)
        read = (usage.get("cache_read_input_tokens") if usage else None)
        if read is None and usage:
            read = (usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}).get("cached_tokens")
        created = usage.get("cache_creation_input_tokens") if usage else None
        input_total = input_raw
        if self.api_type == "anthropic" and input_raw is not None:
            input_total = input_raw + (read or 0) + (created or 0)
        output = usage.get("completion_tokens", usage.get("output_tokens")) if usage else None
        reasoning = None
        if usage:
            reasoning = (usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        _write({"schema_version": 1, "event": "result", "call_id": self.call_id,
                "attempt_id": self.attempt_id, "context": self.context, "protocol": self.api_type,
                "status": status, "http_status": self.status_code, "request_at": self.request_at,
                "response_headers_at": self.response_headers_at, "first_output_at": self.first_output_at,
                "headers_latency_ms": self.headers_latency_ms,
                "first_output_latency_ms": self.first_output_latency_ms,
                "finished_at": _now(), "duration_ms": round((time.monotonic() - self.t0) * 1000, 1),
                "output_text": output_text, "reasoning_text": reasoning_text,
                "finish_reason": _safe(self.finish_reason, self.api_key),
                "response": self.body if keep_raw_output else None,
                "stream_events": self.events or None if keep_raw_output else None,
                "usage_raw": usage,
                "tokens": {"input": input_total, "output": output,
                           "total": usage.get("total_tokens") if usage else None,
                           "reasoning": reasoning, "cached_read": read,
                           "cached_write": created,
                           "cache_hit": usage.get("prompt_cache_hit_tokens") if usage else None,
                           "cache_miss": usage.get("prompt_cache_miss_tokens") if usage else None},
                "error": _safe(str(error), self.api_key) if error is not None else None})
