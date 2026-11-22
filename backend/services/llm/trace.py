"""Private, lossless-on-observed-data LLM request journal.

The JSONL file contains prompts and replies. It is intentionally separate from
application and console logging and must never include authentication material.
"""

import base64
import collections
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
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from openai import DefaultHttpxClient

from core.logger import LOG_DIR as _DEFAULT_LOG_DIR

LOG_DIR = _DEFAULT_LOG_DIR
DEFAULT_MAX_ASSETS_MB = 256.0
_ROLLOVER_PRUNE_GRACE_SECONDS = 60.0
_scope = contextvars.ContextVar("llm_trace_scope", default=None)
_observer = contextvars.ContextVar("llm_archive_observer", default=None)
_lock = threading.RLock()
_asset_lock = threading.Lock()
_in_flight_assets: collections.Counter[str] = collections.Counter()
_prune_seen_assets: dict[int, set[str]] = {}
_prune_scan_seq = 0
_prune_lock = threading.Lock()
_prune_pending: dict[Path, float] = {}
_prune_thread: threading.Thread | None = None
_handler_path = None
_handler = None
_log = logging.getLogger("llm.trace")
_log.propagate = False
_REDACTED = "[REDACTED]"
_IMAGE_OMITTED = "[IMAGE_OMITTED]"
_SECRET_KEYS = {
    "authorization", "proxy-authorization", "x-api-key", "api-key", "api_key", "apikey",
    "x-auth-token", "access_token", "refresh_token", "client_secret", "token", "secret", "password",
}
_METADATA_CONTEXT_KEYS = {
    "scene", "qa_id", "source", "answer_attempt", "mmpg_group_id", "mmpg_revision", "mmpg_seq",
    "session_id", "turn_id", "job_id", "record_id",
}
_TOKEN_COUNT_KEYS = {
    "prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
}
_TOKEN_DETAIL_KEYS = {
    "prompt_tokens_details": "cached_tokens", "input_tokens_details": "cached_tokens",
    "completion_tokens_details": "reasoning_tokens", "output_tokens_details": "reasoning_tokens",
}
_FINISH_REASONS = {
    "stop", "length", "content_filter", "tool_calls", "function_call", "end_turn", "max_tokens",
    "stop_sequence", "tool_use", "pause_turn", "refusal", "completed", "incomplete", "failed",
    "response.completed", "response.incomplete", "response.failed", "model_context_window_exceeded",
}
_DATA_URL = re.compile(r"^data:([^;,]+);base64,([A-Za-z0-9+/=]+)$", re.I)
_URL_CREDENTIALS = re.compile(r"(https?://)[^/?#@\s]+@", re.I)
_QUERY_SECRET = re.compile(r"([?&](?:" + "|".join(sorted(_SECRET_KEYS)) + r")=)[^&#\s]+", re.I)
_AUTH_HEADER = re.compile(r"(\b(?:proxy-)?authorization\s*[:=]\s*(?:(?:bearer|basic)\s+)?)[^\s,;]+", re.I)


def trace_mode() -> str:
    """Return configured LLM trace privacy mode: 'off', 'metadata', or 'full'."""
    raw = os.environ.get("IA_LLM_TRACE", "").strip().lower()
    if raw in ("", "full", "1", "true", "on"):
        return "full"
    if raw in ("0", "off", "false", "none", "disabled"):
        return "off"
    if raw in ("metadata", "meta"):
        return "metadata"
    return "metadata"


def max_assets_bytes() -> int:
    """Return configured total byte cap for llm_assets/ (default 256 MB)."""
    raw = os.environ.get("IA_LLM_TRACE_MAX_ASSETS_MB", "").strip()
    if not raw:
        return int(DEFAULT_MAX_ASSETS_MB * 1024 * 1024)
    try:
        mb = float(raw)
    except ValueError:
        return int(DEFAULT_MAX_ASSETS_MB * 1024 * 1024)
    if not math.isfinite(mb) or mb < 0:
        return int(DEFAULT_MAX_ASSETS_MB * 1024 * 1024)
    return int(mb * 1024 * 1024)


def _enforce_assets_quota_locked(
    assets: Path,
    *,
    admission_limit: int | None = None,
    protected: set[str] | None = None,
) -> bool:
    if not assets.exists():
        return True
    limit = max_assets_bytes() if admission_limit is None else admission_limit
    protected_names = set(protected or ())
    entries: list[tuple[int, str, int, Path]] = []
    total = 0
    protected_bytes = 0
    for asset in assets.iterdir():
        if not asset.is_file():
            continue
        try:
            st = asset.stat()
        except OSError:
            continue
        total += st.st_size
        if asset.name in protected_names:
            protected_bytes += st.st_size
        else:
            entries.append((st.st_mtime_ns, asset.name, st.st_size, asset))
    if total <= limit:
        return True
    # Reject an image before eviction if even removing every eligible file cannot fit it.
    if admission_limit is not None and protected_bytes > limit:
        return False
    entries.sort(key=lambda item: (item[0], item[1]))
    for _mtime, _name, size, asset in entries:
        if total <= limit:
            break
        try:
            asset.unlink()
            total -= size
        except FileNotFoundError:
            total -= size
        except OSError:
            continue
    return total <= limit


def prune_unused_assets(log_dir=None, *, min_age_seconds: float = 0.0):
    """Remove unreferenced assets and enforce the llm_assets/ disk quota."""
    global _prune_scan_seq
    directory = Path(LOG_DIR if log_dir is None else log_dir)
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

    with _asset_lock:
        _prune_scan_seq += 1
        scan_id = _prune_scan_seq
        _prune_seen_assets[scan_id] = set(_in_flight_assets.keys())

    try:
        for logfile in directory.glob("llm.jsonl*"):
            try:
                with logfile.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            visit(json.loads(line))
                        except ValueError:
                            continue
            except FileNotFoundError:
                continue
        now = time.time()
        with _asset_lock:
            protected_in_flight = set(_in_flight_assets.keys()) | _prune_seen_assets.pop(scan_id, set())
            for asset in assets.iterdir():
                if not asset.is_file():
                    continue
                rel = f"llm_assets/{asset.name}"
                if rel in referenced or rel in protected_in_flight:
                    continue
                if min_age_seconds > 0:
                    try:
                        age = now - asset.stat().st_mtime
                    except OSError:
                        continue
                    if age < min_age_seconds:
                        continue
                try:
                    asset.unlink()
                except FileNotFoundError:
                    continue
            _enforce_assets_quota_locked(
                assets,
                protected={Path(rel).name for rel in protected_in_flight},
            )
    except OSError:
        logging.getLogger("llm.trace.failure").exception("Unable to prune LLM assets")
    finally:
        with _asset_lock:
            _prune_seen_assets.pop(scan_id, None)


def _prune_worker() -> None:
    global _prune_thread
    while True:
        with _prune_lock:
            if not _prune_pending:
                _prune_thread = None
                return
            target_dir, grace = _prune_pending.popitem()
        try:
            prune_unused_assets(target_dir, min_age_seconds=grace)
        except Exception:
            logging.getLogger("llm.trace.failure").exception("Background LLM asset prune failed")


def schedule_prune_unused_assets(log_dir=None, *, min_age_seconds: float = 0.0) -> None:
    """Queue asset pruning on a coalesced low-priority daemon thread (never blocks emit)."""
    global _prune_thread
    target = Path(LOG_DIR if log_dir is None else log_dir)
    with _prune_lock:
        existing = _prune_pending.get(target)
        _prune_pending[target] = min_age_seconds if existing is None else min(existing, min_age_seconds)
        if _prune_thread is None or not _prune_thread.is_alive():
            _prune_thread = threading.Thread(
                target=_prune_worker,
                name="llm-trace-asset-pruner",
                daemon=True,
            )
            _prune_thread.start()


def wait_for_pending_prune(timeout: float = 5.0) -> bool:
    """Wait until any queued background asset prune finishes (mainly for tests)."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _prune_lock:
            thread = _prune_thread
            pending = bool(_prune_pending)
        if thread is None and not pending:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if thread is not None:
            thread.join(timeout=min(0.05, remaining))
        else:
            time.sleep(min(0.01, remaining))


class _TraceFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, *args, **kwargs):
        self._log_dir = Path(filename).parent
        self._rolled_over = False
        super().__init__(filename, *args, **kwargs)

    def doRollover(self):
        super().doRollover()
        self._rolled_over = True

    def emit(self, record):
        self._rolled_over = False
        super().emit(record)
        if self._rolled_over:
            self._rolled_over = False
            schedule_prune_unused_assets(self._log_dir, min_age_seconds=_ROLLOVER_PRUNE_GRACE_SECONDS)


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


def _safe_url(url, api_key, *, origin_only=False):
    try:
        parts = urlsplit(str(url))
        authority = _redact_text(parts.netloc.rsplit("@", 1)[-1], api_key)
        if origin_only:
            return urlunsplit((parts.scheme, authority, "", "", ""))
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


def _asset(data_url, *, _created_assets=None):
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
    limit = max_assets_bytes()
    metadata = {"mime": mime, "bytes": len(data), "sha256": digest}
    omitted = {"path": None, **metadata, "omitted": "asset_quota_exceeded"}
    if limit <= 0 or len(data) > limit:
        return omitted
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}.get(mime, "img")
    rel = Path("llm_assets") / f"{digest}.{ext}"
    rel_posix = rel.as_posix()
    path = Path(LOG_DIR) / rel
    with _asset_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        protected = {Path(p).name for p in _in_flight_assets} | {path.name}
        required = 0 if path.exists() else len(data)
        if not _enforce_assets_quota_locked(path.parent, admission_limit=limit - required, protected=protected):
            return omitted
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            try:
                os.utime(path, None)
            except OSError:
                pass
        except OSError:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        for seen_set in _prune_seen_assets.values():
            seen_set.add(rel_posix)
        if _created_assets is not None:
            _in_flight_assets[rel_posix] += 1
            _created_assets.append(rel_posix)
    return {"path": rel_posix, **metadata}


def _safe(value, api_key="", *, _created_assets=None):
    if isinstance(value, dict):
        if value.get("type") == "base64" and str(value.get("media_type", "")).startswith("image/"):
            try:
                asset = _asset(
                    f"data:{value['media_type']};base64,{value.get('data', '')}",
                    _created_assets=_created_assets,
                )
            except OSError:
                asset = None
            value = {**value, "data": asset if asset is not None else _IMAGE_OMITTED}
        return {
            _redact_text(str(key), api_key): (
                _REDACTED
                if str(key).lower() in _SECRET_KEYS
                else _safe(item, api_key, _created_assets=_created_assets)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(v, api_key, _created_assets=_created_assets) for v in value]
    if isinstance(value, str):
        try:
            asset = _asset(value, _created_assets=_created_assets)
        except OSError:
            asset = None
        if asset is not None:
            return asset
        if value[:11].lower() == "data:image/":
            return _IMAGE_OMITTED
        return _redact_text(value, api_key)
    return value


def _safe_metadata(value, api_key=""):
    """Redact secrets and strip any embedded image payloads without writing assets."""
    if isinstance(value, dict):
        if value.get("type") == "base64" and str(value.get("media_type", "")).startswith("image/"):
            return {"type": "base64", "media_type": str(value.get("media_type", "")), "data": _IMAGE_OMITTED}
        return {
            _redact_text(str(key), api_key): (
                _REDACTED if str(key).lower() in _SECRET_KEYS else _safe_metadata(item, api_key)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(v, api_key) for v in value]
    if isinstance(value, str):
        if _DATA_URL.fullmatch(value) and value[:11].lower() == "data:image/":
            return _IMAGE_OMITTED
        return _redact_text(value, api_key)
    return value


def _count_images(value) -> int:
    if isinstance(value, dict):
        if value.get("type") == "base64" and str(value.get("media_type", "")).startswith("image/"):
            return 1
        return sum(_count_images(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_images(item) for item in value)
    if isinstance(value, str):
        return 1 if _DATA_URL.fullmatch(value) and value[:11].lower() == "data:image/" else 0
    return 0


def _summarize_request_metadata(payload, api_key=""):
    summary = {"redacted": True, "image_count": _count_images(payload)}
    if not isinstance(payload, dict):
        return summary
    if isinstance(payload.get("model"), str):
        summary["model"] = _safe_metadata(payload["model"], api_key)
    if "stream" in payload:
        summary["stream"] = bool(payload.get("stream"))
    for key in ("temperature", "max_tokens", "max_completion_tokens", "max_output_tokens"):
        val = payload.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            summary[key] = val
    if isinstance(payload.get("messages"), list):
        summary["message_count"] = len(payload["messages"])
    elif isinstance(payload.get("input"), list):
        summary["message_count"] = len(payload["input"])
    return summary


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


@contextlib.contextmanager
def observe(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def traced(scene):
    """Scope a complete logical call, including generators consumed later."""
    def decorate(func):
        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def generator(*args, **kwargs):
                context = dict(kwargs.pop("trace_context", None) or {})
                context.setdefault("call_id", uuid4().hex)
                with scope(context.pop("scene", scene), **context):
                    yield from func(*args, **kwargs)
            return generator
        @functools.wraps(func)
        def normal(*args, **kwargs):
            context = dict(kwargs.pop("trace_context", None) or {})
            context.setdefault("call_id", uuid4().hex)
            with scope(context.pop("scene", scene), **context):
                return func(*args, **kwargs)
        return normal
    return decorate


def begin(url, api_type, payload, *, api_key="", context=None):
    mode = trace_mode()
    has_observer = _observer.get() is not None
    if mode == "off" and not has_observer:
        return _NullAttempt()
    try:
        return Attempt(
            url, api_type, payload, api_key=api_key, context=context, mode=mode, write_disk=(mode != "off")
        )
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
        if trace_mode() == "off" and _observer.get() is None:
            request.extensions["llm_trace"] = _NullAttempt()
            return
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
        if attempt is None or isinstance(attempt, _NullAttempt):
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
    def __init__(self, url, api_type, payload, *, api_key="", context=None, mode="full", write_disk=True):
        self.mode = mode
        self.write_disk = write_disk
        self.api_key = api_key
        self._observer = _observer.get()
        context = dict(context if context is not None else current_context())
        created_assets: list[str] = []
        try:
            self.model_id = _redact_text(str(payload.get("model", "")), api_key) if isinstance(payload, dict) else ""
            if not self.write_disk:
                self.call_id = _safe_metadata(context.pop("call_id", uuid4().hex), api_key)
                self.context = _safe_metadata(context, api_key)
                safe_request = None
            elif self.mode == "metadata":
                self.call_id = _safe_metadata(context.pop("call_id", uuid4().hex), api_key)
                self.context = {
                    key: _safe_metadata(value, api_key) for key, value in context.items()
                    if key in _METADATA_CONTEXT_KEYS and isinstance(value, (str, int))
                }
                safe_request = _summarize_request_metadata(payload, api_key)
            else:
                self.call_id = _safe(context.pop("call_id", uuid4().hex), api_key, _created_assets=created_assets)
                self.context = _safe(context, api_key, _created_assets=created_assets)
                safe_request = _safe(payload, api_key, _created_assets=created_assets)
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
            self._output_chars = 0
            self._reasoning_chars = 0
            self._stream_event_count = 0
            self.finish_reason = None
            self._finished = False
            if self._observer:
                try:
                    self._observer({"event": "request", "attempt_id": self.attempt_id, "request_at": self.request_at})
                except Exception:
                    logging.getLogger("llm.trace.failure").exception("Archive observer failed")
            if self.write_disk:
                record = {
                    "schema_version": 1,
                    "event": "request",
                    "call_id": self.call_id,
                    "attempt_id": self.attempt_id,
                    "context": self.context,
                    "at": self.request_at,
                    "protocol": api_type,
                    "url": _safe_url(url, api_key, origin_only=self.mode == "metadata"),
                    "request": safe_request,
                }
                if self.mode == "metadata":
                    record["trace_mode"] = "metadata"
                _write(record)
        finally:
            if created_assets:
                with _asset_lock:
                    for rel_posix in created_assets:
                        _in_flight_assets[rel_posix] -= 1
                        if _in_flight_assets[rel_posix] <= 0:
                            del _in_flight_assets[rel_posix]

    def headers(self, status_code):
        self.status_code = status_code
        self.response_headers_at = _now()
        self.headers_latency_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def _record_output_parts(self, out_parts, reason_parts):
        if self.mode == "metadata":
            self._output_chars += sum(len(p) for p in out_parts)
            self._reasoning_chars += sum(len(p) for p in reason_parts)
        if self.mode != "metadata" or self._observer is not None:
            self.output_text.extend(out_parts)
            self.reasoning.extend(reason_parts)

    def _has_output(self) -> bool:
        if self.mode == "metadata":
            return self._output_chars > 0 or self._reasoning_chars > 0
        return any(self.output_text) or any(self.reasoning)

    def response(self, body):
        if self.mode == "full" and self.write_disk:
            self.body = _safe(body, self.api_key)
        if isinstance(body, dict):
            self._usage(body.get("usage"))
            out_parts = []
            reason_parts = []
            if self.api_type == "anthropic":
                out_parts = [_text(b.get("text")) for b in body.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
                reason_parts = [_text(b.get("thinking")) for b in body.get("content", []) if isinstance(b, dict) and b.get("type") == "thinking"]
                self.finish_reason = body.get("stop_reason")
            elif self.api_type == "responses":
                out_parts = [_text(part.get("text")) for item in body.get("output", []) if isinstance(item, dict)
                             for part in item.get("content", []) if isinstance(part, dict) and part.get("type") == "output_text"]
                reason_parts = [_text(part.get("text")) for item in body.get("output", [])
                                if isinstance(item, dict) and item.get("type") == "reasoning"
                                for part in item.get("summary", []) if isinstance(part, dict)]
                self.finish_reason = body.get("status")
            else:
                choices = body.get("choices") or []
                for choice in choices:
                    if isinstance(choice, dict):
                        message = choice.get("message") or {}
                        if isinstance(message, dict):
                            out_parts.extend(_text_parts(message.get("content")))
                            reason_parts.extend(_text_parts(message.get("reasoning_content") or message.get("reasoning")))
                        self.finish_reason = choice.get("finish_reason") or self.finish_reason
            if self.mode == "metadata":
                self._output_chars = sum(len(p) for p in out_parts)
                self._reasoning_chars = sum(len(p) for p in reason_parts)
            if self.mode != "metadata" or self._observer is not None:
                self.output_text = out_parts
                self.reasoning = reason_parts
        if self._has_output() and self.first_output_at is None:
            self.first_output_at = _now()
            self.first_output_latency_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def _usage(self, usage):
        if isinstance(usage, dict):
            if self.usage is None:
                self.usage = {}
            for key, value in usage.items():
                if self.mode == "metadata":
                    if key in _TOKEN_DETAIL_KEYS and isinstance(value, dict):
                        detail = _TOKEN_DETAIL_KEYS[key]
                        if type(value.get(detail)) is int:
                            self.usage[key] = {detail: value[detail]}
                    elif key in _TOKEN_COUNT_KEYS and type(value) is int:
                        self.usage[key] = value
                    continue
                if value is not None:
                    self.usage[key] = value

    def stream_event(self, obj):
        self._stream_event_count += 1
        if self.mode == "full" and self.write_disk:
            self.events.append({"at": _now(), "data": _safe(obj, self.api_key)})
        if not isinstance(obj, dict):
            return
        event = obj.get("type")
        out_parts = []
        reason_parts = []
        if self.api_type == "anthropic":
            if event == "message_start":
                self._usage((obj.get("message") or {}).get("usage"))
            elif event == "message_delta":
                self._usage(obj.get("usage"))
                self.finish_reason = (obj.get("delta") or {}).get("stop_reason") or self.finish_reason
            elif event == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta":
                    out_parts.append(_text(delta.get("text")))
                elif delta.get("type") == "thinking_delta":
                    reason_parts.append(_text(delta.get("thinking")))
        elif self.api_type == "responses":
            if event == "response.output_text.delta":
                out_parts.append(_text(obj.get("delta")))
            elif event in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                reason_parts.append(_text(obj.get("delta")))
            elif event in ("response.completed", "response.incomplete", "response.failed"):
                response = obj.get("response") or {}
                self._usage(response.get("usage"))
                self.finish_reason = response.get("status") or event
        else:
            self._usage(obj.get("usage"))
            for choice in obj.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    out_parts.extend(_text_parts(delta.get("content")))
                    reason_parts.extend(_text_parts(delta.get("reasoning_content") or delta.get("reasoning")))
                self.finish_reason = choice.get("finish_reason") or self.finish_reason
        if out_parts or reason_parts:
            self._record_output_parts(out_parts, reason_parts)
        if self._has_output() and self.first_output_at is None:
            self.first_output_at = _now()
            self.first_output_latency_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def finish(self, status="completed", error=None):
        if self._finished:
            return
        self._finished = True
        usage = (
            self.usage if self.mode == "metadata"
            else (_safe(self.usage, self.api_key) if self.write_disk else _safe_metadata(self.usage, self.api_key))
        )
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
        finished_at = _now()
        duration_ms = round((time.monotonic() - self.t0) * 1000, 1)
        tokens_dict = {
            "input": input_total,
            "output": output,
            "total": usage.get("total_tokens") if usage else None,
            "reasoning": reasoning,
            "cached_read": read,
            "cached_write": created,
            "cache_hit": usage.get("prompt_cache_hit_tokens") if usage else None,
            "cache_miss": usage.get("prompt_cache_miss_tokens") if usage else None,
        }
        output_text = None
        reasoning_text = None
        safe_finish_reason = None
        safe_error = None
        if self.mode == "full" and self.write_disk:
            output_text, split_output_secret = _redact_joined_text(self.output_text, self.api_key)
            reasoning_text, split_reasoning_secret = _redact_joined_text(self.reasoning, self.api_key)
            # Individually safe fragments can still expose a joined credential.
            keep_raw_output = not (split_output_secret or split_reasoning_secret)
            safe_finish_reason = _safe(self.finish_reason, self.api_key)
            safe_error = _safe(str(error), self.api_key) if error is not None else None
            _write({
                "schema_version": 1,
                "event": "result",
                "call_id": self.call_id,
                "attempt_id": self.attempt_id,
                "context": self.context,
                "protocol": self.api_type,
                "status": status,
                "http_status": self.status_code,
                "request_at": self.request_at,
                "response_headers_at": self.response_headers_at,
                "first_output_at": self.first_output_at,
                "headers_latency_ms": self.headers_latency_ms,
                "first_output_latency_ms": self.first_output_latency_ms,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "output_text": output_text,
                "reasoning_text": reasoning_text,
                "finish_reason": safe_finish_reason,
                "response": self.body if keep_raw_output else None,
                "stream_events": self.events or None if keep_raw_output else None,
                "usage_raw": usage,
                "tokens": tokens_dict,
                "error": safe_error,
            })
        elif self.write_disk:
            _write({
                "schema_version": 1,
                "event": "result",
                "trace_mode": "metadata",
                "call_id": self.call_id,
                "attempt_id": self.attempt_id,
                "context": self.context,
                "protocol": self.api_type,
                "status": status,
                "http_status": self.status_code,
                "request_at": self.request_at,
                "response_headers_at": self.response_headers_at,
                "first_output_at": self.first_output_at,
                "headers_latency_ms": self.headers_latency_ms,
                "first_output_latency_ms": self.first_output_latency_ms,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "output_text": None,
                "reasoning_text": None,
                "output_chars": self._output_chars,
                "reasoning_chars": self._reasoning_chars,
                "finish_reason": (self.finish_reason if isinstance(self.finish_reason, str)
                                  and self.finish_reason in _FINISH_REASONS else None),
                "response": None,
                "stream_events": None,
                "stream_event_count": self._stream_event_count,
                "usage_raw": usage,
                "tokens": tokens_dict,
                "error": None,
            })
        if self._observer:
            if output_text is None and (self.mode != "full" or not self.write_disk):
                output_text, _ = _redact_joined_text(self.output_text, self.api_key)
                reasoning_text, _ = _redact_joined_text(self.reasoning, self.api_key)
                safe_finish_reason = _safe_metadata(self.finish_reason, self.api_key)
                safe_error = _safe_metadata(str(error), self.api_key) if error is not None else None
            try:
                self._observer({
                    "schema_version": 1,
                    "event": "result",
                    "call_id": self.call_id,
                    "attempt_id": self.attempt_id,
                    "protocol": self.api_type,
                    "status": status,
                    "http_status": self.status_code,
                    "request_at": self.request_at,
                    "response_headers_at": self.response_headers_at,
                    "first_output_at": self.first_output_at,
                    "headers_latency_ms": self.headers_latency_ms,
                    "first_output_latency_ms": self.first_output_latency_ms,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "output_text": output_text,
                    "reasoning_text": reasoning_text,
                    "finish_reason": safe_finish_reason,
                    "usage_raw": usage,
                    "tokens": tokens_dict,
                    "error": safe_error,
                    "model": self.model_id,
                })
            except Exception:
                logging.getLogger("llm.trace.failure").exception("Archive observer failed")
