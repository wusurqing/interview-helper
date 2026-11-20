"""Discover model IDs without inference, and validate a chosen connection on request."""

import hashlib
import hmac
import json
import re
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic
from types import SimpleNamespace
from typing import Any, Literal, TypeVar
from urllib.parse import urlsplit

import requests
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, ValidationError

from core.config import get_config, get_model_providers, model_health_fingerprint
from services.llm import trace
from services.llm.protocols import (
    API_TYPE_ANTHROPIC,
    API_TYPE_RESPONSES,
    build_chat_payload,
    build_llm_headers,
    chat_url_candidates,
    get_api_type,
)
from services.llm.streaming import _completion_token_kwargs

router = APIRouter(prefix="/model-providers")

_NO_KEY = ("", "__IA_KEEP_EXISTING_API_KEY__", "sk-your-api-key-here")
_VERIFICATION_KEY = secrets.token_bytes(32)
_VERIFICATION_SLOTS = threading.BoundedSemaphore(4)
_VERIFICATION_LOCK = threading.Lock()
_VERIFICATION_RESULTS: OrderedDict[bytes, tuple[object, bool]] = OrderedDict()
_VERIFICATION_CACHE_LIMIT = 4096
_CATALOG_PAGE_LIMIT = 100
_CATALOG_COMPAT_SUFFIXES = (
    "/api/claudecode", "/api/anthropic", "/apps/anthropic", "/api/coding",
    "/claudecode", "/anthropic", "/step_plan", "/coding", "/claude",
)


@dataclass(frozen=True)
class Connection:
    api_type: str
    api_base_url: str
    api_key: str = field(repr=False)


class ConnectionRequest(BaseModel):
    provider_id: str | None = None
    api_type: Literal["openai", "responses", "anthropic"] = "openai"
    api_base_url: str
    api_key: str | None = Field(default=None, repr=False)


class VerificationRequest(ConnectionRequest):
    model: str


_ModelRequest = TypeVar("_ModelRequest", bound=BaseModel)


def validate_model_request(body: Any, schema: type[_ModelRequest]) -> _ModelRequest:
    # Validation details can contain the whole submitted object, including credentials.
    try:
        return schema.model_validate(body)
    except ValidationError:
        raise HTTPException(422, "供应商或模型请求格式无效") from None


class ProviderRequestError(RuntimeError):
    """A stable, user-facing failure that never contains raw upstream data."""


def _connection(provider) -> Connection:
    return Connection(
        get_api_type(provider),
        provider.api_base_url.strip().rstrip("/"),
        provider.api_key.strip(),
    )


def _valid_base_url(base_url: str) -> bool:
    try:
        url = urlsplit(base_url)
        _ = url.port
    except ValueError:
        return False
    return (
        url.scheme in ("http", "https")
        and bool(url.hostname)
        and url.username is None
        and url.password is None
        and "?" not in base_url
        and "#" not in base_url
    )


def _resolve_connection(body: ConnectionRequest) -> Connection:
    base_url = body.api_base_url.strip().rstrip("/")
    if not _valid_base_url(base_url):
        raise HTTPException(422, "请输入有效的 HTTP(S) API Base URL")

    key = body.api_key
    if "api_key" not in body.model_fields_set and body.provider_id:
        provider = next(
            (item for item in get_model_providers(get_config()) if item.id == body.provider_id),
            None,
        )
        if provider is None:
            raise HTTPException(409, "供应商配置已更改，请刷新后重试")
        saved = _connection(provider)
        if (base_url, body.api_type) != (saved.api_base_url, saved.api_type):
            raise HTTPException(409, "接口地址或类型已变更，请填写对应的 API Key")
        key = saved.api_key

    if not isinstance(key, str) or key.strip() in _NO_KEY:
        raise HTTPException(422, "请填写 API Key")
    return Connection(body.api_type, base_url, key.strip())


def _status_error(status: int) -> ProviderRequestError:
    if status in (401, 403):
        detail = "供应商鉴权失败或当前 API Key 无访问权限"
    elif status == 402:
        detail = "供应商账户余额或额度不足"
    elif status == 429:
        detail = "供应商限流或额度不足，请稍后重试"
    elif status in (404, 405):
        detail = "供应商未找到该模型或接口"
    elif status in (400, 422):
        detail = "供应商拒绝请求，请检查接口类型或模型是否支持文本生成"
    elif 300 <= status < 400:
        detail = "供应商返回了重定向，请填写实际 API Base URL"
    else:
        detail = "供应商服务返回错误"
    return ProviderRequestError(f"{detail}（HTTP {status}）")


def _read_json(response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderRequestError("供应商返回的内容不是 JSON") from exc


def _authorization_headers(connection: Connection) -> list[dict[str, str]]:
    headers = [build_llm_headers(connection.api_type, connection.api_key)]
    if connection.api_type == API_TYPE_ANTHROPIC:
        headers.append(build_llm_headers(connection.api_type, connection.api_key, bearer=True))
    return headers


def _request(method, url: str, *, timeout: int = 12, **kwargs):
    try:
        return method(url, allow_redirects=False, timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        raise ProviderRequestError("连接供应商超时，请稍后重试") from exc
    except requests.RequestException as exc:
        raise ProviderRequestError("无法连接供应商，请检查网络和 API Base URL") from exc


def _model_id(item: Any) -> str | None:
    value = item.get("id") if isinstance(item, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> set[str]:
    return {item.strip().lower() for item in value if isinstance(item, str) and item.strip()} if isinstance(value, list) else set()


def _bool_hint(*values: Any) -> bool | None:
    return next((value for value in values if isinstance(value, bool)), None)


def _catalog_entry(item: dict, model_id: str) -> dict:
    result = {"id": model_id}
    for key, candidates in (
        ("name", (item.get("display_name"), item.get("name"))),
        ("owned_by", (item.get("owned_by"), item.get("ownedBy"))),
    ):
        for value in candidates:
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()
                break

    modalities = _object(item.get("modalities"))
    input_modes = _strings(modalities.get("input_modalities")) or _strings(item.get("input_modalities"))
    output_modes = _strings(modalities.get("output_modalities")) or _strings(item.get("output_modalities"))
    capabilities = _object(item.get("capabilities"))
    vision = _bool_hint(_object(capabilities.get("image_input")).get("supported"), item.get("supports_vision"))
    if vision is None and input_modes:
        vision = "image" in input_modes
    if vision is not None:
        result["supports_vision"] = vision

    think_type = _object(_object(item.get("features")).get("thinking")).get("type")
    think = _bool_hint(_object(capabilities.get("thinking")).get("supported"), item.get("supports_think"))
    if think is None and think_type in ("enabled", "disabled"):
        think = think_type == "enabled"
    if think is not None:
        result["supports_think"] = think

    for size in (item.get("context_window"), _object(item.get("token_limits")).get("context_window")):
        if type(size) is int and 0 < size <= 2**53 - 1:
            result["context_window_tokens"] = size
            break

    domain = item.get("domain")
    if (
        isinstance(domain, str) and domain.lower() in ("imagegeneration", "videogeneration", "3dgeneration", "embedding")
        or output_modes and "text" not in output_modes
    ):
        result["incompatible_reason"] = "目录显示该模型不支持文本生成"
    return result


def _catalog_page(response) -> tuple[list[tuple[str, dict]], bool, str | None]:
    body = _read_json(response)
    if isinstance(body, dict):
        if body.get("error"):
            raise ProviderRequestError("供应商返回模型目录错误")
        rows = body.get("data", body.get("models"))
        has_more = body.get("has_more") is True
        cursor = body.get("last_id")
    else:
        rows, has_more, cursor = body, False, None
    if not isinstance(rows, list):
        raise ProviderRequestError("供应商未返回有效的模型目录")
    entries = [(model_id, item) for item in rows if (model_id := _model_id(item)) is not None]
    if rows and not entries:
        raise ProviderRequestError("供应商模型目录缺少有效的 Model ID")
    return entries, has_more, cursor if isinstance(cursor, str) and cursor.strip() else None


def _catalog_url_candidates(base: str, api_type: str) -> list[str]:
    if api_type == API_TYPE_ANTHROPIC:
        paths = [f"{base}/models"] if base.lower().endswith("/v1") else [f"{base}/v1/models", f"{base}/models"]
    elif re.search(r"/v\d+[a-z]*$", base, flags=re.I):
        paths = [f"{base}/models"]
        if not base.lower().endswith("/v1"):
            paths.append(f"{base}/v1/models")
    else:
        paths = [f"{base}/v1/models", f"{base}/models"]

    for suffix in _CATALOG_COMPAT_SUFFIXES:
        if base.endswith(suffix):
            root = base[:-len(suffix)].rstrip("/")
            if _valid_base_url(root):
                paths.extend((f"{root}/v1/models", f"{root}/models"))
            break
    return list(dict.fromkeys(paths))


def fetch_provider_catalog(connection: Connection) -> list[dict]:
    if not _valid_base_url(connection.api_base_url):
        raise ProviderRequestError("API Base URL 无效")
    if connection.api_key in _NO_KEY:
        raise ProviderRequestError("未配置 API Key")

    for url in _catalog_url_candidates(connection.api_base_url, connection.api_type):
        variants = _authorization_headers(connection)
        for index, headers in enumerate(variants):
            response = _request(requests.get, url, headers=headers, timeout=15)
            if response.status_code == 401 and index + 1 < len(variants):
                continue
            break
        if response.status_code in (404, 405):
            continue
        if not 200 <= response.status_code < 300:
            raise _status_error(response.status_code)

        entries: dict[str, dict] = {}
        sent_cursors: set[str] = set()
        for _ in range(_CATALOG_PAGE_LIMIT):
            rows, has_more, cursor = _catalog_page(response)
            for model_id, item in rows:
                entry = entries.setdefault(model_id, {})
                for field_name, value in _catalog_entry(item, model_id).items():
                    entry.setdefault(field_name, value)
            if not has_more:
                return sorted(entries.values(), key=lambda entry: entry["id"].casefold())
            if not cursor or cursor in sent_cursors:
                break
            sent_cursors.add(cursor)
            response = _request(requests.get, url, headers=headers, timeout=15, params={"after_id": cursor})
            if not 200 <= response.status_code < 300:
                raise _status_error(response.status_code)
        raise ProviderRequestError("供应商模型目录分页未完成")
    raise ProviderRequestError("供应商未提供模型目录接口")


def check_catalog_models(indices: list[int] | None = None) -> dict:
    """Check each configured Model ID once using one catalog per connection."""
    models = get_config().models
    targets = range(len(models)) if indices is None else dict.fromkeys(indices)
    groups: dict[Connection, list[int]] = {}
    snapshots: dict[int, tuple[str | None, str]] = {}
    for index in targets:
        if not 0 <= index < len(models) or not models[index].enabled:
            continue
        model = models[index]
        groups.setdefault(_connection(model), []).append(index)
        snapshots[index] = (getattr(model, "id", None), model_health_fingerprint(model))

    states: dict[int, str] = {}
    details: dict[int, str] = {}
    for connection, model_indices in groups.items():
        try:
            model_ids = {entry["id"] for entry in fetch_provider_catalog(connection)}
        except ProviderRequestError as exc:
            for index in model_indices:
                states[index] = "unverified"
                details[index] = str(exc)
        else:
            for index in model_indices:
                visible = models[index].model.strip() in model_ids
                states[index] = "visible" if visible else "not_listed"
                details[index] = "" if visible else "供应商目录未列出该 Model ID"

    latest = get_config().models
    fingerprints: dict[int, str] = {}
    for index, (model_id, fingerprint) in snapshots.items():
        if (
            0 <= index < len(latest)
            and latest[index].enabled
            and getattr(latest[index], "id", None) == model_id
            and model_health_fingerprint(latest[index]) == fingerprint
        ):
            fingerprints[index] = fingerprint
        else:
            states.pop(index, None)
            details.pop(index, None)
    return {"catalog": states, "detail": details, "fingerprint": fingerprints}


def _proof_key(connection: Connection, model_id: str) -> bytes:
    serialized = json.dumps(
        (connection.api_type, connection.api_base_url, connection.api_key, model_id.strip()),
        ensure_ascii=True,
    )
    return hmac.new(_VERIFICATION_KEY, serialized.encode(), hashlib.sha256).digest()


def was_verified(provider, model_id: str) -> bool:
    key = _proof_key(_connection(provider), model_id)
    with _VERIFICATION_LOCK:
        result = _VERIFICATION_RESULTS.get(key)
        if result and result[1]:
            _VERIFICATION_RESULTS.move_to_end(key)
            return True
        return False


def _begin_verification(key: bytes) -> object:
    attempt = object()
    with _VERIFICATION_LOCK:
        _VERIFICATION_RESULTS[key] = (attempt, False)
        _VERIFICATION_RESULTS.move_to_end(key)
        if len(_VERIFICATION_RESULTS) > _VERIFICATION_CACHE_LIMIT:
            _VERIFICATION_RESULTS.popitem(last=False)
    return attempt


def _finish_verification(key: bytes, attempt: object, passed: bool) -> bool:
    with _VERIFICATION_LOCK:
        current = _VERIFICATION_RESULTS.get(key)
        if current is None or current[0] is not attempt:
            return False
        if passed:
            _VERIFICATION_RESULTS[key] = (attempt, True)
        else:
            _VERIFICATION_RESULTS.pop(key, None)
        return True


def _verification_is_current(key: bytes, attempt: object) -> bool:
    with _VERIFICATION_LOCK:
        current = _VERIFICATION_RESULTS.get(key)
        return current is not None and current[0] is attempt


def _verification_thinking(connection: Connection, model_id: str) -> dict:
    host = urlsplit(connection.api_base_url).hostname or ""
    if host == "api.deepseek.com":
        # Official DeepSeek enables thinking by default.
        if connection.api_type == API_TYPE_RESPONSES:
            return {"reasoning": {"effort": "none"}}
        return {"thinking": {"type": "disabled"}}
    if (
        host.startswith("ark.")
        and host.endswith(".volces.com")
        and model_id.lower().startswith(("kimi-k3", "doubao-seed-2", "deepseek-v4"))
    ):
        if connection.api_type == API_TYPE_RESPONSES:
            return {"reasoning": {"effort": "minimal"}}
        return {"thinking": {"type": "disabled"}}
    return {}


def _verification_payload(connection: Connection, model_id: str) -> dict:
    messages = [{"role": "user", "content": "Reply OK."}]
    thinking = _verification_thinking(connection, model_id)
    if connection.api_type in (API_TYPE_ANTHROPIC, API_TYPE_RESPONSES):
        payload = build_chat_payload(
            connection.api_type, model_id, messages, max_tokens=16, think_params=thinking,
        )
        payload.pop("temperature", None)
        return payload
    model = SimpleNamespace(api_type=connection.api_type, model=model_id)
    return {
        "model": model_id,
        "messages": messages,
        "stream": False,
        **_completion_token_kwargs(model, 16),
        **thinking,
    }


def _text_in_parts(parts: Any, part_type: str) -> bool:
    return isinstance(parts, list) and any(
        isinstance(part, dict)
        and part.get("type") == part_type
        and isinstance(part.get("text"), str)
        and part["text"].strip()
        for part in parts
    )


def _has_text_reply(body: Any, api_type: str) -> bool:
    if not isinstance(body, dict) or body.get("error"):
        return False
    if api_type == API_TYPE_ANTHROPIC:
        return _text_in_parts(body.get("content"), "text")
    if api_type == API_TYPE_RESPONSES:
        outputs = body.get("output")
        return (
            body.get("status") not in ("failed", "cancelled", "queued", "in_progress")
            and isinstance(outputs, list)
            and any(
                isinstance(item, dict) and _text_in_parts(item.get("content"), "output_text")
                for item in outputs
            )
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    content = _object(_object(choices[0]).get("message")).get("content")
    return bool(content.strip()) if isinstance(content, str) else _text_in_parts(content, "text")


def _missing_inference_route(status: int, body: Any) -> bool:
    if status == 405:
        return True
    if status != 404:
        return False
    body = _object(body)
    error = _object(body.get("error"))
    code = str(error.get("code", "")).lower()
    message = str(error.get("message", body.get("detail", ""))).lower()
    return not (
        "model" in code
        or "endpoint" in code and "invalid" in code
        or re.search(r"\bmodel\b.*(?:not found|does not exist|not available)", message)
    )


def _post_verification(connection: Connection, url: str, headers: dict, payload: dict) -> tuple[int, Any, bool]:
    attempt = trace.begin(url, connection.api_type, payload, api_key=connection.api_key)
    try:
        response = _request(requests.post, url, headers=headers, json=payload)
    except ProviderRequestError as exc:
        attempt.finish("failed", exc)
        raise

    status = response.status_code
    attempt.headers(status)
    try:
        body = response.json()
    except ValueError:
        body = None
    passed = 200 <= status < 300 and _has_text_reply(body, connection.api_type)
    # A malformed upstream usage/content field must not change the probe outcome.
    try:
        attempt.response(body)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        attempt.finish("completed" if passed else "failed")
    except (AttributeError, TypeError, ValueError):
        pass
    return status, body, passed


@trace.traced("model_verification")
def _run_verification(connection: Connection, model_id: str) -> None:
    payload = _verification_payload(connection, model_id)
    for url in chat_url_candidates(connection.api_base_url, connection.api_type):
        variants = _authorization_headers(connection)
        for index, headers in enumerate(variants):
            status, body, passed = _post_verification(connection, url, headers, payload)
            if status == 401 and index + 1 < len(variants):
                continue
            break
        if _missing_inference_route(status, body):
            continue
        if not 200 <= status < 300:
            raise _status_error(status)
        if not passed:
            raise ProviderRequestError("模型没有返回有效文本，请检查它是否支持文本生成")
        return
    raise ProviderRequestError("未找到供应商的文本生成接口")


@router.post("/catalog")
def api_provider_catalog(body: Any = Body()) -> dict:
    return provider_catalog(validate_model_request(body, ConnectionRequest))


def provider_catalog(body: ConnectionRequest) -> dict:
    connection = _resolve_connection(body)
    try:
        return {"models": fetch_provider_catalog(connection)}
    except ProviderRequestError as exc:
        raise HTTPException(502, str(exc)) from None


@router.post("/verify")
def api_provider_verify_model(body: Any = Body()) -> dict:
    return provider_verify_model(validate_model_request(body, VerificationRequest))


def provider_verify_model(body: VerificationRequest) -> dict:
    connection = _resolve_connection(body)
    model_id = body.model.strip()
    if not model_id:
        raise HTTPException(422, "请填写 Model ID")

    key = _proof_key(connection, model_id)
    attempt = _begin_verification(key)
    passed = False
    detail = ""
    with _VERIFICATION_SLOTS:
        if not _verification_is_current(key, attempt):
            return {"ok": False, "detail": "验证结果已被新的请求替代，请重试", "latency_ms": 0}
        started = monotonic()
        try:
            _run_verification(connection, model_id)
            passed = True
        except ProviderRequestError as exc:
            detail = str(exc)
        finally:
            current = _finish_verification(key, attempt, passed)
        latency_ms = int((monotonic() - started) * 1000)
    if not current:
        return {"ok": False, "detail": "验证结果已被新的请求替代，请重试", "latency_ms": latency_ms}
    return {"ok": passed, "detail": detail, "latency_ms": latency_ms}
