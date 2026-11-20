import json
import socket
import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Body, UploadFile, File, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, StrictInt, StrictStr, ValidationError

from core.auth import get_token, is_auth_disabled, is_loopback_host, loopback_bypass_allowed
from core.config import (
    CANDIDATE_STT_PROVIDER_OPTIONS, DEFAULT_CONTEXT_WINDOW_TOKENS, MMPG_SECRET_FIELDS, MMPG_STT_PROVIDER, Mmpg,
    STT_SECRET_FIELDS, ConfigCorruptError,
    get_config, get_model_providers, model_health_fingerprint, provider_url, update_config, ModelListChanged,
    AssistGeneration, AssistScene, InvalidAssistModelSelection, ThinkEffort,
    POSITION_OPTIONS, LANGUAGE_OPTIONS, STT_PROVIDER_OPTIONS,
    SCREEN_CAPTURE_REGION_OPTIONS,
    normalize_llm_max_tokens, normalize_llm_temperature,
)
from core.env import env_int
from core.prompt_settings import PROMPT_SCENES
from services.audio import AudioCapture
from services.stt import STTError, get_stt_engine, stt_configuration_error
from api.common.config_payload import build_config_payload
from api.common.model_health import (
    get_model_health_snapshot,
    probe_single_model,
)
from api.common.provider_catalog import (
    Connection,
    ProviderRequestError,
    check_catalog_models,
    fetch_provider_catalog,
    router as catalog_router,
    validate_model_request,
)
from services.llm.protocols import get_api_type
from api.common.provider_settings import router as provider_settings_router
from services.storage.resume_history import (
    MAX_UPLOAD_BYTES as RESUME_UPLOAD_MAX_BYTES,
    add_upload,
    apply_entry,
    delete_entry,
    get_entry_detail,
    list_entries,
    update_entry_summary,
)

router = APIRouter()
router.include_router(catalog_router)
router.include_router(provider_settings_router)

# 简历上传阈值从 services.storage.resume_history 引入 (单一来源 / DRY)。
# Router 层做流式校验是为了在拿到 Content-Length 或读到超限时立即拒绝,
# 避免恶意客户端发送 GB 级 body 把整个文件读进内存触发 OOM.
_RESUME_UPLOAD_CHUNK = 1 * 1024 * 1024

def _has_enabled_model(models: list) -> bool:
    return any(bool(getattr(model, "enabled", True)) for model in models)


class ConfigUpdate(BaseModel):
    system_prompts: Optional[dict[str, StrictStr | None]] = None
    models: Optional[list] = None
    assist_model_ids: list[StrictStr] | None = None
    assist_screenshot_model_id: StrictStr | None = None
    written_exam_model_id: StrictStr | None = None
    written_exam_model_ids: list[StrictStr] | None = None
    assist_generation: dict[AssistScene, AssistGeneration] | None = None
    assist_think_mode: bool | None = None
    assist_think_effort: ThinkEffort | None = None
    max_parallel_answers: Optional[int] = None
    active_model: Optional[int] = None
    active_model_id: StrictStr | None = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    think_mode: Optional[bool] = None
    think_effort: Optional[str] = None
    stt_provider: Optional[str] = None
    # Validated inside the handler so malformed credential fields never appear in FastAPI's error body.
    mmpg: Any = None
    doubao_stt_app_id: Optional[str] = None
    doubao_stt_access_token: Optional[str] = None
    doubao_stt_api_key: Optional[str] = None
    doubao_stt_resource_id: Optional[str] = None
    doubao_stt_boosting_table_id: Optional[str] = None
    position: Optional[str] = None
    language: Optional[str] = None
    auto_detect: Optional[bool] = None
    silence_threshold: Optional[float] = None
    silence_duration: Optional[float] = None
    answer_autoscroll_bottom_px: Optional[int] = None
    transcription_min_sig_chars: Optional[int] = None
    assist_transcription_merge_gap_sec: Optional[float] = None
    assist_transcription_merge_max_sec: Optional[float] = None
    assist_vad_max_speech_sec: Optional[float] = None
    assist_vad_min_speech_sec: Optional[float] = None
    assist_asr_confirm_window_sec: Optional[float] = None
    assist_asr_group_max_wait_sec: Optional[float] = None
    assist_asr_interrupt_running: Optional[bool] = None
    assist_high_churn_short_answer: Optional[bool] = None
    assist_realtime_max_tokens: Optional[int] = None
    assist_realtime_high_churn_max_tokens: Optional[int] = None
    assist_stop_answer_wait_sec: Optional[float] = None
    assist_interviewer_asr_drain_timeout_sec: Optional[float] = None
    screen_capture_region: Optional[str] = None
    screen_capture_max_long_edge: Optional[int] = None
    multi_screen_capture_idle_sec: Optional[float] = None
    written_exam_mode: Optional[bool] = None
    written_exam_think: Optional[bool] = None
    generic_stt_api_base_url: Optional[str] = None
    generic_stt_api_key: Optional[str] = None
    generic_stt_model: Optional[str] = None
    generic_stt_custom_headers: Optional[str] = None
    candidate_asr_enabled: Optional[bool] = None
    candidate_stt_provider: Optional[str] = None
    candidate_context_enabled: Optional[bool] = None
    candidate_context_wait_ms: Optional[int] = None
    candidate_context_max_chars: Optional[int] = None
    candidate_context_min_chars: Optional[int] = None
    candidate_mic_compatibility_mode: Optional[bool] = None
    # KB - 详细字段(min_score / OCR / Vision / chunk_size 等)仍走 config.json
    kb_enabled: Optional[bool] = None
    kb_top_k: Optional[int] = None
    kb_deadline_ms: Optional[int] = None
    kb_asr_deadline_ms: Optional[int] = None
    # Review (面试复盘)
    review_enabled: Optional[bool] = None
    review_model_index: Optional[int] = None
    review_model_id: StrictStr | None = None
    # 明确确认以当前配置覆盖已损坏的 config.json（只读保护解锁）
    confirm_overwrite_corrupt: Optional[bool] = None


_MODEL_API_KEY_KEEP = "__IA_KEEP_EXISTING_API_KEY__"


class ModelListRequest(BaseModel):
    api_base_url: str
    api_key: str
    provider_id: StrictStr | None = None
    model_index: Optional[int] = None
    model_id: StrictStr | None = None
    api_type: Optional[str] = None


def _list_remote_models(api_base_url: str, api_key: str, api_type: str = "openai") -> dict:
    entries = fetch_provider_catalog(Connection(api_type, provider_url(api_base_url), api_key.strip()))
    return {"models": [{"id": entry["id"], "owned_by": entry.get("owned_by")} for entry in entries]}



@router.get("/config")
async def api_get_config():
    return build_config_payload(get_config())


@router.get("/prompts")
async def api_get_prompts():
    from services.llm.prompts import assist_prompt_specs
    from services.resume_optimizer import resume_prompt_specs
    from services.review_analysis import review_prompt_specs

    specs = {
        **assist_prompt_specs(get_config()),
        **review_prompt_specs(),
        **resume_prompt_specs(),
    }
    scenes = []
    for scene, (group, label, entry) in PROMPT_SCENES.items():
        default_prompt, output_contract, max_output_tokens = specs[scene]
        scenes.append({
            "id": scene,
            "group": group,
            "label": label,
            "entry": entry,
            "default_prompt": default_prompt,
            "output_contract": output_contract,
            "max_output_tokens": max_output_tokens,
        })
    return {"scenes": scenes}


_MMPG_PROBE_CONFIG_PATH = Path(__file__).resolve().parents[3] / "tools" / "mmpg-probe" / "config.local.json"
_MMPG_PROBE_IMPORT_FIELDS = set(Mmpg.model_fields) - {
    "rtc_app_key", "token_mode", "answer_delay_ms", "followup_window_ms",
}
_MMPG_PROBE_CONFIG_LIMIT = 64 * 1024


def _require_local_mmpg_request(request: Request | None) -> None:
    if request is None or not is_loopback_host(request.url.hostname) or not loopback_bypass_allowed(
        request.client.host if request.client else None,
        request.headers.get("origin"),
        request.url.hostname,
        request.url.scheme,
        request.url.port,
    ):
        raise HTTPException(403, "请在运行后端的电脑上修改豆包 MMPG 设置")


def _mmpg_validation_error(error: ValidationError) -> HTTPException:
    fields = {
        str(item["loc"][-1])
        for item in error.errors(include_input=False, include_context=False, include_url=False)
        if item["loc"] and item["loc"][-1] in Mmpg.model_fields
    }
    detail = "豆包 MMPG 设置格式无效"
    if fields:
        detail += "（" + "、".join(sorted(fields)) + "）"
    return HTTPException(422, detail)


@router.get("/config/models-full")
async def api_get_models_full():
    """Return model fields for local frontend editing without echoing secrets."""
    cfg = get_config()
    return {
        "models": [
            {
                "id": getattr(mdl, "id", ""),
                "name": mdl.name,
                "api_base_url": mdl.api_base_url,
                "api_key": "",
                "model": mdl.model,
                "api_type": getattr(mdl, "api_type", "openai") or "openai",
                "context_window_tokens": getattr(mdl, "context_window_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS),
                "supports_think": mdl.supports_think,
                "supports_vision": mdl.supports_vision,
                "enabled": getattr(mdl, "enabled", True),
                "think_enabled_params": getattr(mdl, "think_enabled_params", {}) or {},
                "think_disabled_params": getattr(mdl, "think_disabled_params", {}) or {},
                "has_key": bool(mdl.api_key and mdl.api_key not in ("", "sk-your-api-key-here")),
            }
            for mdl in cfg.models
        ],
    }


@router.post("/config")
async def api_update_config_request(request: Request, body: Any = Body()):
    try:
        parsed = ConfigUpdate.model_validate(body)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        for error in errors:
            error["loc"] = ("body", *error["loc"])
        raise HTTPException(422, errors) from None
    return await api_update_config(parsed, request)


async def api_update_config(body: ConfigUpdate, request: Request = None):
    from core.config import ModelConfig

    d = body.model_dump(exclude_none=True)
    for field in (
        "active_model",
        "active_model_id",
        "review_model_index",
        "review_model_id",
        *STT_SECRET_FIELDS,
    ):
        if field in body.model_fields_set:
            d[field] = getattr(body, field)
    if "assist_screenshot_model_id" in body.model_fields_set:
        d["assist_screenshot_model_id"] = body.assist_screenshot_model_id
    if "written_exam_model_id" in body.model_fields_set:
        d["written_exam_model_id"] = body.written_exam_model_id
    model_snapshot = None
    model_origins = None
    try:
        if body.mmpg is not None:
            _require_local_mmpg_request(request)
            if not isinstance(body.mmpg, dict) or body.mmpg.keys() - Mmpg.model_fields.keys():
                raise HTTPException(422, "豆包 MMPG 设置格式无效或包含未知字段")
            # Keep explicit nulls for clearing credentials and optional time limits.
            d["mmpg"] = body.mmpg
        if "models" in d:
            raw_models = []
            current_models = list(get_config().models)
            model_snapshot = current_models
            model_origins = []
            by_id = {model.id: i for i, model in enumerate(current_models) if getattr(model, "id", "")}
            by_fingerprint: dict[str, list[int]] = {}
            for index, model in enumerate(current_models):
                by_fingerprint.setdefault(model_health_fingerprint(model), []).append(index)
            retained_origins: set[int] = set()
            for idx, x in enumerate(d["models"]):
                if not isinstance(x, dict):
                    continue
                model_id = x.pop("id", None)
                keep_key = x.get("api_key") == _MODEL_API_KEY_KEEP
                source_idx = x.pop("model_original_index", idx if keep_key else None)
                if model_id:
                    if not isinstance(model_id, str) or model_id not in by_id:
                        raise ModelListChanged("模型目录已更新，请刷新后重试")
                    source_idx = by_id[model_id]
                if model_id or keep_key:
                    if type(source_idx) is not int or not 0 <= source_idx < len(current_models):
                        raise ModelListChanged("模型目录已更新，请刷新后重试")
                row = ModelConfig(**x)
                if keep_key:
                    saved = current_models[source_idx]
                    if provider_url(row.api_base_url) != provider_url(saved.api_base_url) or row.api_type != get_api_type(saved):
                        raise ModelListChanged("接口地址或类型已变更，请填写对应的 API Key")
                    row.api_key = saved.api_key
                if not model_id and not keep_key:
                    # Older editors send row indices, which may refer to the
                    # previous ordering. Match unchanged rows by identity.
                    matches = by_fingerprint.get(model_health_fingerprint(row), [])
                    source_idx = matches[0] if len(matches) == 1 else None
                if source_idx is not None:
                    if source_idx in retained_origins:
                        raise HTTPException(422, "同一个模型不能在目录中重复保存")
                    retained_origins.add(source_idx)
                    row.id = getattr(current_models[source_idx], "id", "")
                model_origins.append(source_idx)
                raw_models.append(row)
            d["models"] = raw_models
            if not d["models"]:
                raise HTTPException(400, "至少保留一个模型")
            if not _has_enabled_model(d["models"]):
                raise HTTPException(422, "至少启用一个模型")
        for field, options in (
            ("stt_provider", STT_PROVIDER_OPTIONS),
            ("candidate_stt_provider", CANDIDATE_STT_PROVIDER_OPTIONS),
        ):
            if field in d and d[field] not in ("", *options):
                raise HTTPException(422, f"{field} 已废弃或不受支持，请选择 {options} 中的引擎")
        if "temperature" in d:
            d["temperature"] = normalize_llm_temperature(d["temperature"])
        if "max_tokens" in d:
            d["max_tokens"] = normalize_llm_max_tokens(d["max_tokens"])
        if "candidate_context_wait_ms" in d:
            d["candidate_context_wait_ms"] = max(0, min(2000, int(d["candidate_context_wait_ms"])))
        if "candidate_context_max_chars" in d:
            d["candidate_context_max_chars"] = max(100, min(4000, int(d["candidate_context_max_chars"])))
        if "candidate_context_min_chars" in d:
            d["candidate_context_min_chars"] = max(1, min(100, int(d["candidate_context_min_chars"])))
        if "max_parallel_answers" in d:
            d["max_parallel_answers"] = max(1, min(8, int(d["max_parallel_answers"])))
        if "answer_autoscroll_bottom_px" in d:
            d["answer_autoscroll_bottom_px"] = max(4, min(400, int(d["answer_autoscroll_bottom_px"])))
        if "transcription_min_sig_chars" in d:
            d["transcription_min_sig_chars"] = max(1, min(50, int(d["transcription_min_sig_chars"])))
        if "assist_transcription_merge_gap_sec" in d:
            d["assist_transcription_merge_gap_sec"] = max(
                0.0, min(15.0, float(d["assist_transcription_merge_gap_sec"]))
            )
        if "assist_transcription_merge_max_sec" in d:
            d["assist_transcription_merge_max_sec"] = max(
                1.0, min(120.0, float(d["assist_transcription_merge_max_sec"]))
            )
        if "assist_vad_max_speech_sec" in d:
            d["assist_vad_max_speech_sec"] = max(
                6.0, min(60.0, float(d["assist_vad_max_speech_sec"]))
            )
        if "assist_vad_min_speech_sec" in d:
            d["assist_vad_min_speech_sec"] = max(
                0.1, min(2.0, float(d["assist_vad_min_speech_sec"]))
            )
        if "assist_asr_confirm_window_sec" in d:
            d["assist_asr_confirm_window_sec"] = max(
                0.0, min(5.0, float(d["assist_asr_confirm_window_sec"]))
            )
        if "assist_asr_group_max_wait_sec" in d:
            d["assist_asr_group_max_wait_sec"] = max(
                0.2, min(8.0, float(d["assist_asr_group_max_wait_sec"]))
            )
        if "assist_realtime_max_tokens" in d:
            d["assist_realtime_max_tokens"] = max(
                256, min(4096, int(d["assist_realtime_max_tokens"]))
            )
        if "assist_realtime_high_churn_max_tokens" in d:
            realtime_cap = int(d.get("assist_realtime_max_tokens", get_config().assist_realtime_max_tokens))
            d["assist_realtime_high_churn_max_tokens"] = max(
                128,
                min(realtime_cap, int(d["assist_realtime_high_churn_max_tokens"])),
            )
        if "assist_stop_answer_wait_sec" in d:
            d["assist_stop_answer_wait_sec"] = max(
                0.0, min(20.0, float(d["assist_stop_answer_wait_sec"]))
            )
        if "assist_interviewer_asr_drain_timeout_sec" in d:
            d["assist_interviewer_asr_drain_timeout_sec"] = max(
                0.5, min(30.0, float(d["assist_interviewer_asr_drain_timeout_sec"]))
            )
        # 非法 enum 值必须明确报错: 静默 pop 会让前端以为保存成功,
        # 但实际配置没变, 用户看到的 UI 状态与后端不一致 (P0 #3 in CR)。
        # 例外: 空字符串视为「重置回默认」, pop 而不是 422, 兼容前端清空字段的语义。
        if d.get("screen_capture_region") == "":
            d.pop("screen_capture_region", None)
        elif "screen_capture_region" in d and d["screen_capture_region"] not in SCREEN_CAPTURE_REGION_OPTIONS:
            raise HTTPException(
                422,
                f"screen_capture_region 必须是 {list(SCREEN_CAPTURE_REGION_OPTIONS)} 之一",
            )
        if "multi_screen_capture_idle_sec" in d:
            d["multi_screen_capture_idle_sec"] = max(
                1.0, min(60.0, float(d["multi_screen_capture_idle_sec"]))
            )
        if "think_effort" in d:
            val = str(d["think_effort"]).strip().lower()
            if val not in ("off", "low", "medium", "high", "xhigh"):
                raise HTTPException(422, "think_effort 必须是 off/low/medium/high/xhigh 之一")
            d["think_effort"] = val
            if val == "off" and d.get("think_mode", True) is True:
                d["think_mode"] = False
            elif val != "off" and d.get("think_mode", False) is False:
                d["think_mode"] = True
        if d.get("language") and d["language"] not in LANGUAGE_OPTIONS:
            raise HTTPException(422, f"language 必须是 {list(LANGUAGE_OPTIONS)} 之一")
        if "kb_top_k" in d:
            d["kb_top_k"] = max(1, min(20, int(d["kb_top_k"])))
        if "kb_deadline_ms" in d:
            d["kb_deadline_ms"] = max(20, min(2000, int(d["kb_deadline_ms"])))
        if "kb_asr_deadline_ms" in d:
            d["kb_asr_deadline_ms"] = max(20, min(1000, int(d["kb_asr_deadline_ms"])))

        confirm_overwrite = bool(d.pop("confirm_overwrite_corrupt", False))
        save_options = {} if model_snapshot is None else {
            "model_origins": model_origins, "model_snapshot": model_snapshot,
        }
        if confirm_overwrite:
            save_options["allow_overwrite_corrupt"] = True
        if "written_exam_mode" in d:
            from api.assist.mmpg_integration import http_lifecycle_lock, manager
            from api.realtime.ws import broadcast
            from core.session import get_session

            async with http_lifecycle_lock:
                changing_workspace = d["written_exam_mode"] != get_config().written_exam_mode
                if changing_workspace and (get_session().is_recording or manager.active):
                    raise HTTPException(409, "请先结束当前会话，再切换答题模式。")
                await run_in_threadpool(update_config, d, **save_options)
                if changing_workspace:
                    broadcast({"type": "workspace_mode", "written_exam_mode": d["written_exam_mode"]})
        else:
            await run_in_threadpool(update_config, d, **save_options)
        if d.get("auto_detect") is False:
            from api.assist.mmpg_integration import disable_automatic_answers

            disable_automatic_answers()
    except HTTPException:
        raise
    except ConfigCorruptError as e:
        raise HTTPException(409, str(e)) from e
    except ModelListChanged as e:
        raise HTTPException(409, str(e)) from e
    except InvalidAssistModelSelection as e:
        raise HTTPException(422, str(e)) from e
    except OSError:
        raise HTTPException(500, "无法保存设置，请检查配置文件目录的写入权限") from None
    except ValidationError as e:
        if "mmpg" in d:
            raise _mmpg_validation_error(e) from None
        raise HTTPException(400, str(e)) from e
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    # 返回更新后的完整配置，而不是简单的 {ok: True}
    return build_config_payload(get_config())


@router.get("/network-info")
async def api_network_info(request: Request):
    """Return LAN IP and port so the frontend can render a scannable QR code.

    若 LAN 鉴权开启,环回访问会在 URL 中追加 ``?t=<token>``,扫码后手机端
    可自动写入 sessionStorage 并附在后续请求里。LAN 客户端拿到的是不带
    token 的 URL,自身就拒绝访问,从而避免凭证泄露。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    # Read PORT from env (set by start.py), fallback to 18080
    import os
    port = env_int("PORT", 18080, minimum=1)
    base = f"http://{ip}:{port}"
    client_host = request.client.host if request.client else None
    url = base
    auth_required = not is_auth_disabled()
    if auth_required and is_loopback_host(client_host):
        url = f"{base}/?t={get_token()}"
    return {
        "ip": ip,
        "port": port,
        "url": url,
        "auth_required": auth_required,
    }


@router.get("/options")
async def api_options():
    return {
        "positions": POSITION_OPTIONS,
        "languages": LANGUAGE_OPTIONS,
        "stt_providers": STT_PROVIDER_OPTIONS,
        "candidate_stt_providers": CANDIDATE_STT_PROVIDER_OPTIONS,
        "screen_capture_regions": SCREEN_CAPTURE_REGION_OPTIONS,
    }


@router.get("/devices")
async def api_devices():
    try:
        devices = AudioCapture.list_devices()
        platform_info = AudioCapture.get_platform_info()
        return {"devices": devices, "platform": platform_info}
    except Exception as e:
        return {"devices": [], "platform": {"platform": "unknown", "needs_virtual_device": False, "instructions": "", "error": str(e)}}


@router.post("/resume")
async def api_upload_resume(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "未选择文件")

    # 早期拦截: 若客户端声明的 Content-Length 已经超限, 直接 413, 不读 body.
    try:
        declared = int(request.headers.get("content-length") or "0")
    except ValueError:
        declared = 0
    if declared > RESUME_UPLOAD_MAX_BYTES:
        raise HTTPException(413, "文件大小不能超过 10MB")

    # 流式累积: chunked 编码或 Content-Length 缺失/伪造时, 累计超限就立即终止读取。
    # max_iters 是 robustness 兜底: 防御异常 file.read 实现 (例如返回 size=0 的非空 chunk)
    # 无限循环, 实际 starlette UploadFile 不会出现这种情况。
    max_iters = (RESUME_UPLOAD_MAX_BYTES // _RESUME_UPLOAD_CHUNK) + 2
    chunks: list[bytes] = []
    total = 0
    for _ in range(max_iters):
        chunk = await file.read(_RESUME_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > RESUME_UPLOAD_MAX_BYTES:
            raise HTTPException(413, "文件大小不能超过 10MB")
        chunks.append(chunk)
    else:
        # for-else: 跑满 max_iters 仍没退出, 视为读取异常
        raise HTTPException(500, "上传读取异常")
    content = b"".join(chunks)

    try:
        return await run_in_threadpool(add_upload, content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.delete("/resume")
async def api_delete_resume():
    await run_in_threadpool(
        update_config, {"resume_text": None, "resume_active_history_id": None}
    )
    return {"ok": True}


@router.get("/resume/history")
async def api_resume_history():
    return {"items": list_entries(), "max": 10}


class ResumeHistoryUpdateBody(BaseModel):
    summary: str = ""


@router.get("/resume/history/{entry_id}")
async def api_resume_history_detail(entry_id: int):
    try:
        return get_entry_detail(entry_id)
    except FileNotFoundError:
        raise HTTPException(404, "记录不存在")


@router.put("/resume/history/{entry_id}")
async def api_resume_history_update(entry_id: int, body: ResumeHistoryUpdateBody):
    try:
        return update_entry_summary(entry_id, body.summary)
    except FileNotFoundError:
        raise HTTPException(404, "记录不存在")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/resume/history/{entry_id}/apply")
async def api_resume_history_apply(entry_id: int):
    """在线程池执行，避免大 PDF 解析阻塞事件循环导致其它请求（含预览）卡死。"""
    try:
        return await run_in_threadpool(apply_entry, entry_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/resume/history/{entry_id}")
async def api_resume_history_delete(entry_id: int):
    delete_entry(entry_id)
    return {"ok": True}


@router.get("/stt/status")
async def api_stt_status():
    from api.assist import mmpg_integration

    cfg = mmpg_integration.pipeline.get_capture_config()
    provider = cfg.stt_provider
    if provider == MMPG_STT_PROVIDER:
        from services.stt.mmpg import MmpgError, validate_config

        try:
            validate_config(cfg.mmpg)
            configured, missing = True, []
        except MmpgError as e:
            configured, missing = False, e.fields
        return {**mmpg_integration.stt_status(), "configured": configured, "missing": missing}
    detail = stt_configuration_error(cfg, provider)
    return {"loaded": not detail, "loading": False, "model": provider, "provider": provider, "detail": detail}


def _import_mmpg_probe_config() -> None:
    try:
        with _MMPG_PROBE_CONFIG_PATH.open("rb") as probe_file:
            content = probe_file.read(_MMPG_PROBE_CONFIG_LIMIT + 1)
        if len(content) > _MMPG_PROBE_CONFIG_LIMIT:
            raise ValueError
        probe = json.loads(content.decode("utf-8-sig"))
        if not isinstance(probe, dict):
            raise ValueError
    except FileNotFoundError:
        raise HTTPException(404, "本机没有找到 tools/mmpg-probe/config.local.json") from None
    except (OSError, ValueError):
        raise HTTPException(422, "无法读取探针配置，请检查 config.local.json 格式") from None
    patch = {
        field: value
        for field, value in probe.items()
        if field in _MMPG_PROBE_IMPORT_FIELDS and (value is not None or field not in MMPG_SECRET_FIELDS)
    }
    if not patch:
        raise HTTPException(422, "探针配置没有可导入的豆包 MMPG 设置")
    update_config({"mmpg": patch})


@router.post("/stt/mmpg/import-probe")
async def api_import_mmpg_probe(request: Request):
    _require_local_mmpg_request(request)
    try:
        await run_in_threadpool(_import_mmpg_probe_config)
    except ConfigCorruptError as e:
        raise HTTPException(409, str(e)) from e
    except OSError:
        raise HTTPException(500, "无法保存设置，请检查配置文件目录的写入权限") from None
    except ValidationError as e:
        raise _mmpg_validation_error(e) from None
    return {"ok": True}


class ModelCatalogCheck(BaseModel):
    indices: list[StrictInt] | None = None


@router.post("/models/health")
async def api_check_models_health(body: ModelCatalogCheck | None = None):
    indices = body.indices if body else None
    return {"ok": True, **await run_in_threadpool(check_catalog_models, indices)}


@router.get("/models/health")
async def api_get_models_health():
    return get_model_health_snapshot()


@router.post("/models/list")
async def api_model_list_request(body: Any = Body()):
    return await api_list_remote_models(validate_model_request(body, ModelListRequest))


async def api_list_remote_models(body: ModelListRequest):
    api_base_url = (body.api_base_url or "").strip()
    api_key = (body.api_key or "").strip()
    api_type = (body.api_type or "openai").strip().lower()
    if api_type not in ("openai", "anthropic", "responses"):
        api_type = "openai"
    if not api_base_url:
        raise HTTPException(400, "API Base URL 不能为空")
    if api_key == _MODEL_API_KEY_KEEP:
        cfg = get_config()
        models = cfg.models
        if body.provider_id is not None:
            saved = next((provider for provider in get_model_providers(cfg) if provider.id == body.provider_id), None)
        elif body.model_id is not None:
            saved = next((model for model in models if getattr(model, "id", None) == body.model_id), None)
        else:
            index = body.model_index
            saved = models[index] if index is not None and 0 <= index < len(models) else None
        if saved is None or provider_url(saved.api_base_url) != provider_url(api_base_url) or get_api_type(saved) != api_type:
            raise HTTPException(409, "模型目录、服务地址或接口类型已变更，请刷新配置或填写对应的 API Key")
        api_key = saved.api_key.strip()
    if not api_key or api_key == "sk-your-api-key-here":
        raise HTTPException(400, "API Key 不能为空")
    try:
        return await run_in_threadpool(_list_remote_models, api_base_url, api_key, api_type)
    except ProviderRequestError as exc:
        raise HTTPException(502, f"获取模型列表失败：{exc}") from None


@router.post("/models/health/{index}")
async def api_check_single_model_health(index: int):
    """Check whether the saved model ID appears in its provider catalog."""
    cfg = get_config()
    if index < 0 or index >= len(cfg.models):
        raise HTTPException(400, f"模型 index {index} 超出范围")
    return {"ok": True, **await run_in_threadpool(check_catalog_models, [index])}


@router.post("/models/probe/{index}")
async def api_probe_single_model(index: int, model_id: str | None = None):
    """Check connectivity and synchronously probe model capabilities."""
    cfg = get_config()
    if index < 0 or index >= len(cfg.models):
        raise HTTPException(400, f"模型 index {index} 超出范围")
    try:
        return await run_in_threadpool(probe_single_model, index, model_id)
    except ModelListChanged as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class SttConnectionTest(BaseModel):
    provider: Literal["doubao", "generic"]


@router.post("/stt/test")
async def api_stt_test(body: SttConnectionTest | None = None):
    """Test one saved segmented ASR connection using a known sample."""
    cfg = get_config()
    provider = cfg.stt_provider if body is None else body.provider
    if provider == MMPG_STT_PROVIDER:
        return {"ok": False, "detail": "请在语音配置中使用豆包 MMPG 的会话诊断"}
    if detail := stt_configuration_error(cfg, provider):
        return {"ok": False, "detail": detail}
    try:
        from services.audio import load_wav_file
        wav_path = Path(__file__).resolve().parents[2] / "assets" / "preflight_phrase.wav"
        audio, sample_rate = await run_in_threadpool(load_wav_file, wav_path)
        engine = get_stt_engine(provider, config_snapshot=cfg)
        result = await run_in_threadpool(engine.transcribe, audio, sample_rate=sample_rate)
    except STTError as error:
        return {"ok": False, "detail": str(error)}
    except (TimeoutError, OSError):
        return {"ok": False, "detail": "ASR 连接失败或超时，请检查网络和服务地址"}
    except Exception:
        return {"ok": False, "detail": "无法完成 ASR 连接测试，请检查语音配置"}
    if not (result or "").strip():
        return {"ok": False, "detail": "引擎返回空文本，请检查接口兼容性或模型配置"}
    return {"ok": True, "text": result}


@router.get("/token/stats")
async def api_token_stats():
    from services.llm import get_token_stats
    return get_token_stats()


@router.post("/config/models-layout")
async def api_models_layout(body: dict):
    """调整模型顺序、开关与并行路数，不丢失各模型 api_key。"""
    cfg = get_config()
    order = body.get("order")
    origins = list(range(len(cfg.models)))
    if order is not None and isinstance(order, list):
        origins = []
        seen = set()
        for i in order:
            # 必须显式排除 bool: Python 中 isinstance(True, int) 为 True,
            # 否则前端误传 [True, False, 0, 1] 时 True/False 会被当成 1/0,
            # 导致模型顺序被悄悄改写或直接丢失.
            if (
                isinstance(i, int)
                and not isinstance(i, bool)
                and 0 <= i < len(cfg.models)
                and i not in seen
            ):
                origins.append(i)
                seen.add(i)
        for i in range(len(cfg.models)):
            if i not in seen:
                origins.append(i)
    models = [cfg.models[i] for i in origins]
    enabled = body.get("enabled")
    if isinstance(enabled, list):
        for i in range(min(len(enabled), len(models))):
            models[i] = models[i].model_copy(update={"enabled": bool(enabled[i])})
    if models and not _has_enabled_model(models):
        raise HTTPException(status_code=422, detail="至少启用一个模型")
    mp = body.get("max_parallel_answers")
    updates: dict = {"models": [m.model_dump() for m in models]}
    if mp is not None:
        try:
            updates["max_parallel_answers"] = max(1, min(8, int(mp)))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="max_parallel_answers must be an integer")
    active = cfg.active_model
    client_active = body.get("active_model")
    # 前端在重排后传入新列表中的优先模型下标，避免同名同 endpoint 模型时 next() 匹配到错误项
    if (
        isinstance(client_active, int)
        and not isinstance(client_active, bool)
        and len(models) > 0
    ):
        ca = int(client_active)
        if 0 <= ca < len(models):
            updates["active_model"] = ca
    elif active is not None and isinstance(order, list) and order:
        try:
            old = cfg.models[active] if active is not None and 0 <= active < len(cfg.models) else None
            if old:
                updates["active_model"] = next(
                    (
                        j
                        for j, m in enumerate(models)
                        if m.name == old.name
                        and m.model == old.model
                        and m.api_base_url == old.api_base_url
                    ),
                    min(active, len(models) - 1),
                )
            else:
                updates["active_model"] = min(active, len(models) - 1)
        except Exception:
            updates["active_model"] = 0
    try:
        await run_in_threadpool(update_config, updates, model_origins=origins, model_snapshot=cfg.models)
    except ModelListChanged as e:
        raise HTTPException(409, str(e)) from e
    except InvalidAssistModelSelection as e:
        raise HTTPException(422, str(e)) from e
    return {"ok": True}
