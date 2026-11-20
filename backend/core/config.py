from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Callable, Literal, Optional, Any, get_args
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid4, uuid5
import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import threading

from pydantic import BaseModel, Field, StrictBool, StrictStr, StringConstraints, ValidationError, field_validator, model_validator

from core.logger import get_logger
from core.prompt_settings import patch_system_prompts

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_WINDOW_TOKENS = 262_144
_MODEL_HEALTH_FINGERPRINT_KEY = secrets.token_bytes(32)

# LLM 接口协议（模型级配置 api_type 的取值范围）
API_TYPE_OPENAI = "openai"
API_TYPE_ANTHROPIC = "anthropic"
API_TYPE_RESPONSES = "responses"
API_TYPE_OPTIONS = [API_TYPE_OPENAI, API_TYPE_ANTHROPIC, API_TYPE_RESPONSES]


def _finite_number(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def normalize_llm_temperature(value: Any) -> float:
    parsed = _finite_number(value, DEFAULT_TEMPERATURE)
    return max(0.0, min(2.0, parsed))


def normalize_llm_max_tokens(value: Any) -> int:
    parsed = _finite_number(value, DEFAULT_MAX_TOKENS)
    return max(256, min(32768, int(parsed)))

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.environ.get("IA_CONFIG_PATH") or os.path.join(_BACKEND_DIR, "config.json")
CONFIG_EXAMPLE = os.path.join(_BACKEND_DIR, "config.example.json")
_ASR_CONFIG_VERSION = 1


class ModelConfig(BaseModel):
    id: str = ""
    provider_id: str = ""
    name: str = "Default"
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = Field(default="", repr=False)
    model: str = "gpt-4o-mini"
    context_window_tokens: int = Field(default=DEFAULT_CONTEXT_WINDOW_TOKENS, strict=True, ge=1, le=2**53 - 1)
    # 接口协议：openai=Chat Completions，anthropic=Messages，responses=Responses
    api_type: str = API_TYPE_OPENAI
    supports_think: bool = False
    supports_vision: bool = False
    enabled: bool = True
    verified: bool = False
    think_enabled_params: dict[str, Any] = Field(default_factory=dict)
    think_disabled_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_type", mode="before")
    @classmethod
    def _normalize_api_type(cls, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in API_TYPE_OPTIONS:
            return API_TYPE_OPENAI
        return normalized


class ModelProvider(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    api_type: Literal["openai", "anthropic", "responses"]
    api_base_url: str
    api_key: str = Field(default="", repr=False)


def provider_url(url: str) -> str:
    return url.strip().rstrip("/")


def provider_connection(provider: ModelProvider | ModelConfig) -> tuple[str, str, str]:
    return (provider_url(provider.api_base_url), provider.api_type, provider.api_key)


def model_health_fingerprint(model: Any) -> str:
    """Return a process-local opaque identity for health-check ownership.

    The API key participates so changing credentials invalidates an old
    health result, but the key itself (and a reusable unsalted hash of it) is
    never exposed to the frontend.
    """

    fields = (
        str(getattr(model, "name", "") or "").strip(),
        str(getattr(model, "api_base_url", "") or "").strip().rstrip("/"),
        str(getattr(model, "model", "") or "").strip(),
        str(getattr(model, "api_type", "") or "").strip() or API_TYPE_OPENAI,
        str(getattr(model, "api_key", "") or ""),
    )
    payload = "\x1f".join(fields).encode("utf-8", errors="surrogatepass")
    return hmac.new(
        _MODEL_HEALTH_FINGERPRINT_KEY,
        payload,
        hashlib.sha256,
    ).hexdigest()[:24]


def _default_model_config() -> ModelConfig:
    return ModelConfig(
        name="请在 config.json 中配置",
        api_base_url="https://api.openai.com/v1",
        api_key="",
        model="gpt-4o-mini",
    )


MMPG_STT_PROVIDER = "doubao_mmpg"
SEGMENT_STT_PROVIDERS = ("doubao", "generic")
STT_PROVIDER_OPTIONS = [*SEGMENT_STT_PROVIDERS, MMPG_STT_PROVIDER]
CANDIDATE_STT_PROVIDER_OPTIONS = STT_PROVIDER_OPTIONS.copy()
MMPG_SECRET_FIELDS = frozenset({
    "volc_access_key", "volc_secret_key", "rtc_app_key", "rtc_token", "doubao_api_key",
})
STT_SECRET_FIELDS = frozenset({
    "doubao_stt_access_token",
    "doubao_stt_api_key",
    "generic_stt_api_key",
    "generic_stt_custom_headers",
})
_MmpgText = Annotated[str, StringConstraints(strip_whitespace=True, strict=True)]
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_AnswerWindowMs = Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]


class Mmpg(BaseModel):
    volc_access_key: _MmpgText = Field(default="", repr=False)
    volc_secret_key: _MmpgText = Field(default="", repr=False)
    rtc_app_id: _MmpgText = ""
    rtc_app_key: _MmpgText = Field(default="", repr=False)
    rtc_room_id: _MmpgText = ""
    rtc_user_id: _MmpgText = ""
    rtc_bot_id: _MmpgText = ""
    rtc_token: _MmpgText = Field(default="", repr=False)
    rtc_token_expires_at: _PositiveInt | None = None
    doubao_api_key: _MmpgText = Field(default="", repr=False)
    doubao_resource_id: _MmpgText = "volc.seedasr.sauc.duration"
    aivad: StrictBool = True
    silence_ms: Annotated[int, Field(strict=True, ge=500, lt=3000)] = 800
    answer_delay_ms: _AnswerWindowMs = 1500
    followup_window_ms: _AnswerWindowMs = 1500
    twopass: StrictBool = True
    enable_prefetch: StrictBool = True
    expire_time_ms: _PositiveInt | None = None
    force_end: StrictBool = False
    token_mode: Literal["auto", "manual"] = "auto"


class KnowledgeAnalysisSettings(BaseModel):
    model_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    model_id: StrictStr | None = None
    system_prompt: StrictStr | None = None
    temperature: float = Field(default=0.3, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, strict=True, ge=256, le=32768)
    think_effort: Literal["off", "low", "medium", "high", "xhigh"] = "off"


AssistScene = Literal["assist_realtime", "assist_manual", "assist_screen", "assist_exam"]
ASSIST_SCENES = get_args(AssistScene)
ThinkEffort = Literal["off", "low", "medium", "high", "xhigh"]


class AssistGeneration(BaseModel):
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    max_output_tokens: int = Field(strict=True, ge=128, le=32768)


class AppConfig(BaseModel):
    system_prompts: dict[str, StrictStr] = Field(default_factory=dict)
    models: list[ModelConfig] = Field(default_factory=lambda: [_default_model_config()])
    model_providers: list[ModelProvider] | None = None
    active_model: int | None = 0
    active_model_id: StrictStr | None = None
    assist_model_ids: list[StrictStr] | None = None
    assist_screenshot_model_id: StrictStr | None = None
    written_exam_model_id: StrictStr | None = None
    written_exam_model_ids: list[StrictStr] = Field(default_factory=list)
    assist_generation: dict[AssistScene, AssistGeneration] = Field(default_factory=dict)
    assist_think_mode: StrictBool | None = None
    assist_think_effort: ThinkEffort | None = None
    knowledge_practice_system_prompt: StrictStr | None = None

    temperature: float = 0.5
    max_tokens: int = 4096
    think_mode: bool = False
    # 推理强度: off=关闭, low/medium/high/xhigh 分别对应低/中/高/超高强度推理
    think_effort: Literal["off", "low", "medium", "high", "xhigh"] = "off"

    # 两路各自选择云端引擎。空值表示用户尚未选择。
    stt_provider: str = ""
    mmpg: Mmpg = Field(default_factory=Mmpg)
    # 豆包语音识别 API（当 stt_provider=doubao 时使用），使用小时版 + WebSocket 双流式
    doubao_stt_app_id: str = ""
    doubao_stt_access_token: str = Field(default="", repr=False)
    # 新版控制台 API Key（优先于 app_id + access_token）
    doubao_stt_api_key: str = Field(default="", repr=False)
    # 资源 ID：豆包流式语音识别模型2.0 小时版
    doubao_stt_resource_id: str = "volc.seedasr.sauc.duration"
    # 热词表 ID：在自学习平台上传热词文件后获得；有则传入请求，没有则不传
    doubao_stt_boosting_table_id: str = ""
    # 通用 ASR（OpenAI-compatible multipart: POST /audio/transcriptions）
    generic_stt_api_base_url: str = ""
    generic_stt_api_key: str = Field(default="", repr=False)
    generic_stt_model: str = ""
    # 自定义 HTTP header，JSON 格式如 {"X-Custom":"value"} 或每行 Key: Value
    generic_stt_custom_headers: str = Field(default="", repr=False)
    # 我的回答需要用户显式选择引擎并启用。
    candidate_asr_enabled: bool = False
    candidate_stt_provider: str = ""
    # 候选人真实回答上下文：下一轮问题可携带，追问时强优先，非追问时作为可忽略背景。
    candidate_context_enabled: bool = True
    candidate_context_wait_ms: int = 200
    candidate_context_max_chars: int = 900
    candidate_context_min_chars: int = 6
    # 只用于“我的麦克风”：始终共享读取；开启后冲突时会尝试更保守的采样与默认输入设备。
    candidate_mic_compatibility_mode: bool = True

    position: str = "后端开发"
    language: str = "Python"
    resume_text: Optional[str] = None
    # 当前生效的简历对应的历史记录 id（写入 config.json；简历正文仍不入库）
    resume_active_history_id: Optional[int] = None

    auto_detect: bool = True
    silence_threshold: float = 0.01
    # Lower default reduces turn latency after interviewer finishes speaking.
    silence_duration: float = 1.2
    # 同时生成答案的最大路数（受可用模型数限制）
    max_parallel_answers: int = 2
    # 答案区流式输出时，距底部小于该像素则自动滚到底（调小便于手动上滑回看）
    answer_autoscroll_bottom_px: int = 40
    # 转写广播/入历史/自动答题：去标点后的有效字符（中英数字）至少为该值；过滤「嗯」等
    transcription_min_sig_chars: int = 2
    # 实时辅助：多段 ASR 合并后再写入转写/触发自动答题。上一段结束后若超过该秒数仍无新段则送出；0=关闭合并（恢复每段立即发送）
    assist_transcription_merge_gap_sec: float = 2.0
    # 从第一段 ASR 起最长等待（秒），超时强制送出，避免对方长停顿导致永远不触发
    assist_transcription_merge_max_sec: float = 12.0
    # 主链路 VAD 单段最长语音（秒），避免面试官连续讲话被攒成超长音频导致远程 ASR 超时
    assist_vad_max_speech_sec: float = 18.0
    # 主链路 VAD 单段最短语音（秒）；短于该值的片段视为噪声/语气词丢弃，不送 STT。
    # 调小可减少短句被误丢（漏听），调大可减少噪声片段送 STT。原硬编码 0.5。
    assist_vad_min_speech_sec: float = 0.3
    # 实时辅助：问句候选组在最后一条有效追问后静默超过该秒数再确认提交
    assist_asr_confirm_window_sec: float = 0.45
    # 实时辅助：候选问句组从第一条有效追问开始的最长等待时间
    assist_asr_group_max_wait_sec: float = 1.2
    # 实时辅助：确认新问句组后是否中断仍在生成的旧 ASR 回答
    assist_asr_interrupt_running: bool = True
    # 实时辅助：为每轮语音回答使用简短提示词
    assist_high_churn_short_answer: bool = False
    # 实时语音问答默认输出 token 上限，避免首题展开过长拖慢后续题。
    assist_realtime_max_tokens: int = 720
    # 语音简短回答的 token 上限。
    assist_realtime_high_churn_max_tokens: int = 320
    # stop 时等待 answer worker 收尾的最长时间；压测 interviewer 转写链路时可设为 0。
    assist_stop_answer_wait_sec: float = 3.0
    # stop 时等待面试官 ASR segment worker 清空队列的最长时间；本地 Whisper 需要更宽裕。
    assist_interviewer_asr_drain_timeout_sec: float = 6.0
    # 电脑截图区域：full=全屏，left_half/right_half/top_half/bottom_half=对应半屏
    screen_capture_region: str = "left_half"
    # 截图送入识图模型前的最长边限制；0=不缩放。默认 1600 兼顾题面可读性与 token/带宽。
    screen_capture_max_long_edge: int = 1600
    # 多图截图判题：最后一次截图后等待多少秒再提交整批图片
    multi_screen_capture_idle_sec: float = 10.0
    # 笔试模式：截屏后选择题直接输出答案，编程题直接输出代码，不做分析
    written_exam_mode: bool = False
    # 笔试模式下是否开启 think（深度思考），默认关闭以加快响应
    written_exam_think: bool = False

    # --- Knowledge Base ---
    # 路径相对 backend/ 目录解析; 默认值与 .gitignore (backend/data/kb/) 对齐。
    kb_enabled: bool = False
    kb_dir: str = "data/kb"
    kb_db_path: str = "data/kb.sqlite"
    kb_cache_dir: str = "data/kb_cache"
    kb_top_k: int = 4
    kb_deadline_ms: int = 150
    kb_asr_deadline_ms: int = 80
    # BM25 对稀疏语料区分度低 (常返回很小的分数), MVP 只靠 top_k + 排序,
    # 将 min_score 设为 0 关闭阈值过滤; 上线后看热点 query 再调。
    kb_min_score: float = 0.0
    kb_chunk_max_chars: int = 800
    kb_prompt_excerpt_chars: int = 300
    kb_trigger_modes: list[str] = Field(
        default_factory=lambda: ["asr_realtime", "manual_text", "written_exam"]
    )
    kb_file_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".txt", ".log", ".docx", ".pdf"]
    )
    kb_ocr_enabled: bool = False
    kb_vision_caption_enabled: bool = False
    kb_max_upload_bytes: int = 20 * 1024 * 1024
    kb_recent_hits_capacity: int = 50
    kb_asr_min_query_chars: int = 6

    # --- Review (面试复盘) ---
    review_enabled: bool = False
    review_model_index: int | None = 0
    review_model_id: StrictStr | None = None
    knowledge_analysis: KnowledgeAnalysisSettings = Field(default_factory=KnowledgeAnalysisSettings)

    @field_validator("temperature", mode="before")
    @classmethod
    def _normalize_temperature_field(cls, value: Any) -> float:
        return normalize_llm_temperature(value)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _normalize_max_tokens_field(cls, value: Any) -> int:
        return normalize_llm_max_tokens(value)

    @model_validator(mode="before")
    @classmethod
    def _retire_legacy_candidate_route(cls, data):
        # Old builds used a separate permission flag before routing a segment to the cloud.
        if (isinstance(data, dict) and data.get("candidate_remote_stt_enabled") is False
                and data.get("candidate_stt_provider") in SEGMENT_STT_PROVIDERS):
            return {**data, "candidate_asr_enabled": False, "candidate_stt_provider": ""}
        return data

    @model_validator(mode="after")
    def _ensure_valid_models(self):
        if self.stt_provider not in STT_PROVIDER_OPTIONS:
            self.stt_provider = ""
        if self.candidate_stt_provider not in CANDIDATE_STT_PROVIDER_OPTIONS:
            self.candidate_stt_provider = ""
            self.candidate_asr_enabled = False
        self.temperature = normalize_llm_temperature(self.temperature)
        self.max_tokens = normalize_llm_max_tokens(self.max_tokens)
        self.candidate_context_wait_ms = max(0, min(2000, int(self.candidate_context_wait_ms or 0)))
        self.candidate_context_max_chars = max(100, min(4000, int(self.candidate_context_max_chars or 900)))
        self.candidate_context_min_chars = max(1, min(100, int(self.candidate_context_min_chars or 6)))
        self.assist_realtime_max_tokens = max(256, min(4096, int(self.assist_realtime_max_tokens or 720)))
        self.assist_realtime_high_churn_max_tokens = max(
            128,
            min(self.assist_realtime_max_tokens, int(self.assist_realtime_high_churn_max_tokens or 320)),
        )
        self.candidate_mic_compatibility_mode = bool(self.candidate_mic_compatibility_mode)
        if self.model_providers is None and not self.models:
            self.models = [_default_model_config()]
        if self.model_providers is not None:
            if "models" not in self.model_fields_set:
                self.models = []
            providers = {provider.id: provider for provider in self.model_providers}
            if len(providers) != len(self.model_providers):
                raise ValueError("供应商 ID 重复")
            self.models = [_hydrate_model(model, providers) for model in self.models]
        self.screen_capture_max_long_edge = max(0, min(4000, int(self.screen_capture_max_long_edge or 0)))
        self.assist_vad_max_speech_sec = max(6.0, min(60.0, float(self.assist_vad_max_speech_sec or 18.0)))
        self.assist_vad_min_speech_sec = max(0.1, min(2.0, float(self.assist_vad_min_speech_sec or 0.3)))
        self.assist_stop_answer_wait_sec = max(0.0, min(20.0, float(self.assist_stop_answer_wait_sec)))
        self.assist_interviewer_asr_drain_timeout_sec = max(
            0.5,
            min(30.0, float(self.assist_interviewer_asr_drain_timeout_sec or 6.0)),
        )
        if self.model_providers is None:
            if self.active_model is not None:
                self.active_model = max(0, min(self.active_model, len(self.models) - 1))
                if not self.models[self.active_model].enabled:
                    self.active_model = next(
                        (i for i, model in enumerate(self.models) if model.enabled), self.active_model,
                    )
            if self.review_model_index is not None:
                self.review_model_index = max(0, min(self.review_model_index, len(self.models) - 1))
        self._migrate_assist_settings()
        return self

    def _migrate_assist_settings(self) -> None:
        used_ids: set[str] = set()
        reserved_ids = {model.id for model in self.models if model.id}
        for index, model in enumerate(self.models):
            if not model.id or model.id in used_ids:
                # An old config can be read across restarts before its first save.
                seed = f"interview-helper:legacy-model:{index}"
                candidate = uuid5(NAMESPACE_URL, seed).hex
                while candidate in used_ids or candidate in reserved_ids:
                    candidate = uuid5(NAMESPACE_URL, candidate).hex
                model = model.model_copy(update={"id": candidate})
                self.models[index] = model
            used_ids.add(model.id)

        self._sync_model_references()
        legacy_order = list(self.models)
        if self.active_model is not None:
            legacy_order.insert(0, legacy_order.pop(self.active_model))
        if self.assist_model_ids is None:
            self.assist_model_ids = [model.id for model in legacy_order if model.enabled]
        else:
            self.assist_model_ids = list(dict.fromkeys(
                model_id for model_id in self.assist_model_ids if model_id in used_ids
            ))
        vision_ids = {model.id for model in self.models if model.supports_vision}
        if "assist_screenshot_model_id" not in self.model_fields_set:
            self.assist_screenshot_model_id = next(
                (model.id for model in legacy_order if model.id in vision_ids and knowledge_model_available(model)),
                None,
            )
        elif self.assist_screenshot_model_id not in vision_ids:
            self.assist_screenshot_model_id = None
        if "written_exam_model_ids" not in self.model_fields_set:
            ready_vision_ids = {model.id for model in self.models if written_exam_model_available(model)}
            primary_id = self.written_exam_model_id
            if "written_exam_model_id" not in self.model_fields_set:
                primary_id = self.assist_screenshot_model_id if self.assist_screenshot_model_id in ready_vision_ids else None
            exam_ids = [primary_id] if primary_id in used_ids else []
            if primary_id in ready_vision_ids:
                exam_ids.extend(model_id for model_id in self.assist_model_ids if model_id in ready_vision_ids)
        else:
            exam_ids = [model_id for model_id in self.written_exam_model_ids if model_id in used_ids]
        self.written_exam_model_ids = list(dict.fromkeys(exam_ids))
        self.written_exam_model_id = self.written_exam_model_ids[0] if self.written_exam_model_ids else None
        for scene in ASSIST_SCENES:
            if scene not in self.assist_generation:
                limit = min(self.max_tokens, self.assist_realtime_max_tokens) if scene == "assist_realtime" else self.max_tokens
                self.assist_generation[scene] = AssistGeneration(temperature=self.temperature, max_output_tokens=limit)
        if self.assist_think_mode is None:
            self.assist_think_mode = self.think_mode
        if self.assist_think_effort is None:
            self.assist_think_effort = self.think_effort
        if "knowledge_practice_system_prompt" not in self.model_fields_set:
            self.knowledge_practice_system_prompt = self.system_prompts.get("assist_manual")

    def _sync_model_references(self) -> None:
        if self.model_providers is None:
            if "active_model_id" not in self.model_fields_set:
                self.active_model_id = self._model_id_at(self.active_model)
            if "review_model_id" not in self.model_fields_set:
                self.review_model_id = self._model_id_at(self.review_model_index)
            if "model_id" not in self.knowledge_analysis.model_fields_set:
                self.knowledge_analysis.model_id = self._model_id_at(self.knowledge_analysis.model_index)
            return

        enabled_indices = {model.id: i for i, model in enumerate(self.models) if model.enabled}
        self.active_model = enabled_indices.get(self.active_model_id)
        self.review_model_index = enabled_indices.get(self.review_model_id)
        self.knowledge_analysis.model_index = enabled_indices.get(self.knowledge_analysis.model_id)
        if self.active_model is None:
            self.active_model_id = None
        if self.review_model_index is None:
            self.review_model_id = None
        if self.knowledge_analysis.model_index is None:
            self.knowledge_analysis.model_id = None

    def _model_id_at(self, index: int | None) -> str | None:
        return self.models[index].id if index is not None and 0 <= index < len(self.models) else None

    def get_active_model(self) -> ModelConfig | None:
        index = self.active_model
        return self.models[index] if index is not None else None

    def get_review_model(self) -> ModelConfig | None:
        index = self.review_model_index
        return self.models[index] if index is not None else None


_config: Optional[AppConfig] = None
_config_corrupt_state: Optional[dict[str, Any]] = None
_config_lock = threading.RLock()


class ConfigCorruptError(RuntimeError):
    """配置文件损坏处于只读保护态时，拒绝未确认的覆写保存。"""


def _current_config_file_key() -> str:
    return str(Path(CONFIG_FILE).resolve())


def get_config_corrupt_state() -> Optional[dict[str, Any]]:
    with _config_lock:
        if _config_corrupt_state is None:
            return None
        if _config_corrupt_state.get("config_file") != _current_config_file_key():
            return None
        return {
            "corrupt": True,
            "error": _config_corrupt_state["error"],
            "backup_path": _config_corrupt_state.get("backup_path"),
            "backup_filename": _config_corrupt_state.get("backup_filename"),
        }


def _assert_config_writable(*, allow_overwrite_corrupt: bool = False) -> None:
    corrupt = get_config_corrupt_state()
    if corrupt is None or allow_overwrite_corrupt:
        return
    backup_hint = (
        f"（原文件已备份为 {corrupt['backup_filename']}）"
        if corrupt.get("backup_filename")
        else ""
    )
    raise ConfigCorruptError(
        f"配置文件加载失败，当前处于只读保护模式{backup_hint}。"
        "请先修复或恢复 config.json，或在页面明确确认覆盖后再保存。"
    )


def get_config() -> AppConfig:
    global _config
    with _config_lock:
        if _config is None:
            _config = _load_config()
        return _config


class ModelListChanged(ValueError):
    pass


class InvalidAssistModelSelection(ValueError):
    pass


def _hydrate_model(model: ModelConfig, providers: dict[str, ModelProvider]) -> ModelConfig:
    provider = providers.get(model.provider_id)
    if provider is None:
        raise ValueError("模型引用的供应商不存在")
    return model.model_copy(update={
        "api_base_url": provider.api_base_url, "api_key": provider.api_key, "api_type": provider.api_type,
    })


def _provider_name(model: ModelConfig) -> str:
    try:
        host = urlsplit(model.api_base_url).hostname
    except ValueError:
        return model.name
    return {
        "api.openai.com": "OpenAI",
        "api.deepseek.com": "DeepSeek",
        "api.anthropic.com": "Anthropic",
    }.get(host, "火山方舟" if host and host.startswith("ark.") and host.endswith(".volces.com") else host or model.name)


def get_provider_layout(cfg: AppConfig) -> tuple[list[ModelProvider], list[ModelConfig]]:
    if cfg.model_providers is not None:
        return list(cfg.model_providers), list(cfg.models)
    groups: dict[tuple[str, str, str], ModelProvider] = {}
    models = []
    for model in cfg.models:
        if not model.api_key and model.name == "请在 config.json 中配置":
            continue
        connection = provider_connection(model)
        provider = groups.get(connection)
        if provider is None:
            provider = ModelProvider(
                id=uuid5(NAMESPACE_URL, f"interview-helper:legacy-provider:{model.id}").hex,
                name=_provider_name(model),
                api_base_url=connection[0],
                api_type=connection[1],
                api_key=connection[2],
            )
            groups[connection] = provider
        models.append(model.model_copy(update={"provider_id": provider.id, "verified": False}))
    return list(groups.values()), models


def get_model_providers(cfg: AppConfig) -> list[ModelProvider]:
    return get_provider_layout(cfg)[0]


def get_model_usages(cfg: AppConfig) -> dict[str, list[str]]:
    usages: dict[str, list[str]] = {}
    for name, model_ids in (
        ("实时辅助", cfg.assist_model_ids or []),
        ("截图答题", [cfg.assist_screenshot_model_id]),
        ("笔试", cfg.written_exam_model_ids),
        ("复盘", [cfg.review_model_id]),
        ("知识分析", [cfg.knowledge_analysis.model_id]),
        ("通用默认模型", [cfg.active_model_id]),
    ):
        for model_id in model_ids:
            if model_id is not None:
                usages.setdefault(model_id, []).append(name)
    return usages


def provider_layout_revision(cfg: AppConfig) -> str:
    providers, models = get_provider_layout(cfg)
    data = [
        [provider.model_dump() for provider in providers],
        [model.model_dump() for model in models],
        get_model_usages(cfg),
        cfg.assist_model_ids,
        cfg.written_exam_model_ids,
    ]
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hmac.new(_MODEL_HEALTH_FINGERPRINT_KEY, payload, hashlib.sha256).hexdigest()


def save_provider_layout(
    revision: str,
    providers: list[ModelProvider],
    models: list[ModelConfig],
    confirmed_remove_ids: set[str],
    active_model_id: str | None,
    was_verified: Callable[[ModelProvider, str], bool],
) -> AppConfig:
    global _config
    with _config_lock:
        cfg = get_config()
        _assert_config_writable()
        if revision != provider_layout_revision(cfg):
            raise ModelListChanged("供应商或模型用途已变更，请刷新并重新确认")
        previous_providers, previous_models = get_provider_layout(cfg)
        previous_provider_by_id = {provider.id: provider for provider in previous_providers}
        providers_by_id = {provider.id: provider for provider in providers}
        if len(providers_by_id) != len(providers):
            raise ValueError("供应商 ID 重复")
        if any(not provider.name.strip() for provider in providers):
            raise ValueError("请输入供应商名称")
        for provider in providers:
            previous = previous_provider_by_id.get(provider.id)
            if previous is None or provider_url(previous.api_base_url) != provider_url(provider.api_base_url):
                _validate_provider_url(provider.api_base_url)
        models_by_id = {model.id: model for model in models}
        if len(models_by_id) != len(models) or any(not model.id.strip() or not model.model.strip() for model in models):
            raise ValueError("模型 ID 不能为空或重复")
        previous_by_id = {model.id: model for model in previous_models}
        removed = {
            model.id for model in previous_models
            if model.id not in models_by_id or (model.enabled and not models_by_id[model.id].enabled)
        }
        if (removed & get_model_usages(cfg).keys()) - confirmed_remove_ids:
            raise ModelListChanged("模型正在被使用，请确认移除后重试")
        verified_models = []
        for model in models:
            provider = providers_by_id.get(model.provider_id)
            if provider is None:
                raise ValueError("模型引用的供应商不存在")
            previous = previous_by_id.get(model.id)
            connection_changed = (
                previous is None or previous.model != model.model
                or previous.provider_id != model.provider_id
                or provider_connection(previous_provider_by_id[previous.provider_id]) != provider_connection(provider)
            )
            has_proof = was_verified(provider, model.model)
            if model.enabled and (connection_changed or not previous.enabled) and not has_proof:
                raise ValueError("请在当前供应商连接下验证模型后再勾选")
            verified_models.append(model.model_copy(update={
                "verified": has_proof or (not connection_changed and previous.verified),
            }))
        if active_model_id in removed:
            active_model_id = None
        if active_model_id is not None:
            active = models_by_id.get(active_model_id)
            if active is None or not active.enabled:
                raise ValueError("请选择已勾选的通用默认模型")
        data = cfg.model_dump()
        data.update({
            "model_providers": providers,
            "models": verified_models,
            "active_model_id": active_model_id,
            "assist_model_ids": [mid for mid in cfg.assist_model_ids or [] if mid not in removed],
            "written_exam_model_ids": [mid for mid in cfg.written_exam_model_ids if mid not in removed],
        })
        if cfg.assist_screenshot_model_id in removed:
            data["assist_screenshot_model_id"] = None
        if cfg.review_model_id in removed:
            data["review_model_id"] = None
        if cfg.knowledge_analysis.model_id in removed:
            data["knowledge_analysis"]["model_id"] = None
        updated = AppConfig.model_validate(data)
        _save_config(updated)
        _config = updated
        return updated


def _validate_provider_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        # urllib.parse validates the range only when the port is read.
        _ = parsed.port
        valid = (
            bool(parsed.hostname)
            and parsed.scheme in ("http", "https")
            and parsed.username is None
            and parsed.password is None
            and "?" not in url
            and "#" not in url
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("API Base URL 必须是有效的 HTTP 或 HTTPS 地址")


def _map_single_model_reference(
    data: dict, updates: dict, model_rows: list[dict], id_field: str, index_field: str,
) -> None:
    if id_field not in updates and index_field in updates:
        index = updates[index_field]
        if index is not None and not (type(index) is int and 0 <= index < len(model_rows)):
            raise InvalidAssistModelSelection("模型已变更，请重新选择")
        data[id_field] = model_rows[index]["id"] if index is not None else None
    if id_field not in updates and index_field not in updates:
        return
    model_id = data[id_field]
    model = next((model for model in model_rows if model["id"] == model_id), None)
    if model_id is not None and (model is None or not model["enabled"]):
        raise InvalidAssistModelSelection("请选择已勾选的模型")
    if id_field in updates:
        data[index_field] = model_rows.index(model) if model is not None else None


def _validate_legacy_model_changes(cfg: AppConfig, rows: list[dict]) -> None:
    previous_by_id = {model.id: model for model in cfg.models}
    candidates = [ModelConfig.model_validate(row) for row in rows]
    by_id = {model.id: model for model in candidates}
    if len(by_id) != len(candidates):
        raise InvalidAssistModelSelection("模型 ID 不能重复")
    for model_id in get_model_usages(cfg):
        previous = previous_by_id.get(model_id)
        if previous is None:
            continue
        candidate = by_id.get(model_id)
        if candidate is None or (previous.enabled and not candidate.enabled):
            raise ModelListChanged("模型正在被使用，请在模型供应商配置中确认移除")
    for candidate in candidates:
        previous = previous_by_id.get(candidate.id)
        changed = previous is None or (
            previous.model != candidate.model or provider_connection(previous) != provider_connection(candidate)
        )
        reused_key_at_new_address = previous is not None and bool(previous.api_key.strip()) and (
            previous.api_key == candidate.api_key
            and provider_connection(previous)[:2] != provider_connection(candidate)[:2]
        )
        needs_proof = candidate.enabled and (changed or not previous.enabled)
        if needs_proof or reused_key_at_new_address:
            from api.common.provider_catalog import was_verified

            if not was_verified(candidate, candidate.model):
                raise InvalidAssistModelSelection("请在模型供应商配置中使用当前连接验证模型")


def update_config(
    updates: dict,
    *,
    model_origins: list[int | None] | None = None,
    model_snapshot: list[ModelConfig] | None = None,
    allow_overwrite_corrupt: bool = False,
) -> AppConfig:
    global _config
    with _config_lock:
        cfg = get_config()
        _assert_config_writable(allow_overwrite_corrupt=allow_overwrite_corrupt)
        if model_snapshot is not None and model_snapshot != cfg.models:
            raise ModelListChanged("模型列表已更新，请刷新后重试")
        data = cfg.model_dump()
        for key, value in updates.items():
            if key == "mmpg":
                # Merge under the same lock used to persist settings so concurrent
                # patches preserve fields and credentials they don't edit.
                current = data["mmpg"]
                previous_token = current["rtc_token"]
                for field, setting in value.items():
                    if field in MMPG_SECRET_FIELDS:
                        if isinstance(setting, str):
                            setting = setting.strip()
                            if not setting:
                                continue
                        if setting is None:
                            setting = ""
                    current[field] = setting
                if current["rtc_token"] != previous_token and "rtc_token_expires_at" not in value:
                    current["rtc_token_expires_at"] = None
            elif key in STT_SECRET_FIELDS:
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        continue
                    data[key] = stripped
                elif value is None:
                    data[key] = ""
                else:
                    data[key] = value
            elif key == "system_prompts":
                data[key] = patch_system_prompts(data[key], value)
            elif key == "assist_generation":
                data[key].update(value)
            elif hasattr(cfg, key):
                data[key] = value
        legacy_exam_update = "written_exam_model_id" in updates and "written_exam_model_ids" not in updates
        if legacy_exam_update:
            primary_id = updates["written_exam_model_id"]
            data["written_exam_model_ids"] = [] if primary_id is None else [primary_id, *cfg.written_exam_model_ids]
        if "models" in updates:
            model_rows = [
                model.model_dump() if isinstance(model, ModelConfig) else dict(model)
                for model in data["models"]
            ]
            for index, model in enumerate(model_rows):
                if not model.get("id"):
                    origin = model_origins[index] if model_origins is not None and index < len(model_origins) else None
                    model["id"] = cfg.models[origin].id if origin is not None else uuid4().hex
            if cfg.model_providers is not None:
                previous_by_id = {model.id: model for model in cfg.models}
                if len(model_rows) != len(previous_by_id) or {row["id"] for row in model_rows} != previous_by_id.keys():
                    raise ModelListChanged("请在模型供应商配置中添加或移除模型")
                for row in model_rows:
                    previous = previous_by_id[row["id"]]
                    candidate = ModelConfig.model_validate(row)
                    if (candidate.enabled != previous.enabled or candidate.model != previous.model
                            or provider_connection(candidate) != provider_connection(previous)):
                        raise ModelListChanged("请在模型供应商配置中变更供应商连接或勾选模型")
                    row["provider_id"] = previous.provider_id
                    row["verified"] = previous.verified
            else:
                _validate_legacy_model_changes(cfg, model_rows)
            data["models"] = model_rows
            by_id = {model["id"]: index for index, model in enumerate(model_rows)}
            for field in ("active_model", "review_model_index"):
                if field not in updates and (previous_index := getattr(cfg, field)) is not None:
                    old_id = cfg.models[previous_index].id
                    data[field] = by_id.get(old_id, data[field])
        model_rows = data["models"]
        for id_field, index_field in (("active_model_id", "active_model"), ("review_model_id", "review_model_index")):
            _map_single_model_reference(data, updates, model_rows, id_field, index_field)
        if cfg.model_providers is None:
            if "active_model_id" not in updates:
                data.pop("active_model_id", None)
            if "review_model_id" not in updates:
                data.pop("review_model_id", None)
        if "assist_model_ids" in updates or "assist_screenshot_model_id" in updates:
            catalog = {
                model.id if isinstance(model, ModelConfig) else model["id"]: model
                for model in data["models"]
            }
            if "assist_model_ids" in updates and any(model_id not in catalog for model_id in updates["assist_model_ids"]):
                raise ModelListChanged("答题模型目录已更新，请刷新后重试")
            if cfg.model_providers is not None and "assist_model_ids" in updates and any(
                not catalog[model_id]["enabled"] and model_id not in (cfg.assist_model_ids or [])
                for model_id in updates["assist_model_ids"]
            ):
                raise InvalidAssistModelSelection("实时辅助请选择已勾选的模型")
            selected_vision = updates.get("assist_screenshot_model_id")
            if selected_vision is not None:
                model = catalog.get(selected_vision)
                vision = model.supports_vision if isinstance(model, ModelConfig) else (model or {}).get("supports_vision")
                if not vision:
                    raise InvalidAssistModelSelection("截图答题请选择支持识图的模型")
                if cfg.model_providers is not None and selected_vision != cfg.assist_screenshot_model_id:
                    if not written_exam_model_available(ModelConfig.model_validate(model)):
                        raise InvalidAssistModelSelection("截图答题请选择已勾选、已配置 API Key 且支持识图的模型")
        updated = AppConfig(**data)
        if legacy_exam_update or "written_exam_model_ids" in updates:
            exam_catalog = {model.id: model for model in updated.models}
            requested_ids = data["written_exam_model_ids"][:1] if legacy_exam_update else updates["written_exam_model_ids"]
            for model_id in requested_ids:
                model = exam_catalog.get(model_id)
                retained = not legacy_exam_update and model_id in cfg.written_exam_model_ids
                if model is None or (not retained and not written_exam_model_available(model)):
                    raise InvalidAssistModelSelection("笔试答题请选择已启用、已配置 API Key 且支持图片输入的模型")
        index = cfg.knowledge_analysis.model_index
        if cfg.model_providers is None and "models" in updates and index is not None:
            if model_origins is not None:
                matches = [new for new, old in enumerate(model_origins) if old == index]
            elif index < len(cfg.models):
                # Full-list edits may carry indices from an older model layout.
                fingerprint = model_health_fingerprint(cfg.models[index])
                matches = [
                    new for new, model in enumerate(updated.models)
                    if model_health_fingerprint(model) == fingerprint
                ]
            else:
                matches = []
            updated.knowledge_analysis.model_index = matches[0] if len(matches) == 1 else None
            updated.knowledge_analysis.model_id = updated._model_id_at(updated.knowledge_analysis.model_index)
        _save_config(updated, allow_overwrite_corrupt=allow_overwrite_corrupt)
        _config = updated
        return updated


def knowledge_model_available(model: ModelConfig) -> bool:
    return bool(model.enabled and model.api_key.strip() not in ("", "sk-your-api-key-here"))


def written_exam_model_available(model: ModelConfig) -> bool:
    return model.supports_vision and knowledge_model_available(model)


def update_knowledge_analysis_settings(
    settings: KnowledgeAnalysisSettings, fingerprint: str | None,
) -> AppConfig:
    with _config_lock:
        cfg = get_config()
        models = cfg.models
        index = settings.model_index
        if settings.model_id is not None:
            index = next((i for i, model in enumerate(models) if model.id == settings.model_id), None)
            if index is None:
                raise ModelListChanged("模型列表已更新，请重新选择分析模型")
        if index is not None:
            if index >= len(models) or fingerprint != model_health_fingerprint(models[index]):
                raise ModelListChanged("模型列表已更新，请重新选择分析模型")
            if not knowledge_model_available(models[index]):
                raise ValueError("请选择已启用且配置了 API Key 的分析模型")
            if settings.think_effort != "off" and not models[index].supports_think:
                raise ValueError("该模型未启用推理能力，请将推理强度设为关闭")
        data = settings.model_dump()
        data["model_index"] = index
        data["model_id"] = models[index].id if index is not None else None
        return update_config({"knowledge_analysis": data})


def _backup_corrupt_config(target: Path) -> Optional[Path]:
    if not target.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(f"{target.name}.corrupt.{stamp}")
    suffix = 1
    while backup.exists():
        backup = target.with_name(f"{target.name}.corrupt.{stamp}.{suffix}")
        suffix += 1
    temp_path = None
    try:
        raw = target.read_bytes()
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, backup)
        temp_path = None
        return backup
    except OSError as exc:
        logger.error("备份损坏的配置文件失败: %s", exc)
        return None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _load_config() -> AppConfig:
    global _config_corrupt_state
    _config_corrupt_state = None
    _custom_path = os.environ.get("IA_CONFIG_PATH")
    if not os.path.exists(CONFIG_FILE):
        if _custom_path:
            logger.warning("IA_CONFIG_PATH=%s does not exist, using defaults", _custom_path)
            return AppConfig()
        if os.path.exists(CONFIG_EXAMPLE):
            shutil.copy2(CONFIG_EXAMPLE, CONFIG_FILE)
            logger.info("已从 config.example.json 创建 config.json，请在设置中添加模型供应商")
    if os.path.exists(CONFIG_FILE):
        target = Path(CONFIG_FILE)
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("配置文件根节点必须是 JSON 对象")
            if data.pop("asr_config_version", None) == _ASR_CONFIG_VERSION:
                data.pop("candidate_remote_stt_enabled", None)
            else:
                # In older releases, omitting this flag meant the candidate still used local ASR.
                data["candidate_remote_stt_enabled"] = data.get("candidate_remote_stt_enabled") is True
            return AppConfig(**data)
        except ValidationError as e:
            backup = _backup_corrupt_config(target)
            fields = sorted({
                ".".join(str(part) for part in item.get("loc", ()))
                for item in e.errors(include_input=False, include_context=False, include_url=False)
                if item.get("loc")
            })
            field_suffix = f"：{', '.join(fields[:5])}" if fields else ""
            summary = f"配置校验失败（{e.error_count()} 项无效{field_suffix}）"
            logger.error("%s，已备份至 %s 并进入只读保护模式", summary, backup or "备份失败")
            _config_corrupt_state = {
                "config_file": _current_config_file_key(),
                "error": summary,
                "backup_path": str(backup) if backup else None,
                "backup_filename": backup.name if backup else None,
            }
        except Exception as e:
            backup = _backup_corrupt_config(target)
            summary = f"配置文件解析失败（{type(e).__name__}）"
            logger.error("%s，已备份至 %s 并进入只读保护模式: %s", summary, backup or "备份失败", e)
            _config_corrupt_state = {
                "config_file": _current_config_file_key(),
                "error": summary,
                "backup_path": str(backup) if backup else None,
                "backup_filename": backup.name if backup else None,
            }
    return AppConfig()


def _save_config(cfg: AppConfig, *, allow_overwrite_corrupt: bool = False) -> None:
    global _config_corrupt_state
    _assert_config_writable(allow_overwrite_corrupt=allow_overwrite_corrupt)
    target = Path(CONFIG_FILE)
    data = cfg.model_dump()
    data["asr_config_version"] = _ASR_CONFIG_VERSION
    data.pop("resume_text", None)
    if cfg.model_providers is None:
        for field in ("model_providers", "active_model_id", "review_model_id"):
            data.pop(field, None)
        data["knowledge_analysis"].pop("model_id", None)
        for model in data["models"]:
            for field in ("provider_id", "verified"):
                model.pop(field, None)
    else:
        _backup_legacy_model_config(target)
        for key in ("active_model", "review_model_index"):
            data.pop(key, None)
        data["knowledge_analysis"].pop("model_index", None)
        for model in data["models"]:
            for field in ("api_base_url", "api_type", "api_key"):
                model.pop(field, None)
    temp_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as f:
            temp_path = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
        temp_path = None
        if _config_corrupt_state is not None and _config_corrupt_state.get("config_file") == _current_config_file_key():
            _config_corrupt_state = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("无法清理配置临时文件")


def _backup_legacy_model_config(target: Path) -> None:
    backup = target.with_name(target.name + ".pre-model-providers.bak")
    if backup.exists() or not target.exists():
        return
    original = target.read_bytes()
    try:
        prev = json.loads(original.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise OSError("无法读取旧配置以备份") from exc
    if not isinstance(prev, dict):
        raise OSError("无法读取旧配置以备份")
    if isinstance(prev.get("model_providers"), list):
        return
    temp_path = None
    try:
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, backup)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


POSITION_OPTIONS = [
    "前端开发", "后端开发", "全栈开发", "算法工程师", "测试开发",
    "机器学习工程师", "数据开发", "DBA", "产品经理", "项目经理",
]
LANGUAGE_OPTIONS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript",
    "Go", "SQL",
]
# 远程 ASR 失败降级目标：本地引擎或关闭
# FunASR 本地模型（中文面试场景推荐 paraformer-zh：准确率高、CPU 快、支持热词）
# 电脑截图区域
SCREEN_CAPTURE_REGION_OPTIONS = ["full", "left_half", "right_half", "top_half", "bottom_half"]
