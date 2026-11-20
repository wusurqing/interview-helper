from pydantic import BaseModel, Field, StrictBool, StrictStr, StringConstraints, ValidationError, field_validator, model_validator
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, Optional, Any, get_args
from uuid import NAMESPACE_URL, uuid4, uuid5
import hashlib
import hmac
import json
import math
import os
import secrets

from core.logger import get_logger
from core.prompt_settings import patch_system_prompts

logger = get_logger(__name__)
import shutil
import threading

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
    name: str = "Default"
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    context_window_tokens: int = Field(default=DEFAULT_CONTEXT_WINDOW_TOKENS, strict=True, ge=1, le=2**53 - 1)
    # 接口协议：openai=Chat Completions，anthropic=Messages，responses=Responses
    api_type: str = API_TYPE_OPENAI
    supports_think: bool = False
    supports_vision: bool = False
    enabled: bool = True
    think_enabled_params: dict[str, Any] = Field(default_factory=dict)
    think_disabled_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_type", mode="before")
    @classmethod
    def _normalize_api_type(cls, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in API_TYPE_OPTIONS:
            return API_TYPE_OPENAI
        return normalized


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
    active_model: int = 0
    assist_model_ids: list[StrictStr] | None = None
    assist_screenshot_model_id: StrictStr | None = None
    written_exam_model_id: StrictStr | None = None
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
    doubao_stt_access_token: str = ""
    # 新版控制台 API Key（优先于 app_id + access_token）
    doubao_stt_api_key: str = ""
    # 资源 ID：豆包流式语音识别模型2.0 小时版
    doubao_stt_resource_id: str = "volc.seedasr.sauc.duration"
    # 热词表 ID：在自学习平台上传热词文件后获得；有则传入请求，没有则不传
    doubao_stt_boosting_table_id: str = ""
    # 通用 ASR（OpenAI-compatible multipart: POST /audio/transcriptions）
    generic_stt_api_base_url: str = ""
    generic_stt_api_key: str = ""
    generic_stt_model: str = ""
    # 自定义 HTTP header，JSON 格式如 {"X-Custom":"value"} 或每行 Key: Value
    generic_stt_custom_headers: str = ""
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
    review_model_index: int = 0
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
        if not self.models:
            self.models = [_default_model_config()]
        self.screen_capture_max_long_edge = max(0, min(4000, int(self.screen_capture_max_long_edge or 0)))
        self.assist_vad_max_speech_sec = max(6.0, min(60.0, float(self.assist_vad_max_speech_sec or 18.0)))
        self.assist_vad_min_speech_sec = max(0.1, min(2.0, float(self.assist_vad_min_speech_sec or 0.3)))
        self.assist_stop_answer_wait_sec = max(0.0, min(20.0, float(self.assist_stop_answer_wait_sec)))
        self.assist_interviewer_asr_drain_timeout_sec = max(
            0.5,
            min(30.0, float(self.assist_interviewer_asr_drain_timeout_sec or 6.0)),
        )
        self.active_model = max(0, min(int(self.active_model), len(self.models) - 1))
        if not getattr(self.models[self.active_model], "enabled", True):
            for i, model in enumerate(self.models):
                if getattr(model, "enabled", True):
                    self.active_model = i
                    break
        self.review_model_index = max(0, min(int(self.review_model_index), len(self.models) - 1))
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

        legacy_order = [self.models[self.active_model]] + [
            model for index, model in enumerate(self.models) if index != self.active_model
        ]
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
        if "written_exam_model_id" not in self.model_fields_set:
            self.written_exam_model_id = next((
                model.id for model in self.models
                if model.id == self.assist_screenshot_model_id and written_exam_model_available(model)
            ), None)
        elif self.written_exam_model_id not in used_ids:
            self.written_exam_model_id = None
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

    def get_active_model(self) -> ModelConfig:
        idx = max(0, min(self.active_model, len(self.models) - 1))
        return self.models[idx]

    def get_review_model(self) -> ModelConfig:
        idx = max(0, min(self.review_model_index, len(self.models) - 1))
        return self.models[idx]


_config: Optional[AppConfig] = None
_config_lock = threading.RLock()


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


def update_config(
    updates: dict,
    *,
    model_origins: list[int | None] | None = None,
    model_snapshot: list[ModelConfig] | None = None,
) -> AppConfig:
    global _config
    with _config_lock:
        cfg = get_config()
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
            elif key == "system_prompts":
                data[key] = patch_system_prompts(data[key], value)
            elif key == "assist_generation":
                data[key].update(value)
            elif hasattr(cfg, key):
                data[key] = value
        if "models" in updates:
            model_rows = [
                model.model_dump() if isinstance(model, ModelConfig) else dict(model)
                for model in data["models"]
            ]
            for index, model in enumerate(model_rows):
                if not model.get("id"):
                    origin = model_origins[index] if model_origins is not None and index < len(model_origins) else None
                    model["id"] = cfg.models[origin].id if origin is not None else uuid4().hex
            data["models"] = model_rows
            by_id = {model["id"]: index for index, model in enumerate(model_rows)}
            for field in ("active_model", "review_model_index"):
                if field not in updates:
                    old_id = cfg.models[getattr(cfg, field)].id
                    data[field] = by_id.get(old_id, data[field])
        if "assist_model_ids" in updates or "assist_screenshot_model_id" in updates:
            catalog = {
                model.id if isinstance(model, ModelConfig) else model["id"]: model
                for model in data["models"]
            }
            if "assist_model_ids" in updates and any(model_id not in catalog for model_id in updates["assist_model_ids"]):
                raise ModelListChanged("答题模型目录已更新，请刷新后重试")
            selected_vision = updates.get("assist_screenshot_model_id")
            if selected_vision is not None:
                model = catalog.get(selected_vision)
                vision = model.supports_vision if isinstance(model, ModelConfig) else (model or {}).get("supports_vision")
                if not vision:
                    raise InvalidAssistModelSelection("截图答题请选择支持识图的模型")
        updated = AppConfig(**data)
        exam_model_id = updates.get("written_exam_model_id")
        if exam_model_id is not None:
            exam_model = next((model for model in updated.models if model.id == exam_model_id), None)
            if exam_model is None or not written_exam_model_available(exam_model):
                raise InvalidAssistModelSelection("笔试答题请选择已启用、已配置 API Key 且支持图片输入的模型")
        index = cfg.knowledge_analysis.model_index
        if "models" in updates and index is not None:
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
        _save_config(updated)
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
        models = get_config().models
        index = settings.model_index
        if index is not None:
            if index >= len(models) or fingerprint != model_health_fingerprint(models[index]):
                raise ModelListChanged("模型列表已更新，请重新选择分析模型")
            if not knowledge_model_available(models[index]):
                raise ValueError("请选择已启用且配置了 API Key 的分析模型")
            if settings.think_effort != "off" and not models[index].supports_think:
                raise ValueError("该模型未启用推理能力，请将推理强度设为关闭")
        return update_config({"knowledge_analysis": settings.model_dump()})


def _load_config() -> AppConfig:
    _custom_path = os.environ.get("IA_CONFIG_PATH")
    if not os.path.exists(CONFIG_FILE):
        if _custom_path:
            logger.warning("IA_CONFIG_PATH=%s does not exist, using defaults", _custom_path)
            return AppConfig()
        if os.path.exists(CONFIG_EXAMPLE):
            shutil.copy2(CONFIG_EXAMPLE, CONFIG_FILE)
            print("[Config] 已从 config.example.json 创建 config.json，请填入你的 API Key")
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.pop("asr_config_version", None) == _ASR_CONFIG_VERSION:
                data.pop("candidate_remote_stt_enabled", None)
            else:
                # In older releases, omitting this flag meant the candidate still used local ASR.
                data["candidate_remote_stt_enabled"] = data.get("candidate_remote_stt_enabled") is True
            return AppConfig(**data)
        except ValidationError as e:
            logger.warning("配置包含 %s 项无效设置，使用默认配置", e.error_count())
        except Exception as e:
            print(f"[Config] 配置文件解析失败: {e}，使用默认配置")
    return AppConfig()


def _save_config(cfg: AppConfig) -> None:
    target = Path(CONFIG_FILE)
    data = cfg.model_dump()
    data["asr_config_version"] = _ASR_CONFIG_VERSION
    data.pop("resume_text", None)
    temp_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as f:
            temp_path = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("无法清理配置临时文件")


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
