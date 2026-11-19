"""Configure and call the selected segmented ASR service."""

from collections.abc import Callable

import numpy as np
from websocket import WebSocketBadStatusException, WebSocketConnectionClosedException, WebSocketTimeoutException

from core.config import MMPG_STT_PROVIDER, SEGMENT_STT_PROVIDERS, STT_PROVIDER_OPTIONS, get_config
from core.logger import get_logger
from .engines import DoubaoSTT, GenericHTTPSTT, STTError
from .mmpg import MmpgError, validate_config

_log = get_logger("stt.factory")


def stt_configuration_error(config, provider: str) -> str:
    if provider == "doubao":
        if getattr(config, "doubao_stt_api_key", "").strip():
            return ""
        if getattr(config, "doubao_stt_app_id", "").strip() and getattr(config, "doubao_stt_access_token", "").strip():
            return ""
        return "请输入火山原生豆包的 API Key 或 App ID 与 Access Token"
    if provider == "generic":
        if all(getattr(config, f"generic_stt_{field}", "").strip() for field in ("api_base_url", "api_key", "model")):
            return ""
        return "请输入 OpenAI 兼容 ASR 的 Base URL、API Key 和模型名"
    if provider == MMPG_STT_PROVIDER:
        try:
            validate_config(getattr(config, "mmpg", None))
        except MmpgError:
            return "请检查豆包 MMPG 的引擎配置和 Token"
        return ""
    return "请在实时辅助中选择 ASR 引擎"


def configured_stt_routes(config) -> dict[str, bool]:
    return {provider: not stt_configuration_error(config, provider) for provider in STT_PROVIDER_OPTIONS}


def get_stt_engine(
    provider: str | None = None, *, config_snapshot=None,
) -> DoubaoSTT | GenericHTTPSTT:
    cfg = get_config() if config_snapshot is None else config_snapshot
    provider = cfg.stt_provider if provider is None else provider
    if provider == "doubao":
        return DoubaoSTT(
            app_id=cfg.doubao_stt_app_id.strip(),
            access_token=cfg.doubao_stt_access_token.strip(),
            api_key=cfg.doubao_stt_api_key.strip(),
            resource_id=cfg.doubao_stt_resource_id,
            boosting_table_id=cfg.doubao_stt_boosting_table_id,
        )
    if provider == "generic":
        return GenericHTTPSTT(
            api_base_url=cfg.generic_stt_api_base_url.strip(),
            api_key=cfg.generic_stt_api_key.strip(),
            model=cfg.generic_stt_model.strip(),
            custom_headers=cfg.generic_stt_custom_headers,
        )
    raise STTError("请选择火山原生豆包或 OpenAI 兼容 ASR")


def _provider_error(exc: Exception) -> STTError:
    if isinstance(exc, STTError):
        return exc
    if isinstance(exc, WebSocketBadStatusException):
        status = exc.status_code
        return STTError(
            f"火山原生豆包 ASR 连接失败（HTTP {status}）",
            retryable=status in (408, 429) or (isinstance(status, int) and status >= 500),
        )
    if isinstance(exc, (TimeoutError, WebSocketTimeoutException)):
        return STTError("ASR 请求超时", retryable=True)
    if isinstance(exc, (OSError, WebSocketConnectionClosedException)):
        return STTError("ASR 连接中断，请检查网络", retryable=True)
    return STTError("ASR 请求失败，请检查引擎配置")


def transcribe_segment(
    audio: np.ndarray,
    sample_rate: int,
    *,
    position: str = "后端开发",
    language: str = "Python",
    provider: str | None = None,
    scope: str = "interviewer",
    config_snapshot=None,
    is_current: Callable[[], bool] = lambda: True,
) -> str:
    """Retry once within the same provider, with an ownership check before sending audio."""
    cfg = get_config() if config_snapshot is None else config_snapshot
    provider = cfg.stt_provider if provider is None else provider
    if provider not in SEGMENT_STT_PROVIDERS:
        raise STTError("请选择火山原生豆包或 OpenAI 兼容 ASR")
    if detail := stt_configuration_error(cfg, provider):
        raise STTError(detail)
    engine = get_stt_engine(provider, config_snapshot=cfg)
    samples = audio
    for attempt in range(2):
        if not is_current():
            return ""
        try:
            text = engine.transcribe(samples, sample_rate, position=position, language=language)
        except Exception as exc:
            failure = _provider_error(exc)
            if not is_current():
                return ""
            if attempt or not failure.retryable:
                raise failure from None
            _log.info("Retrying ASR provider=%s scope=%s cause=%s", provider, scope, type(exc).__name__)
        else:
            if not is_current():
                return ""
            if text or attempt or scope != "interviewer" or not 0 < len(audio) <= 6 * sample_rate:
                return text or ""
            # Short utterances sometimes need trailing silence for the provider to finish the turn.
            samples = np.concatenate((audio, np.zeros(int(sample_rate * 0.24), dtype=audio.dtype)))
    return ""
