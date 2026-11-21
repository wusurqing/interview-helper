"""Segmented cloud ASR connections for Doubao and OpenAI-compatible services."""

from http.cookiejar import DefaultCookiePolicy
import io
import gzip
import json
import struct
import threading
import uuid
import wave
import numpy as np
from typing import Optional, Any

import requests

from core.logger import get_logger
from .text_utils import _postprocess

_log = get_logger("stt.engines")
_ORIG_REQUESTS_POST = requests.post
_stt_http_local = threading.local()


def _get_generic_stt_session() -> requests.Session:
    session = getattr(_stt_http_local, "session", None)
    if session is None:
        session = requests.Session()
        session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _stt_http_local.session = session
    else:
        session.cookies.clear()
    return session

try:
    import websocket
except ImportError:
    websocket = None


class STTError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Shared audio helpers
# ---------------------------------------------------------------------------

def _audio_to_pcm_int16(audio: np.ndarray) -> np.ndarray:
    if not audio.size:
        return np.empty(0, dtype=np.int16)
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        if audio.max() <= 1.0 and audio.min() >= -1.0:
            return (audio * 32767).astype(np.int16)
        return np.clip(audio, -32768, 32767).astype(np.int16)
    if audio.dtype != np.int16:
        return audio.astype(np.int16)
    return audio


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = _audio_to_pcm_int16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate or 16000))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _resample_pcm_int16(pcm: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    if not pcm.size or sample_rate <= 0 or sample_rate == target_rate:
        return pcm
    n_orig = len(pcm)
    if n_orig == 1:
        return pcm
    n_new = max(1, int(round(n_orig * target_rate / sample_rate)))
    samples = pcm.astype(np.float64)
    if sample_rate > target_rate and n_orig > 4:
        cutoff_norm = 0.47 * (target_rate / sample_rate)
        num_taps = min(63, n_orig if n_orig % 2 == 1 else max(3, n_orig - 1))
        n = np.arange(num_taps) - (num_taps - 1) / 2.0
        fir = 2.0 * cutoff_norm * np.sinc(2.0 * cutoff_norm * n) * np.blackman(num_taps)
        fir_sum = fir.sum()
        if fir_sum != 0:
            fir /= fir_sum
        samples = np.convolve(samples, fir, mode="same")
    x_old = np.arange(n_orig, dtype=np.float64)
    x_new = np.linspace(0.0, float(n_orig - 1), n_new)
    resampled = np.interp(x_new, x_old, samples)
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# DoubaoSTT – Volcengine streaming ASR
# ---------------------------------------------------------------------------

DOUBAO_ASR_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
MSG_FULL_CLIENT_REQUEST = 0x01
MSG_AUDIO_ONLY = 0x02
MSG_FULL_SERVER_RESPONSE = 0x09
MSG_ERROR = 0x0F
# 客户端帧 flags（byte1 低 4 位）：带序号=0b0001；最后一包音频再或上 0b0010（且序号取负）。
# 参考 issue #22 / 火山文档：服务端只在收到「负序号结束包」后才出结果；旧实现最后一包
# 不带负序号，服务端会干等约 8 秒（错误码 45000081）再断会话。
FLAG_SEQUENCE = 0x01
FLAG_LAST = 0x02
COMPRESSION_GZIP = 0x01
CHUNK_MS = 200
CHUNK_SAMPLES = 16000 * CHUNK_MS // 1000


def _doubao_language(language: str) -> str:
    """把配置中的语言码映射成火山原生豆包的 language 参数。"""
    lang = (language or "").strip().lower()
    if lang.startswith("zh"):
        return "zh-CN"
    if lang.startswith("en"):
        return "en-US"
    return "zh-CN"


def _build_ws_frame_full_request(
    uid: str,
    boosting_table_id: str,
    language: str = "zh-CN",
) -> bytes:
    req: dict[str, Any] = {
        "user": {"uid": (uid or "").strip() or "interview-helper"},
        "audio": {
            "format": "pcm",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
            "language": language if language and language != "auto" else "zh-CN",
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
        },
    }
    if boosting_table_id and boosting_table_id.strip():
        req["request"]["corpus"] = {"boosting_table_id": boosting_table_id.strip()}
    payload_raw = json.dumps(req, ensure_ascii=False).encode("utf-8")
    payload = gzip.compress(payload_raw)
    # header(4) + sequence(4, 全量请求固定 1) + payload_size(4) + payload
    header = bytes([0x11, 0x10 | FLAG_SEQUENCE, 0x11, 0x00])
    return header + struct.pack(">i", 1) + struct.pack(">I", len(payload)) + payload


def _build_ws_frame_audio(pcm_chunk: bytes, sequence: int, is_last: bool = False) -> bytes:
    payload = gzip.compress(pcm_chunk)
    flags = FLAG_SEQUENCE | (FLAG_LAST if is_last else 0)
    header = bytes([0x11, 0x20 | flags, 0x01, 0x00])
    # 最后一包序号取负，服务端据此识别音频结束并立即出结果
    return header + struct.pack(">i", -sequence if is_last else sequence) + struct.pack(">I", len(payload)) + payload


def _parse_ws_response(data: bytes) -> tuple[int, Optional[dict], bool]:
    """解析服务端帧；返回 (msg_type, payload, is_last)。

    带 sequence 的帧（flags 0b0001/0b0011）：header(4) + sequence(4) + size(4) + payload；
    不带 sequence 的帧：header(4) + size(4) + payload。
    服务端最后一包 flags 带 0b0010，收到后调用方应立即返回，不再等 recv 超时。
    """
    if len(data) < 4:
        return 0, None, False
    msg_type = (data[1] >> 4) & 0x0F
    flags = data[1] & 0x0F
    compression = data[2] & 0x0F
    is_last = bool(flags & FLAG_LAST)
    if msg_type == MSG_ERROR:
        return MSG_ERROR, None, False
    if msg_type == MSG_FULL_SERVER_RESPONSE:
        size_offset = 8 if flags & FLAG_SEQUENCE else 4
        if len(data) < size_offset + 4:
            return msg_type, None, is_last
        payload_size = struct.unpack(">I", data[size_offset : size_offset + 4])[0]
        if len(data) < size_offset + 4 + payload_size:
            return msg_type, None, is_last
        raw = data[size_offset + 4 : size_offset + 4 + payload_size]
        if compression == COMPRESSION_GZIP:
            try:
                raw = gzip.decompress(raw)
            except Exception:
                _log.warning("_parse_ws_response gzip decompress failed")
                return msg_type, None, is_last
        try:
            payload = json.loads(raw.decode("utf-8"))
            return msg_type, payload, is_last
        except Exception:
            _log.warning("_parse_ws_response invalid JSON at offset %d", len(raw))
            return msg_type, None, is_last
    return msg_type, None, is_last


class DoubaoSTT:
    """豆包流式语音识别模型，WebSocket 双流式。

    支持两种鉴权方式（自动选择）：
    - 新版控制台：X-Api-Key（api_key 非空时优先使用）
    - 旧版控制台：X-Api-App-Key + X-Api-Access-Key
    """

    def __init__(
        self,
        app_id: str = "",
        access_token: str = "",
        api_key: str = "",
        resource_id: str = "volc.seedasr.sauc.duration",
        boosting_table_id: str = "",
    ):
        self.app_id = app_id or ""
        self.access_token = access_token or ""
        self.api_key = api_key or ""
        self.resource_id = (resource_id or "").strip() or "volc.seedasr.sauc.duration"
        self.boosting_table_id = boosting_table_id or ""

    @property
    def model_size(self) -> str:
        return "doubao"

    @property
    def is_loaded(self) -> bool:
        return bool(self.api_key or (self.app_id and self.access_token))

    def _build_headers(self) -> dict:
        if self.api_key:
            return {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }
        return {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   position: str = "后端开发", language: str = "Python") -> str:
        if not self.is_loaded:
            raise STTError("请输入火山原生豆包的 API Key 或 App ID 与 Access Token")
        if websocket is None:
            raise STTError("火山原生豆包的 WebSocket 客户端不可用")
        user_uid = (self.app_id or "").strip() or "interview-helper"
        pcm = _audio_to_pcm_int16(audio)
        if sample_rate != 16000:
            pcm = _resample_pcm_int16(pcm, sample_rate, 16000)

        first_frame = _build_ws_frame_full_request(
            user_uid,
            self.boosting_table_id,
            language=_doubao_language(language),
        )
        n_samples = len(pcm)
        chunks = []
        offset = 0
        sequence = 1  # 全量请求占用序号 1，音频包从 2 开始递增
        while offset < n_samples:
            take = min(CHUNK_SAMPLES, n_samples - offset)
            chunk_bytes = pcm[offset : offset + take].tobytes()
            is_last = offset + take >= n_samples
            sequence += 1
            chunks.append(_build_ws_frame_audio(chunk_bytes, sequence, is_last=is_last))
            offset += take
        if not chunks:
            pad = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
            chunks = [_build_ws_frame_audio(pad.tobytes(), 2, is_last=True)]

        ws = websocket.create_connection(
            DOUBAO_ASR_WS_URL,
            header=[f"{k}: {v}" for k, v in self._build_headers().items()],
            timeout=10,
        )
        try:
            # 整段连发：不按 100-200ms 间隔空等，最后一包负序号即结束包
            ws.send_binary(first_frame)
            for frame in chunks:
                ws.send_binary(frame)
            final_text = ""
            ws.settimeout(5)
            first_packet = True
            while True:
                raw = ws.recv()
                if first_packet:
                    ws.settimeout(15)
                    first_packet = False
                if not raw:
                    raise ConnectionError("豆包 ASR 在最终结果前断开")
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                msg_type, payload, is_last = _parse_ws_response(raw)
                if msg_type == MSG_ERROR:
                    # error 帧也可能带 sequence（flags 0b0001），字段整体后移 4 字节
                    err_off = 8 if (len(raw) > 4 and raw[1] & FLAG_SEQUENCE) else 4
                    code = struct.unpack_from(">I", raw, err_off)[0] if len(raw) >= err_off + 4 else 0
                    raise STTError(f"火山原生豆包 ASR 返回错误（{code}）")
                if msg_type == MSG_FULL_SERVER_RESPONSE and payload:
                    res = payload.get("result") or {}
                    text = (res.get("text") or "").strip()
                    if text:
                        final_text = text
                if is_last:
                    # 服务端结束包（flags 0b0010/0b0011）：立即返回，不再空等到超时
                    if msg_type == MSG_FULL_SERVER_RESPONSE and payload is None:
                        raise STTError("火山原生豆包 ASR 返回了无效的最终结果")
                    break
            return _postprocess(final_text) if final_text else ""
        finally:
            ws.close()



# ---------------------------------------------------------------------------
# GenericHTTPSTT – OpenAI-compatible multipart ASR
# ---------------------------------------------------------------------------

class GenericHTTPSTT:
    """通用 HTTP ASR: POST {base_url}/audio/transcriptions with multipart file."""

    def __init__(self, api_base_url: str, api_key: str, model: str,
                 custom_headers: str = ""):
        self.api_base_url = (api_base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.custom_headers = custom_headers or ""

    @property
    def model_size(self) -> str:
        return "generic"

    @property
    def is_loaded(self) -> bool:
        return bool(self.api_base_url and self.api_key and self.model)

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return data.strip()
        if not isinstance(data, dict):
            return ""
        for key in ("text", "transcript", "result", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        result = data.get("result")
        if isinstance(result, dict):
            for key in ("text", "transcript", "content"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   position: str = "后端开发", language: str = "Python") -> str:
        if not self.api_base_url:
            raise RuntimeError("通用 ASR Base URL 未配置")
        if not self.api_key:
            raise RuntimeError("通用 ASR API Key 未配置")
        if not self.model:
            raise RuntimeError("通用 ASR Model 未配置")

        url = f"{self.api_base_url}/audio/transcriptions"
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=sample_rate)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.custom_headers.strip():
            try:
                extra = json.loads(self.custom_headers)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        key_clean = str(k).strip()
                        if key_clean and key_clean.lower() != "content-type":
                            headers[key_clean] = str(v)
            except (json.JSONDecodeError, ValueError):
                for line in self.custom_headers.strip().splitlines():
                    line = line.strip()
                    if ":" in line:
                        k, v = line.split(":", 1)
                        key_clean = k.strip()
                        if key_clean and key_clean.lower() != "content-type":
                            headers[key_clean] = v.strip()
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "response_format": "json"}
        post_fn = (
            requests.post
            if requests.post is not _ORIG_REQUESTS_POST
            else _get_generic_stt_session().post
        )
        try:
            resp = post_fn(url, headers=headers, files=files, data=data, timeout=(10, 60))
            if resp.status_code >= 400:
                status = resp.status_code
                raise STTError(
                    f"OpenAI 兼容 ASR 请求失败（HTTP {status}）",
                    retryable=status in (408, 429) or status >= 500,
                )
            ctype = resp.headers.get("content-type", "")
            if "json" in ctype.lower():
                payload = resp.json()
            else:
                try:
                    payload = resp.json()
                except Exception:
                    payload = resp.text
            text = self._extract_text(payload)
            return _postprocess(text) if text else ""
        except requests.exceptions.Timeout:
            raise STTError("OpenAI 兼容 ASR 请求超时", retryable=True) from None
        except requests.exceptions.ConnectionError:
            raise STTError("OpenAI 兼容 ASR 连接失败", retryable=True) from None
        except requests.exceptions.RequestException:
            raise STTError("OpenAI 兼容 ASR 请求失败") from None
