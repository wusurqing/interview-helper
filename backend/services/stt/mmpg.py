"""Single-owner RTC / MMPG recognition sessions for the local desktop client."""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hmac
import json
import logging
import re
import secrets
import struct
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from core.config import Mmpg
from core.logger import get_logger


MMPG_PATH = "/api/v2/mmpg"
MMPG_URL = "wss://interactai.volcvideo.com" + MMPG_PATH
TOKEN_TTL_SECONDS = 3600
HANDSHAKE_TIMEOUT = 20
RTC_RECOVERY_TIMEOUT = 10
RECOVERY_STABLE_SECONDS = 30
AUDIO_STALL_TIMEOUT = 2
DIAGNOSTIC_TIMEOUT = 60
BROWSER_UNDERRUN_WINDOW = 5
BROWSER_UNDERRUN_LIMIT = 3
PCM_BYTES_PER_SECOND = 16_000 * 4  # 16 kHz mono float32 PCM.
MAX_AUDIO_BYTES = PCM_BYTES_PER_SECOND
# The browser worklet can queue one second of PCM before RTC playback. Allow
# that queue and a short cloud event grace after the required silence is sent.
ROLE_DRAIN_PLAYBACK_GRACE_SECONDS = 1.24
_RTC_ID = re.compile(r"[a-zA-Z0-9_@.\-]{1,128}\Z")
_SINGLE_MIC_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
_RTC_FAULTS = {
    "rtc_error", "rtc_join_failed", "rtc_token_update_failed",
    "rtc_audio_unavailable", "rtc_token_expired", "rtc_token_invalid",
}
_RTC_STAGES = {"sdk_loading", "track_ready", "room_joining", "room_joined", "publishing", "token_updated"}
_RETRYABLE = {
    "rtc_error", "rtc_join_failed", "rtc_token_update_failed", "rtc_token_expired",
    "rtc_join_timeout", "rtc_recovery_timeout", "audio_overrun", "audio_stalled", "audio_underrun",
    "cloud_connection_failed", "cloud_connection_lost", "session_create_timeout",
}
_DIAGNOSTIC_EVENTS = {
    "user.audio.transcription.started": "speech_recognized",
    "user.audio.transcription.segment": "speech_paused",
    "user.conversation.canceled": "turn_cancelled",
}
_DEFAULTS = {
    "token_mode": "auto",
    "doubao_resource_id": "volc.seedasr.sauc.duration",
    "aivad": True,
    "silence_ms": 800,
    "twopass": True,
    "enable_prefetch": True,
    "expire_time_ms": None,
    "force_end": False,
}
# The application's root logger runs at DEBUG. The transport library logs
# handshake headers and WebSocket payloads at that level.
_WIRE_LOGGER = logging.Logger("mmpg.wire", logging.CRITICAL + 1)
_WIRE_LOGGER.addHandler(logging.NullHandler())
_log = get_logger("mmpg.diagnostics")


class MmpgError(ValueError):
    def __init__(self, code, fields=None):
        self.code = code if isinstance(code, str) and _SAFE_CODE.fullmatch(code) else "mmpg_error"
        self.fields = list(fields or ())
        super().__init__(self.code)


def valid_single_mic_id(value: object) -> bool:
    return isinstance(value, str) and _SINGLE_MIC_ID.fullmatch(value) is not None


def _expiry_timestamp(value):
    if value is None or value == "":
        return None
    try:
        if type(value) is int and 0 < value <= 0xFFFFFFFF:
            return value
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.timestamp()
    except (ValueError, OverflowError):
        pass
    raise MmpgError("config_invalid", ["rtc_token_expires_at"])


def _display_expiry(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else None


def _manual_expiry(config):
    supplied = _expiry_timestamp(config.get("rtc_token_expires_at"))
    token = config["rtc_token"]
    if not token.startswith("001"):
        return supplied  # The provider validates opaque or future token formats.
    try:
        # Version + 24-byte AppID + two uint16-prefixed fields (payload and MAC).
        # Room/user IDs and the five official privileges make a normal token
        # shorter than 600 characters. Bound input before decoding.
        if len(token) > 1024 or token[3:27] != config["rtc_app_id"]:
            raise ValueError
        contents = base64.b64decode(token[27:], validate=True)
        message_size = struct.unpack_from("<H", contents)[0]
        signature_size = struct.unpack_from("<H", contents, 2 + message_size)[0]
        if signature_size != 32 or len(contents) != 4 + message_size + signature_size:
            raise ValueError
        message = contents[2:2 + message_size]
        _, _, expires = struct.unpack_from("<III", message)
        room_size = struct.unpack_from("<H", message, 12)[0]
        user_size_offset = 14 + room_size
        user_size = struct.unpack_from("<H", message, user_size_offset)[0]
        user_start = user_size_offset + 2
        privileges_offset = user_start + user_size
        privilege_count = struct.unpack_from("<H", message, privileges_offset)[0]
        if len(message) != privileges_offset + 2 + privilege_count * 6:
            raise ValueError
        room = message[14:user_size_offset].decode("utf-8")
        user = message[user_start:privileges_offset].decode("utf-8")
        if room not in (config["rtc_room_id"], "*") or user != config["rtc_user_id"]:
            raise ValueError
        if supplied is not None and expires:
            return min(supplied, expires)
        return supplied or expires or None
    except (ValueError, binascii.Error, struct.error):
        raise MmpgError("rtc_token_invalid", ["rtc_token"]) from None


def refresh_manual_rtc_token(snapshot: Mmpg, current: Mmpg) -> Mmpg:
    """Keep the session's RTC identity and recognition options when renewing its Token."""
    if (
        snapshot.token_mode == current.token_mode == "manual"
        and snapshot.rtc_app_id == current.rtc_app_id
        and snapshot.rtc_room_id == current.rtc_room_id
        and snapshot.rtc_user_id == current.rtc_user_id
    ):
        return snapshot.model_copy(update={
            "rtc_token": current.rtc_token,
            "rtc_token_expires_at": current.rtc_token_expires_at,
        })
    return snapshot


def validate_config(config):
    """Return an independent, validated server-side settings snapshot."""
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    if not isinstance(config, dict):
        raise MmpgError("config_missing")
    config = {**_DEFAULTS, **copy.deepcopy(config)}
    mode = config["token_mode"]
    if mode not in ("auto", "manual"):
        raise MmpgError("config_invalid", ["token_mode"])
    required = [
        "volc_access_key", "volc_secret_key", "rtc_app_id", "rtc_room_id",
        "rtc_user_id", "rtc_bot_id", "doubao_api_key",
        "rtc_app_key" if mode == "auto" else "rtc_token",
    ]
    missing = []
    for key in required:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
        else:
            value = value.strip()
            if value.lower().startswith(("your_", "paste_", "replace_")) or "请填写" in value:
                missing.append(key)
            config[key] = value
    if missing:
        raise MmpgError("config_missing", missing)
    if not re.fullmatch(r"[A-Za-z0-9]{24}", config["rtc_app_id"]):
        raise MmpgError("config_invalid", ["rtc_app_id"])
    for key in ("rtc_room_id", "rtc_user_id", "rtc_bot_id"):
        if not _RTC_ID.fullmatch(config[key]):
            raise MmpgError("config_invalid", [key])
    if config["rtc_user_id"] == config["rtc_bot_id"]:
        raise MmpgError("config_invalid", ["rtc_bot_id"])
    if config["doubao_resource_id"] not in ("volc.seedasr.sauc.duration", "volc.seedasr.sauc.concurrent"):
        raise MmpgError("config_invalid", ["doubao_resource_id"])
    for key in ("aivad", "twopass", "enable_prefetch", "force_end"):
        if type(config[key]) is not bool:
            raise MmpgError("config_invalid", [key])
    if type(config["silence_ms"]) is not int or not 500 <= config["silence_ms"] < 3000:
        raise MmpgError("config_invalid", ["silence_ms"])
    expire_ms = config["expire_time_ms"]
    if expire_ms is not None and (type(expire_ms) is not int or expire_ms <= 0):
        raise MmpgError("config_invalid", ["expire_time_ms"])
    config["rtc_token_expires_at"] = _manual_expiry(config) if mode == "manual" else None
    if config["rtc_token_expires_at"] is not None and config["rtc_token_expires_at"] <= time.time():
        raise MmpgError("rtc_token_expired", ["rtc_token_expires_at"])
    return config


def sign_rtc_token(app_id, app_key, room_id, user_id, *, issued_at, expires_at, nonce=None):
    """Generate the official RTC AccessToken v001 wire format.

    Reference: python/src/AccessToken.py in RTC_Token.zip from
    https://docs.volcengine.com/docs/real-time-communication/token-authentication
    This publisher grants the official PrivPublishStream permission group.
    """
    def pack_bytes(value):
        return struct.pack("<H", len(value)) + value

    if nonce is None:
        nonce = secrets.randbelow(99_999_999) + 1
    privileges = (0, 1, 2, 3)  # Publish stream, audio, video and data.
    message = (
        struct.pack("<III", nonce, issued_at, expires_at)
        + pack_bytes(room_id.encode("utf-8"))
        + pack_bytes(user_id.encode("utf-8"))
        + struct.pack("<H", len(privileges))
        + b"".join(struct.pack("<HI", privilege, expires_at) for privilege in privileges)
    )
    signature = hmac.digest(app_key.encode("utf-8"), message, "sha256")
    return "001" + app_id + base64.b64encode(pack_bytes(message) + pack_bytes(signature)).decode("ascii")


def _event(kind, **fields):
    return {"type": kind, "event_id": uuid.uuid4().hex, **fields}


def build_session_create(config, room_id):
    vad = {
        "SilenceTime": config["silence_ms"],
        "AIVAD": config["aivad"],
        "ForceEnd": config["force_end"],
        "EnablePrefetch": config["aivad"] and config["enable_prefetch"],
    }
    if config["expire_time_ms"] is not None:
        vad["ExpireTime"] = config["expire_time_ms"]
    return _event("session.create", session={
        "AgentConfig": {
            "TransmitConfig": {
                "Type": "rtc", "AppId": config["rtc_app_id"],
                "RoomId": room_id, "UserId": config["rtc_bot_id"],
                "TargetUserId": [config["rtc_user_id"]],
            },
            "IdleTimeout": 180, "TurnDetectionMode": 0,
            "SubtitleConfig": {"DisableRTSSubtitle": True, "UserTranscriptMode": 1},
        },
        "AudioConfig": {
            "AudioFormat": {
                "UserAudioFormat": "pcm16", "AssistantAudioFormat": "pcm24", "AnsMode": 0,
            },
            "UserAudioCallbackMode": 0, "VADConfig": vad,
        },
        "ASRConfig": {"Provider": "volcano", "ProviderParams": {
            "Mode": "bigmodel", "StreamMode": 2,
            "Credential": {"ApiKey": config["doubao_api_key"], "ApiResourceId": config["doubao_resource_id"]},
            "VolcanoASRParameters": json.dumps({"request": {"enable_nonstream": config["twopass"]}}),
        }},
    })


async def connect_mmpg(config):
    try:
        from volcenginesdkcore.signv4 import SignerV4
        from websockets.asyncio.client import connect
    except ImportError:
        raise MmpgError("dependencies_missing") from None
    try:
        headers = {}
        SignerV4.sign(MMPG_PATH, "GET", headers, "", None, {},
                      config["volc_access_key"], config["volc_secret_key"], "cn-north-1", "rtc")
        return await connect(MMPG_URL, additional_headers=headers, open_timeout=12,
                             close_timeout=3, ping_interval=20, ping_timeout=20, max_size=1024 * 1024,
                             logger=_WIRE_LOGGER)
    except Exception as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) in (400, 401, 403):
            raise MmpgError("cloud_auth_failed") from None
        raise MmpgError("cloud_connection_failed") from None


@dataclass
class _Run:
    mode: str
    device_ids: list[int]
    candidate_mic_id: int | None
    single_mic_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stop_reason: str | None = None
    started: bool = False
    deadline: asyncio.TimerHandle | None = None

    def capture_options(self) -> dict[str, str]:
        return {"single_mic_id": self.single_mic_id} if self.single_mic_id is not None else {}


@dataclass
class _Generation:
    number: int
    config: dict = field(repr=False)
    room_id: str
    rtc_ready: asyncio.Event = field(default_factory=asyncio.Event)
    fault: asyncio.Event = field(default_factory=asyncio.Event)
    expiry_changed: asyncio.Event = field(default_factory=asyncio.Event)
    recovery_timer: asyncio.TimerHandle | None = None
    error: str | None = None
    expires_at: float | None = None
    warned_expiry: bool = False
    capture_entered: bool = False
    cleaning: bool = False
    cloud_ready: bool = False
    ready_since: float | None = None
    was_stable: bool = False
    muted: bool = False
    recovering: bool = False
    seen_done: set[str] = field(default_factory=set)
    cancelled: set[str] = field(default_factory=set)
    open_conversations: set[str] = field(default_factory=set)
    conversation_numbers: dict[str, int] = field(default_factory=dict)
    pause_received_at: dict[str, float] = field(default_factory=dict)
    recent_underruns: deque = field(default_factory=lambda: deque(maxlen=BROWSER_UNDERRUN_LIMIT))
    token_issued_at: int | None = None


class MmpgManager:
    """Owns one recognition run and the browser publisher feeding that run."""

    def __init__(self, config_getter, on_transcript, *, on_state=None,
                 on_capture_start=None, on_capture_stop=None, connector=connect_mmpg,
                 retry_delays=(1, 2, 4), test_timeout=DIAGNOSTIC_TIMEOUT, on_turn=None):
        self._get_config = config_getter
        self._on_transcript = on_transcript
        self._on_turn = on_turn
        self._on_state = on_state
        self._capture_start = on_capture_start
        self._capture_stop = on_capture_stop
        self._connector = connector
        self._retry_delays = retry_delays
        self._test_timeout = test_timeout
        self._owner = None
        self._send = None
        self._run = None
        self._task = None
        self._current = None
        self._generation = 0
        self._closed = False
        self._state = "idle"
        self._error = self._reason = self._mode = None
        self._browser_underruns = self._browser_overrun_samples = 0

        # The capture thread reserves capacity before scheduling a loop wakeup.
        # Include the PCM chunk currently in the WebSocket send in that limit.
        self._audio_lock = threading.Lock()
        self._audio = deque()
        self._queued_bytes = self._inflight_bytes = 0
        self._audio_dropped = 0
        self._audio_generation = None
        self._audio_gate_revision = 0
        self._audio_sent_bytes = 0
        self._last_audio = None
        self._audio_failed = False
        self._wake_pending = False
        self._audio_ready = None
        self._loop = None

    @property
    def active(self):
        return self._run is not None

    @property
    def single_mic_id(self) -> str | None:
        return self._run.single_mic_id if self._run is not None else None

    def status(self):
        current = self._current
        with self._audio_lock:
            dropped = self._audio_dropped
        return {
            "state": self._state, "error": self._error, "reason": self._reason,
            "generation": self._generation, "mode": self._mode,
            "token_expires_at": _display_expiry(current.expires_at) if current else None,
            "audio_dropped": dropped, "browser_underruns": self._browser_underruns,
            "browser_overrun_samples": self._browser_overrun_samples,
        }

    def attach(self, owner, send):
        if self._closed or self._owner is not None:
            return False
        self._owner, self._send = owner, send
        return True

    async def detach(self, owner):
        if owner is self._owner:
            self._send = None
            await self.stop("owner_disconnected")
            self._owner = None

    def start(self, owner, mode, device_ids, candidate_mic_id=None, *, preflight=None, single_mic_id=None):
        """Reserve the lifecycle immediately while the WS reads RTC acknowledgments."""
        if self._closed or owner is not self._owner:
            raise MmpgError("owner_unavailable")
        if self.active:
            raise MmpgError("mmpg_busy")
        if single_mic_id is not None and (mode == "test" or not valid_single_mic_id(single_mic_id)):
            raise MmpgError("invalid_command")
        self._loop = asyncio.get_running_loop()
        self._audio_ready = asyncio.Event()
        run = self._run = _Run(mode, list(device_ids), candidate_mic_id, single_mic_id)
        self._mode = mode
        self._browser_underruns = self._browser_overrun_samples = 0
        with self._audio_lock:
            self._audio_dropped = 0
        self._task = self._loop.create_task(self._run_session(run, preflight))
        if mode == "test":
            run.deadline = self._loop.call_later(
                self._test_timeout, lambda: self._loop.create_task(self._timeout_test(run)),
            )

    async def _timeout_test(self, run):
        if run is self._run:
            await self.stop("test_timeout")

    async def stop(self, reason="stopped"):
        run, task = self._run, self._task
        if run is None:
            return
        if run.stop_reason is None:
            run.stop_reason = reason
            if self._current is not None:
                self._interrupt_conversations(self._current)
            else:
                self._set_audio_gate(None)
            # A start immediately followed by stop can precede the task's first
            # scheduling turn. Let it enter its cleanup block before cancelling.
            if run.started and (self._current is None or not self._current.cleaning):
                task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.shield(task)

    async def close(self):
        self._closed = True
        await self.stop("shutdown")

    async def _emit(self, payload):
        send = self._send
        if send is None:
            raise MmpgError("owner_unavailable")
        try:
            await send(payload)
        except Exception:
            raise MmpgError("owner_unavailable") from None

    async def _publish(self, state, *, error=None, reason=None):
        self._state, self._error, self._reason = state, error, reason
        snapshot = self.status()
        if self._run is not None:
            _log.info(
                "session=%s generation=%s state=%s mode=%s error=%s reason=%s",
                self._run.trace_id, self._generation, state, self._mode, error, reason,
            )
        if self._on_state is not None:
            self._on_state(snapshot.copy())
        with suppress(MmpgError):
            await self._emit({"type": "state", **snapshot})

    def _set_audio_gate(self, generation):
        with self._audio_lock:
            self._audio_gate_revision += 1
            self._audio_generation = generation
            self._last_audio = time.monotonic() if generation is not None else None
            self._audio.clear()
            self._queued_bytes = 0
            self._audio_failed = False

    def offer_audio(self, samples):
        """Queue one 16 kHz mono chunk from a capture thread without blocking on I/O."""
        with self._audio_lock:
            generation = self._audio_generation
            gate_revision = self._audio_gate_revision
        if generation is None:
            return False
        if isinstance(samples, bytes):
            payload = samples
        else:
            array = np.asarray(samples)
            if array.ndim != 1:
                return False
            payload = array.astype("<f4", copy=False).tobytes()
        size = len(payload)
        if not size or size % 4:
            return False
        with self._audio_lock:
            if self._audio_gate_revision != gate_revision or self._audio_failed:
                return False
            if self._queued_bytes + self._inflight_bytes + size > MAX_AUDIO_BYTES:
                self._audio_dropped += (self._queued_bytes + size) // 4
                self._audio.clear()
                self._queued_bytes = 0
                self._audio_failed = True
            else:
                self._audio.append(payload)
                self._queued_bytes += size
                self._last_audio = time.monotonic()
            if not self._wake_pending:
                self._wake_pending = True
                self._loop.call_soon_threadsafe(self._wake_audio)
            return not self._audio_failed

    def _wake_audio(self):
        with self._audio_lock:
            self._wake_pending = False
        self._audio_ready.set()

    async def _send_audio(self, generation):
        while True:
            await self._audio_ready.wait()
            self._audio_ready.clear()
            while True:
                with self._audio_lock:
                    if self._audio_generation != generation.number:
                        break
                    if self._audio_failed:
                        raise MmpgError("audio_overrun")
                    if not self._audio:
                        break
                    payload = self._audio.popleft()
                    self._queued_bytes -= len(payload)
                    self._inflight_bytes = len(payload)
                try:
                    await self._emit(payload)
                    with self._audio_lock:
                        self._audio_sent_bytes += len(payload)
                finally:
                    with self._audio_lock:
                        self._inflight_bytes = 0

    async def _watch_audio(self, generation):
        """Fail a live generation if raw capture stops delivering any PCM."""
        while True:
            await asyncio.sleep(AUDIO_STALL_TIMEOUT / 2)
            with self._audio_lock:
                last_audio = self._last_audio if self._audio_generation == generation.number else None
            if last_audio is not None and time.monotonic() - last_audio >= AUDIO_STALL_TIMEOUT:
                raise MmpgError("audio_stalled")

    async def drain_transcripts(self):
        """Send silence after capture stops so the provider can finish the current turn."""
        generation = self._current
        if generation is None or self._state != "listening":
            return

        started = self._loop.time()
        deadline = started + 3
        grace = started + 0.24
        silence = bytes(640 * 4)  # 40 ms of 16 kHz float32 PCM.
        while self._current is generation and self._state == "listening":
            now = self._loop.time()
            if now >= deadline or (not generation.open_conversations and now >= grace):
                break
            if not self.offer_audio(silence):
                break
            await asyncio.sleep(0.04)

    async def drain_single_mic_input(self, session_id: str, *, timeout: float) -> bool:
        """Feed silence while a paused physical source has a final turn to finish."""
        if self.single_mic_id != session_id or self._state != "listening":
            return False
        drain = asyncio.create_task(self.finish_role_input(session_id, timeout=timeout))
        silence = bytes(640 * 4)  # 40 ms of 16 kHz float32 PCM.
        try:
            while not drain.done() and self.offer_audio(silence):
                await asyncio.wait({drain}, timeout=0.04)
            return drain.done() and drain.result()
        finally:
            if not drain.done():
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)

    async def finish_role_input(self, single_mic_id: str, *, timeout: float) -> bool:
        """Wait for routed silence and final ASR events without closing RTC."""
        run, generation = self._run, self._current
        if not single_mic_id or run is None or run.single_mic_id != single_mic_id or generation is None:
            return False
        with self._audio_lock:
            if self._audio_generation != generation.number:
                return False
            gate_revision = self._audio_gate_revision
            # The caller has already stopped feeding spoken PCM for this role.
            # Bytes after this fence are silence, even if more are still queued.
            pcm_fence = self._audio_sent_bytes + self._queued_bytes + self._inflight_bytes
        silence_bytes = generation.config["silence_ms"] * (PCM_BYTES_PER_SECOND // 1000)
        deadline = self._loop.time() + timeout
        silence_sent_at = None
        while (self._run is run and self._current is generation
               and run.stop_reason is None and not generation.fault.is_set()
               and self._state == "listening"):
            now = self._loop.time()
            if now >= deadline:
                break
            with self._audio_lock:
                if self._audio_gate_revision != gate_revision or self._audio_failed:
                    return False
                if silence_sent_at is None and self._audio_sent_bytes >= pcm_fence + silence_bytes:
                    silence_sent_at = now
            if (silence_sent_at is not None and now - silence_sent_at >= ROLE_DRAIN_PLAYBACK_GRACE_SECONDS
                    and not generation.open_conversations):
                return True
            await asyncio.sleep(min(0.02, deadline - now))
        return False

    def discard_turns(self):
        if self._current is not None:
            self._discard_conversations(self._current)

    def _discard_conversations(self, generation):
        if self._on_turn and self._run and self._run.mode != "test":
            self._on_turn("reset", "")
        generation.cancelled.update(generation.open_conversations)
        generation.open_conversations.clear()

    def _interrupt_conversations(self, generation):
        self._discard_conversations(generation)
        generation.recent_underruns.clear()
        self._set_audio_gate(None)

    async def set_muted(self, muted):
        generation = self._current
        if generation is None or generation.muted == muted:
            return
        generation.muted = muted
        if muted:
            self._interrupt_conversations(generation)
        if generation.cloud_ready and not generation.recovering and not generation.fault.is_set():
            if not muted:
                self._set_audio_gate(generation.number)
            await self._publish("muted" if muted else "listening")

    def _fault(self, generation, code):
        if generation is not self._current or generation.fault.is_set():
            return
        generation.error = code
        self._interrupt_conversations(generation)
        generation.fault.set()

    def _issue_token(self, generation):
        config = generation.config
        if config["token_mode"] == "manual":
            generation.expires_at = config["rtc_token_expires_at"]
            return config["rtc_token"]
        issued = int(time.time())
        generation.token_issued_at = issued
        generation.expires_at = issued + TOKEN_TTL_SECONDS
        generation.expiry_changed.set()
        return sign_rtc_token(
            config["rtc_app_id"], config["rtc_app_key"], generation.room_id, config["rtc_user_id"],
            issued_at=issued, expires_at=generation.expires_at,
        )

    async def rtc_event(self, owner, message):
        generation = self._current
        if (
            owner is not self._owner or generation is None or generation.cleaning or generation.fault.is_set()
            or self._run.stop_reason is not None
            or type(message.get("generation")) is not int or message["generation"] != generation.number
        ):
            return False
        kind = message["type"]
        if kind == "rtc_ready":
            generation.rtc_ready.set()
            _log.info("session=%s generation=%s event=rtc_ready", self._run.trace_id, generation.number)
        elif kind == "rtc_stage":
            stage = message.get("stage")
            if isinstance(stage, str) and stage in _RTC_STAGES:
                _log.info("session=%s generation=%s event=rtc_stage stage=%s",
                          self._run.trace_id, generation.number, stage)
        elif kind == "rtc_fault":
            code = message.get("code")
            self._fault(generation, code if isinstance(code, str) and code in _RTC_FAULTS else "rtc_error")
        elif kind == "rtc_reconnecting" and not generation.recovering:
            generation.recovering = True
            generation.ready_since = None
            self._interrupt_conversations(generation)
            generation.recovery_timer = self._loop.call_later(
                RTC_RECOVERY_TIMEOUT, self._fault, generation, "rtc_recovery_timeout",
            )
            await self._publish("reconnecting")
        elif kind == "rtc_recovered" and generation.recovering:
            generation.recovering = False
            generation.recovery_timer.cancel()
            generation.recovery_timer = None
            if generation.cloud_ready:
                generation.ready_since = self._loop.time()
                if not generation.muted:
                    self._set_audio_gate(generation.number)
                await self._publish("muted" if generation.muted else "listening")
        elif kind == "renew" and generation.config["token_mode"] == "auto" and generation.rtc_ready.is_set():
            if int(time.time()) - generation.token_issued_at >= 1:
                token = self._issue_token(generation)
                _log.info("session=%s generation=%s event=token_issued", self._run.trace_id, generation.number)
                await self._emit({
                    "type": "rtc_token", "generation": generation.number,
                    "token": token, "expires_at": _display_expiry(generation.expires_at),
                })
        elif kind == "renew" and generation.config["token_mode"] == "manual" and not generation.warned_expiry:
            generation.warned_expiry = True
            await self._emit({
                "type": "notice", "generation": generation.number, "code": "manual_token_renewal_required",
            })
        elif kind == "audio_status":
            self._record_audio_health(generation, message)
        return True

    def _record_audio_health(self, generation, message):
        def count(field):
            value = message.get(field)
            return value if type(value) is int and 0 <= value <= 1_000_000 else 0

        underruns = count("underruns")
        overrun_samples = count("overrun_samples")
        self._browser_underruns += underruns
        self._browser_overrun_samples += overrun_samples
        if not generation.cloud_ready or generation.recovering or generation.muted:
            return
        with self._audio_lock:
            accepting_audio = self._audio_generation == generation.number
        if not accepting_audio:
            return
        if underruns or overrun_samples:
            _log.info(
                "session=%s generation=%s event=audio_buffer underruns=%s overrun_samples=%s",
                self._run.trace_id, generation.number, underruns, overrun_samples,
            )
        if overrun_samples:
            self._fault(generation, "audio_overrun")
        elif underruns:
            now = self._loop.time()
            recent = generation.recent_underruns
            while recent and now - recent[0] > BROWSER_UNDERRUN_WINDOW:
                recent.popleft()
            recent.extend([now] * min(underruns, BROWSER_UNDERRUN_LIMIT))
            if len(recent) == BROWSER_UNDERRUN_LIMIT:
                self._fault(generation, "audio_underrun")

    async def _watch_expiry(self, generation):
        while generation.expires_at is not None:
            try:
                await asyncio.wait_for(generation.expiry_changed.wait(),
                                       max(0, generation.expires_at - time.time()))
            except asyncio.TimeoutError:
                raise MmpgError("rtc_token_expired") from None
            generation.expiry_changed.clear()
        await asyncio.Future()

    @staticmethod
    async def _get_cloud_event(cloud, timeout=None):
        try:
            payload = await asyncio.wait_for(cloud.recv(), timeout)
        except asyncio.TimeoutError:
            raise MmpgError("session_create_timeout") from None
        except Exception:
            raise MmpgError("cloud_connection_lost") from None
        try:
            event = json.loads(payload)
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError
        except (TypeError, ValueError, UnicodeError):
            raise MmpgError("cloud_protocol_error") from None
        if event["type"] == "error":
            raise MmpgError("cloud_error")
        if event["type"] == "session.closed":
            raise MmpgError("cloud_connection_lost")
        return event

    async def _read_cloud(self, run, generation, cloud):
        while True:
            event = await self._get_cloud_event(cloud)
            kind, conv_id = event["type"], event.get("conv_id")
            if not isinstance(conv_id, str) or not conv_id:
                continue
            with self._audio_lock:
                audio_overrun = self._audio_failed
            if (
                generation is not self._current or run.stop_reason or generation.fault.is_set()
                or generation.recovering or generation.muted or audio_overrun
            ):
                generation.cancelled.add(conv_id)
                generation.open_conversations.discard(conv_id)
                continue
            if conv_id in generation.cancelled or conv_id in generation.seen_done:
                continue
            if kind not in _DIAGNOSTIC_EVENTS and kind not in {
                "user.conversation.created", "user.audio.transcription.done",
            }:
                continue
            turn = generation.conversation_numbers.setdefault(conv_id, len(generation.conversation_numbers) + 1)
            trace = f"{run.trace_id}/{generation.number}/{turn}"
            diagnostic = _DIAGNOSTIC_EVENTS.get(kind)
            if kind == "user.conversation.canceled":
                generation.cancelled.add(conv_id)
                generation.open_conversations.discard(conv_id)
                generation.pause_received_at.pop(conv_id, None)
                if self._on_turn and run.mode != "test":
                    self._on_turn("cancelled", trace)
            if kind == "user.audio.transcription.segment":
                generation.pause_received_at[conv_id] = self._loop.time()
            if diagnostic or kind == "user.conversation.created":
                _log.info("turn=%s event=%s", trace, diagnostic or "speech_started")
            if run.mode == "test" and diagnostic:
                await self._emit({
                    "type": "diagnostic", "generation": generation.number, "event": diagnostic,
                })
            if kind in {"user.conversation.created", "user.audio.transcription.started",
                        "user.audio.transcription.segment"}:
                if self._on_turn and run.mode != "test" and kind != "user.audio.transcription.segment":
                    self._on_turn("started", trace)
                generation.open_conversations.add(conv_id)
            if kind != "user.audio.transcription.done":
                continue
            generation.seen_done.add(conv_id)
            generation.open_conversations.discard(conv_id)
            text = event.get("transcript")
            pause_at = generation.pause_received_at.pop(conv_id, None)
            _log.info(
                "turn=%s event=asr_done chars=%s since_last_pause_ms=%s",
                trace, len(text.strip()) if isinstance(text, str) else 0,
                round(1000 * (self._loop.time() - pause_at)) if pause_at is not None else None,
            )
            if not isinstance(text, str) or not text.strip():
                if self._on_turn and run.mode != "test":
                    self._on_turn("cancelled", trace)
                continue
            text = text.strip()
            for key in ("volc_access_key", "volc_secret_key", "rtc_app_key", "rtc_token", "doubao_api_key"):
                secret = generation.config.get(key)
                if isinstance(secret, str) and secret:
                    text = text.replace(secret, "[REDACTED]")
            if run.mode == "test":
                await self._emit({
                    "type": "transcript", "generation": generation.number,
                    "conv_id": conv_id, "transcript": text,
                })
            else:
                self._on_transcript(text, trace)

    async def _run_transport(self, run, generation, cloud):
        try:
            await asyncio.wait_for(generation.rtc_ready.wait(), HANDSHAKE_TIMEOUT)
        except asyncio.TimeoutError:
            raise MmpgError("rtc_join_timeout") from None
        try:
            cloud.append(await self._connector(generation.config))
            await asyncio.wait_for(
                cloud[0].send(json.dumps(build_session_create(generation.config, generation.room_id))),
                timeout=5,
            )
        except MmpgError:
            raise
        except Exception:
            raise MmpgError("cloud_connection_failed") from None
        deadline = self._loop.time() + HANDSHAKE_TIMEOUT
        while True:
            event = await self._get_cloud_event(cloud[0], max(0, deadline - self._loop.time()))
            if event["type"] == "session.created":
                _log.info("session=%s generation=%s event=cloud_ready", run.trace_id, generation.number)
                break
        if self._capture_start is not None:
            generation.capture_entered = True
            start = asyncio.create_task(self._capture_start(
                run.mode, run.device_ids, run.candidate_mic_id, **run.capture_options(),
            ))
            try:
                await asyncio.shield(start)
            except asyncio.CancelledError:
                # A device-open call running off-loop can still complete. Release
                # capture only after it settles to prevent a handle leaking.
                with suppress(Exception):
                    await start
                raise
            except MmpgError:
                raise
            except Exception:
                raise MmpgError("capture_failed") from None
        if run.stop_reason or generation.fault.is_set():
            return
        generation.cloud_ready = True
        if not generation.recovering:
            generation.ready_since = self._loop.time()
            await self._publish("muted" if generation.muted else "listening")
        if (not generation.muted and not generation.recovering
                and not generation.fault.is_set() and not run.stop_reason):
            self._set_audio_gate(generation.number)
        tasks = {
            asyncio.create_task(self._read_cloud(run, generation, cloud[0])),
            asyncio.create_task(self._send_audio(generation)),
            asyncio.create_task(self._watch_audio(generation)),
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _attempt(self, run, generation):
        config = generation.config
        cloud = []
        tasks = set()
        reason = "stopped"
        try:
            token = self._issue_token(generation)
            await self._emit({
                "type": "rtc_join", "generation": generation.number,
                "appId": config["rtc_app_id"], "roomId": generation.room_id,
                "userId": config["rtc_user_id"], "token": token,
                "expires_at": _display_expiry(generation.expires_at),
            })
            transport = asyncio.create_task(self._run_transport(run, generation, cloud))
            fault = asyncio.create_task(generation.fault.wait())
            expiry = asyncio.create_task(self._watch_expiry(generation))
            tasks = {transport, fault, expiry}
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if fault in done:
                raise MmpgError(generation.error)
            for task in done:
                task.result()
            raise MmpgError("cloud_connection_lost")
        except MmpgError as exc:
            reason = exc.code
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            reason = "mmpg_error"
            raise MmpgError(reason) from None
        finally:
            generation.cleaning = True
            generation.was_stable = (
                generation.ready_since is not None
                and self._loop.time() - generation.ready_since >= RECOVERY_STABLE_SECONDS
            )
            reason = run.stop_reason or reason
            _log.info(
                "session=%s generation=%s event=closed reason=%s audio_dropped=%s underruns=%s overrun_samples=%s",
                run.trace_id, generation.number, reason, self._audio_dropped,
                self._browser_underruns, self._browser_overrun_samples,
            )
            generation.cloud_ready = False
            self._interrupt_conversations(generation)
            if generation.recovery_timer is not None:
                generation.recovery_timer.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                with suppress(MmpgError):
                    await self._emit({"type": "rtc_leave", "generation": generation.number, "reason": reason})
                if generation.capture_entered and self._capture_stop is not None:
                    await self._capture_stop(run.mode, reason, **run.capture_options())
            finally:
                if cloud:
                    with suppress(Exception):
                        await asyncio.wait_for(cloud[0].send(json.dumps(_event("session.close"))), timeout=2)
                    with suppress(Exception):
                        await asyncio.wait_for(cloud[0].close(), timeout=4)
                self._current = None

    async def _run_session(self, run, preflight):
        error = None
        try:
            run.started = True
            if run.stop_reason:
                return
            await self._publish("connecting")
            if preflight is not None:
                await preflight(run.mode, run.device_ids, run.candidate_mic_id, **run.capture_options())
            config = validate_config(self._get_config())
            _log.info(
                "session=%s event=config device=%s token_mode=%s aivad=%s silence_ms=%s "
                "twopass=%s prefetch=%s force_end=%s expire_ms=%s",
                run.trace_id, run.device_ids[0], config["token_mode"], config["aivad"], config["silence_ms"],
                config["twopass"], config["enable_prefetch"], config["force_end"], config["expire_time_ms"],
            )
            retry_index = 0
            while not run.stop_reason:
                if (expiry := config["rtc_token_expires_at"]) is not None and expiry <= time.time():
                    raise MmpgError("rtc_token_expired")
                self._generation += 1
                room_id = config["rtc_room_id"]
                if config["token_mode"] == "auto":
                    room_id = f"{room_id[:95]}-{uuid.uuid4().hex}"
                generation = self._current = _Generation(self._generation, config, room_id)
                try:
                    await self._attempt(run, generation)
                except MmpgError as exc:
                    error = exc.code
                    if error not in _RETRYABLE or (error == "rtc_token_expired" and config["token_mode"] == "manual"):
                        break
                    if generation.was_stable:
                        retry_index = 0
                    if retry_index >= len(self._retry_delays):
                        break
                    _log.info("session=%s generation=%s event=retry attempt=%s delay_s=%s error=%s",
                              run.trace_id, self._generation, retry_index + 1, self._retry_delays[retry_index], error)
                    await self._publish("reconnecting", error=error)
                    await asyncio.sleep(self._retry_delays[retry_index])
                    retry_index += 1
        except asyncio.CancelledError:
            if run.stop_reason is None:
                raise
        except MmpgError as exc:
            error = exc.code
        except Exception:
            error = "mmpg_error"
        finally:
            if run.deadline is not None:
                run.deadline.cancel()
            try:
                if run.stop_reason:
                    await self._publish("idle", reason=run.stop_reason)
                else:
                    await self._publish("error", error=error or "mmpg_error")
            finally:
                self._run = self._task = None
