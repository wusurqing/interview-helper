from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.config import AppConfig
from services.stt.engines import GenericHTTPSTT, STTError
from services.stt import factory as stt_factory


def test_generic_http_stt_posts_openai_compatible_multipart(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        headers = {"content-type": "application/json"}

        def json(self):
            return {"text": "请介绍一下 Redis。"}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        captured.update({
            "url": url,
            "headers": headers,
            "files": files,
            "data": data,
            "timeout": timeout,
        })
        return _Resp()

    monkeypatch.setattr("services.stt.engines.requests.post", fake_post)

    engine = GenericHTTPSTT(
        api_base_url="https://asr.example.com/v1/",
        api_key="sk-test",
        model="asr-model",
    )
    audio = np.zeros(1600, dtype=np.float32)

    assert engine.transcribe(audio, sample_rate=16000) == "请介绍一下 Redis。"
    assert captured["url"] == "https://asr.example.com/v1/audio/transcriptions"
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}
    assert captured["data"] == {"model": "asr-model", "response_format": "json"}
    assert captured["files"]["file"][0] == "audio.wav"
    assert captured["files"]["file"][2] == "audio/wav"
    assert captured["files"]["file"][1].startswith(b"RIFF")


def test_generic_http_stt_extracts_nested_result_text(monkeypatch):
    class _Resp:
        status_code = 200
        headers = {"content-type": "application/json"}

        def json(self):
            return {"result": {"text": "你好。"}}

    monkeypatch.setattr("services.stt.engines.requests.post", lambda **_kwargs: _Resp())
    monkeypatch.setattr(
        "services.stt.engines.requests.post",
        lambda *args, **kwargs: _Resp(),
    )

    engine = GenericHTTPSTT("https://asr.example.com/v1", "sk", "m")
    assert engine.transcribe(np.zeros(100, dtype=np.float32), sample_rate=16000) == "你好。"


def configured_generic():
    return AppConfig(
        stt_provider="generic", generic_stt_api_base_url="https://asr.example/v1",
        generic_stt_api_key="test-key", generic_stt_model="test-model",
    )


def test_segment_retries_with_the_same_provider_and_session_configuration(monkeypatch):
    cfg = configured_generic()
    events = []

    def transcribe(_audio, _rate, **_context):
        events.append("attempt")
        if len(events) == 1:
            raise TimeoutError("request timed out")
        return "完整转写。"

    def engine(provider, *, config_snapshot):
        assert provider == "generic"
        assert config_snapshot is cfg
        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setattr(stt_factory, "get_stt_engine", engine)
    monkeypatch.setattr(stt_factory, "get_config", lambda: pytest.fail("Must use session snapshot"))
    assert stt_factory.transcribe_segment(np.zeros(16000), 16000, config_snapshot=cfg) == "完整转写。"
    assert events == ["attempt", "attempt"]


def test_auth_failure_does_not_retry_or_expose_provider_response(monkeypatch):
    cfg = configured_generic()
    attempts = []

    def post(*_args, **_kwargs):
        attempts.append("request")
        return SimpleNamespace(status_code=401, text="a private diagnostic")

    monkeypatch.setattr("services.stt.engines.requests.post", post)
    with pytest.raises(STTError, match="HTTP 401") as exc:
        stt_factory.transcribe_segment(np.zeros(16000), 16000, config_snapshot=cfg)
    assert "private" not in str(exc.value)
    assert attempts == ["request"]


def test_failed_session_does_not_retry_after_losing_capture_ownership(monkeypatch):
    cfg = configured_generic()
    active = True
    attempts = []

    def transcribe(*_args, **_kwargs):
        nonlocal active
        attempts.append("request")
        active = False
        raise ConnectionError("interrupted")

    monkeypatch.setattr(stt_factory, "get_stt_engine", lambda *_args, **_options: SimpleNamespace(transcribe=transcribe))
    assert stt_factory.transcribe_segment(
        np.zeros(16000), 16000, config_snapshot=cfg, is_current=lambda: active,
    ) == ""
    assert attempts == ["request"]


@pytest.mark.parametrize(("scope", "expected"), [
    ("interviewer", [16000, 19840]),
    ("candidate", [16000]),
])
def test_only_short_interviewer_empty_segment_is_retried_with_silence(monkeypatch, scope, expected):
    cfg = configured_generic()
    lengths = []

    def transcribe(audio, *_args, **_kwargs):
        lengths.append(len(audio))
        return ""

    monkeypatch.setattr(stt_factory, "get_stt_engine", lambda *_args, **_options: SimpleNamespace(transcribe=transcribe))
    assert stt_factory.transcribe_segment(np.zeros(16000), 16000, config_snapshot=cfg, scope=scope) == ""
    assert lengths == expected
