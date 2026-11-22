from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.audio import VADBuffer
from services.audio import AudioCapture
from services.audio import _audio_device_sort_key


def test_vad_flushes_when_max_speech_duration_reached():
    vad = VADBuffer(
        sample_rate=10,
        silence_threshold=0.01,
        silence_duration=99.0,
        min_speech_duration=0.1,
        max_speech_duration=0.5,
    )

    result = vad.feed(np.ones(5, dtype=np.float32) * 0.1)

    assert result is not None
    assert len(result) == 5


def test_vad_rollover_preserves_tail_after_max_speech_flush():
    vad = VADBuffer(
        sample_rate=10,
        silence_threshold=0.01,
        silence_duration=99.0,
        min_speech_duration=0.1,
        max_speech_duration=0.5,
        rollover_duration=0.2,
    )

    first = vad.feed(np.ones(5, dtype=np.float32) * 0.1)
    assert first is not None
    assert len(first) == 5

    vad.feed(np.ones(1, dtype=np.float32) * 0.1)
    pending = vad.pending_audio()

    assert pending is not None
    assert len(pending) == 3


def test_vad_min_speech_duration_drops_short_segment():
    """短于 min_speech_duration 的片段应被丢弃; 长于则保留。

    注意: speech_duration 把尾部静音 padding 也计入 (VADBuffer 既有行为),
    所以这里用 (语音+静音) 总采样数控制阈值两侧。
    """
    # sample_rate=10; min_speech_duration=0.5 -> 阈值 5 采样 (含尾部静音)
    vad = VADBuffer(
        sample_rate=10,
        silence_threshold=0.01,
        silence_duration=0.0,  # 第二静音帧即触发判定
        min_speech_duration=0.5,
    )

    # 2 语音 + 2 静音 = 4 采样 (0.4s < 0.5) -> 丢弃
    vad.feed(np.ones(2, dtype=np.float32) * 0.1)          # 语音帧
    vad.feed(np.zeros(1, dtype=np.float32))               # 静音帧1: 起计时
    dropped = vad.feed(np.zeros(1, dtype=np.float32))     # 静音帧2: 判定 -> 丢弃
    assert dropped is None

    # 4 语音 + 2 静音 = 6 采样 (0.6s >= 0.5) -> 保留
    vad.feed(np.ones(4, dtype=np.float32) * 0.1)          # 语音帧
    vad.feed(np.zeros(1, dtype=np.float32))               # 静音帧1
    kept = vad.feed(np.zeros(1, dtype=np.float32))        # 静音帧2: 判定 -> 保留
    assert kept is not None
    assert len(kept) == 6


def test_vad_preroll_preserves_leading_audio_before_voice_threshold():
    vad = VADBuffer(
        sample_rate=10,
        silence_threshold=0.01,
        silence_duration=0.0,
        min_speech_duration=0.2,
        preroll_duration=0.2,
    )

    vad.feed(np.ones(1, dtype=np.float32) * 0.005)       # 预卷帘静音/低声
    vad.feed(np.ones(2, dtype=np.float32) * 0.1)         # 开始说话
    vad.feed(np.zeros(1, dtype=np.float32))              # 触发静音
    kept = vad.feed(np.zeros(1, dtype=np.float32))       # flush

    assert kept is not None
    assert len(kept) == 5


def test_vad_silence_duration_uses_audio_samples_not_wall_clock():
    vad = VADBuffer(
        sample_rate=10,
        silence_threshold=0.01,
        silence_duration=0.2,
        min_speech_duration=0.2,
    )

    vad.feed(np.ones(2, dtype=np.float32) * 0.1)
    assert vad.feed(np.zeros(1, dtype=np.float32)) is None
    kept = vad.feed(np.zeros(1, dtype=np.float32))

    assert kept is not None
    assert len(kept) == 4


def test_audio_capture_push_splits_large_chunk_before_queue():
    capture = AudioCapture()
    large = np.ones(capture.MAX_QUEUE_CHUNK_SAMPLES * 2 + 100, dtype=np.float32) * 0.1

    capture._push(large)

    chunks = capture.drain_audio_chunks(timeout=0.0, max_chunks=10)
    assert len(chunks) == 3
    assert max(len(chunk) for chunk in chunks) == capture.MAX_QUEUE_CHUNK_SAMPLES
    assert sum(len(chunk) for chunk in chunks) == len(large)


def test_audio_capture_stats_snapshot_reports_queue_chunk_and_discontinuity():
    capture = AudioCapture()
    capture._loopback_discontinuity_count = 3
    capture._push(np.ones(1280, dtype=np.float32) * 0.1)

    stats = capture.capture_stats_snapshot()

    assert stats["loopback_discontinuity_count"] == 3
    assert stats["max_queue_chunk_samples"] == 1280
    assert stats["last_queue_chunk_samples"] == 1280


def test_audio_capture_loopback_min_chunk_samples_is_smaller_than_max():
    capture = AudioCapture()

    assert capture.MIN_LOOPBACK_QUEUE_CHUNK_SAMPLES < capture.MAX_QUEUE_CHUNK_SAMPLES


def test_audio_device_sort_keeps_default_loopback_first():
    devices = [
        {"id": 1, "name": "Alpha Mic", "is_loopback": False},
        {"id": 20001, "name": "ZZZ Speakers (系统音频)", "is_loopback": True, "is_default_output": True},
        {"id": 20000, "name": "AAA Monitor (系统音频)", "is_loopback": True, "is_default_output": False},
    ]

    ordered = sorted(devices, key=_audio_device_sort_key)

    assert [item["id"] for item in ordered] == [20001, 20000, 1]

def test_windows_microphone_initializes_com_on_stream_open_and_close(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from services import audio

    operations = []
    ole = SimpleNamespace(
        CoInitializeEx=lambda *_: operations.append("init") or 0,
        CoUninitialize=lambda: operations.append("uninit"),
    )
    stream = SimpleNamespace(
        start=lambda: operations.append("start"),
        stop=lambda: operations.append("stop"),
        close=lambda: operations.append("close"),
    )
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(ole32=ole), raising=False)
    monkeypatch.setattr(audio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(audio.sd, "query_devices", lambda *_: {"default_samplerate": 48000, "hostapi": 0})
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda *_: {"name": "Windows WASAPI"})
    monkeypatch.setattr(audio.sd, "InputStream", lambda **_: stream)

    capture = AudioCapture()
    capture.start(9, owner="assist")
    capture.stop(owner="assist")

    assert operations == ["init", "start", "uninit", "init", "stop", "close", "uninit"]


def _patch_fake_sounddevice(monkeypatch, *, fail_devices: set[int] | None = None):
    from types import SimpleNamespace
    from services import audio

    fail = fail_devices or set()
    monkeypatch.setattr(audio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audio.sd,
        "query_devices",
        lambda dev=None, *_args: {"default_samplerate": 16000, "hostapi": 0},
    )
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda *_: {"name": "ALSA"})

    def make_stream(**kwargs):
        if kwargs["device"] in fail:
            raise OSError(f"device {kwargs['device']} failed")
        return SimpleNamespace(start=lambda: None, stop=lambda: None, close=lambda: None)

    monkeypatch.setattr(audio.sd, "InputStream", make_stream)


def test_microphone_compatibility_falls_back_to_default_input_when_selected_stream_fails(monkeypatch):
    from types import SimpleNamespace
    from services import audio

    _patch_fake_sounddevice(monkeypatch, fail_devices={9})
    # Like sounddevice.default.device this pair supports [0] and [1] but is
    # neither a list nor a tuple. The output endpoint must never be opened.
    pair = range(3, 5)
    monkeypatch.setattr(audio.sd, "default", SimpleNamespace(device=pair))

    def query(device=None, kind=None):
        if device is None:
            assert kind == "input"
            device = pair[0]
        return {"index": device, "default_samplerate": 48000, "hostapi": 0}

    monkeypatch.setattr(audio.sd, "query_devices", query)
    make_stream = audio.sd.InputStream
    opened = []

    def observe(**kwargs):
        opened.append((kwargs["device"], kwargs["samplerate"]))
        return make_stream(**kwargs)

    monkeypatch.setattr(audio.sd, "InputStream", observe)
    capture = AudioCapture()
    try:
        capture.start(9, owner="fallback-test", mic_compatibility_mode=True)
        assert opened == [(9, 48000), (9, 16000), (3, 48000)]
        assert AudioCapture.active_device_owner(3) == "fallback-test"
    finally:
        capture.stop(owner="fallback-test")
    assert not AudioCapture.is_device_busy(3)


def test_global_device_lock_blocks_separate_audio_capture_instances_on_same_device(monkeypatch):
    import pytest
    from services.audio import AudioBusyError

    _patch_fake_sounddevice(monkeypatch)
    first = AudioCapture()
    second = AudioCapture()
    other_dev = AudioCapture()

    first.start(7, owner="assist")
    try:
        assert AudioCapture.is_device_busy(7) is True
        assert AudioCapture.active_device_owner(7) == "assist"

        # Different device on a second instance must still succeed (e.g. loopback + candidate mic).
        other_dev.start(8, owner="assist-candidate")
        assert AudioCapture.is_device_busy(8) is True

        # Same device on another instance must raise AudioBusyError even with same or different owner.
        with pytest.raises(AudioBusyError) as exc:
            second.start(7, owner="audio-test")
        assert exc.value.current_owner == "assist"
        assert exc.value.requested_owner == "audio-test"

        # Calling stop() on the failed second instance must NOT release first's claim.
        second.stop(owner="audio-test")
        assert AudioCapture.is_device_busy(7) is True
        assert AudioCapture.active_device_owner(7) == "assist"
    finally:
        other_dev.stop(owner="assist-candidate")
        first.stop(owner="assist")

    assert AudioCapture.is_device_busy(7) is False
    assert AudioCapture.is_device_busy(8) is False


def test_global_device_lock_releases_on_startup_failure(monkeypatch):
    import pytest

    _patch_fake_sounddevice(monkeypatch, fail_devices={11})
    cap1 = AudioCapture()
    with pytest.raises(RuntimeError):
        cap1.start(11, owner="audio-test")

    assert AudioCapture.is_device_busy(11) is False
    assert cap1.is_running is False
    assert cap1.current_owner is None

    # Now allow device 11 to succeed and verify another instance can acquire it.
    _patch_fake_sounddevice(monkeypatch)
    cap2 = AudioCapture()
    cap2.start(11, owner="assist")
    try:
        assert AudioCapture.is_device_busy(11) is True
    finally:
        cap2.stop(owner="assist")
    assert AudioCapture.is_device_busy(11) is False


def test_sound_test_local_captures_raise_audio_busy_and_routes_map_to_409(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import pytest
    from fastapi import HTTPException
    from api.assist import routes, sound_test
    from services.audio import AudioBusyError

    _patch_fake_sounddevice(monkeypatch)
    monkeypatch.setattr(routes, "get_session", lambda: SimpleNamespace(is_recording=False, is_paused=False))
    monkeypatch.setattr(
        routes,
        "get_config",
        lambda: SimpleNamespace(stt_provider="doubao", doubao_stt_api_key="test-key"),
    )

    holder = AudioCapture()
    holder.start(12, owner="assist")
    try:
        with pytest.raises(AudioBusyError) as input_busy:
            sound_test.test_input_audio(12, duration_sec=0.5)
        assert input_busy.value.current_owner == "assist"

        with pytest.raises(AudioBusyError) as monitor_busy:
            sound_test.start_input_level_monitor(12)
        assert monitor_busy.value.current_owner == "assist"

        with pytest.raises(AudioBusyError) as replay_busy:
            sound_test.replay_preflight_capture_stt(12, repeats=1)
        assert replay_busy.value.current_owner == "assist"

        with pytest.raises(HTTPException) as route_input:
            asyncio.run(routes.api_audio_test_input({"device_id": 12}))
        assert route_input.value.status_code == 409

        with pytest.raises(HTTPException) as route_monitor:
            asyncio.run(routes.api_audio_test_input_start({"device_id": 12}))
        assert route_monitor.value.status_code == 409

        with pytest.raises(HTTPException) as route_replay:
            asyncio.run(routes.api_preflight_replay({"device_id": 12, "repeats": 1}))
        assert route_replay.value.status_code == 409
    finally:
        holder.stop(owner="assist")
    assert AudioCapture.is_device_busy(12) is False


def test_soundcard_reader_crash_clears_owner_and_stale_token_cannot_revoke_restarted_claim(monkeypatch):
    import threading
    from services import audio

    _patch_fake_sounddevice(monkeypatch)
    cap = AudioCapture()
    cap.start(15, owner="assist")
    try:
        old_token = cap._sc_stop
        cap.stop(owner="assist")
        assert AudioCapture.is_device_busy(15) is False

        # Restart same instance on device 15 -> gets a fresh token.
        cap.start(15, owner="assist-v2")
        assert cap._sc_stop is not old_token
        # Simulated delayed release from an older reader token must NOT revoke the active claim.
        cap._release_devices({15}, token=old_token)
        assert AudioCapture.is_device_busy(15) is True
        assert AudioCapture.active_device_owner(15) == "assist-v2"
    finally:
        cap.stop(owner="assist-v2")
    assert AudioCapture.is_device_busy(15) is False

    # Also exercise actual soundcard _reader crash path releasing _ACTIVE_DEVICES and clearing owner.
    crashed = threading.Event()
    monkeypatch.setitem(audio._SC_ID_MAP, 20099, "fake-guid")

    class _BoomSC:
        @staticmethod
        def get_microphone(_guid, include_loopback=True):
            crashed.set()
            raise RuntimeError("WASAPI device disconnected")

    monkeypatch.setattr(audio, "_sc", _BoomSC, raising=False)
    sc_cap = AudioCapture()
    sc_cap.start(20099, owner="assist")
    assert crashed.wait(timeout=2.0)
    sc_cap._sc_thread.join(timeout=2.0)
    assert sc_cap.is_running is False
    assert sc_cap.current_owner is None
    assert AudioCapture.is_device_busy(20099) is False


def test_concurrent_input_level_monitor_start_serializes_and_never_orphans_device(monkeypatch):
    import threading
    from api.assist import sound_test

    _patch_fake_sounddevice(monkeypatch)
    orig_start = AudioCapture.start
    entered_first = threading.Event()
    release_first = threading.Event()

    def slow_start(self, dev_id, *args, **kwargs):
        orig_start(self, dev_id, *args, **kwargs)
        if dev_id == 41:
            entered_first.set()
            release_first.wait(timeout=2.0)

    monkeypatch.setattr(AudioCapture, "start", slow_start)

    t1 = threading.Thread(target=sound_test.start_input_level_monitor, args=(41,), daemon=True)
    t2 = threading.Thread(target=sound_test.start_input_level_monitor, args=(42,), daemon=True)
    t1.start()
    assert entered_first.wait(timeout=2.0)
    t2.start()
    release_first.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    sound_test.stop_input_level_monitor()
    assert AudioCapture.is_device_busy(41) is False
    assert AudioCapture.is_device_busy(42) is False
    assert sound_test.get_input_level_status()["running"] is False


def test_soundcard_hot_path_never_calls_catch_warnings(monkeypatch):
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace
    from services import audio

    recorded_chunks: list[np.ndarray] = []
    enough = threading.Event()
    calls = {"count": 0}

    def forbid_catch_warnings(*_args, **_kwargs):
        raise AssertionError("warnings.catch_warnings must not be called on 64ms WASAPI hot path")

    class _FakeRecorder:
        def record(self, numframes: int):
            calls["count"] += 1
            if calls["count"] >= 4:
                enough.set()
            return np.ones((numframes, 1), dtype=np.float32) * 0.05

    class _FakeMic:
        name = "Fake Loopback"

        @contextmanager
        def recorder(self, **_kwargs):
            yield _FakeRecorder()

    class _FakeSC:
        @staticmethod
        def get_microphone(_guid, include_loopback=True):
            return _FakeMic()

    monkeypatch.setattr(audio.warnings, "catch_warnings", forbid_catch_warnings)
    monkeypatch.setattr(audio, "_sc", _FakeSC, raising=False)
    monkeypatch.setitem(audio._SC_ID_MAP, 20088, "guid-88")

    cap = AudioCapture()
    cap.start(20088, on_audio=recorded_chunks.append, owner="assist")
    try:
        assert enough.wait(timeout=2.0)
    finally:
        cap.stop(owner="assist")

    assert calls["count"] >= 4
    assert len(recorded_chunks) >= 1


def test_soundcard_device_cache_concurrent_read_write_is_thread_safe(monkeypatch):
    import threading
    from types import SimpleNamespace
    from services import audio

    speakers = [
        SimpleNamespace(id=f"guid-{i}", name=f"Speaker {i}", channels=2)
        for i in range(6)
    ]

    class _FakeSC:
        @staticmethod
        def default_speaker():
            return speakers[0]

        @staticmethod
        def all_speakers():
            return list(speakers)

    monkeypatch.setattr(audio, "_HAS_SC", True)
    monkeypatch.setattr(audio, "_sc", _FakeSC, raising=False)
    monkeypatch.setattr(audio, "_SC_DEVICE_CACHE", {})
    monkeypatch.setattr(audio, "_SC_ID_MAP", {})

    errors: list[Exception] = []
    barrier = threading.Barrier(6)

    def writer():
        try:
            barrier.wait(timeout=5.0)
            for _ in range(40):
                devs = audio._get_sc_loopback_devices()
                assert len(devs) == 6
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            barrier.wait(timeout=5.0)
            for _ in range(80):
                for idx in range(6):
                    guid = audio._lookup_sc_speaker_guid(audio._SC_ID_BASE + idx)
                    assert guid is None or guid.startswith("guid-")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(3)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    for idx in range(6):
        assert audio._lookup_sc_speaker_guid(audio._SC_ID_BASE + idx) is not None


def test_loopback_device_enumeration_preserves_speaker_channels(monkeypatch):
    from types import SimpleNamespace
    from services import audio

    speaker = SimpleNamespace(name="Test speaker", id="loopback-test-id", channels=6)
    monkeypatch.setattr(audio, "_HAS_SC", True)
    monkeypatch.setattr(audio, "_sc", SimpleNamespace(
        default_speaker=lambda: speaker, all_speakers=lambda: [speaker],
    ), raising=False)
    monkeypatch.setattr(audio, "_SC_DEVICE_CACHE", {})
    monkeypatch.setattr(audio, "_SC_ID_MAP", {})
    monkeypatch.setattr(audio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(audio.sd, "query_devices", lambda: [])

    [device] = AudioCapture.list_devices()
    assert device["is_default_output"] and device["is_loopback"]
    assert device["channels"] == 6 and device["sample_rate"] == 16000
    assert audio._lookup_sc_speaker_guid(device["id"]) == speaker.id


def test_microphone_compatibility_ignores_broken_default_input_metadata(monkeypatch):
    from services import audio

    _patch_fake_sounddevice(monkeypatch)

    def query(device=None, kind=None):
        if device is None:
            assert kind == "input"
            return {"index": 3, "default_samplerate": 48000, "hostapi": 99}
        return {"index": device, "default_samplerate": 48000, "hostapi": 0}

    def query_hostapis(idx):
        if idx == 99:
            raise KeyError(99)
        return {"name": "ALSA"}

    monkeypatch.setattr(audio.sd, "query_devices", query)
    monkeypatch.setattr(audio.sd, "query_hostapis", query_hostapis)

    attempts = AudioCapture()._sounddevice_attempts(9, True)
    assert [(item["device"], item["samplerate"]) for item in attempts] == [(9, 48000), (9, 16000)]


def test_headless_portaudio_import_failure_degrades_gracefully(tmp_path):
    fake_site = tmp_path / "fake_site"
    fake_site.mkdir()
    (fake_site / "sounddevice.py").write_text(
        "class PortAudioError(Exception):\n"
        "    pass\n\n"
        "raise PortAudioError(\n"
        "    'Error initializing PortAudio: Unanticipated host error [PaErrorCode -9999]: '\n"
        "    \"'PulseAudio_Initialize: Can't connect to server'\"\n"
        ")\n",
        encoding="utf-8",
    )

    wav_path = tmp_path / "probe.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)

    log_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.json"
    log_dir.mkdir()
    data_dir.mkdir()

    script = f"""
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, {str(BACKEND_DIR)!r})
sys.path.insert(0, {str(fake_site)!r})

import core.logger as _logger
import core.config as _config
_logger.LOG_DIR = {str(log_dir)!r}
_config.CONFIG_FILE = {str(config_path)!r}

import services.audio as audio
import main

assert main.app is not None
assert audio._SOUNDDEVICE_IMPORT_ERROR is not None
assert type(audio._SOUNDDEVICE_IMPORT_ERROR).__name__ == "PortAudioError"
assert "PulseAudio_Initialize: Can't connect to server" in str(audio._SOUNDDEVICE_IMPORT_ERROR)
assert audio.sd.default.device == (-1, -1)
assert audio.sd.query_devices() == []

for call in (
    lambda: audio.sd.query_devices(0),
    lambda: audio.sd.query_devices(kind="input"),
    lambda: audio.sd.query_hostapis(0),
    lambda: audio.sd.InputStream(),
    lambda: audio.sd.WasapiSettings(),
    lambda: audio.sd.play(b"", 16000),
    lambda: audio.sd.wait(),
):
    with pytest.raises(RuntimeError):
        call()

audio._HAS_SC = False
assert audio.AudioCapture.list_devices() == []

with pytest.raises(RuntimeError):
    audio.AudioCapture().start(0)

with pytest.raises(RuntimeError):
    audio.play_audio_file({str(wav_path)!r})

# Verify audio.sd remains a monkeypatchable proxy across all expected attributes.
with pytest.MonkeyPatch.context() as mp:
    mp.setattr(audio.sd, "query_devices", lambda *args, **kwargs: (
        {{"index": 0, "name": "Fake Mic", "max_input_channels": 1, "default_samplerate": 16000, "hostapi": 0}}
        if args or kwargs else
        [{{"name": "Fake Mic", "max_input_channels": 1, "default_samplerate": 16000, "hostapi": 0}}]
    ))
    mp.setattr(audio.sd, "query_hostapis", lambda *_: {{"name": "ALSA"}})
    mp.setattr(audio.sd, "InputStream", lambda **_: SimpleNamespace(start=lambda: None, stop=lambda: None, close=lambda: None))
    mp.setattr(audio.sd, "WasapiSettings", lambda **_: SimpleNamespace())
    played = []
    waited = []
    mp.setattr(audio.sd, "play", lambda a, sr: played.append((len(a), sr)))
    mp.setattr(audio.sd, "wait", lambda: waited.append(True))
    mp.setattr(audio.sd, "default", SimpleNamespace(device=(0, 0)))

    assert audio.AudioCapture.list_devices()[0]["name"] == "Fake Mic"
    cap = audio.AudioCapture()
    cap.start(0, owner="headless-monkeypatch-test")
    cap.stop(owner="headless-monkeypatch-test")
    audio.play_audio_file({str(wav_path)!r})
    assert played == [(160, 16000)]
    assert waited == [True]
    assert audio.sd.default.device == (0, 0)

# After MonkeyPatch restores audio.sd, fallback behavior returns intact.
with pytest.raises(RuntimeError):
    audio.AudioCapture().start(0)
"""

    env = os.environ.copy()
    env["IA_CONFIG_PATH"] = str(config_path)
    env["IA_DATA_DIR"] = str(data_dir)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_list_devices_handles_sounddevice_query_error(monkeypatch):
    from types import SimpleNamespace
    from services import audio

    def broken_query(*_args, **_kwargs):
        raise RuntimeError("PulseAudio_Initialize: Can't connect to server")

    monkeypatch.setattr(audio.sd, "query_devices", broken_query)
    monkeypatch.setattr(audio.platform, "system", lambda: "Linux")
    assert AudioCapture.list_devices() == []

    speaker = SimpleNamespace(name="Fallback Speaker", id="loopback-fallback-id", channels=2)
    monkeypatch.setattr(audio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(audio, "_HAS_SC", True)
    monkeypatch.setattr(
        audio,
        "_sc",
        SimpleNamespace(default_speaker=lambda: speaker, all_speakers=lambda: [speaker]),
        raising=False,
    )
    monkeypatch.setattr(audio, "_SC_DEVICE_CACHE", {})
    monkeypatch.setattr(audio, "_SC_ID_MAP", {})

    devices = AudioCapture.list_devices()
    assert len(devices) == 1
    assert devices[0]["is_loopback"] is True
    assert "Fallback Speaker" in devices[0]["name"]


def test_start_ensure_python_deps_checks_sounddevice_without_importing(monkeypatch):
    start_path = BACKEND_DIR.parent / "start.py"
    spec = importlib.util.spec_from_file_location("start_headless_check", start_path)
    start_mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(start_mod)

    orig_find_spec = importlib.util.find_spec
    find_spec_calls: list[str] = []
    import_calls: list[str] = []
    sounddevice_installed = True

    def fake_find_spec(name: str, package=None):
        find_spec_calls.append(name)
        if name == "sounddevice":
            return object() if sounddevice_installed else None
        return orig_find_spec(name, package)

    def fake_import(name: str, *args, **kwargs):
        import_calls.append(name)
        if name == "sounddevice":
            raise RuntimeError(
                "PortAudioError: Error initializing PortAudio: PulseAudio_Initialize: Can't connect to server"
            )
        return object()

    pip_calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        nonlocal sounddevice_installed
        pip_calls.append(cmd)
        sounddevice_installed = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(start_mod.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(start_mod, "__import__", fake_import, raising=False)
    monkeypatch.setattr(start_mod.subprocess, "run", fake_run)

    # Case 1: sounddevice is installed; ensure_python_deps uses find_spec and never imports sounddevice.
    start_mod.ensure_python_deps()
    assert "sounddevice" in find_spec_calls
    assert "sounddevice" not in import_calls
    assert pip_calls == []

    # Case 2: sounddevice is initially missing, triggers pip install, then passes re-check via find_spec.
    find_spec_calls.clear()
    import_calls.clear()
    sounddevice_installed = False
    start_mod.ensure_python_deps()
    assert find_spec_calls.count("sounddevice") == 2
    assert "sounddevice" not in import_calls
    assert len(pip_calls) == 1
    assert pip_calls[0][:6] == [sys.executable, "-m", "pip", "install", "-r", str(BACKEND_DIR / "requirements.txt")]
