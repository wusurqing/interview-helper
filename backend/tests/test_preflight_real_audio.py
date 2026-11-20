from __future__ import annotations

from pathlib import Path
import importlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

sound_test = importlib.import_module('api.assist.sound_test')


def test_normalize_phrase_strips_punctuation_and_spaces():
    assert sound_test.normalize_phrase(' 请 介绍一下，你最近做过的项目！ ') == '请介绍一下你最近做过的项目'


def test_phrase_match_accepts_expected_substring():
    ok, detail = sound_test.match_phrase('请介绍一下你最近做过的项目', '请介绍一下你最近做过的项目。')
    assert ok is True
    assert '识别匹配' in detail


def test_phrase_match_rejects_mismatch():
    ok, detail = sound_test.match_phrase('请介绍一下你最近做过的项目', '今天天气很好')
    assert ok is False
    assert '未匹配' in detail


def test_collect_capture_audio_merges_chunks(monkeypatch: pytest.MonkeyPatch):
    chunks = [np.ones(3, dtype=np.float32), np.ones(2, dtype=np.float32), None]
    now = {'v': 0.0}

    def fake_monotonic():
        now['v'] += 0.06
        return now['v']

    monkeypatch.setattr(sound_test.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sound_test.time, 'monotonic', fake_monotonic)

    class FakeCapture:
        def get_audio_chunk(self):
            return chunks.pop(0) if chunks else None

    merged = sound_test.collect_capture_audio(FakeCapture(), duration_sec=0.2, poll_interval=0.05)
    assert merged is not None
    assert len(merged) == 5


def test_input_audio_returns_level_metrics(monkeypatch: pytest.MonkeyPatch):
    starts: list[tuple[int, str | None, dict]] = []
    stops: list[str | None] = []

    class FakeCapture:
        def start(self, device_id, owner=None, **kwargs):
            starts.append((device_id, owner, kwargs))

        def stop(self, owner=None):
            stops.append(owner)

        @staticmethod
        def compute_energy(audio):
            return float(np.sqrt(np.mean(audio ** 2)))

    monkeypatch.setattr(sound_test, 'AudioCapture', FakeCapture)
    monkeypatch.setattr(sound_test.time, 'sleep', lambda _: None)
    monkeypatch.setattr(
        sound_test,
        'collect_capture_audio',
        lambda cap, duration_sec, poll_interval=0.05: np.array([0.0, 0.02, -0.04], dtype=np.float32),
    )

    result = sound_test.test_input_audio(7, duration_sec=0.8)

    assert result['ok'] is True
    assert result['device_id'] == 7
    assert result['has_signal'] is True
    assert result['rms'] > 0
    assert result['peak'] == pytest.approx(0.04)
    assert starts == [(7, 'audio-test', {'mic_compatibility_mode': True})]
    assert stops == ['audio-test']


def test_input_level_monitor_updates_until_stopped(monkeypatch: pytest.MonkeyPatch):
    starts: list[tuple[int, str | None, dict]] = []
    stops: list[str | None] = []

    class FakeCapture:
        def start(self, device_id, owner=None, **kwargs):
            starts.append((device_id, owner, kwargs))

        def stop(self, owner=None):
            stops.append(owner)

        def get_audio_chunk(self, max_chunks=None):
            return np.array([0.0, 0.03, -0.05], dtype=np.float32)

        @staticmethod
        def compute_energy(audio):
            return float(np.sqrt(np.mean(audio ** 2)))

    monkeypatch.setattr(sound_test, 'AudioCapture', FakeCapture)

    sound_test.start_input_level_monitor(9)
    deadline = time.monotonic() + 1.0
    status = sound_test.get_input_level_status()
    while time.monotonic() < deadline and status['level_pct'] <= 0:
        time.sleep(0.02)
        status = sound_test.get_input_level_status()
    stopped = sound_test.stop_input_level_monitor()

    assert starts == [(9, 'audio-level-test', {'mic_compatibility_mode': True})]
    assert stops == ['audio-level-test']
    assert status['level_pct'] > 0
    assert status['has_signal'] is True
    assert stopped['running'] is False


def test_resolve_preflight_scenario_falls_back_to_recommended(monkeypatch: pytest.MonkeyPatch):
    scenarios = [
        {"id": "a", "label": "A", "question": "A", "recommended": False},
        {"id": "b", "label": "B", "question": "B", "recommended": True},
    ]
    monkeypatch.setattr(sound_test, 'PREFLIGHT_SCENARIOS', scenarios)

    assert sound_test.resolve_preflight_scenario('a') is scenarios[0]
    assert sound_test.resolve_preflight_scenario('missing') is scenarios[1]


def test_run_preflight_updates_status_and_completes(monkeypatch: pytest.MonkeyPatch):
    events: list[dict] = []
    llm_questions: list[str] = []
    monkeypatch.setattr(sound_test, 'broadcast', lambda data: events.append(data))
    monkeypatch.setattr(sound_test, 'play_preflight_audio', lambda: 1.0)
    monkeypatch.setattr(
        sound_test,
        'collect_capture_audio_during_playback',
        lambda cap, play_fn, trailing_sec=0.45, poll_interval=0.02: (
            np.ones(1600, dtype=np.float32) * 0.1,
            float(play_fn()),
        ),
    )
    monkeypatch.setattr(sound_test, 'match_phrase', lambda expected, actual: (True, '识别匹配'))

    class FakeCapture:
        SAMPLE_RATE = 16000

        def start(self, device_id):
            self.device_id = device_id
        def stop(self):
            return None
        @staticmethod
        def compute_energy(audio):
            return 0.1

    model = SimpleNamespace(name='demo', model='demo-model', api_key='test', supports_vision=False)
    cfg = SimpleNamespace(stt_provider='generic', models=[model], active_model=0, get_active_model=lambda: model)
    monkeypatch.setattr(sound_test, 'get_config', lambda: cfg)
    monkeypatch.setattr(sound_test, 'AudioCapture', FakeCapture)
    monkeypatch.setattr(sound_test, 'transcribe_segment', lambda *args, **kwargs: sound_test.PREFLIGHT_EXPECTED_PHRASE)
    monkeypatch.setattr(
        sound_test,
        'generate_preflight_answer',
        lambda cfg, model_cfg, question: (
            llm_questions.append(question)
            or {'answer': '真实答题链路回答', 'first_token_ms': 120, 'total_ms': 880}
        ),
    )

    sound_test._run_preflight(1, 'self_intro')

    status = sound_test.get_preflight_status()
    assert status['running'] is False
    assert status['captured_transcript'] == sound_test.PREFLIGHT_EXPECTED_PHRASE
    assert status['match_ok'] is True
    assert llm_questions == [sound_test.PREFLIGHT_EXPECTED_PHRASE]
    assert any(e.get('step') == 'playback' for e in events)
    assert any(e.get('step') == 'done' for e in events)
    assert any(e.get('type') == 'preflight_step' for e in events)


def test_generate_preflight_answer_uses_voice_profile(monkeypatch: pytest.MonkeyPatch):
    from core.config import AssistGeneration

    calls: dict = {}
    cfg = SimpleNamespace(
        assist_high_churn_short_answer=True,
        assist_realtime_max_tokens=720,
        assist_realtime_high_churn_max_tokens=320,
        assist_generation={"assist_realtime": AssistGeneration(temperature=0.6, max_output_tokens=600)},
        system_prompts={
            'assist_realtime': '标准语音规则',
            'assist_realtime_short': '简短语音规则',
            'assist_manual': '手动提问规则',
        },
    )
    model_cfg = SimpleNamespace(name='demo', model='demo-model', supports_vision=False)

    def fake_stream(model, messages, **kwargs):
        calls.update(model=model, messages=messages, **kwargs)
        yield ('think', 'internal reasoning')
        yield ('text', '# 真实')
        yield ('text', '回答')

    monkeypatch.setattr(sound_test, 'chat_stream_single_model', fake_stream)
    ticks = iter([10.0, 10.4, 10.9])
    monkeypatch.setattr(sound_test.time, 'monotonic', lambda: next(ticks))

    result = sound_test.generate_preflight_answer(cfg, model_cfg, '请做一下自我介绍')

    assert result == {'answer': '# 真实回答', 'first_token_ms': 400, 'total_ms': 900}
    assert calls['model'] is model_cfg
    assert calls['messages'] == [{'role': 'user', 'content': '【面试官｜当前发言】"请做一下自我介绍"'}]
    assert calls['system_prompt'] == '标准语音规则'
    assert calls['override_max_tokens'] == 600
    assert calls['config_snapshot'].temperature == 0.6


def test_replay_preflight_capture_stt_collects_round_stats(monkeypatch: pytest.MonkeyPatch):
    starts: list[tuple[int, str | None, dict]] = []
    stops: list[str | None] = []
    playback_paths: list[str | Path | None] = []
    transcripts = iter([
        sound_test.PREFLIGHT_EXPECTED_PHRASE,
        "识别失败内容",
        sound_test.PREFLIGHT_EXPECTED_PHRASE,
    ])

    class FakeCapture:
        SAMPLE_RATE = 16000
        def __init__(self):
            self._stats_calls = 0

        def start(self, device_id, owner=None, **kwargs):
            starts.append((device_id, owner, kwargs))

        def stop(self, owner=None):
            stops.append(owner)

        @staticmethod
        def compute_energy(audio):
            return float(np.sqrt(np.mean(audio ** 2)))

        def capture_stats_snapshot(self):
            self._stats_calls += 1
            return {
                "dropped_chunks_count": 0,
                "loopback_discontinuity_count": 0,
            }

    monkeypatch.setattr(sound_test, "AudioCapture", FakeCapture)
    monkeypatch.setattr(
        sound_test,
        "get_config",
        lambda: SimpleNamespace(stt_provider="doubao"),
    )
    monkeypatch.setattr(
        sound_test,
        "transcribe_segment",
        lambda *args, **kwargs: next(transcripts),
    )

    def fake_play_audio_fixture(audio_path=None):
        playback_paths.append(audio_path)
        return 0.5

    monkeypatch.setattr(sound_test, "play_audio_fixture", fake_play_audio_fixture)
    monkeypatch.setattr(
        sound_test,
        "play_audio_file",
        lambda *_args, **_kwargs: pytest.fail("replay unit test reached real audio playback"),
    )
    monkeypatch.setattr(
        sound_test,
        "collect_capture_audio_during_playback",
        lambda cap, play_fn, trailing_sec=0.45, poll_interval=0.02: (
            np.ones(1600, dtype=np.float32) * 0.1,
            float(play_fn()),
        ),
    )
    monkeypatch.setattr(sound_test.time, "sleep", lambda _: None)

    result = sound_test.replay_preflight_capture_stt(12, repeats=3, gap_sec=0.1)

    assert result["device_id"] == 12
    assert result["repeats"] == 3
    assert result["success_count"] == 2
    assert result["failure_count"] == 1
    assert result["provider"] == "doubao"
    assert "summary" in result
    assert result["summary"]["avg_stt_ms"] >= 0
    assert result["summary"]["avg_total_ms"] >= 0
    assert result["summary"]["max_raw_queue_drop_count"] == 0
    assert result["summary"]["max_loopback_discontinuity_count"] == 0
    assert [round_item["ok"] for round_item in result["rounds"]] == [True, False, True]
    assert all("stt_ms" in round_item for round_item in result["rounds"])
    assert all("total_ms" in round_item for round_item in result["rounds"])
    assert all("capture" in round_item for round_item in result["rounds"])
    assert starts == [(12, "audio-preflight-replay", {"mic_compatibility_mode": True})]
    assert stops == ["audio-preflight-replay"]
    assert playback_paths == [None, None, None]


def test_collect_capture_audio_during_playback_captures_while_playing(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        np.ones(3, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        None,
        None,
    ]

    class FakeCapture:
        def get_audio_chunk(self, timeout=0.02):
            return chunks.pop(0) if chunks else None

    monkeypatch.setattr(sound_test.time, "sleep", lambda _: None)

    captured, playback_sec = sound_test.collect_capture_audio_during_playback(
        FakeCapture(),
        lambda: 0.5,
        trailing_sec=0.0,
        poll_interval=0.0,
    )

    assert playback_sec == 0.5
    assert captured is not None
    assert len(captured) == 5


def test_resolve_preflight_fixture_enforces_allowlist_and_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assert sound_test.resolve_preflight_fixture(None) == sound_test.PREFLIGHT_AUDIO_PATH.resolve()
    assert sound_test.resolve_preflight_fixture("") == sound_test.PREFLIGHT_AUDIO_PATH.resolve()
    assert sound_test.resolve_preflight_fixture("default") == sound_test.PREFLIGHT_AUDIO_PATH.resolve()

    with pytest.raises(ValueError):
        sound_test.resolve_preflight_fixture("nonexistent")

    outside = tmp_path / "escape.wav"
    outside.write_bytes(b"RIFF")
    monkeypatch.setitem(sound_test.PREFLIGHT_REPLAY_FIXTURES, "escaped", outside)
    with pytest.raises(ValueError):
        sound_test.resolve_preflight_fixture("escaped")


@pytest.mark.parametrize(
    "bad_path",
    [
        "\\\\server\\share\\probe.wav",
        "//server/share/probe.wav",
        "C:\\Windows\\win.ini",
        "notes.txt",
        "",
    ],
)
def test_play_audio_fixture_and_load_wav_reject_unc_and_non_wav(bad_path: str):
    from services.audio import load_wav_file

    with pytest.raises(ValueError):
        sound_test.play_audio_fixture(bad_path)
    with pytest.raises(ValueError):
        load_wav_file(bad_path)
