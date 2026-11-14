import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from services.stt import local_runtime


def test_funasr_uses_configured_cache_and_preserves_explicit_environment(tmp_path, monkeypatch):
    cfg = SimpleNamespace(stt_model_cache_root=str(tmp_path))
    monkeypatch.delenv('MODELSCOPE_CACHE', raising=False)
    local_runtime.configure_funasr_cache(cfg)
    assert os.environ['MODELSCOPE_CACHE'] == str(tmp_path / 'modelscope')
    monkeypatch.setenv('MODELSCOPE_CACHE', 'external-cache')
    local_runtime.configure_funasr_cache(cfg)
    assert os.environ['MODELSCOPE_CACHE'] == 'external-cache'


def test_whisper_cache_is_separate_from_qwen(tmp_path):
    cfg = SimpleNamespace(stt_model_cache_root=str(tmp_path))
    assert local_runtime.whisper_cache_path(cfg) == str(tmp_path / 'huggingface' / 'hub')
    assert local_runtime.whisper_cache_path(SimpleNamespace(stt_model_cache_root='')) is None


def test_funasr_prefers_complete_cached_model_for_offline_use(tmp_path, monkeypatch):
    monkeypatch.setenv('MODELSCOPE_CACHE', str(tmp_path))
    repo = 'iic/SenseVoiceSmall'
    cached = tmp_path / 'models' / 'iic--SenseVoiceSmall' / 'snapshots' / 'master'
    cached.mkdir(parents=True)
    (cached / 'config.yaml').write_text('model: test', encoding='utf-8')
    assert local_runtime.funasr_model_reference(repo) == repo
    (cached / 'model.pt').write_bytes(b'test')
    assert local_runtime.funasr_model_reference(repo) == str(cached)


def test_cuda11_runtime_rejects_missing_directory(tmp_path):
    cfg = SimpleNamespace(stt_model_cache_root=str(tmp_path), whisper_cuda11_compat=True)
    with patch.object(local_runtime.os, 'name', 'nt'):
        try:
            local_runtime.configure_whisper_cuda11(cfg)
        except RuntimeError as exc:
            assert 'CUDA 11' in str(exc)
        else:
            raise AssertionError('missing runtime must not be silently ignored')


def test_cuda11_runtime_sets_paths_for_native_dlls(tmp_path, monkeypatch):
    runtime = tmp_path / 'whisper-cuda11'
    (runtime / 'ctranslate2').mkdir(parents=True)
    cudnn = tmp_path / 'cudnn8' / 'nvidia' / 'cudnn' / 'bin'
    cudnn.mkdir(parents=True)
    cfg = SimpleNamespace(stt_model_cache_root=str(tmp_path), whisper_cuda11_compat=True)
    monkeypatch.setattr(local_runtime, '_configured_runtime', None)
    monkeypatch.setattr(local_runtime, '_dll_handles', [])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delenv('PATH', raising=False)
    with patch.object(local_runtime.os, 'name', 'nt'), patch.object(local_runtime.os, 'add_dll_directory', create=True) as add_dll:
        local_runtime.configure_whisper_cuda11(cfg)
    assert sys.path[0] == str(runtime)
    assert str(cudnn) in os.environ['PATH']
    add_dll.assert_called_once_with(str(cudnn))


def test_switching_whisper_model_releases_unconfigured_old_engine(monkeypatch):
    from services.stt import factory
    from core import config
    cfg = SimpleNamespace(stt_provider='whisper', whisper_model='base', whisper_language='auto',
                          candidate_stt_provider='whisper', candidate_whisper_model='',
                          candidate_whisper_language='')
    class FakeEngine:
        def __init__(self, model_size='base', language='auto'):
            self.model_size, self.language = model_size, language
    monkeypatch.setattr(config, 'get_config', lambda: cfg)
    monkeypatch.setattr(factory, 'STTEngine', FakeEngine)
    monkeypatch.setattr(factory, '_engine', None)
    monkeypatch.setattr(factory, '_whisper_engines', {})
    factory.get_stt_engine()
    cfg.whisper_model = 'medium'
    medium = factory.get_stt_engine()
    assert factory._engine is medium
    assert list(factory._whisper_engines) == [('medium', 'auto')]


def test_streaming_paraformer_flushes_last_audio_segment():
    from services.stt.engines import FunASRSTT
    seen = {}
    class FakeModel:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return [{'text': '完整的问题'}]
    engine = FunASRSTT('paraformer-zh-streaming')
    engine._model = FakeModel()
    with patch.object(engine, '_hotword_string', return_value=''):
        assert engine.transcribe(np.zeros(1600, dtype=np.float32)) == '完整的问题'
    assert seen['is_final'] is True
    assert seen['cache'] == {}


def test_evicting_funasr_releases_unused_cuda_cache(monkeypatch):
    from services.stt import factory
    released = []
    monkeypatch.setattr(factory, '_funasr_engines', {'sensevoice-small': object()})
    monkeypatch.setattr(factory, '_engine', None)
    monkeypatch.setattr(factory, '_whisper_engines', {})
    monkeypatch.setattr(factory, '_qwen_engine', None)
    monkeypatch.setattr(factory, '_release_torch_cuda_cache', lambda: released.append(True))
    with factory._engine_lock:
        factory._evict_other_local_engine_locked('whisper', candidate='whisper')
    assert factory._funasr_engines == {}
    assert released == [True]
