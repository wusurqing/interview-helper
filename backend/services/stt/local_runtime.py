"""Optional per-installation cache and CUDA 11 compatibility paths."""

import os
from pathlib import Path
import sys
import threading

_runtime_lock = threading.Lock()
_dll_handles = []
_configured_runtime = None


def whisper_cache_path(cfg):
    root = getattr(cfg, 'stt_model_cache_root', '') or ''
    return str(Path(root) / 'huggingface' / 'hub') if root else None


def configure_funasr_cache(cfg):
    root = getattr(cfg, 'stt_model_cache_root', '') or ''
    if root:
        os.environ.setdefault('MODELSCOPE_CACHE', str(Path(root) / 'modelscope'))


def funasr_model_reference(repo):
    """Use an already downloaded ModelScope snapshot without network access."""
    cache = os.environ.get('MODELSCOPE_CACHE')
    if cache:
        snapshot = Path(cache) / 'models' / repo.replace('/', '--') / 'snapshots' / 'master'
        if (snapshot / 'config.yaml').is_file() and (snapshot / 'model.pt').is_file():
            return str(snapshot)
    return repo


def configure_whisper_cuda11(cfg):
    global _configured_runtime
    if not getattr(cfg, 'whisper_cuda11_compat', False):
        return
    root = Path(getattr(cfg, 'stt_model_cache_root', '') or '')
    runtime = root / 'whisper-cuda11'
    cudnn = root / 'cudnn8' / 'nvidia' / 'cudnn' / 'bin'
    if os.name != 'nt' or not (runtime / 'ctranslate2').is_dir() or not cudnn.is_dir():
        raise RuntimeError('Whisper CUDA 11 兼容运行库不完整，请检查本地模型缓存目录。')
    with _runtime_lock:
        if _configured_runtime is not None:
            if _configured_runtime != str(runtime):
                raise RuntimeError('Whisper CUDA 11 运行库路径已更改，请重启应用。')
            return
        if 'ctranslate2' in sys.modules or 'faster_whisper' in sys.modules:
            raise RuntimeError('CTranslate2 已加载，请重启应用后再使用 CUDA 11 兼容模式。')
        _dll_handles.append(os.add_dll_directory(str(cudnn)))
        # cuDNN 8 loads its component DLLs with LoadLibrary at inference time;
        # on Windows those lookups also need the process PATH.
        dll_paths = [str(cudnn)]
        torch = sys.modules.get('torch')
        torch_file = getattr(torch, '__file__', None)
        if torch_file:
            dll_paths.append(str(Path(torch_file).parent / 'lib'))
        os.environ['PATH'] = os.pathsep.join(dll_paths + [os.environ.get('PATH', '')])
        sys.path.insert(0, str(runtime))
        _configured_runtime = str(runtime)
