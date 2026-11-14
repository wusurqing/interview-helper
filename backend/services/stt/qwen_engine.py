"""Local Qwen3-ASR-0.6B using the official native Transformers checkpoint."""

import math
import threading

import numpy as np

from core.logger import get_logger
from .text_utils import build_tech_hotwords, custom_hotwords_from_config, join_transcription_fragments

_log = get_logger("stt.qwen")
MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"


def _projector_input_dtype(module, args):
    """Bridge the FP32 audio encoder to the FP16 multimodal projector."""
    return (args[0].to(module.linear_1.weight.dtype),)


class QwenASRSTT:
    model_size = "Qwen3-ASR-0.6B"
    device = "cuda:0"
    compute_type = "float32(audio)+float16(text)"

    def __init__(self):
        self._model = None
        self._processor = None
        self._loading = False
        self._retired = False
        self._lock = threading.RLock()
        self.load_error = ""

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def is_loading(self):
        return self._loading

    def load_model(self):
        with self._lock:
            if self._retired:
                raise RuntimeError("Qwen3-ASR 引擎已卸载，请重新获取当前语音引擎。")
            if self.is_loaded:
                return
            self._loading = True
            self.load_error = ""
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("Qwen3-ASR 需要 CUDA GPU，请安装兼容驱动的 CUDA 版 PyTorch。")
                # Initialize cuBLAS before allocating weights on small GPUs.
                # Loading its CUDA modules late can fail under memory pressure.
                warmup = torch.ones((1, 1), device=self.device)
                (warmup @ warmup).item()
                del warmup
                from transformers import AutoModelForMultimodalLM, AutoProcessor

                processor = AutoProcessor.from_pretrained(MODEL_ID)
                # Turing's FP16 audio encoder produces NaNs. Keep only that
                # tower in FP32 to fit 4 GiB VRAM; the text decoder stays FP16.
                major, _ = torch.cuda.get_device_capability(0)
                self.compute_type = "bfloat16" if major >= 8 else "float32(audio)+float16(text)"
                dtype = torch.bfloat16 if major >= 8 else torch.float16
                model = AutoModelForMultimodalLM.from_pretrained(
                    MODEL_ID, dtype=dtype, attn_implementation="sdpa",
                ).to(self.device).eval()
                if major < 8:
                    model.model.audio_tower.float()
                    model.model.multi_modal_projector.register_forward_pre_hook(_projector_input_dtype)
                torch.cuda.empty_cache()
                self._processor = processor
                self._model = model
                _log.info("Qwen ASR loaded model=%s device=%s dtype=%s gpu=%s",
                          MODEL_ID, self.device, self.compute_type, torch.cuda.get_device_name(0))
            except ImportError as exc:
                self.load_error = "Qwen3-ASR 缺少依赖，请安装 requirements-qwen.txt 和 CUDA 版 PyTorch。"
                raise RuntimeError(self.load_error) from exc
            except Exception as exc:
                self.load_error = str(exc)
                raise
            finally:
                self._loading = False

    def unload(self):
        """Release GPU weights when the primary local provider is changed."""
        with self._lock:
            self._retired = True
            self._model = None
            self._processor = None
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   position: str = "后端开发", language: str = "Python") -> str:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not audio.size:
            return ""
        audio_f = audio.astype(np.float32)
        if np.issubdtype(audio.dtype, np.integer):
            audio_f /= max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        if audio_f.ndim == 2:
            audio_f = audio_f.mean(axis=1)
        if not np.isfinite(audio_f).all():
            raise ValueError("audio contains non-finite samples")
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            divisor = math.gcd(sample_rate, 16000)
            audio_f = resample_poly(audio_f, 16000 // divisor, sample_rate // divisor)
        import torch

        words = build_tech_hotwords(position, language, limit=400, custom=custom_hotwords_from_config())
        # The settings test and interviewer/candidate workers can overlap.
        # Serialize them, including loading/unloading, to bound GPU memory.
        with self._lock:
            if self._retired:
                raise RuntimeError("Qwen3-ASR 引擎已卸载，请重新获取当前语音引擎。")
            if not self.is_loaded:
                self.load_model()
            parts = []
            offset = 0
            while offset < len(audio_f):
                # Bound encoder memory and avoid truncating sustained candidate answers.
                chunk = audio_f[offset:offset + 16000 * 30]
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "Vocabulary: " + ", ".join(words)}]},
                    {"role": "user", "content": [{"type": "audio", "audio": chunk}]},
                ]
                inputs = self._processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(self._model.device, self._model.dtype)
                with torch.inference_mode():
                    output_ids = self._model.generate(**inputs, max_new_tokens=768, do_sample=False)
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                text = self._processor.decode(generated_ids, return_format="transcription_only")[0]
                parts.append(text.strip())
                if offset + len(chunk) >= len(audio_f):
                    break
                offset += 16000 * 29  # one second of context across chunk boundaries
            return join_transcription_fragments(parts)
