"""Run the real Qwen model twice on a WAV and report CUDA placement and latency."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", default=str(ROOT / "backend/assets/preflight_phrase.wav"))
    args = parser.parse_args()
    import torch
    from services.audio import load_wav_file
    from services.stt.qwen_engine import QwenASRSTT
    from services.stt.text_utils import transcription_significant_len

    print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}, ensure_ascii=False), flush=True)
    engine = QwenASRSTT()
    started = time.perf_counter()
    engine.load_model()
    print(json.dumps({"load_seconds": round(time.perf_counter() - started, 3),
                      "gpu": torch.cuda.get_device_name(0),
                      "device": str(next(engine._model.parameters()).device),
                      "parameter_devices": sorted({str(p.device) for p in engine._model.parameters()}),
                      "compute_type": engine.compute_type}, ensure_ascii=False), flush=True)
    audio, rate = load_wav_file(Path(args.wav))
    for run in range(2):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        text = engine.transcribe(audio, rate)
        torch.cuda.synchronize()
        print(json.dumps({"run": run + 1, "audio_seconds": round(len(audio) / rate, 3),
                          "inference_seconds": round(time.perf_counter() - started, 3),
                          "peak_allocated_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
                          "text": text}, ensure_ascii=False), flush=True)
        if transcription_significant_len(text) < 2:
            raise RuntimeError("Qwen returned no meaningful transcript")
        if Path(args.wav).resolve() == (ROOT / "backend/assets/preflight_phrase.wav").resolve() and "项目" not in text:
            raise RuntimeError("Qwen did not recognize the bundled project's test phrase")
