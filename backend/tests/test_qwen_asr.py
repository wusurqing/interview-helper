"""Qwen's local-engine contract; no model downloads or GPU are needed here."""

import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.config import AppConfig
from services.stt import factory


class QwenRoutingTests(unittest.TestCase):
    def test_qwen_primary_uses_local_engine_and_candidate_follows_it(self):
        cfg = AppConfig(stt_provider="qwen3_asr", candidate_stt_provider="whisper")
        with patch("core.config.get_config", return_value=cfg):
            engine = factory.get_stt_engine()
        self.assertEqual(engine.model_size, "Qwen3-ASR-0.6B")
        self.assertEqual(factory._candidate_local_provider(cfg), "qwen3_asr")
        self.assertIn("qwen3_asr", factory.LOCAL_STT_PROVIDERS)
        self.assertIsNot(factory._local_infer_lock("qwen3_asr"), factory._local_infer_lock("whisper"))

    def test_candidate_qwen_survives_config_validation(self):
        cfg = AppConfig(stt_provider="qwen3_asr", candidate_stt_provider="qwen3_asr")
        self.assertEqual(cfg.candidate_stt_provider, "qwen3_asr")

    def test_candidate_partial_transcription_can_be_enabled_for_qwen(self):
        from api.assist.pipeline import _candidate_provider_config, _candidate_streaming_config
        cfg = AppConfig(stt_provider="qwen3_asr", candidate_stt_provider="whisper",
                        candidate_asr_enabled=True, candidate_streaming_asr_enabled=True)
        provider = _candidate_provider_config(cfg)[0]
        self.assertEqual(provider, "qwen3_asr")
        enabled, _ = _candidate_streaming_config(cfg, provider)
        self.assertTrue(enabled)


class QwenEngineTests(unittest.TestCase):
    def make_engine(self):
        from services.stt.qwen_engine import QwenASRSTT
        return QwenASRSTT()

    def test_empty_audio_does_not_download_or_load_model(self):
        engine = self.make_engine()
        self.assertEqual(engine.transcribe(np.empty(0, np.float32), 16000), "")
        self.assertFalse(engine.is_loaded)

    def test_missing_cuda_is_reported_instead_of_silently_using_cpu(self):
        engine = self.make_engine()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                engine.load_model()
        self.assertFalse(engine.is_loaded)
        self.assertFalse(engine.is_loading)
        self.assertIn("CUDA", engine.load_error)

    def test_turing_gpu_keeps_audio_float32_and_decoder_float16(self):
        engine = self.make_engine()

        class Warmup:
            def __matmul__(self, other):
                return self

            def item(self):
                return 1

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda i: (7, 5),
                                 get_device_name=lambda i: "GTX 1650", empty_cache=lambda: None),
            float16="fp16", float32="fp32", bfloat16="bf16",
            ones=lambda shape, device: Warmup(),
        )

        class Loader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                if kwargs.get("dtype") != "fp16":
                    raise ValueError("Full FP32 exceeds the small GPU's memory budget")
                tower = SimpleNamespace(dtype="fp16")
                tower.float = lambda: setattr(tower, "dtype", "fp32")
                projector = SimpleNamespace()
                projector.register_forward_pre_hook = lambda hook: setattr(projector, "hook", hook)
                result = SimpleNamespace(model=SimpleNamespace(audio_tower=tower, multi_modal_projector=projector))
                result.to = lambda device: result
                result.eval = lambda: result
                return result

        fake_transformers = SimpleNamespace(AutoModelForMultimodalLM=Loader,
                                           AutoProcessor=SimpleNamespace(from_pretrained=lambda model_id: object()))
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            engine.load_model()
        self.assertTrue(engine.is_loaded)
        self.assertEqual(engine._model.model.audio_tower.dtype, "fp32")
        self.assertIn("float16", engine.compute_type)
        projector = engine._model.model.multi_modal_projector
        projector.linear_1 = SimpleNamespace(weight=SimpleNamespace(dtype="fp16"))
        tensor = SimpleNamespace(to=lambda dtype: dtype)
        self.assertEqual(projector.hook(projector, (tensor,)), ("fp16",))

    def test_pcm_is_normalized_and_answer_decoded_without_prompt_tokens(self):
        engine = self.make_engine()
        observed = {}

        class Batch(dict):
            def to(self, device, dtype):
                observed["input_device"] = device
                return self

        class Processor:
            def apply_transcription_request(self, *, audio, **kwargs):
                # Transformers 5.13 does not put the prompt kwarg in a message.
                observed["context"] = ""
                observed["audio"] = audio
                return Batch(input_ids=np.zeros((1, 3), dtype=np.int64))

            def apply_chat_template(self, messages, **kwargs):
                observed["context"] = messages[0]["content"][0]["text"]
                observed["audio"] = messages[1]["content"][0]["audio"]
                observed.setdefault("chunks", []).append(len(observed["audio"]))
                return Batch(input_ids=np.zeros((1, 3), dtype=np.int64))

            def decode(self, tokens, return_format):
                observed["tokens"] = tokens.tolist()
                self.format = return_format
                return ["  解释一下数据库事务。  "]

        class Model:
            device = "cuda:0"
            dtype = "float16"

            def generate(self, **kwargs):
                observed["token_budget"] = kwargs["max_new_tokens"]
                return np.array([[0, 0, 0, 11, 12]])

        engine._processor = Processor()
        engine._model = Model()
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        pcm = np.array([0, 16384, -32768], dtype=np.int16)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            "services.stt.qwen_engine.custom_hotwords_from_config", return_value={"通用": "AcmeVector"},
        ):
            text = engine.transcribe(pcm, 16000)
        self.assertEqual(text, "解释一下数据库事务。")
        np.testing.assert_allclose(observed["audio"], [0, 0.5, -1.0])
        self.assertEqual(observed["tokens"], [[11, 12]])
        self.assertEqual(observed["input_device"], "cuda:0")
        self.assertIn("AcmeVector", observed["context"])
        # A sustained candidate answer must be covered by bounded GPU chunks.
        with patch.dict(sys.modules, {"torch": fake_torch}):
            engine.transcribe(np.zeros(16000 * 120, np.float32), 16000)
        long_chunks = observed["chunks"][1:]
        self.assertGreater(len(long_chunks), 1)
        self.assertLessEqual(max(long_chunks), 16000 * 30)
        self.assertGreaterEqual(sum(long_chunks), 16000 * 120)

    def test_evicted_model_cannot_reload_through_a_stale_reference(self):
        engine = self.make_engine()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: False, empty_cache=lambda: None,
        ))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            engine.unload()
            with self.assertRaisesRegex(RuntimeError, "已卸载"):
                engine.load_model()

    def test_invalid_sample_rate_is_rejected_before_loading(self):
        engine = self.make_engine()
        with self.assertRaises(ValueError):
            engine.transcribe(np.ones(160, np.float32), 0)


if __name__ == "__main__":
    unittest.main()
