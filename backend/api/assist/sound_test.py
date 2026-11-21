"""Pre-flight real audio diagnostic: Playback → Capture → STT → LLM → WebSocket."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from api.realtime.ws import broadcast
from api.common.model_health import get_model_health
from api.assist.scheduler import pick_model_index
from core.config import get_config
from core.logger import get_logger
from services.audio import AudioCapture, play_audio_file
from services.llm import (
    LLMError,
    PROMPT_MODE_ASR_REALTIME,
    build_system_prompt,
    chat_stream_single_model,
    create_answer_stream_sanitizer,
)
from services.llm.interview_budget import stream_interview_answer
from services.llm.prompts import (
    build_assist_user_context,
    config_for_assist_generation,
    format_interview_utterance,
)
from services.stt import transcribe_segment

PREFLIGHT_SCENARIOS = [
    {
        "id": "self_intro",
        "label": "项目经历",
        "question": "请介绍一下你最近做过的项目",
        "recommended": True,
    }
]

PREFLIGHT_EXPECTED_PHRASE = "请介绍一下你最近做过的项目"
PREFLIGHT_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "assets"
PREFLIGHT_AUDIO_PATH = PREFLIGHT_FIXTURES_DIR / "preflight_phrase.wav"
PREFLIGHT_REPLAY_FIXTURES: dict[str, Path] = {
    "default": PREFLIGHT_AUDIO_PATH,
    "preflight_phrase": PREFLIGHT_AUDIO_PATH,
}

_running = False
_replay_running = False
_lock = threading.Lock()
_log = get_logger("assist.preflight")
_status: dict = {
    "running": False,
    "scenario_id": None,
    "device_id": None,
    "expected_phrase": PREFLIGHT_EXPECTED_PHRASE,
    "captured_transcript": None,
    "match_ok": None,
    "steps": {},
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_level_lock = threading.Lock()
_level_lifecycle_lock = threading.Lock()
_level_stop_event = threading.Event()
_level_thread: Optional[threading.Thread] = None
_level_status: dict = {
    "running": False,
    "device_id": None,
    "rms": 0.0,
    "peak": 0.0,
    "level_pct": 0,
    "has_signal": False,
    "error": None,
    "started_at": None,
    "updated_at": None,
}


def normalize_phrase(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    punctuation = "，。！？；：、,.!?;:'\"()[]{}<>-—_~`"
    return "".join(ch for ch in text if ch not in punctuation and not ch.isspace())


def match_phrase(expected: str, actual: str) -> tuple[bool, str]:
    expected_norm = normalize_phrase(expected)
    actual_norm = normalize_phrase(actual)
    if not actual_norm:
        return False, "未识别到有效文本"
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True, "识别匹配"
    overlap = sum(1 for ch in expected_norm if ch in actual_norm)
    if expected_norm and overlap / max(len(expected_norm), 1) >= 0.7:
        return True, "识别近似匹配"
    return False, f"识别未匹配（识别结果：{actual.strip() or '空'}）"


def _set_status(**updates):
    with _lock:
        _status.update(updates)


def _set_step(step: str, status: str, detail: str = "", extra: Optional[dict] = None):
    entry = {"status": status, "detail": detail}
    if extra:
        entry.update(extra)
    with _lock:
        steps = dict(_status.get("steps") or {})
        steps[step] = entry
        _status["steps"] = steps
    msg = {"type": "preflight_step", "step": step, "status": status, "detail": detail}
    if extra:
        msg.update(extra)
    broadcast(msg)


def is_preflight_replay_running() -> bool:
    with _lock:
        return bool(_replay_running)


def begin_preflight_replay() -> bool:
    global _replay_running
    with _lock:
        if _running or _replay_running:
            return False
        _replay_running = True
        return True


def end_preflight_replay() -> None:
    global _replay_running
    with _lock:
        _replay_running = False


def get_preflight_status() -> dict:
    with _lock:
        return {
            "running": _status.get("running", False),
            "scenario_id": _status.get("scenario_id"),
            "device_id": _status.get("device_id"),
            "expected_phrase": _status.get("expected_phrase", PREFLIGHT_EXPECTED_PHRASE),
            "captured_transcript": _status.get("captured_transcript"),
            "match_ok": _status.get("match_ok"),
            "steps": dict(_status.get("steps") or {}),
            "error": _status.get("error"),
            "started_at": _status.get("started_at"),
            "finished_at": _status.get("finished_at"),
        }


def collect_capture_audio(cap: AudioCapture, duration_sec: float, poll_interval: float = 0.05) -> Optional[np.ndarray]:
    deadline = time.monotonic() + duration_sec
    chunks: list[np.ndarray] = []
    while time.monotonic() < deadline:
        chunk = cap.get_audio_chunk()
        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)
        time.sleep(poll_interval)
    if not chunks:
        return None
    return np.concatenate(chunks)


def resolve_preflight_fixture(fixture_id: Optional[str] = None) -> Path:
    """将 HTTP 回放请求中的 fixture_id 解析为受控目录内的 WAV 文件。"""
    if fixture_id is None:
        key = "default"
    elif isinstance(fixture_id, str):
        key = fixture_id.strip() or "default"
    else:
        raise ValueError("fixture_id 必须是字符串")
    candidate = PREFLIGHT_REPLAY_FIXTURES.get(key)
    if candidate is None:
        raise ValueError("不支持的测试音频 fixture_id")
    raw = str(candidate)
    if raw.startswith(("\\\\", "//")):
        raise ValueError("不支持网络共享路径")
    root = PREFLIGHT_FIXTURES_DIR.resolve()
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("测试音频路径越出受控目录")
    if resolved.suffix.lower() != ".wav":
        raise ValueError("测试音频必须是 .wav 文件")
    return resolved


def play_preflight_audio() -> float:
    if not PREFLIGHT_AUDIO_PATH.exists():
        raise FileNotFoundError("内置测试音频不存在")
    started = time.monotonic()
    play_audio_file(PREFLIGHT_AUDIO_PATH)
    return time.monotonic() - started


def play_audio_fixture(audio_path: str | Path | None = None) -> float:
    if audio_path is not None:
        raw = str(audio_path).strip()
        if not raw or raw.startswith(("\\\\", "//")):
            raise ValueError("不支持的测试音频路径")
        target = Path(raw)
        if target.suffix.lower() != ".wav":
            raise ValueError("仅支持 .wav 测试音频文件")
    else:
        target = PREFLIGHT_AUDIO_PATH
    if not target.exists():
        raise FileNotFoundError("测试音频不存在")
    started = time.monotonic()
    play_audio_file(target)
    return time.monotonic() - started


def collect_capture_audio_during_playback(
    cap: AudioCapture,
    play_fn,
    *,
    trailing_sec: float = 0.45,
    poll_interval: float = 0.02,
) -> tuple[Optional[np.ndarray], float]:
    chunks: list[np.ndarray] = []
    playback_queue: queue.Queue[tuple[str, float | Exception]] = queue.Queue(maxsize=1)

    def _append_chunk(timeout: float) -> None:
        chunk = cap.get_audio_chunk(timeout=max(0.0, timeout))
        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)

    def _drain_ready_chunks() -> None:
        drain_chunks = getattr(cap, "drain_audio_chunks", None)
        if callable(drain_chunks):
            try:
                ready_chunks = drain_chunks(timeout=0.0, max_chunks=32) or []
            except TypeError:
                ready_chunks = drain_chunks(timeout=0.0) or []
            for chunk in ready_chunks:
                if chunk is not None and len(chunk) > 0:
                    chunks.append(chunk)
            return
        while True:
            chunk = cap.get_audio_chunk(timeout=0.0)
            if chunk is None or len(chunk) <= 0:
                return
            chunks.append(chunk)

    def _runner():
        try:
            playback_queue.put(("ok", float(play_fn() or 0.0)))
        except Exception as exc:
            playback_queue.put(("err", exc))

    thread = threading.Thread(target=_runner, daemon=True)
    started = time.monotonic()
    thread.start()

    while thread.is_alive():
        _append_chunk(poll_interval)
    thread.join(timeout=0.1)
    playback_elapsed = max(0.0, time.monotonic() - started)
    try:
        status, payload = playback_queue.get_nowait()
    except queue.Empty:
        status, payload = ("ok", playback_elapsed)
    if status == "err":
        raise payload  # type: ignore[misc]

    # Playback completion and scheduler jitter can leave queued chunks unread
    # even when trailing_sec is 0; drain all currently available chunks first.
    _drain_ready_chunks()

    trailing_deadline = time.monotonic() + max(0.0, float(trailing_sec or 0.0))
    while time.monotonic() < trailing_deadline:
        _append_chunk(poll_interval)
        _drain_ready_chunks()

    if not chunks:
        return None, float(payload or playback_elapsed)
    return np.concatenate(chunks), float(payload or playback_elapsed)


def replay_preflight_capture_stt(
    device_id: int,
    *,
    repeats: int = 3,
    gap_sec: float = 0.25,
    audio_path: str | Path | None = None,
    expected_phrase: Optional[str] = None,
) -> dict:
    rounds: list[dict] = []
    cfg = get_config()
    cap = AudioCapture()
    normalized_repeats = max(1, min(20, int(repeats or 1)))
    normalized_gap_sec = max(0.0, min(2.0, float(gap_sec or 0.0)))
    target_phrase = (expected_phrase or PREFLIGHT_EXPECTED_PHRASE or "").strip()
    try:
        cap.start(int(device_id), owner="audio-preflight-replay", mic_compatibility_mode=True)
        time.sleep(0.15)
        for idx in range(normalized_repeats):
            round_started = time.monotonic()
            captured, playback_elapsed = collect_capture_audio_during_playback(
                cap,
                lambda: play_audio_fixture(audio_path),
                trailing_sec=0.45,
            )
            transcript = ""
            rms = 0.0
            peak = 0.0
            stt_ms = 0
            capture_stats = {}
            if captured is not None and len(captured) > 0:
                audio = captured.astype(np.float32)
                rms = float(AudioCapture.compute_energy(audio))
                peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                snapshot_fn = getattr(cap, "capture_stats_snapshot", None)
                if callable(snapshot_fn):
                    try:
                        capture_stats = dict(snapshot_fn() or {})
                    except Exception:
                        capture_stats = {}
                stt_started = time.monotonic()
                transcript = transcribe_segment(
                    audio,
                    AudioCapture.SAMPLE_RATE,
                    position=getattr(cfg, "position", ""),
                    language=getattr(cfg, "language", ""),
                    scope="interviewer",
                    config_snapshot=cfg,
                ) or ""
                stt_ms = int((time.monotonic() - stt_started) * 1000)
            ok, detail = match_phrase(target_phrase, transcript)
            total_ms = int((time.monotonic() - round_started) * 1000)
            rounds.append(
                {
                    "round": idx + 1,
                    "ok": ok,
                    "detail": detail,
                    "transcript": transcript,
                    "rms": rms,
                    "peak": peak,
                    "playback_sec": playback_elapsed,
                    "stt_ms": stt_ms,
                    "total_ms": total_ms,
                    "provider": cfg.stt_provider,
                    "capture": capture_stats,
                }
            )
            if idx + 1 < normalized_repeats and normalized_gap_sec > 0:
                time.sleep(normalized_gap_sec)
    finally:
        cap.stop(owner="audio-preflight-replay")

    success_count = sum(1 for item in rounds if item["ok"])
    stt_values = [int(item.get("stt_ms", 0)) for item in rounds if int(item.get("stt_ms", 0)) > 0]
    total_values = [int(item.get("total_ms", 0)) for item in rounds if int(item.get("total_ms", 0)) > 0]
    raw_drop_values = [
        int((item.get("capture") or {}).get("dropped_chunks_count", 0))
        for item in rounds
    ]
    discontinuity_values = [
        int((item.get("capture") or {}).get("loopback_discontinuity_count", 0))
        for item in rounds
    ]
    return {
        "device_id": int(device_id),
        "repeats": normalized_repeats,
        "success_count": success_count,
        "failure_count": len(rounds) - success_count,
        "provider": cfg.stt_provider,
        "expected_phrase": target_phrase,
        "audio_path": str(Path(audio_path)) if audio_path else str(PREFLIGHT_AUDIO_PATH),
        "summary": {
            "avg_stt_ms": int(sum(stt_values) / len(stt_values)) if stt_values else 0,
            "max_stt_ms": max(stt_values) if stt_values else 0,
            "avg_total_ms": int(sum(total_values) / len(total_values)) if total_values else 0,
            "max_total_ms": max(total_values) if total_values else 0,
            "max_raw_queue_drop_count": max(raw_drop_values) if raw_drop_values else 0,
            "max_loopback_discontinuity_count": max(discontinuity_values) if discontinuity_values else 0,
        },
        "rounds": rounds,
    }


def test_input_audio(device_id: int, duration_sec: float = 1.2) -> dict:
    cap = AudioCapture()
    started = time.monotonic()
    try:
        cap.start(int(device_id), owner="audio-test", mic_compatibility_mode=True)
        time.sleep(0.1)
        captured = collect_capture_audio(
            cap,
            duration_sec=max(0.5, min(3.0, float(duration_sec or 1.2))),
        )
    finally:
        cap.stop(owner="audio-test")

    elapsed_sec = time.monotonic() - started
    if captured is None or len(captured) == 0:
        return {
            "ok": False,
            "device_id": int(device_id),
            "elapsed_sec": elapsed_sec,
            "rms": 0.0,
            "peak": 0.0,
            "has_signal": False,
            "detail": "未捕获到音频，请检查麦克风权限或设备选择",
        }

    audio = captured.astype(np.float32)
    rms = float(AudioCapture.compute_energy(audio))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    has_signal = rms > 0.003 or peak > 0.02
    return {
        "ok": True,
        "device_id": int(device_id),
        "elapsed_sec": elapsed_sec,
        "samples": int(len(audio)),
        "rms": rms,
        "peak": peak,
        "has_signal": has_signal,
        "detail": (
            f"已捕获输入信号（RMS {rms:.4f}，峰值 {peak:.3f}）"
            if has_signal
            else f"已打开麦克风，但音量偏低（RMS {rms:.4f}，峰值 {peak:.3f}）"
        ),
    }


test_input_audio.__test__ = False


def _input_level_pct(rms: float, peak: float) -> int:
    return max(0, min(100, int(round(max(rms * 2500, peak * 250)))))


def _set_level_status(**updates) -> None:
    with _level_lock:
        _level_status.update(updates)
        _level_status["updated_at"] = time.time()


def get_input_level_status() -> dict:
    with _level_lock:
        return dict(_level_status)


def _run_input_level_monitor(
    device_id: int,
    stop_event: threading.Event,
    cap: Optional[AudioCapture] = None,
) -> None:
    started_here = cap is None
    if cap is None:
        cap = AudioCapture()
    rms = 0.0
    peak = 0.0
    try:
        if started_here:
            cap.start(int(device_id), owner="audio-level-test", mic_compatibility_mode=True)
        _set_level_status(running=True, error=None)
        while not stop_event.is_set():
            chunk = cap.get_audio_chunk(max_chunks=4)
            if chunk is not None and len(chunk) > 0:
                audio = chunk.astype(np.float32)
                next_rms = float(AudioCapture.compute_energy(audio))
                next_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                rms = (rms * 0.35) + (next_rms * 0.65)
                peak = max(next_peak, peak * 0.72)
            else:
                rms *= 0.82
                peak *= 0.72
            _set_level_status(
                rms=rms,
                peak=peak,
                level_pct=_input_level_pct(rms, peak),
                has_signal=rms > 0.003 or peak > 0.02,
            )
            stop_event.wait(0.08)
    except Exception as exc:
        _log.warning("input level monitor failed device=%s: %s", device_id, exc, exc_info=True)
        _set_level_status(error=str(exc), running=False)
    finally:
        try:
            cap.stop(owner="audio-level-test")
        finally:
            current = threading.current_thread()
            with _level_lock:
                if _level_thread is current or _level_thread is None:
                    _level_status["running"] = False
                    _level_status["updated_at"] = time.time()


def _stop_input_level_monitor_locked() -> dict:
    global _level_thread
    with _level_lock:
        thread = _level_thread
        _level_stop_event.set()
    if thread and thread.is_alive():
        thread.join(timeout=1.5)
    with _level_lock:
        if _level_thread is thread:
            _level_thread = None
        _level_status["running"] = False
        _level_status["updated_at"] = time.time()
    return get_input_level_status()


def start_input_level_monitor(device_id: int) -> dict:
    global _level_thread, _level_stop_event, _level_status
    with _level_lifecycle_lock:
        _stop_input_level_monitor_locked()
        cap = AudioCapture()
        try:
            cap.start(int(device_id), owner="audio-level-test", mic_compatibility_mode=True)
        except Exception as exc:
            _set_level_status(device_id=int(device_id), running=False, error=str(exc))
            raise
        with _level_lock:
            _level_stop_event = threading.Event()
            _level_status = {
                "running": True,
                "device_id": int(device_id),
                "rms": 0.0,
                "peak": 0.0,
                "level_pct": 0,
                "has_signal": False,
                "error": None,
                "started_at": time.time(),
                "updated_at": time.time(),
            }
            thread = threading.Thread(
                target=_run_input_level_monitor,
                args=(int(device_id), _level_stop_event, cap),
                daemon=True,
                name="audio-level-test",
            )
            _level_thread = thread
        try:
            thread.start()
        except Exception:
            with _level_lock:
                if _level_thread is thread:
                    _level_thread = None
                _level_status["running"] = False
                _level_status["updated_at"] = time.time()
            cap.stop(owner="audio-level-test")
            raise
        return get_input_level_status()


def stop_input_level_monitor() -> dict:
    with _level_lifecycle_lock:
        return _stop_input_level_monitor_locked()


def resolve_preflight_scenario(scenario_id: str) -> dict:
    for scenario in PREFLIGHT_SCENARIOS:
        if scenario.get("id") == scenario_id:
            return scenario
    for scenario in PREFLIGHT_SCENARIOS:
        if scenario.get("recommended"):
            return scenario
    return PREFLIGHT_SCENARIOS[0]


def generate_preflight_answer(cfg, model_cfg, question: str) -> dict:
    prompt_mode = PROMPT_MODE_ASR_REALTIME
    generation_cfg = config_for_assist_generation(cfg, prompt_mode)
    system_prompt = build_system_prompt(mode=prompt_mode, cfg=cfg)
    user_context = build_assist_user_context(cfg, mode=prompt_mode)
    context_prefix = f"[本轮任务资料]\n{user_context}\n\n[真实对话]" if user_context else None
    turn = format_interview_utterance(question, speaker="interviewer", current=True)
    sanitizer = create_answer_stream_sanitizer(prompt_mode, plain_text=True)
    answer_parts: list[str] = []
    started = time.monotonic()
    first_token_mono: Optional[float] = None
    for chunk_type, chunk_text in stream_interview_answer(
        model_cfg,
        ((turn,),),
        system_prompt=system_prompt,
        context_prefix=context_prefix,
        stream_fn=chat_stream_single_model,
        override_max_tokens=generation_cfg.max_tokens,
        config_snapshot=generation_cfg,
        trace_context={"scene": "assist_realtime", "source": "audio_preflight"},
    ):
        if chunk_type != "text":
            continue
        visible = sanitizer.push(chunk_text)
        if first_token_mono is None and visible.strip():
            first_token_mono = time.monotonic()
        answer_parts.append(visible)
    total_ms = int((time.monotonic() - started) * 1000)
    first_token_ms = int((first_token_mono - started) * 1000) if first_token_mono is not None else total_ms
    return {
        "answer": "".join(answer_parts) + sanitizer.finish(),
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
    }


def generate_selected_preflight_answer(cfg, question: str):
    task = (question, None, False, "conversation_loopback", {})
    attempted: set[int] = set()
    reason = "请在实时辅助中选择已配置的答题模型"
    while (index := pick_model_index(task, set(), cfg, get_model_health, excluded_indices=attempted)) is not None:
        attempted.add(index)
        model = cfg.models[index]
        try:
            result = generate_preflight_answer(cfg, model, question)
        except LLMError as exc:
            reason = exc.user_msg
        else:
            if result["answer"].strip():
                return model, result
            reason = "模型回答为空"
    raise RuntimeError(reason)


def _run_preflight(device_id: Optional[int], scenario_id: str):
    global _running
    started_mono = time.monotonic()
    scenario = resolve_preflight_scenario(scenario_id)
    resolved_scenario_id = scenario.get("id", scenario_id)
    expected_phrase = scenario.get("expected_phrase") or PREFLIGHT_EXPECTED_PHRASE
    llm_question = scenario.get("question") or expected_phrase
    _set_status(
        running=True,
        scenario_id=resolved_scenario_id,
        device_id=device_id,
        expected_phrase=expected_phrase,
        captured_transcript=None,
        match_ok=None,
        steps={},
        error=None,
        started_at=time.time(),
        finished_at=None,
    )
    try:
        _log.info("PREFLIGHT_START scenario=%s requested=%s device_id=%s", resolved_scenario_id, scenario_id, device_id)
        if device_id is None:
            raise RuntimeError("未选择音频设备，无法进行真实音频链路测试")

        cfg = get_config()
        cap = AudioCapture()
        _set_step("playback", "running", "准备播放测试音频…")
        _set_step("capture", "running", "准备捕获真实音频…")
        try:
            try:
                cap.start(device_id, owner="audio-preflight")
            except TypeError:
                cap.start(device_id)
            time.sleep(0.15)
            captured, playback_elapsed = collect_capture_audio_during_playback(
                cap,
                play_preflight_audio,
                trailing_sec=0.45,
            )
            _set_step("playback", "pass", f"测试音频已播放（{playback_elapsed:.2f}s）")
        finally:
            try:
                cap.stop(owner="audio-preflight")
            except TypeError:
                cap.stop()

        if captured is None or len(captured) == 0:
            _set_step("capture", "fail", "未捕获到有效音频")
            raise RuntimeError("未捕获到有效音频")

        energy = float(AudioCapture.compute_energy(captured))
        if energy <= 0.003:
            _set_step("capture", "fail", f"捕获音量过低（RMS {energy:.4f}）")
            raise RuntimeError("捕获音量过低，请确认选择带 ★ 的当前系统输出设备，并调高播放音量；Windows soundcard 模式无需 Stereo Mix")
        _set_step("capture", "pass", f"已捕获真实音频（RMS {energy:.4f}）")

        _set_step("stt", "running", "正在识别测试音频…")
        transcript = transcribe_segment(
            captured.astype(np.float32),
            AudioCapture.SAMPLE_RATE,
            position=getattr(cfg, "position", ""),
            language=getattr(cfg, "language", ""),
            scope="interviewer",
            config_snapshot=cfg,
        ) or ""
        _set_status(captured_transcript=transcript)
        ok, detail = match_phrase(expected_phrase, transcript)
        _set_step("stt", "pass" if transcript.strip() else "fail", transcript.strip() or "识别结果为空", {"transcript": transcript})
        _set_status(match_ok=ok)
        _set_step("match", "pass" if ok else "fail", detail, {"transcript": transcript, "expected_phrase": expected_phrase})
        if not ok:
            raise RuntimeError(detail)

        _set_step("llm", "running", "正在通过真实答题链路检测 LLM…")
        model_cfg, llm_result = generate_selected_preflight_answer(cfg, llm_question)
        answer = llm_result["answer"]
        if not answer:
            raise RuntimeError("模型回答为空")
        _log.info(
            "PREFLIGHT_LLM_OK model=%s question=%r first_token=%dms total=%dms answer_len=%d",
            model_cfg.name,
            llm_question,
            llm_result["first_token_ms"],
            llm_result["total_ms"],
            len(answer),
        )
        _set_step(
            "llm",
            "pass",
            f"首 token {llm_result['first_token_ms']}ms · 完整 {llm_result['total_ms']}ms",
            {
                "answer": answer,
                "question": llm_question,
                "first_token_ms": llm_result["first_token_ms"],
                "total_ms": llm_result["total_ms"],
                "model_name": model_cfg.name,
            },
        )

        _set_step("ws", "pass", "WebSocket 链路正常（您看到这条就说明已通）")
        _set_step("done", "done", "真实音频链路检测完成")
        _log.info("PREFLIGHT_DONE scenario=%s elapsed=%.0fms", resolved_scenario_id, (time.monotonic() - started_mono) * 1000)
    except Exception as e:
        _log.error("PREFLIGHT_ERROR scenario=%s requested=%s device_id=%s: %s", resolved_scenario_id, scenario_id, device_id, e, exc_info=True)
        _set_status(error=str(e))
        _set_step("error", "fail", f"检测异常: {e}")
    finally:
        with _lock:
            _status["running"] = False
            _status["finished_at"] = time.time()
            _running = False


def start_preflight(device_id: Optional[int] = None, scenario_id: str = "self_intro"):
    global _running
    scenario = resolve_preflight_scenario(scenario_id)
    resolved_scenario_id = scenario.get("id", scenario_id)
    expected_phrase = scenario.get("expected_phrase") or PREFLIGHT_EXPECTED_PHRASE
    with _lock:
        if _running or _replay_running:
            return False
        _running = True
        _status.update(
            running=True,
            scenario_id=resolved_scenario_id,
            device_id=device_id,
            expected_phrase=expected_phrase,
            captured_transcript=None,
            match_ok=None,
            steps={},
            error=None,
            started_at=time.time(),
            finished_at=None,
        )
    try:
        t = threading.Thread(target=_run_preflight, args=(device_id, scenario_id), daemon=True)
        t.start()
    except Exception:
        with _lock:
            _running = False
            _status["running"] = False
            _status["finished_at"] = time.time()
        raise
    return True


def get_scenarios():
    return PREFLIGHT_SCENARIOS
