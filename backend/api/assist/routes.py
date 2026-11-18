"""Interview assist HTTP routes."""

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from core.config import get_config
from core.logger import get_logger
from core.session import get_session, snapshot_session
from services.audio import AudioBusyError
from services.stt.mmpg import valid_single_mic_id
from . import candidate_mmpg
from . import mmpg_integration as mmpg
from .answer_worker import prompt_server_screen_code
from .pipeline import (
    broadcast_single_mic,
    cancel_answer_work,
    is_paused,
    is_streaming_interview,
    pause_interview,
    pick_model_index,
    start_nonblocking,
    stop_interview_loop,
    submit_answer_task,
    unpause_interview,
    _bump_generation,
)
from .single_mic import Role, RoleSwitch, SingleMicConflict, SingleMicSession

router = APIRouter()
_rlog = get_logger("assist.routes")
_role_switches: dict[RoleSwitch, asyncio.Task[tuple[int, dict | None]]] = {}


class ManualQuestion(BaseModel):
    text: str = Field(..., max_length=10000)
    image: Optional[str] = None
    override_think_mode: Optional[bool] = None
    purpose: Literal["interview", "knowledge_practice"] = "interview"


class SingleMicRoleRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    role: Role


class MultiServerScreenQuestion(BaseModel):
    images: list[str]
    override_think_mode: Optional[bool] = None


class ServerScreenQuestion(BaseModel):
    override_think_mode: Optional[bool] = None


@router.post("/start")
async def api_start(body: dict):
    async with mmpg.http_lifecycle_lock:
        return await _api_start(body)


async def _api_start(body: dict):
    from .exam_test import is_exam_preflight_running

    if is_exam_preflight_running():
        raise HTTPException(409, "笔试链路检测正在运行，请等待检测结束后再开始")
    cfg = get_config()
    written_exam = bool(getattr(cfg, "written_exam_mode", False))
    if mmpg.manager.active:
        raise HTTPException(409, "豆包语义判停会话正在连接或运行")
    single_mic_id = body.get("single_mic_id")
    if single_mic_id is not None and not valid_single_mic_id(single_mic_id):
        raise HTTPException(400, "无效的单人测试会话标识")
    session = get_session()
    if session.single_mic is not None or (single_mic_id is not None and session.is_recording):
        raise HTTPException(409, "请先结束当前面试")
    if single_mic_id is not None and written_exam:
        raise HTTPException(400, "单人测试适用于语音面试")
    uses_mmpg = getattr(cfg, "stt_provider", None) == mmpg.PROVIDER
    if not written_exam and uses_mmpg:
        raise HTTPException(409, "请从桌面主窗口或本机浏览器开始豆包语义判停")
    device_id = None if written_exam and uses_mmpg else body.get("device_id")
    candidate_mic_device_id = body.get("candidate_mic_device_id")
    if not written_exam and device_id is None:
        raise HTTPException(400, "\u8bf7\u9009\u62e9\u97f3\u9891\u8bbe\u5907")
    if device_id is not None:
        try:
            dev = int(device_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "device_id \u5fc5\u987b\u662f\u6574\u6570")
    else:
        dev = None
    if candidate_mic_device_id is not None:
        try:
            candidate_dev = int(candidate_mic_device_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "candidate_mic_device_id \u5fc5\u987b\u662f\u6574\u6570")
    else:
        candidate_dev = None
    try:
        await candidate_mmpg.stop("stopped")
        if single_mic_id is not None:
            await run_in_threadpool(
                start_nonblocking, dev, config_snapshot=cfg, single_mic_id=single_mic_id,
            )
            mmpg.watch_single_mic(get_session().single_mic)
        elif dev is not None and not written_exam:
            await run_in_threadpool(start_nonblocking, dev, candidate_dev, config_snapshot=cfg)
        elif candidate_dev is None:
            await run_in_threadpool(start_nonblocking, dev)
        else:
            await run_in_threadpool(start_nonblocking, dev, candidate_dev)
        return {"ok": True}
    except AudioBusyError as e:
        raise HTTPException(409, str(e))
    except SingleMicConflict as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        _rlog.error("api_start failed: %s", e, exc_info=True)
        raise HTTPException(500, "启动失败，请重试")


@router.post("/stop")
async def api_stop():
    async with mmpg.http_lifecycle_lock:
        single_mic = get_session().single_mic
        if single_mic is not None:
            await mmpg.drain_single_mic_transports(single_mic, "stopped")
        await asyncio.gather(
            mmpg.manager.stop("stopped"),
            candidate_mmpg.stop("stopped", drain=single_mic is None),
        )
        await run_in_threadpool(stop_interview_loop)
    return {"ok": True}


@router.post("/pause")
async def api_pause(body: dict | None = None):
    async with mmpg.http_lifecycle_lock:
        session = get_session()
        single_mic_id = (body or {}).get("single_mic_id")
        if single_mic_id is not None and (session.single_mic is None or session.single_mic.id != single_mic_id):
            raise HTTPException(409, "单人测试会话已变化")
        if session.single_mic is not None:
            if not session.is_recording or session.is_paused:
                raise HTTPException(400, "单人测试未在收音")
            await mmpg.drain_single_mic_transports(session.single_mic, "paused")
            await run_in_threadpool(pause_interview)
            await asyncio.gather(mmpg.manager.stop("paused"), candidate_mmpg.stop("paused"))
            return {"ok": True}
        if is_streaming_interview() or mmpg.manager.active:
            await asyncio.gather(mmpg.manager.stop("paused"), candidate_mmpg.stop("paused", drain=True))
            return {"ok": True}
        session = get_session()
        if not session.is_recording:
            raise HTTPException(400, "面试未在进行中")
        if is_paused():
            raise HTTPException(400, "已经处于暂停状态")
        await asyncio.gather(run_in_threadpool(pause_interview), candidate_mmpg.stop("paused", drain=True))
    return {"ok": True}


@router.post("/unpause")
async def api_resume_interview(body: Optional[dict] = None):
    async with mmpg.http_lifecycle_lock:
        return await _api_resume_interview(body)


async def _api_resume_interview(body: Optional[dict] = None):
    if is_streaming_interview():
        raise HTTPException(409, "请从桌面主窗口或本机浏览器恢复豆包语义判停")
    session = get_session()
    if not session.is_recording:
        raise HTTPException(400, "\u9762\u8bd5\u672a\u5728\u8fdb\u884c\u4e2d")
    if not is_paused():
        raise HTTPException(400, "\u9762\u8bd5\u672a\u5904\u4e8e\u6682\u505c\u72b6\u6001")
    single_mic_id = (body or {}).get("single_mic_id")
    if session.single_mic is not None or single_mic_id is not None:
        if session.single_mic is None or session.single_mic.id != single_mic_id:
            raise HTTPException(409, "单人测试会话已变化")
        try:
            await run_in_threadpool(unpause_interview)
        except (AudioBusyError, SingleMicConflict) as e:
            raise HTTPException(409, str(e))
        except Exception:
            _rlog.exception("single microphone resume failed")
            raise HTTPException(500, "麦克风启动失败，请检查设备后重试")
        return {"ok": True}
    device_id = (body or {}).get("device_id")
    candidate_mic_device_id = (body or {}).get("candidate_mic_device_id")
    if device_id is not None:
        try:
            int(device_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "device_id \u5fc5\u987b\u662f\u6574\u6570")
    if candidate_mic_device_id is not None:
        try:
            int(candidate_mic_device_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "candidate_mic_device_id \u5fc5\u987b\u662f\u6574\u6570")
    try:
        if candidate_mic_device_id is None:
            await run_in_threadpool(
                unpause_interview,
                int(device_id) if device_id is not None else None,
            )
        else:
            await run_in_threadpool(
                unpause_interview,
                int(device_id) if device_id is not None else None,
                int(candidate_mic_device_id),
            )
    except AudioBusyError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


async def _complete_role_switch(single_mic: SingleMicSession, operation: RoleSwitch) -> tuple[int, dict | None]:
    try:
        try:
            timeout = max(6.0, get_config().assist_interviewer_asr_drain_timeout_sec)
            ready = await mmpg.finish_single_mic_role(single_mic, operation, timeout)
        except Exception:
            _rlog.exception("single microphone ASR did not finish")
            ready = False
        async with mmpg.http_lifecycle_lock:
            if get_session().single_mic is not single_mic or not single_mic.owns_switch(operation):
                return 409, None
            if ready and operation.streaming and operation.from_role == "interviewer":
                try:
                    mmpg.finish_interviewer_role(single_mic)
                except Exception:
                    _rlog.exception("single microphone interviewer role did not finish")
                    ready = False
            if not single_mic.finish_switch(operation, ready):
                return 409, None
            broadcast_single_mic(single_mic)
            return (200 if ready else 504), single_mic.snapshot()
    finally:
        _role_switches.pop(operation, None)


@router.post("/single-mic/role")
async def api_single_mic_role(body: SingleMicRoleRequest):
    async with mmpg.http_lifecycle_lock:
        session = get_session()
        single_mic = session.single_mic
        if single_mic is None or single_mic.id != body.session_id or not session.is_recording:
            raise HTTPException(409, "单人测试会话已变化")
        try:
            operation = single_mic.begin_switch(body.role)
        except SingleMicConflict as e:
            raise HTTPException(409, str(e))
        if operation is None:
            return {"ok": True, "single_mic": single_mic.snapshot()}
        task = _role_switches.get(operation)
        if task is None:
            broadcast_single_mic(single_mic)
            task = _role_switches[operation] = asyncio.create_task(_complete_role_switch(single_mic, operation))
    status, snapshot = await asyncio.shield(task)
    if status == 409:
        raise HTTPException(409, "切换已取消，请确认当前角色后重试")
    if status == 504:
        raise HTTPException(504, "上一句转写未能及时完成，请重试切换")
    return {"ok": True, "single_mic": snapshot}


@router.post("/clear")
async def api_clear():
    from api.realtime.ws import broadcast
    async with mmpg.http_lifecycle_lock:
        single_mic = get_session().single_mic
        if single_mic is not None:
            single_mic.pause()
        mmpg.clear_turns()
        candidate_mmpg.manager.discard_turns()
        cancel_answer_work(invalidate_capture=True)
        await asyncio.gather(mmpg.manager.stop("stopped"), candidate_mmpg.stop("stopped"))
        await run_in_threadpool(stop_interview_loop)
        cancel_answer_work(reset_session_data=True)
        broadcast({"type": "session_cleared"})
    return {"ok": True}


@router.post("/ask/cancel")
async def api_ask_cancel():
    mmpg.clear_turns()
    _bump_generation()
    cancel_answer_work(reset_session_data=False)
    return {"ok": True}


@router.post("/ask")
async def api_ask(body: ManualQuestion):
    if not body.text.strip() and not body.image:
        raise HTTPException(400, "问题不能为空")
    text = body.text.strip() or "粘贴的图片"
    src = "manual_image" if body.image else "manual_text"
    meta = {"origin": "manual"}
    if body.purpose == "knowledge_practice":
        meta["purpose"] = body.purpose
    if body.override_think_mode is not None:
        meta["override_think_mode"] = bool(body.override_think_mode)
    queued = submit_answer_task((text, body.image, True, src, meta))
    if not queued:
        raise HTTPException(503, "没有可用的答题模型，请先启用并配置至少一个模型")
    return {"ok": True}


@router.post("/ask-from-server-screen")
async def api_ask_from_server_screen(body: Optional[ServerScreenQuestion] = None):
    from services.llm import has_vision_model
    from services.capture import ScreenCaptureError, capture_primary_left_half_data_url

    if not has_vision_model():
        raise HTTPException(400, "请至少配置一个支持识图且已填写 API Key 的模型")
    try:
        data_url = capture_primary_left_half_data_url()
    except ScreenCaptureError as e:
        raise HTTPException(503, str(e))
    cfg = get_config()
    region = getattr(cfg, "screen_capture_region", "left_half") or "left_half"
    text = prompt_server_screen_code(region)
    meta = {"origin": "server_screen"}
    if body is not None and body.override_think_mode is not None:
        meta["override_think_mode"] = bool(body.override_think_mode)
    if pick_model_index((text, data_url, True, "server_screen_left", meta), set()) is None:
        raise HTTPException(400, "没有可用的识图模型，请检查启用状态与 API Key")
    # Written exam: keep in-flight answers so multi-shot screenshots can run in parallel.
    # Non-exam interview: cancel previous generation (screenshot replaces the live answer).
    if not bool(getattr(cfg, "written_exam_mode", False)):
        cancel_answer_work(reset_session_data=False)
    queued = submit_answer_task((text, data_url, True, "server_screen_left", meta))
    if not queued:
        raise HTTPException(503, "没有可用的识图模型，请检查启用状态与 API Key")
    return {"ok": True}


@router.post("/capture-server-screen")
async def api_capture_server_screen():
    from services.capture import ScreenCaptureError, capture_primary_left_half_data_url

    try:
        return {"ok": True, "image": capture_primary_left_half_data_url()}
    except ScreenCaptureError as e:
        raise HTTPException(503, str(e))


@router.post("/ask-from-server-screens")
async def api_ask_from_server_screens(body: MultiServerScreenQuestion):
    from services.llm import has_vision_model

    images = [img for img in body.images if img]
    if not images:
        raise HTTPException(400, "至少需要一张截图")
    if len(images) > 8:
        raise HTTPException(400, "一次最多提交 8 张截图")
    if not has_vision_model():
        raise HTTPException(400, "请至少配置一个支持识图且已填写 API Key 的模型")

    cfg = get_config()
    region = getattr(cfg, "screen_capture_region", "left_half") or "left_half"
    text = prompt_server_screen_code(region, image_count=len(images))
    meta = {"origin": "server_screen", "image_count": len(images)}
    if body.override_think_mode is not None:
        meta["override_think_mode"] = bool(body.override_think_mode)
    task = (text, images, True, "server_screen_multi", meta)
    if pick_model_index(task, set()) is None:
        raise HTTPException(400, "没有可用的识图模型，请检查启用状态与 API Key")
    if not bool(getattr(cfg, "written_exam_mode", False)):
        cancel_answer_work(reset_session_data=False)
    queued = submit_answer_task(task)
    if not queued:
        raise HTTPException(503, "没有可用的识图模型，请检查启用状态与 API Key")
    return {"ok": True}


@router.get("/preflight/scenarios")
async def api_preflight_scenarios():
    from .sound_test import get_scenarios
    return {"scenarios": get_scenarios()}


@router.post("/preflight/run")
async def api_preflight_run(body: dict):
    _check_segment_preflight()
    from .sound_test import start_preflight
    scenario_id = body.get("scenario_id", "self_intro")
    device_id = body.get("device_id")
    if device_id is not None:
        try:
            device_id = int(device_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "device_id 必须是整数")
    ok = start_preflight(
        device_id=device_id,
        scenario_id=scenario_id,
    )
    if not ok:
        raise HTTPException(409, "链路检测已在运行中")
    return {"ok": True}


@router.get("/preflight/status")
async def api_preflight_status():
    from .sound_test import get_preflight_status
    return get_preflight_status()


@router.post("/preflight/replay")
async def api_preflight_replay(body: dict):
    _check_segment_preflight()
    session = get_session()
    if session.is_recording and not session.is_paused:
        raise HTTPException(409, "请先暂停或结束面试，再做高频回放检测")
    device_id = body.get("device_id")
    if device_id is None:
        raise HTTPException(400, "device_id 必须提供")
    try:
        device_id = int(device_id)
    except (TypeError, ValueError):
        raise HTTPException(400, "device_id 必须是整数")
    repeats = body.get("repeats", 3)
    gap_sec = body.get("gap_sec", 0.25)
    audio_path = body.get("audio_path")
    expected_phrase = body.get("expected_phrase")
    try:
        repeats = int(repeats)
    except (TypeError, ValueError):
        raise HTTPException(400, "repeats 必须是整数")
    try:
        gap_sec = float(gap_sec)
    except (TypeError, ValueError):
        raise HTTPException(400, "gap_sec 必须是数字")
    try:
        from .sound_test import replay_preflight_capture_stt
        return await run_in_threadpool(
            replay_preflight_capture_stt,
            device_id,
            repeats=repeats,
            gap_sec=gap_sec,
            audio_path=audio_path,
            expected_phrase=expected_phrase,
        )
    except AudioBusyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        _rlog.warning("preflight replay failed device=%s: %s", device_id, e, exc_info=True)
        raise HTTPException(503, f"高频回放检测失败: {e}")


@router.post("/exam-preflight/run")
async def api_exam_preflight_run():
    from .exam_test import start_exam_preflight

    session = get_session()
    if session.is_recording or mmpg.manager.active:
        raise HTTPException(409, "请先结束当前面试或笔试，再运行链路检测")
    preflight_id = start_exam_preflight()
    if not preflight_id:
        if get_session().is_recording:
            raise HTTPException(409, "请先结束当前面试或笔试，再运行链路检测")
        raise HTTPException(409, "笔试链路检测已在运行中")
    return {"ok": True, "preflight_id": preflight_id}


def _check_segment_preflight():
    if get_config().stt_provider == mmpg.PROVIDER:
        raise HTTPException(409, "豆包语义判停请在语音设置中使用「保存并测试」")
    if mmpg.manager.active:
        raise HTTPException(409, "请先结束豆包语义判停会话或检测")


@router.get("/exam-preflight/status")
async def api_exam_preflight_status():
    from .exam_test import get_exam_preflight_status
    return get_exam_preflight_status()


@router.post("/audio-test/output")
async def api_audio_test_output():
    session = get_session()
    if session.is_recording and not session.is_paused:
        raise HTTPException(409, "请先暂停或结束面试，再测试音频输出")
    try:
        from .sound_test import play_preflight_audio
        elapsed_sec = await run_in_threadpool(play_preflight_audio)
        return {"ok": True, "elapsed_sec": elapsed_sec}
    except Exception as e:
        _rlog.warning("audio output test failed: %s", e, exc_info=True)
        raise HTTPException(503, f"测试音频播放失败: {e}")


@router.post("/audio-test/input")
async def api_audio_test_input(body: dict):
    session = get_session()
    if session.is_recording and not session.is_paused:
        raise HTTPException(409, "请先暂停或结束面试，再测试麦克风输入")
    device_id = body.get("device_id")
    if device_id is None:
        raise HTTPException(400, "device_id 必须提供")
    try:
        device_id = int(device_id)
    except (TypeError, ValueError):
        raise HTTPException(400, "device_id 必须是整数")
    duration_sec = body.get("duration_sec", 1.2)
    try:
        duration_sec = float(duration_sec)
    except (TypeError, ValueError):
        duration_sec = 1.2
    try:
        from .sound_test import test_input_audio
        return await run_in_threadpool(test_input_audio, device_id, duration_sec)
    except AudioBusyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        _rlog.warning("audio input test failed device=%s: %s", device_id, e, exc_info=True)
        raise HTTPException(503, f"麦克风输入测试失败: {e}")


@router.post("/audio-test/input/start")
async def api_audio_test_input_start(body: dict):
    session = get_session()
    if session.is_recording and not session.is_paused:
        raise HTTPException(409, "请先暂停或结束面试，再测试麦克风输入")
    device_id = body.get("device_id")
    if device_id is None:
        raise HTTPException(400, "device_id 必须提供")
    try:
        device_id = int(device_id)
    except (TypeError, ValueError):
        raise HTTPException(400, "device_id 必须是整数")
    try:
        from .sound_test import start_input_level_monitor
        return await run_in_threadpool(start_input_level_monitor, device_id)
    except Exception as e:
        _rlog.warning("audio input monitor start failed device=%s: %s", device_id, e, exc_info=True)
        raise HTTPException(503, f"麦克风输入监听失败: {e}")


@router.get("/audio-test/input/status")
async def api_audio_test_input_status():
    from .sound_test import get_input_level_status
    return get_input_level_status()


@router.post("/audio-test/input/stop")
async def api_audio_test_input_stop():
    from .sound_test import stop_input_level_monitor
    return await run_in_threadpool(stop_input_level_monitor)


@router.get("/session")
async def api_session():
    return snapshot_session()
