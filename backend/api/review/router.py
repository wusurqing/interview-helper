import re
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.logger import get_logger
from services.storage import review
from services.storage import job_tracker
from services.storage import review_archive
from services import review_analysis, review_async_analysis

logger = get_logger(__name__)
router = APIRouter()


@router.get("/review/sessions/{session_id}/turns/{turn_id}/archive")
async def get_turn_archive(session_id: int, turn_id: int):
    detail = await run_in_threadpool(review_archive.get_archive, session_id, turn_id)
    if detail is None:
        raise HTTPException(404, "此记录未保存归档详情")
    return detail


@router.get("/review/sessions/{session_id}/turns/{turn_id}/audio/{asset_id}")
async def get_turn_audio(session_id: int, turn_id: int, asset_id: str):
    path = await run_in_threadpool(review_archive.asset_path, session_id, turn_id, asset_id)
    if path is None:
        raise HTTPException(404, "录音片段不存在")
    return FileResponse(path, media_type="audio/ogg", headers={"Cache-Control": "private, max-age=3600"})


class UpdateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    company: str | None = Field(default=None, max_length=120)
    role: str | None = Field(default=None, max_length=120)
    application_id: int | None = None


class ManualReviewRequest(BaseModel):
    transcript: str = Field(max_length=200_000)
    title: str | None = Field(default=None, max_length=200)
    company: str | None = Field(default=None, max_length=120)
    role: str | None = Field(default=None, max_length=120)
    analyze: bool = True


class AsrCorrectionTestRequest(BaseModel):
    question: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=20_000)


_QUESTION_PREFIX = re.compile(r"^\s*(?:Q(?:uestion)?\s*\d*|问题\s*\d*|问|面试官|Interviewer|HR)\s*[:：]\s*(.*)$", re.I)
_ANSWER_PREFIX = re.compile(r"^\s*(?:A(?:nswer)?\s*\d*|回答\s*\d*|答|候选人|Candidate|我)\s*[:：]\s*(.*)$", re.I)


def _parse_manual_turns(transcript: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    current_question = ""
    answer_parts: list[str] = []

    def flush() -> None:
        nonlocal current_question, answer_parts
        answer = "\n".join(part.strip() for part in answer_parts if part.strip()).strip()
        if current_question.strip() and answer:
            turns.append({
                "question_text": current_question.strip(),
                "candidate_answer_text": answer,
            })
        current_question = ""
        answer_parts = []

    for raw_line in transcript.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        question_match = _QUESTION_PREFIX.match(line)
        answer_match = _ANSWER_PREFIX.match(line)

        if question_match:
            flush()
            current_question = question_match.group(1).strip()
            continue
        if answer_match:
            if current_question:
                answer_parts.append(answer_match.group(1).strip())
            continue
        if current_question:
            answer_parts.append(line)

    flush()
    return turns


def _has_generated_analysis(detail: dict) -> bool:
    if detail.get("summary_markdown") or detail.get("avg_score") is not None:
        return True
    for turn in detail.get("turns", []):
        if turn.get("analysis_status") == "completed" and (
            turn.get("strengths") or turn.get("risks") or turn.get("scorecard")
        ):
            return True
    return False


def _create_manual_review_sync(req: ManualReviewRequest, turns: list[dict[str, str]]) -> dict:
    session_id = review.create_session(
        started_at=time.time(),
        interviewer_enabled=True,
        candidate_enabled=True,
        source="manual",
        title=req.title or "手动复盘",
        company=req.company or "",
        role=req.role or "",
    )
    for idx, turn in enumerate(turns, start=1):
        review.add_turn(
            session_id=session_id,
            qa_id=f"manual-{session_id}-{idx}",
            seq=idx,
            question_text=turn["question_text"],
            candidate_answer_text=turn["candidate_answer_text"],
            analysis_status="pending",
        )
    review.end_session(session_id, status="analyzing" if req.analyze else "completed", ended_at=time.time())

    if req.analyze:
        try:
            review_async_analysis.analyze_session_async(session_id)
        except Exception:
            logger.exception("Failed to submit manual review analysis for session_id=%d", session_id)
            try:
                review.update_session_status(session_id, "analysis_failed")
            except Exception:
                pass
            raise HTTPException(500, "复盘分析启动失败，请稍后重试")

    return {"session_id": session_id, "turn_count": len(turns), "status": "started" if req.analyze else "created"}


def _update_session_sync(session_id: int, req: UpdateSessionRequest) -> dict:
    if not review.get_session_detail(session_id):
        raise HTTPException(404, "Session not found")
    updates = req.model_dump(exclude_unset=True)
    if "application_id" in updates and updates["application_id"] is not None:
        if not job_tracker.get_application(int(updates["application_id"])):
            raise HTTPException(404, "Application not found")
    update_kwargs = {
        "session_id": session_id,
        "title": updates.get("title"),
        "company": updates.get("company"),
        "role": updates.get("role"),
    }
    if "application_id" in updates:
        update_kwargs["application_id"] = updates["application_id"]
    review.update_session_info(**update_kwargs)
    detail = review.get_session_detail(session_id)
    auto_sync_eligible = review.is_auto_sync_eligible_session(detail)
    return {
        "success": True,
        "synced_todos": bool(detail and detail.get("application_id") and auto_sync_eligible),
        "auto_sync_eligible": auto_sync_eligible,
    }


def _trigger_review_analysis_sync(session_id: int) -> dict:
    detail = review.get_session_detail(session_id)
    if not detail:
        raise HTTPException(404, "Session not found")

    if detail.get("archive_status") in ("recording", "draining"):
        raise HTTPException(409, "面试记录仍在收集或保存，请保存完成后再生成复盘")

    status = detail.get("status", "")

    # 如果正在分析，返回 pending
    if status == "analyzing":
        return {"status": "pending", "message": "分析正在进行中"}

    # completed 只有在已有分析内容时才表示真正完成；历史数据可能只是结束录制。
    if status == "completed" and _has_generated_analysis(detail):
        return {"status": "done", "message": "复盘已完成"}

    # 允许触发的状态：已录制待分析、历史 completed 空复盘、部分录制、分析失败或旧的 recording 归档。
    if status not in review.ANALYZABLE_SESSION_STATUSES:
        raise HTTPException(400, f"当前状态 {status} 不支持触发分析")

    # 原子条件 SQL 占位，防止并发双击重复启动分析任务
    claimed = review.claim_session_for_analysis(session_id)
    if not claimed:
        latest = review.get_session_detail(session_id)
        if not latest:
            raise HTTPException(404, "Session not found")
        if latest.get("archive_status") in ("recording", "draining"):
            raise HTTPException(409, "面试记录仍在收集或保存，请保存完成后再生成复盘")
        latest_status = latest.get("status", "")
        if latest_status == "analyzing":
            return {"status": "pending", "message": "分析正在进行中"}
        if latest_status == "completed" and _has_generated_analysis(latest):
            return {"status": "done", "message": "复盘已完成"}
        raise HTTPException(400, f"当前状态 {latest_status} 不支持触发分析")

    try:
        # 调用异步分析任务（有界线程池）
        review_async_analysis.analyze_session_async(session_id)
        return {"status": "started", "message": "复盘分析已开始"}
    except Exception:
        logger.exception("Failed to submit review analysis for session_id=%d", session_id)
        try:
            review.update_session_status(session_id, status)
        except Exception:
            pass
        raise HTTPException(500, "复盘分析启动失败，请稍后重试")


@router.get("/review/sessions")
async def list_sessions(page: int = 1, page_size: int = 20):
    """列表页（分页上下界由 review.list_sessions 统一钳制至 [1, MAX_REVIEW_PAGE_SIZE]）"""
    return await run_in_threadpool(review.list_sessions, page, page_size)


@router.post("/review/sessions/manual")
async def create_manual_review(req: ManualReviewRequest):
    """从粘贴的逐字稿/问答文本创建手动复盘。"""
    transcript = (req.transcript or "").strip()
    if len(transcript) < 12:
        raise HTTPException(400, "复盘文本太短，请粘贴至少一组问答")

    turns = _parse_manual_turns(transcript)
    if not turns:
        raise HTTPException(400, "未识别到问答结构，请使用“面试官: ... / 候选人: ...”或“Q: ... / A: ...”格式")

    return await run_in_threadpool(_create_manual_review_sync, req, turns)


@router.get("/review/sessions/{session_id}")
async def get_session(session_id: int):
    """详情页"""
    detail = await run_in_threadpool(review.get_session_detail, session_id)
    if not detail:
        raise HTTPException(404, "Session not found")
    return detail


@router.patch("/review/sessions/{session_id}")
async def update_session(session_id: int, req: UpdateSessionRequest):
    """更新会话信息（标题、公司、岗位）"""
    try:
        return await run_in_threadpool(_update_session_sync, session_id, req)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update review session session_id=%d", session_id)
        raise HTTPException(500, "更新复盘会话失败，请稍后重试")


@router.post("/review/asr-correction-test")
async def test_asr_correction(req: AsrCorrectionTestRequest):
    """测试当前复盘模型是否可用于 ASR 纠错（离事件循环执行，避免同步 LLM HTTP 阻塞全站）。"""
    return await run_in_threadpool(
        review_analysis.run_asr_correction_check,
        question=req.question or "请介绍一下你做过的缓存优化。",
        candidate_answer=req.answer or "",
    )


@router.get("/review/current")
async def get_current_session():
    """当前进行中的 session"""
    from services.review_integration import get_archive_status

    current = await run_in_threadpool(review.get_current_session)
    archive = await run_in_threadpool(get_archive_status)
    return {"session": current, "archive": archive}


@router.get("/review/profile")
async def get_profile():
    """长期画像（首版返回空）"""
    return {
        "session_count": 0,
        "summary": None,
        "strengths": [],
        "weaknesses": [],
        "behavior_traits": [],
        "domain_mastery": {},
        "trend": [],
    }


@router.post("/review/sessions/{session_id}/generate")
async def trigger_review_analysis(session_id: int):
    """手动触发复盘分析（用于一开始未开启但已落库的 session）"""
    return await run_in_threadpool(_trigger_review_analysis_sync, session_id)
