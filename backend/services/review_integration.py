"""
Review 模块与 Assist 生命周期集成
在 assist start/stop 时自动创建和结束 review session
"""
import time
from typing import Optional

from core.logger import get_logger
from core.session import Session
from core.config import get_config
from services.storage import review
from services import review_async_analysis

logger = get_logger(__name__)

_current_review_session_id: Optional[int] = None
_existing_qa_ids: frozenset[str] = frozenset()
_archive_status = None


def get_archive_status():
    from core.session import get_session
    archive = get_session().archive
    return archive.snapshot() if archive else _archive_status


def should_create_review_session(
    interviewer_device_id: Optional[int],
    candidate_device_id: Optional[int],
    candidate_asr_enabled: bool,
    written_exam_mode: bool = False,
    *,
    single_mic: bool = False,
) -> bool:
    """Speech archive requires interviewer capture only; written exams need neither input."""
    return written_exam_mode or interviewer_device_id is not None


def on_assist_start(
    interviewer_device_id: Optional[int],
    candidate_device_id: Optional[int],
    candidate_asr_enabled: bool,
    written_exam_mode: bool = False,
    *,
    single_mic: bool = False,
    existing_qa_ids: frozenset[str] = frozenset(),
) -> Optional[int]:
    """
    Assist 启动时调用，创建 review session
    返回 session_id 或 None
    """
    global _current_review_session_id, _existing_qa_ids

    if not should_create_review_session(
        interviewer_device_id,
        candidate_device_id,
        candidate_asr_enabled,
        written_exam_mode=written_exam_mode,
        single_mic=single_mic,
    ):
        logger.info("Review session not created: conditions not met")
        return None

    try:
        session_id = review.create_session(
            started_at=time.time(),
            interviewer_enabled=interviewer_device_id is not None,
            candidate_enabled=(
                candidate_device_id is not None
                and candidate_asr_enabled
                and (single_mic or candidate_device_id != interviewer_device_id)
            ),
            source="written_exam" if written_exam_mode else "assist",
        )
        _current_review_session_id = session_id
        _existing_qa_ids = existing_qa_ids
        if not written_exam_mode:
            from api.realtime.ws import broadcast
            from services.interview_archive.manager import InterviewArchive
            from services.storage import review_archive

            def archive_event(event):
                global _archive_status
                if _archive_status is None or _archive_status["session_id"] <= session_id:
                    _archive_status = {k: v for k, v in event.items() if k != "type"}
                broadcast(event)

            def completed():
                state = archive.snapshot()
                partial = bool(state["warning"])
                review.end_session(session_id, status="partial_capture" if partial else "recorded")
                review_archive.set_state(session_id, "partial" if partial else "complete", state["warning"] or None)
                if _should_auto_analyze(review_enabled=bool(get_config().review_enabled),
                                        turn_count=state["persisted_turn_count"]):
                    review_async_analysis.analyze_session_async(session_id)

            archive = InterviewArchive(session_id, candidate_enabled=bool(candidate_device_id is not None and candidate_asr_enabled),
                                       broadcast=archive_event, on_complete=completed)
            from core.session import get_session
            get_session().archive = archive
            review_archive.set_state(session_id, "recording")
        logger.info("Review session created: session_id=%d", session_id)
        return session_id
    except Exception as e:
        logger.error("Failed to create review session: %s", e, exc_info=True)
        return None


def update_candidate_capture(archive, *, enabled, available):
    archive.candidate_enabled = available
    archive.candidate_unavailable = enabled and not available
    try:
        from services.storage import review_archive
        review_archive.set_candidate_capture(archive.session_id, available)
    except Exception:
        archive.warning("候选人采集状态保存失败")
        logger.exception("Unable to update archive candidate capture flags")


def on_assist_stop(session: Session) -> Optional[int]:
    """
    Assist 停止时调用，结束 review session 并保存所有 turns
    返回已结束的 session_id 或 None
    """
    global _current_review_session_id, _existing_qa_ids

    if _current_review_session_id is None:
        return None

    session_id = _current_review_session_id
    _current_review_session_id = None
    existing_qa_ids = _existing_qa_ids
    _existing_qa_ids = frozenset()

    if session.archive is not None:
        from services.storage import review_archive
        archive, session.archive = session.archive, None
        review_archive.set_state(session_id, "draining")
        archive.stop(timeout=3)
        return session_id

    qa_pairs = [qa for qa in session.qa_pairs if qa.id not in existing_qa_ids]

    try:
        for idx, qa in enumerate(qa_pairs, start=1):
            candidate_answer = session.get_candidate_answer_for_qa(qa.id, max_chars=2000)
            evidence = _build_turn_evidence(qa)
            review.add_turn(
                session_id=session_id,
                qa_id=qa.id,
                seq=idx,
                question_text=qa.question,
                candidate_answer_text=candidate_answer,
                reference_answer_text=qa.answer,  # LLM 给出的参考答案
                duration_ms=0,  # 暂时无法精确计算每题时长
                is_partial=False,
                analysis_status="pending",
                evidence=evidence,
            )

        # 结束录制。review_enabled 控制是否立即进入分析队列; 关闭时只落盘为
        # recorded, 保留 ended_at, 等待前端手动触发 (POST /review/sessions/{id}/generate)。
        # 笔试练习即使只有 1 题也自动生成报告；看板自动同步仍由 AUTO_REVIEW_SYNC_MIN_TURNS 控制。
        # completed 只表示分析结果已经生成。
        turn_count = len(qa_pairs)
        source = str((review.get_session_detail(session_id) or {}).get("source") or "assist")
        auto_analyze = _should_auto_analyze(
            review_enabled=bool(get_config().review_enabled),
            turn_count=turn_count,
            source=source,
        )
        review.end_session(
            session_id=session_id,
            status="analyzing" if auto_analyze else "recorded",
            ended_at=time.time(),
        )

        logger.info(
            "Review session ended: session_id=%d, turn_count=%d, auto_analyze=%s",
            session_id,
            turn_count,
            auto_analyze,
        )

        if auto_analyze:
            # 启动后台分析
            review_async_analysis.analyze_session_async(session_id)
        else:
            logger.info(
                "Review session %d recorded (auto-analysis skipped; review_enabled=%s, turn_count=%d, min_turns=%d)",
                session_id,
                bool(get_config().review_enabled),
                turn_count,
                review.AUTO_REVIEW_SYNC_MIN_TURNS,
            )

        return session_id
    except Exception as e:
        logger.error("Failed to end review session %d: %s", session_id, e, exc_info=True)
        return None


def get_current_review_session_id() -> Optional[int]:
    """返回当前进行中的 review session id"""
    return _current_review_session_id


def _build_turn_evidence(qa) -> dict:
    verdict = str(getattr(qa, "vision_verify_verdict", "") or "").strip().upper()
    if verdict not in ("PASS", "FAIL", "UNKNOWN"):
        return {}
    return {
        "vision_verify": {
            "verdict": verdict,
            "reason": str(getattr(qa, "vision_verify_reason", "") or "").strip(),
        },
    }


def _should_auto_analyze(*, review_enabled: bool, turn_count: int, source: str = "assist") -> bool:
    if not review_enabled or turn_count <= 0:
        return False
    if str(source or "").strip() == "written_exam":
        return True
    return turn_count >= review.AUTO_REVIEW_SYNC_MIN_TURNS


def on_assist_pause():
    """Assist 暂停时调用，暂不需要特殊处理"""
    pass


def on_assist_resume():
    """Assist 恢复时调用，暂不需要特殊处理"""
    pass
