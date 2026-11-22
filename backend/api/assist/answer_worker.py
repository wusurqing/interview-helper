"""Answer generation worker for the assist pipeline."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from core.config import get_config
from core.logger import get_logger
from core.session import get_session, conversation_lock
from services.stt import classify_followup, normalize_transcription_for_analysis
from services.llm import (
    PROMPT_MODE_ASR_REALTIME,
    PROMPT_MODE_MANUAL_TEXT,
    PROMPT_MODE_SERVER_SCREEN,
    PROMPT_MODE_WRITTEN_EXAM,
    PromptMode,
    build_system_prompt,
    chat_stream_single_model,
    create_answer_stream_sanitizer,
    get_token_stats,
    LLMError,
    postprocess_answer_for_mode,
)
from services.llm.interview_budget import stream_interview_answer
from services.llm.prompts import (
    assist_prompt_scene,
    build_assist_user_context,
    config_for_assist_generation,
    format_interview_utterance,
)
from api.assist.scheduler import EXAM_MODEL_STALE, TaskPayload, task_is_written_exam


@dataclass(frozen=True)
class AnswerCommit:
    apply: Callable[[], None]
    cancel: Callable[[], None]

    def __call__(self) -> None:
        self.apply()


@dataclass(frozen=True)
class AnswerWorkerDeps:
    abort_check: Callable[[], bool]
    is_session_current: Callable[[int], bool]
    flush_commit: Callable[[int, Callable[[], None]], None]
    mark_seq_skipped: Callable[[int], None]
    submit_knowledge_record: Callable[[str, str, str, str], bool]
    broadcast: Callable[[dict], None]
    logger: Any
    error_logger: Any
    start_abort_check: Optional[Callable[[], bool]] = None
    model_cfg_snapshot: Any = None
    # Exam options originate at submission; other options originate at dispatch.
    # Session state and exam catalog availability are still checked at runtime.
    config_snapshot: Any = None
    reserve_retry: Callable[[set[int], Callable[[], None]], int | None] | None = None
    resolve_exam_model: Callable[[int], Any] | None = None


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    think: str
    first_token_at: float | None
    prompt_tokens: int
    completion_tokens: int


class EmptyAnswer(LLMError):
    user_msg = "模型未返回有效答案，请重试或更换模型。"


class ExamModelUnavailable(LLMError):
    user_msg = EXAM_MODEL_STALE


def _screen_region_label(region: str) -> str:
    labels = {
        "full": "主显示器全屏",
        "left_half": "主显示器左半屏",
        "right_half": "主显示器右半屏",
        "top_half": "主显示器上半屏",
        "bottom_half": "主显示器下半屏",
    }
    return labels.get(region, "主显示器左半屏")


def prompt_server_screen_code(region: str, *, image_count: int = 1) -> str:
    where = _screen_region_label(region)
    if image_count > 1:
        return f"电脑截图（{where}，共 {image_count} 张连续截图，按页序排列）"
    return f"电脑截图（{where}）"


def _normalize_task_images(image: Any) -> list[str]:
    if isinstance(image, list):
        return [str(x) for x in image if x]
    if image:
        return [str(image)]
    return []


def _image_payload_chars(images: list[str]) -> int:
    return sum(len(img or "") for img in images)


def _message_text_chars(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(str(part.get("text") or ""))
    return total


def _clip_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip() + "…"


def _context_excerpt(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").split())
    budget = max(80, int(max_chars or 0))
    if len(cleaned) <= budget:
        return cleaned
    if budget <= 140:
        return _clip_text(cleaned, budget)
    head_budget = max(60, min(120, budget // 2))
    tail_budget = max(60, budget - head_budget - 3)
    return f"{cleaned[:head_budget].rstrip()}…{cleaned[-tail_budget:].lstrip()}"


def _written_exam_question_context_label(question: str, source: str) -> str:
    cleaned = " ".join(str(question or "").split())
    if (
        cleaned.startswith(("下图来自运行本后端", "电脑截图（"))
        or "请基于图中可见信息作答" in cleaned
    ):
        if source == "server_screen_multi":
            return "上一批连续截图题面（无 OCR 文本）"
        return "上一张截图题面（无 OCR 文本）"
    return _clip_text(cleaned, 260)


def _written_exam_followup_context(session_ref, *, source: str, image_count: int) -> str:
    if image_count <= 0 or not (source or "").startswith("server_screen_"):
        return ""
    recent_qas = [
        qa
        for qa in session_ref.qa_pairs[-4:]
        if (qa.question or qa.answer)
        and qa.written_exam
        and (qa.source or "").startswith("server_screen_")
        and not (qa.source or "").endswith("exam_preflight")
    ][-2:]
    if not recent_qas:
        return ""

    lines = [
        "[笔试连续截图上下文]",
        "最近上一版答案参考:",
    ]
    for idx, qa in enumerate(recent_qas, start=1):
        lines.append(
            f"{idx}. 题目/截图: {_written_exam_question_context_label(qa.question, qa.source)}"
        )
        lines.append(f"   上一版答案: {_clip_text(qa.answer, 900)}")
    return "\n".join(lines)


def _candidate_context_settings(cfg) -> tuple[bool, int, int, int]:
    enabled = bool(getattr(cfg, "candidate_asr_enabled", False)) and bool(
        getattr(cfg, "candidate_context_enabled", True)
    )
    return (
        enabled,
        max(0, min(2000, int(getattr(cfg, "candidate_context_wait_ms", 200) or 0))),
        max(100, min(4000, int(getattr(cfg, "candidate_context_max_chars", 900) or 900))),
        max(1, min(100, int(getattr(cfg, "candidate_context_min_chars", 6) or 6))),
    )


_CANDIDATE_BACKGROUND_STOP_TERMS = frozenset(
    {
        "一个",
        "一下",
        "上一个",
        "上一轮",
        "上一",
        "上轮",
        "上面",
        "下一个",
        "这个",
        "那个",
        "这些",
        "那些",
        "怎么",
        "如何",
        "什么",
        "为什么",
        "为什",
        "可以",
        "能不能",
        "你们",
        "我们",
        "他们",
        "里面",
        "就是",
        "如果",
        "然后",
        "还有",
        "另外",
        "但是",
        "不过",
        "具体",
        "详细",
        "展开",
        "补充",
        "介绍",
        "讲讲",
        "说说",
        "回答",
        "问题",
        "项目",
        "经历",
        "业务",
        "核心",
        "实现",
        "处理",
        "方案",
        "使用",
        "用了",
        "实际",
        "讲的是",
        "候选",
        "候选人",
    }
)
_CANDIDATE_BACKGROUND_EN_STOP_TERMS = frozenset(
    {
        "and",
        "are",
        "can",
        "could",
        "did",
        "does",
        "for",
        "how",
        "that",
        "the",
        "this",
        "what",
        "when",
        "why",
        "would",
        "you",
        "your",
    }
)


def _candidate_background_topic_terms(text: str) -> set[str]:
    normalized = normalize_transcription_for_analysis(text).lower()
    if not normalized:
        return set()
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_+#.-]{1,}", normalized):
        cleaned = token.strip("._-")
        if len(cleaned) >= 2 and cleaned not in _CANDIDATE_BACKGROUND_EN_STOP_TERMS:
            terms.add(cleaned)

    for block in re.findall(r"[\u4e00-\u9fff]+", normalized):
        max_size = min(4, len(block))
        for size in range(2, max_size + 1):
            for start in range(0, len(block) - size + 1):
                term = block[start: start + size]
                if term not in _CANDIDATE_BACKGROUND_STOP_TERMS:
                    terms.add(term)
    return terms


def _candidate_background_matches_current_question(
    question_text: str,
    last_qa: Any,
    actual_spoken_answer: str,
) -> bool:
    question_terms = _candidate_background_topic_terms(question_text)
    if not question_terms:
        return False
    previous_terms = _candidate_background_topic_terms(
        f"{getattr(last_qa, 'question', '')} {actual_spoken_answer}"
    )
    return bool(question_terms & previous_terms)


_FOLLOWUP_BRIDGE_PREFIXES = (
    "那",
    "那么",
    "然后",
    "所以",
    "但是",
    "不过",
    "另外",
    "还有",
    "这个",
    "那个",
    "这样",
    "这种",
    "刚才",
    "前面",
    "上面",
    "how",
    "why",
    "what about",
    "then",
    "so",
    "and",
    "also",
    "could you",
    "can you",
)
_FOLLOWUP_BRIDGE_PHRASES = (
    "刚才说的",
    "前面说的",
    "上一个",
    "上一轮",
    "这个怎么",
    "那个怎么",
    "怎么验证",
    "为什么不",
    "展开讲",
    "详细讲",
    "具体讲",
    "举个例子",
    "补充一下",
    "接着说",
    "继续说",
    "you just mentioned",
    "you said",
    "previous one",
    "last round",
    "how did you",
    "how would you",
    "how do you",
    "why did you",
    "why not",
    "can you elaborate",
    "could you elaborate",
    "give an example",
    "tell me more",
    "continue",
)

_RESUME_CONTEXT_CUES = (
    "简历",
    "实习",
    "项目经历",
    "项目经验",
    "实习经历",
    "工作经历",
    "过往经历",
    "个人经历",
    "你的经历",
    "你过去经历",
    "自我介绍",
    "介绍一下自己",
    "介绍下自己",
    "介绍你自己",
    "介绍一下你自己",
    "简单介绍",
    "先介绍一下",
    "先做个介绍",
    "先做个自我介绍",
    "说说你自己",
    "聊聊你自己",
    "你做过",
    "你之前做",
    "你负责",
    "你们当时",
    "上一家公司",
    "项目里",
    "项目中",
    "项目上",
    "你的项目",
    "你们项目",
    "你在项目",
    "做过的项目",
    "负责的项目",
    "参与的项目",
    "最近的项目",
    "讲讲项目",
    "说说项目",
    "resume",
    "cv",
    "tell me about yourself",
    "introduce yourself",
    "introduce yourself briefly",
    "quick introduction",
    "brief introduction",
    "walk me through your resume",
    "walk me through your cv",
    "walk me through your background",
    "walk through your resume",
    "walk through your background",
    "share your background",
    "about your background",
    "my background",
    "your background",
    "my experience",
    "your experience",
    "my project",
    "your project",
    "previous project",
    "project experience",
    "internship",
    "work experience",
    "previous company",
    "what you built",
    "what you worked on",
)
_RESUME_CONTEXT_NEGATIONS = (
    "不要结合项目",
    "不结合项目",
    "不用结合项目",
    "别结合项目",
    "不要参考项目",
    "不参考项目",
    "不用参考项目",
    "别参考项目",
    "不要带项目",
    "不带项目",
    "不用带项目",
    "别带项目",
    "先不说项目",
    "先别说项目",
    "和项目无关",
    "跟项目无关",
    "与项目无关",
    "不要结合简历",
    "不结合简历",
    "不用结合简历",
    "别结合简历",
    "不要参考简历",
    "不参考简历",
    "不用参考简历",
    "别参考简历",
    "不要看简历",
    "不看简历",
    "不用看简历",
    "别看简历",
    "和简历无关",
    "跟简历无关",
    "与简历无关",
    "without resume",
    "without my resume",
    "without your resume",
    "not based on resume",
    "not based on my resume",
    "not based on your resume",
    "ignore resume",
    "ignore my resume",
    "ignore your resume",
    "do not use resume",
    "don't use resume",
    "without project context",
    "no project context",
    "not based on project",
    "not based on my project",
    "not based on your project",
    "ignore project",
    "ignore my project",
    "ignore your project",
    "do not use project",
    "don't use project",
    "do not relate to my project",
    "don't relate to my project",
    "do not relate to your project",
    "don't relate to your project",
    "do not relate it to my project",
    "don't relate it to my project",
    "do not relate it to your project",
    "don't relate it to your project",
    "do not combine with project",
    "don't combine with project",
)
_CHINESE_RESUME_NEGATION_ACTIONS = (
    "不要结合",
    "不结合",
    "不用结合",
    "别结合",
    "不要参考",
    "不参考",
    "不用参考",
    "别参考",
    "不要带",
    "不带",
    "不用带",
    "别带",
    "不要看",
    "不看",
    "不用看",
    "别看",
    "不要基于",
    "不基于",
    "不用基于",
    "别基于",
    "先不说",
    "先别说",
)
_CHINESE_RESUME_NEGATION_TARGETS = ("项目", "简历")
_CHINESE_RESUME_UNRELATED_PREFIXES = ("和", "跟", "与")
_CHINESE_RESUME_UNRELATED_MARKERS = ("无关", "没关系", "没有关系")


def _contains_resume_context_negation(normalized_lc: str) -> bool:
    if any(phrase in normalized_lc for phrase in _RESUME_CONTEXT_NEGATIONS):
        return True
    for action in _CHINESE_RESUME_NEGATION_ACTIONS:
        start = normalized_lc.find(action)
        while start >= 0:
            tail = normalized_lc[start + len(action): start + len(action) + 14]
            if any(target in tail for target in _CHINESE_RESUME_NEGATION_TARGETS):
                return True
            start = normalized_lc.find(action, start + 1)
    for prefix in _CHINESE_RESUME_UNRELATED_PREFIXES:
        start = normalized_lc.find(prefix)
        while start >= 0:
            tail = normalized_lc[start + len(prefix): start + len(prefix) + 14]
            if (
                any(target in tail for target in _CHINESE_RESUME_NEGATION_TARGETS)
                and any(marker in tail for marker in _CHINESE_RESUME_UNRELATED_MARKERS)
            ):
                return True
            start = normalized_lc.find(prefix, start + 1)
    return False


def _followup_needs_bridge(question_text: str) -> bool:
    normalized = normalize_transcription_for_analysis(question_text)
    if not normalized:
        return False
    normalized_lc = normalized.lower()
    if len(normalized) <= 18:
        return True
    if any(phrase in normalized_lc for phrase in _FOLLOWUP_BRIDGE_PHRASES) and len(normalized) <= 64:
        return True
    if any(normalized_lc.startswith(prefix) for prefix in _FOLLOWUP_BRIDGE_PREFIXES) and len(normalized) <= 56:
        return True
    return False


def _question_explicitly_requests_resume_context(question_text: str) -> bool:
    normalized = normalize_transcription_for_analysis(question_text)
    if not normalized:
        return False
    normalized_lc = normalized.lower()
    if _contains_resume_context_negation(normalized_lc):
        return False
    return any(phrase in normalized_lc for phrase in _RESUME_CONTEXT_CUES)


def _question_rejects_resume_context(question_text: str) -> bool:
    normalized = normalize_transcription_for_analysis(question_text)
    if not normalized:
        return False
    return _contains_resume_context_negation(normalized.lower())


def _should_include_resume_context(
    prompt_mode: PromptMode,
    question_text: str,
    previous_question: str,
    is_followup: bool,
) -> bool:
    if _question_rejects_resume_context(question_text):
        return False
    if prompt_mode != PROMPT_MODE_ASR_REALTIME:
        return True
    if _question_explicitly_requests_resume_context(question_text):
        return True
    return (
        is_followup
        and _followup_needs_bridge(question_text)
        and _question_explicitly_requests_resume_context(previous_question)
    )


def _history_context_options(prompt_mode: PromptMode) -> dict[str, Any]:
    if prompt_mode == PROMPT_MODE_SERVER_SCREEN:
        return {
            "profile": "screen_light",
            "turns": 1,
            "max_chars_per_message": 700,
            "include_summary": False,
            "total_char_budget": 900,
        }
    if prompt_mode == PROMPT_MODE_MANUAL_TEXT:
        return {
            "profile": "manual_balanced",
            "turns": 3,
            "max_chars_per_message": 900,
            "include_summary": True,
            "total_char_budget": 2600,
        }
    return {
        "profile": "default",
        "turns": None,
        "max_chars_per_message": None,
        "include_summary": True,
        "total_char_budget": None,
    }


def _wait_for_candidate_context_if_pending(
    session_ref,
    qa_id: str,
    wait_ms: int,
    abort_check: Optional[Callable[[], bool]] = None,
) -> None:
    if wait_ms <= 0 or not qa_id:
        return
    deadline = time.monotonic() + (wait_ms / 1000.0)
    while time.monotonic() < deadline:
        if abort_check is not None and abort_check():
            return
        if not getattr(session_ref, "has_candidate_asr_pending_for_qa", lambda *_args, **_kwargs: False)(qa_id):
            return
        time.sleep(0.025)


def _supports_candidate_context_source(source: str, meta: dict[str, Any]) -> bool:
    if source == "manual_text":
        return True
    if source in ("asr", "conversation_loopback", "conversation_mic"):
        return True
    return meta.get("origin") == "asr"


def prompt_mode_for_task(
    source: str,
    manual_input: bool,
    written_exam: bool = False,
) -> PromptMode:
    if written_exam:
        return PROMPT_MODE_WRITTEN_EXAM
    if (source or "").startswith("server_screen_"):
        return PROMPT_MODE_SERVER_SCREEN
    if manual_input:
        return PROMPT_MODE_MANUAL_TEXT
    return PROMPT_MODE_ASR_REALTIME


def _process_question_parallel(
    task: TaskPayload,
    seq: int,
    model_idx: int,
    sess_v: int,
    deps: AnswerWorkerDeps,
):
    question_text, image, manual_input, source, meta = task
    mmpg_trace = meta.get("mmpg_trace")
    mmpg_request = meta.get("mmpg_request")
    exam_preflight_id = str(meta.get("exam_preflight_id") or "") if meta.get("exam_preflight") else ""
    qa_id = f"qa-{seq}-{int(time.time() * 1000)}"
    attempt_number = 1

    def _broadcast(data: dict) -> None:
        if data["type"].startswith("answer_"):
            data = {**data, "attempt": attempt_number}
        if exam_preflight_id:
            data = {**data, "exam_preflight_id": exam_preflight_id}
        if mmpg_request:
            with mmpg_request.lock:
                if mmpg_request.cancelled:
                    return
                if data["type"] == "answer_start":
                    mmpg_request.shown_qa_id = qa_id
                if mmpg_request.updates_qa_id and data["type"] in ("answer_start", "answer_retry", "answer_done"):
                    data = {**data, "updates_qa_id": mmpg_request.updates_qa_id}
                deps.broadcast(data)
        else:
            deps.broadcast(data)
        if mmpg_trace and data["type"] in ("answer_done", "answer_error", "answer_cancelled"):
            get_logger("mmpg.diagnostics").info(
                "turn=%s event=%s seq=%s stage=%s", mmpg_trace, data["type"], seq, data.get("stage"),
            )
        if exam_preflight_id:
            try:
                from .exam_test import record_exam_preflight_answer_event

                record_exam_preflight_answer_event(data)
            except Exception as exc:  # noqa: BLE001
                deps.error_logger.warning("exam preflight event record failed: %s", exc)

    # Prefer the dispatch-time snapshot.  Falling back to get_config keeps
    # this worker usable from older callers/tests that invoke it directly.
    cfg = deps.config_snapshot if deps.config_snapshot is not None else get_config()
    model_cfg = deps.model_cfg_snapshot
    if model_cfg is None and (model_idx < 0 or model_idx >= len(cfg.models)):
        deps.error_logger.warning(
            "ANSWER_MODEL_STALE id=%s seq=%d model_idx=%d model_count=%d",
            qa_id,
            seq,
            model_idx,
            len(cfg.models),
        )
        _broadcast(
            {
                "type": "answer_error",
                "id": qa_id,
                "stage": "generation",
                "message": "答题模型配置已变更，请重新提交问题。",
            }
        )
        deps.mark_seq_skipped(seq)
        return
    if model_cfg is None:
        model_cfg = cfg.models[model_idx]
    if (deps.start_abort_check is not None and deps.start_abort_check()) or deps.abort_check():
        deps.mark_seq_skipped(seq)
        return
    if deps.resolve_exam_model is not None:
        live_model = deps.resolve_exam_model(model_idx)
        if live_model is None:
            _broadcast({"type": "answer_error", "id": qa_id, "stage": "generation", "message": EXAM_MODEL_STALE})
            deps.mark_seq_skipped(seq)
            return
        model_cfg = live_model

    written_exam = task_is_written_exam(task, cfg)
    written_exam_think = bool(getattr(cfg, "written_exam_think", False))
    prompt_mode = prompt_mode_for_task(source, manual_input, written_exam=written_exam)
    knowledge_practice = meta.get("purpose") == "knowledge_practice"
    plain_text = prompt_mode == PROMPT_MODE_ASR_REALTIME or (
        prompt_mode == PROMPT_MODE_MANUAL_TEXT
        and source == "manual_text"
        and not image
        and not knowledge_practice
    )
    format_payload = {"answer_format": "plain_text"} if plain_text else {}
    scene = assist_prompt_scene(prompt_mode)
    customized_prompt = (
        cfg.knowledge_practice_system_prompt is not None
        if knowledge_practice and hasattr(cfg, "knowledge_practice_system_prompt")
        else scene in getattr(cfg, "system_prompts", {})
    )
    generation_cfg = cfg if knowledge_practice else config_for_assist_generation(cfg, prompt_mode)

    kb_hits: list = []
    kb_latency_ms = 0
    kb_degraded = False
    if (
        bool(getattr(cfg, "kb_enabled", False))
        and prompt_mode in set(getattr(cfg, "kb_trigger_modes", []) or [])
        and (question_text or "").strip()
    ):
        try:
            from services.kb.retriever import retrieve as _kb_retrieve

            deadline_ms = (
                int(getattr(cfg, "kb_asr_deadline_ms", 80) or 80)
                if prompt_mode == PROMPT_MODE_ASR_REALTIME
                else int(getattr(cfg, "kb_deadline_ms", 150) or 150)
            )
            t0 = time.monotonic()
            kb_hits = _kb_retrieve(
                question_text,
                k=int(getattr(cfg, "kb_top_k", 4) or 4),
                deadline_ms=deadline_ms,
                mode=prompt_mode,
            )
            kb_latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as exc:
            deps.error_logger.warning("kb retrieve in answer worker failed: %s", exc)
            kb_hits = []
            kb_degraded = True

    if deps.abort_check():
        deps.mark_seq_skipped(seq)
        return

    images = _normalize_task_images(image)
    realtime_voice = prompt_mode == PROMPT_MODE_ASR_REALTIME and not images
    transcript_anchor = meta.get("transcript_anchor")

    if images:
        user_for_llm: Any = [
            {"type": "text", "text": question_text},
        ]
        for data_url in images:
            user_for_llm.append({"type": "image_url", "image_url": {"url": data_url}})
    else:
        user_for_llm = question_text

    with conversation_lock:
        session_ref = get_session()
        transcript = session_ref.transcript_snapshot(transcript_anchor) if realtime_voice and transcript_anchor else ()
        if realtime_voice:
            base_messages = []
            written_followup_context = ""
        elif written_exam:
            base_messages = []
            history_stats = {"messages": 0, "history_messages": 0, "stripped_images": 0}
            written_followup_context = (
                ""
                if exam_preflight_id
                else _written_exam_followup_context(
                    session_ref,
                    source=source,
                    image_count=len(images),
                )
            )
        else:
            history_options = _history_context_options(prompt_mode)
            base_messages = list(session_ref.get_conversation_messages_for_llm(**history_options))
            history_stats = dict(getattr(session_ref, "last_llm_history_stats", {}) or {})
            written_followup_context = ""
        last_qa = session_ref.get_last_qa()
        prior_candidate_qa_id = getattr(session_ref, "current_candidate_qa_id", "") or ""
        candidate_context_enabled, candidate_wait_ms, candidate_max_chars, candidate_min_chars = _candidate_context_settings(cfg)
        candidate_source_ok = _supports_candidate_context_source(source, meta)
        should_use_candidate_context = bool(
            not realtime_voice
            and not written_exam
            and not images
            and last_qa
            and candidate_source_ok
            and candidate_context_enabled
        )
        actual_spoken_answer = (
            session_ref.get_candidate_answer_for_qa(last_qa.id, max_chars=candidate_max_chars)
            if last_qa
            and should_use_candidate_context
            else ""
        )

    if should_use_candidate_context and last_qa:
        try:
            _wait_for_candidate_context_if_pending(
                session_ref,
                last_qa.id,
                candidate_wait_ms,
                abort_check=deps.abort_check,
            )
        except TypeError:
            _wait_for_candidate_context_if_pending(session_ref, last_qa.id, candidate_wait_ms)
        if deps.abort_check():
            deps.mark_seq_skipped(seq)
            return
        with conversation_lock:
            actual_spoken_answer = session_ref.get_candidate_answer_for_qa(
                last_qa.id,
                max_chars=candidate_max_chars,
            )
    if len(actual_spoken_answer.strip()) < candidate_min_chars:
        actual_spoken_answer = ""
    candidate_context_chars = len(actual_spoken_answer[:candidate_max_chars]) if actual_spoken_answer else 0

    previous_question = ""
    is_followup = False
    if realtime_voice and len(transcript) > 1:
        previous_round = transcript[-2]
        previous_question = "\n".join(entry.text for entry in previous_round if entry.speaker == "interviewer")
        previous_spoken_answer = "\n".join(entry.text for entry in previous_round if entry.speaker == "candidate")
        if previous_question:
            is_followup = classify_followup(question_text, previous_question, previous_spoken_answer)
    if (
        not realtime_voice
        and not written_exam
        and not images
        and last_qa
        and _supports_candidate_context_source(source, meta)
        and classify_followup(question_text, last_qa.question, actual_spoken_answer or last_qa.answer[:500])
    ):
        is_followup = True
        followup_needs_bridge = _followup_needs_bridge(question_text)
        prev_answer_budget = 0
        prev_answer_summary = ""
        if followup_needs_bridge:
            prev_answer_budget = max(160, min(280, candidate_max_chars // 3 if candidate_max_chars > 0 else 160))
            prev_answer_summary = _context_excerpt(last_qa.answer, prev_answer_budget)
        if not followup_needs_bridge:
            user_for_llm = (
                f"[追问上下文] 上一个问题（主题锚点）：{last_qa.question}\n\n"
                f"现在面试官追问：{question_text}"
            )
        elif not candidate_context_enabled:
            user_for_llm = (
                f"[追问上下文] 上一个问题：{last_qa.question}\n"
                f"助手上一轮建议答案：{prev_answer_summary}\n\n"
                f"现在面试官追问：{question_text}"
            )
        else:
            actual_block = (
                f"候选人麦克风转写（辅助参考，可能有识别误差）：{actual_spoken_answer[:candidate_max_chars]}\n"
                if actual_spoken_answer
                else "候选人麦克风转写：未捕获到。"
            )
            user_for_llm = (
                f"[追问上下文] 上一个问题：{last_qa.question}\n"
                f"{actual_block}\n"
                f"助手上一轮建议答案：{prev_answer_summary}\n\n"
                f"现在面试官追问：{question_text}"
            )
    elif (
        should_use_candidate_context
        and last_qa
        and actual_spoken_answer
        and _candidate_background_matches_current_question(question_text, last_qa, actual_spoken_answer)
    ):
        user_for_llm = (
            f"[候选人回答辅助背景] 上一个问题：{last_qa.question}\n"
            f"候选人上一轮麦克风转写（可能有识别误差）：{actual_spoken_answer[:candidate_max_chars]}\n\n"
            f"现在面试官问题：{question_text}"
        )

    if written_followup_context:
        if isinstance(user_for_llm, list):
            if user_for_llm and isinstance(user_for_llm[0], dict) and user_for_llm[0].get("type") == "text":
                user_for_llm[0] = {
                    **user_for_llm[0],
                    "text": f"{written_followup_context}\n\n[当前截图/题面]\n{question_text}",
                }
            else:
                user_for_llm.insert(
                    0,
                    {
                        "type": "text",
                        "text": f"{written_followup_context}\n\n[当前截图/题面]\n{question_text}",
                    },
                )
        else:
            user_for_llm = f"{written_followup_context}\n\n[当前题面]\n{question_text}"

    aborted_before_start = False
    with conversation_lock:
        if deps.abort_check():
            aborted_before_start = True
        elif prior_candidate_qa_id and session_ref.current_candidate_qa_id == prior_candidate_qa_id:
            session_ref.close_candidate_answer_window()
    if aborted_before_start:
        deps.mark_seq_skipped(seq)
        return

    system_prompt = build_system_prompt(
        manual_input=manual_input,
        mode=prompt_mode,
        cfg=cfg,
        structured_manual=prompt_mode == PROMPT_MODE_MANUAL_TEXT and not plain_text,
        knowledge_practice=knowledge_practice,
    )
    user_context = build_assist_user_context(
        cfg,
        mode=prompt_mode,
        kb_hits=kb_hits,
        include_resume=_should_include_resume_context(
            prompt_mode,
            question_text,
            previous_question,
            is_followup,
        ),
    )
    if realtime_voice:
        voice_rounds = tuple(
            tuple(
                format_interview_utterance(
                    entry.text,
                    speaker=entry.speaker,
                    current=entry.id == transcript_anchor,
                )
                for entry in spoken_round
            )
            for spoken_round in transcript
        )
        if not voice_rounds:
            voice_rounds = ((format_interview_utterance(question_text, speaker="interviewer", current=True),),)
        voice_prefix = f"[本轮任务资料]\n{user_context}\n\n[真实对话]" if user_context else None
    else:
        if user_context:
            context_prefix = f"[本轮任务资料]\n{user_context}\n\n[当前输入]\n"
            if isinstance(user_for_llm, list):
                user_for_llm[0]["text"] = context_prefix + user_for_llm[0]["text"]
            else:
                user_for_llm = context_prefix + user_for_llm

        messages_for_llm = base_messages + [{"role": "user", "content": user_for_llm}]
        deps.logger.info(
            "LLM_INPUT_STATS source=%s prompt_mode=%s written_exam=%s image_count=%d "
            "image_payload_chars=%d history_used=%s history_profile=%s history_messages=%d "
            "historical_images_stripped=%d history_text_raw_chars=%d "
            "history_text_trimmed_chars=%d candidate_context_chars=%d "
            "message_count=%d text_chars=%d prompt_scene=%s customized_prompt=%s",
            source,
            prompt_mode,
            written_exam,
            len(images),
            _image_payload_chars(images),
            bool(base_messages),
            str(history_stats.get("profile", "default")),
            int(history_stats.get("history_messages", 0) or 0),
            int(history_stats.get("stripped_images", 0) or 0),
            int(history_stats.get("raw_text_chars", 0) or 0),
            int(history_stats.get("trimmed_text_chars", 0) or 0),
            candidate_context_chars,
            len(messages_for_llm),
            _message_text_chars(messages_for_llm),
            scene,
            customized_prompt,
        )

    if deps.abort_check():
        deps.mark_seq_skipped(seq)
        return

    if len(images) > 1:
        display_question = f"{question_text} [📷 多图 x{len(images)}]"
    else:
        display_question = question_text + (" [📷 附图]" if images else "")
    deps.logger.info(
        "ANSWER_START id=%s model=%s source=%s followup=%s q=%r",
        qa_id,
        model_cfg.name,
        source,
        is_followup,
        question_text[:120],
    )
    _broadcast(
        {
            "type": "answer_start",
            "id": qa_id,
            "question": display_question,
            "source": source,
            "model_name": model_cfg.name,
            "model_index": model_idx,
            **format_payload,
        }
    )
    # 保留答题卡关联，供复盘、知识记录和手动提问使用。
    should_open_candidate_window = (
        _supports_candidate_context_source(source, meta)
        and (
            candidate_context_enabled
            or bool(getattr(cfg, "candidate_asr_enabled", False))
        )
    )
    if should_open_candidate_window and not deps.abort_check():
        with conversation_lock:
            if not deps.abort_check():
                get_session().open_candidate_answer_window(qa_id)

    if (kb_hits or kb_degraded) and not deps.abort_check():
        try:
            from services.kb.ws import build_kb_hits_payload as _kb_payload

            _broadcast(
                _kb_payload(
                    qa_id=qa_id,
                    hits=kb_hits,
                    latency_ms=kb_latency_ms,
                    degraded=kb_degraded,
                    excerpt_chars=int(
                        getattr(cfg, "kb_prompt_excerpt_chars", 300) or 300
                    ),
                )
            )
        except Exception as exc:
            deps.error_logger.warning("broadcast kb_hits failed: %s", exc)

    gen_start = time.monotonic()
    if mmpg_trace and not mmpg_request:
        get_logger("mmpg.diagnostics").info("turn=%s event=llm_start seq=%s", mmpg_trace, seq)
    # Precedence: per-request override, exam screenshot Think, then the assist setting.
    if meta.get("override_think_mode") is not None:
        think_override = bool(meta["override_think_mode"])
    elif prompt_mode == PROMPT_MODE_WRITTEN_EXAM and (source or "").startswith("server_screen_"):
        think_override = written_exam_think
    else:
        think_override = None

    flush_failed_attempt: Optional[Callable[[], None]] = None

    def _generate_attempt() -> GeneratedAnswer | None:
        nonlocal flush_failed_attempt
        flush_failed_attempt = None
        sanitizer = create_answer_stream_sanitizer(
            prompt_mode, preserve_format=customized_prompt, plain_text=plain_text,
        )
        raw_chunks: list[str] = []
        visible_chunks: list[str] = []
        think_chunks: list[str] = []
        pending_chunks: list[str] = []
        token_prompt_delta = token_completion_delta = 0
        first_token_mono: float | None = None

        def record_usage(prompt_tokens: int, completion_tokens: int, _model_name: str) -> None:
            nonlocal token_prompt_delta, token_completion_delta
            token_prompt_delta += int(prompt_tokens or 0)
            token_completion_delta += int(completion_tokens or 0)

        def flush_chunks() -> None:
            if pending_chunks:
                _broadcast({"type": "answer_chunk", "id": qa_id, "chunk": "".join(pending_chunks)})
                pending_chunks.clear()

        def emit_tail() -> None:
            tail = sanitizer.finish()
            if tail:
                visible_chunks.append(tail)
                _broadcast({"type": "answer_chunk", "id": qa_id, "chunk": tail})

        stream_options = dict(
            system_prompt=system_prompt,
            abort_check=deps.abort_check,
            override_think_mode=think_override,
            usage_callback=record_usage,
            override_max_tokens=int(getattr(generation_cfg, "max_tokens", 4096)),
            trace_context={
                "scene": scene, "qa_id": qa_id, "source": source, "answer_attempt": attempt_number,
                "mmpg_group_id": getattr(mmpg_request, "group_id", None),
                "mmpg_revision": getattr(mmpg_request, "revision", None),
                "mmpg_seq": getattr(mmpg_request, "seq", None),
            },
            mmpg_request=mmpg_request,
            config_snapshot=generation_cfg,
        )
        if realtime_voice:
            stream = stream_interview_answer(
                model_cfg, voice_rounds, context_prefix=voice_prefix,
                stream_fn=chat_stream_single_model, **stream_options,
            )
        else:
            stream = chat_stream_single_model(
                model_cfg, messages_for_llm, protect_output_limit=True, **stream_options,
            )
        try:
            for chunk_type, chunk_text in stream:
                if deps.abort_check():
                    return None
                if chunk_type == "think":
                    think_chunks.append(chunk_text)
                    if prompt_mode != PROMPT_MODE_WRITTEN_EXAM or len(think_chunks) == 1:
                        _broadcast({
                            "type": "answer_think_chunk", "id": qa_id,
                            "chunk": "思考中..." if prompt_mode == PROMPT_MODE_WRITTEN_EXAM else chunk_text,
                        })
                    continue
                clean_chunk = sanitizer.push(chunk_text)
                if first_token_mono is None and (clean_chunk.strip() if plain_text else chunk_text):
                    first_token_mono = time.monotonic()
                    if mmpg_trace:
                        if mmpg_request:
                            get_logger("mmpg.diagnostics").info(
                                "turn=%s group=%s revision=%s event=llm_first_token seq=%s elapsed_ms=%.1f since_final_ms=%.1f",
                                mmpg_trace, mmpg_request.group_id, mmpg_request.revision, seq,
                                (first_token_mono - (mmpg_request.started_at or gen_start)) * 1000,
                                (first_token_mono - mmpg_request.final_at) * 1000,
                            )
                        else:
                            get_logger("mmpg.diagnostics").info(
                                "turn=%s event=llm_first_token seq=%s", mmpg_trace, seq,
                            )
                raw_chunks.append(chunk_text)
                if not clean_chunk:
                    continue
                if plain_text:
                    visible_chunks.append(clean_chunk)
                    _broadcast({"type": "answer_chunk", "id": qa_id, "chunk": clean_chunk})
                else:
                    pending_chunks.append(clean_chunk)
                    if len(pending_chunks) >= 5:
                        flush_chunks()
        except Exception:
            if not deps.abort_check():
                def _flush_residual() -> None:
                    flush_chunks()
                    emit_tail()

                flush_failed_attempt = _flush_residual
            raise
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        if deps.abort_check():
            return None
        flush_chunks()
        emit_tail()
        answer = (
            "".join(visible_chunks) if plain_text else
            postprocess_answer_for_mode("".join(raw_chunks), prompt_mode, preserve_format=customized_prompt)
        )
        if not answer.strip():
            raise EmptyAnswer("No visible answer generated")
        return GeneratedAnswer(
            answer, "".join(think_chunks), first_token_mono, token_prompt_delta, token_completion_delta,
        )

    generated = None
    attempted: set[int] = set()
    while not deps.abort_check():
        first_attempt = not attempted
        attempted.add(model_idx)
        try:
            if deps.resolve_exam_model is not None:
                live_model = deps.resolve_exam_model(model_idx)
                if live_model is None:
                    raise ExamModelUnavailable(EXAM_MODEL_STALE)
                model_cfg = live_model
            generated = _generate_attempt()
        except Exception as exc:
            if deps.abort_check():
                break
            deps.error_logger.error(
                "ANSWER_ATTEMPT_FAILED id=%s attempt=%d model=%s kind=%s",
                qa_id, attempt_number, model_cfg.name, type(exc).__name__,
                exc_info=not isinstance(exc, LLMError),
            )
            retry_attempt = attempt_number + 1

            def announce_retry(model_name: str = "", *, attempt: int = retry_attempt) -> None:
                nonlocal attempt_number
                if deps.abort_check():
                    return
                attempt_number = attempt
                _broadcast({
                    "type": "answer_retry", "id": qa_id, "question": display_question, "source": source,
                    "model_name": model_name, **format_payload,
                })

            next_index = None
            primary_lost = first_attempt and isinstance(exc, ExamModelUnavailable)
            if isinstance(exc, LLMError) and not primary_lost and not knowledge_practice and deps.reserve_retry is not None:
                next_index = deps.reserve_retry(attempted, announce_retry)
            if deps.abort_check():
                break
            if next_index is None:
                if flush_failed_attempt is not None:
                    flush_failed_attempt()
                    flush_failed_attempt = None
                _broadcast({
                    "type": "answer_error", "id": qa_id, "stage": "generation",
                    "message": exc.user_msg if isinstance(exc, LLMError) else "生成答案失败，请稍后重试。",
                })
                deps.mark_seq_skipped(seq)
                return
            flush_failed_attempt = None
            model_idx = next_index
            model_cfg = cfg.models[next_index]
            retry_model_name = model_cfg.name
            if deps.resolve_exam_model is not None:
                live_model = deps.resolve_exam_model(next_index)
                retry_model_name = live_model.name if live_model is not None else ""
            deps.logger.info("ANSWER_RETRY id=%s attempt=%d model=%s", qa_id, retry_attempt, retry_model_name or model_cfg.name)
            announce_retry(retry_model_name)
        else:
            break

    if generated is None or deps.abort_check():
        deps.logger.info("ANSWER_CANCEL id=%s after=%.0fms", qa_id, (time.monotonic() - gen_start) * 1000)
        _broadcast({"type": "answer_cancelled", "id": qa_id})
        deps.mark_seq_skipped(seq)
        return

    if mmpg_request and mmpg_request.started_at is not None:
        gen_start = mmpg_request.started_at
    gen_elapsed = (time.monotonic() - gen_start) * 1000
    first_token_ms = (
        (generated.first_token_at - gen_start) * 1000 if generated.first_token_at is not None else gen_elapsed
    )
    full_answer, full_think = generated.text, generated.think
    token_prompt_delta, token_completion_delta = generated.prompt_tokens, generated.completion_tokens

    def _cancel_visible_answer():
        _broadcast({"type": "answer_cancelled", "id": qa_id})

    def _commit():
        if not deps.is_session_current(sess_v) or (mmpg_request and mmpg_request.cancelled):
            if plain_text:
                _cancel_visible_answer()
            return
        session = get_session()
        if exam_preflight_id:
            if deps.abort_check():
                return
            stats = get_token_stats()
            deps.logger.info(
                "EXAM_PREFLIGHT_ANSWER_DONE id=%s model=%s first_token=%.0fms total=%.0fms answer_len=%d",
                qa_id,
                model_cfg.name,
                first_token_ms,
                gen_elapsed,
                len(full_answer),
            )
            _broadcast(
                {
                    "type": "answer_done",
                    "id": qa_id,
                    "question": display_question,
                    "answer": full_answer,
                    "think": full_think,
                    "model_name": model_cfg.name,
                    "first_token_ms": int(first_token_ms),
                    "total_ms": int(gen_elapsed),
                }
            )
            deps.broadcast(
                {
                    "type": "token_update",
                    "prompt": stats["prompt"],
                    "completion": stats["completion"],
                    "total": stats["total"],
                    "by_model": stats.get("by_model", {}),
                }
            )
            return
        user_msg_content = f"{question_text} [图片已省略 x{len(images)}]" if images else question_text
        stale_before_commit = False
        try:
            with conversation_lock:
                if not deps.is_session_current(sess_v) or (mmpg_request and mmpg_request.cancelled):
                    stale_before_commit = True
                else:
                    prev_history = list(session.conversation_history)
                    prev_qa_pairs = list(session.qa_pairs)
                    prev_completed_qa_id = mmpg_request.completed_qa_id if mmpg_request else None
                    try:
                        session.add_user_message(user_msg_content)
                        session.add_assistant_message(full_answer)
                        session.add_qa(
                            display_question,
                            full_answer,
                            qa_id=qa_id,
                            source=source,
                            model_name=model_cfg.name,
                            written_exam=written_exam,
                            **({"answer_format": "plain_text", "first_token_ms": int(first_token_ms),
                                "total_ms": int(gen_elapsed)} if plain_text else {}),
                            **({"updates_qa_id": mmpg_request.updates_qa_id} if mmpg_request and mmpg_request.updates_qa_id else {}),
                        )
                        if mmpg_request:
                            mmpg_request.completed_qa_id = qa_id
                    except Exception:
                        if mmpg_request:
                            mmpg_request.completed_qa_id = prev_completed_qa_id
                        session.qa_pairs[:] = [qa for qa in prev_qa_pairs if qa.id != qa_id]
                        session.conversation_history[:] = prev_history
                        raise
        except Exception as exc:
            deps.error_logger.error(
                "_commit failed for id=%s seq=%d: %s",
                qa_id, seq, exc, exc_info=True,
            )
            _broadcast({
                "type": "answer_error",
                "id": qa_id,
                "stage": "persistence",
                "message": "答案保存失败",
            })
            return

        if stale_before_commit:
            if plain_text:
                _cancel_visible_answer()
            return

        # Persistence succeeded; post-commit side effects must degrade independently
        # and never roll back the committed QA or emit a contradictory answer_error.
        stats = None
        try:
            stats = get_token_stats()
            deps.logger.info(
                "ANSWER_DONE id=%s model=%s first_token=%.0fms total=%.0fms "
                "answer_len=%d think_len=%d tokens_prompt_delta=%d tokens_completion_delta=%d "
                "tokens_prompt=%d tokens_completion=%d",
                qa_id,
                model_cfg.name,
                first_token_ms,
                gen_elapsed,
                len(full_answer),
                len(full_think),
                token_prompt_delta,
                token_completion_delta,
                stats["prompt"],
                stats["completion"],
            )
        except Exception as exc:  # noqa: BLE001
            deps.error_logger.warning("ANSWER_DONE stats/log failed id=%s: %s", qa_id, exc)

        try:
            _broadcast(
                {
                    "type": "answer_done",
                    "id": qa_id,
                    "question": display_question,
                    "answer": full_answer,
                    "think": full_think,
                    "model_name": model_cfg.name,
                    "first_token_ms": int(first_token_ms),
                    "total_ms": int(gen_elapsed),
                    **format_payload,
                    **({"source": source} if plain_text else {}),
                }
            )
        except Exception as exc:  # noqa: BLE001
            deps.error_logger.warning("broadcast answer_done failed id=%s: %s", qa_id, exc)

        if stats is not None:
            try:
                deps.broadcast(
                    {
                        "type": "token_update",
                        "prompt": stats["prompt"],
                        "completion": stats["completion"],
                        "total": stats["total"],
                        "by_model": stats.get("by_model", {}),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                deps.error_logger.warning("broadcast token_update failed id=%s: %s", qa_id, exc)

        try:
            candidate_answer_for_record = ""
            with conversation_lock:
                candidate_answer_for_record = session.get_candidate_answer_for_qa(
                    qa_id,
                    max_chars=2400,
                )
            if not deps.submit_knowledge_record(question_text, full_answer, qa_id, candidate_answer_for_record):
                deps.error_logger.warning(
                    "KNOWLEDGE_ENQUEUE_DROP id=%s question=%r",
                    qa_id,
                    question_text[:80],
                )
        except Exception as exc:  # noqa: BLE001
            deps.error_logger.warning("KNOWLEDGE_ENQUEUE_FAIL id=%s err=%s", qa_id, exc)

        if prompt_mode in (PROMPT_MODE_SERVER_SCREEN, PROMPT_MODE_WRITTEN_EXAM) and images:
            try:
                from services.vision_verify import schedule_self_verify

                schedule_self_verify(
                    qa_id=qa_id,
                    answer=full_answer,
                    image_data_url=images,
                    broadcast_callable=deps.broadcast,
                    model_cfg=model_cfg,
                )
            except Exception as exc:  # noqa: BLE001
                deps.error_logger.warning(
                    "VISION_VERIFY_SCHEDULE_FAIL id=%s err=%s",
                    qa_id,
                    exc,
                )

    def _commit_current_request():
        with mmpg_request.lock:
            _commit()

    apply_commit = _commit_current_request if mmpg_request else _commit
    deps.flush_commit(seq, AnswerCommit(apply_commit, _cancel_visible_answer) if plain_text else apply_commit)


def process_question_parallel(task, seq, model_idx, sess_v, deps):
    """Observe all exits without changing answer generation/commit scheduling."""
    from services.llm import trace
    from urllib.parse import urlsplit

    meta = task[4]
    archive, run_id = meta.get("_archive_owner"), meta.get("_archive_run_id")
    if archive is None or not run_id:
        return _process_question_parallel(task, seq, model_idx, sess_v, deps)
    cfg = deps.config_snapshot if deps.config_snapshot is not None else get_config()
    round_id = meta.get("_archive_round_id")

    def model_snapshot(model):
        if model is None:
            return {}
        key = getattr(model, "api_key", "")
        try:
            endpoint = urlsplit(getattr(model, "api_base_url", "")).hostname or ""
        except ValueError:
            endpoint = ""
        return dict(
            name=trace._redact_text(model.name, key),
            model=trace._redact_text(model.model, key),
            protocol=getattr(model, "api_type", ""),
            endpoint=trace._redact_text(endpoint, key),
        )

    model = deps.model_cfg_snapshot
    if model is None and 0 <= model_idx < len(cfg.models):
        model = cfg.models[model_idx]
    archive.start_ai(round_id, run_id, model_snapshot(model))

    def broadcast(event):
        deps.broadcast(event)
        try:
            archive.ai_event(run_id, event)
        except Exception:
            deps.error_logger.exception("archive answer observation failed")

    resolve_exam_model = deps.resolve_exam_model
    if resolve_exam_model is not None:
        orig_resolve_exam_model = resolve_exam_model

        def resolve_exam_model(index):
            live_model = orig_resolve_exam_model(index)
            if live_model is not None:
                try:
                    archive.start_ai(round_id, run_id, model_snapshot(live_model))
                except Exception:
                    deps.error_logger.exception("archive answer model update failed")
            return live_model

    reserve_retry = deps.reserve_retry
    if reserve_retry is not None:
        orig_reserve_retry = reserve_retry

        def reserve_retry(attempted, announce_retry):
            next_index = orig_reserve_retry(attempted, announce_retry)
            if next_index is not None and 0 <= next_index < len(cfg.models):
                try:
                    archive.start_ai(round_id, run_id, model_snapshot(cfg.models[next_index]))
                except Exception:
                    deps.error_logger.exception("archive answer retry model update failed")
            return next_index

    commit_submitted = False

    def flush_commit(index, commit):
        nonlocal commit_submitted
        commit_submitted = True
        archive.mark_generation_ended(run_id)

        def apply():
            try:
                commit()
            finally:
                archive.settle_unreported_ai(run_id, deps.abort_check())

        def cancel():
            if isinstance(commit, AnswerCommit):
                commit.cancel()
            archive.settle_unreported_ai(run_id, cancelled=True)

        deps.flush_commit(index, AnswerCommit(apply, cancel))

    try:
        with trace.observe(lambda event: archive.observe_attempt(run_id, event)):
            return _process_question_parallel(
                task,
                seq,
                model_idx,
                sess_v,
                replace(
                    deps,
                    broadcast=broadcast,
                    flush_commit=flush_commit,
                    resolve_exam_model=resolve_exam_model,
                    reserve_retry=reserve_retry,
                ),
            )
    finally:
        if not commit_submitted:
            request = meta.get("mmpg_request")
            archive.settle_unreported_ai(run_id, deps.abort_check() or bool(request and request.cancelled))
