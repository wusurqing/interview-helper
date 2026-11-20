"""
Review 模块 LLM 分析服务（含 ASR 纠错功能）
使用当前激活的 LLM 模型进行逐题分析和整场总结
"""
import json
import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import Any
from openai import OpenAI

from core.logger import get_logger
from core.config import get_config
from core.prompt_settings import resolve_system_prompt
from services.llm.streaming import (
    _build_think_params,
    get_client_for_model,
)

logger = get_logger(__name__)

DEFAULT_ASR_CORRECTION_SAMPLE = (
    "我做过 red 地址缓存，麦 SQL 查询会先看布隆过绿器，"
    "如果空只命中就直接返回，避免打到数据库。"
)


def get_review_model_cfg():
    """获取 review 配置的模型（未配置 Key 时抛 ValueError）"""
    cfg = get_config()
    review_model = cfg.get_review_model()

    if review_model is None or not getattr(review_model, "enabled", True):
        raise ValueError("请先选择复盘分析模型")
    if not review_model.api_key or review_model.api_key in ("", "sk-your-api-key-here"):
        raise ValueError("Review 模型未配置有效的 API Key")

    return review_model


def get_active_llm_client() -> tuple[OpenAI, str]:
    """兼容旧调用：返回 review 模型的 SDK 客户端和模型名"""
    review_model = get_review_model_cfg()
    client = get_client_for_model(review_model)
    return client, review_model.model


def _review_think_params(model_cfg, max_tokens: int) -> dict[str, Any]:
    """Review 路径固定关闭思考；返回统一 call 需要的 think 参数。"""
    request_cfg = SimpleNamespace(
        think_mode=False,
        think_effort="off",
        max_tokens=max_tokens,
    )
    return _build_think_params(model_cfg, request_cfg)


def _review_chat(model_cfg, messages: list, *, temperature: float, max_tokens: int, trace_context=None):
    """统一非流式调用：api_type=anthropic/responses 的模型也能用。"""
    from services.llm.streaming import chat_completion_create

    return chat_completion_create(
        model_cfg,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        think_params=_review_think_params(model_cfg, max_tokens),
        trace_context=trace_context,
    )


def _build_asr_correction_prompt(question: str, candidate_answer: str) -> str:
    return f"""你是一位语音识别纠错助手。请根据面试问题的上下文，纠正候选人回答中可能存在的语音识别错误。

面试问题：
{question}

候选人回答（可能包含 ASR 错误）：
{candidate_answer}

常见 ASR 错误类型：
- 同音字错误（如：在这"快"领域 → 在这"块"领域）
- 专业术语识别错误（如：red 地址 → Redis）
- 标点符号缺失或错误
- 口语化表达（如："那个"、"嗯"等语气词过多）

请返回纠正后的文本，要求：
1. 只纠正明显的错误，不改变原意
2. 保留技术术语的正确拼写
3. 适当添加标点符号提高可读性
4. 去除过多的语气词和重复表达
5. 如果没有明显错误，返回原文
6. 不要添加原文中没有的内容

直接返回纠正后的文本，不要添加任何解释或标记。
"""


def _safe_error_detail(exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return detail[:500]


def run_asr_correction_check(
    question: str = "",
    candidate_answer: str = "",
) -> dict[str, Any]:
    """Run ASR correction and return a structured result for UI/API diagnostics."""
    cfg = get_config()
    review_model = cfg.get_review_model()
    original = (candidate_answer or "").strip() or DEFAULT_ASR_CORRECTION_SAMPLE
    result: dict[str, Any] = {
        "ok": False,
        "model_name": review_model.name if review_model else "",
        "model": review_model.model if review_model else "",
        "original": original,
        "corrected": original,
        "changed": False,
        "detail": "",
    }

    if len(original.strip()) < 10:
        result.update({"ok": True, "detail": "文本过短，未调用模型"})
        return result

    try:
        review_model = get_review_model_cfg()
    except ValueError as exc:
        result["detail"] = _safe_error_detail(exc)
        return result

    try:
        response = _review_chat(
            review_model,
            [
                {
                    "role": "system",
                    "content": "你是一位专业的语音识别纠错助手，擅长根据上下文纠正 ASR 错误。",
                },
                {"role": "user", "content": _build_asr_correction_prompt(question, original)},
            ],
            temperature=0.1,
            max_tokens=800,
            trace_context={"scene": "asr_correction"},
        )

        content = response.choices[0].message.content
        if not content or not content.strip():
            result["detail"] = "模型返回为空"
            return result

        corrected = content.strip()
        if len(corrected) < len(original) * 0.7:
            result.update({"ok": True, "detail": "纠错结果过短，已保留原文"})
            return result

        similarity = SequenceMatcher(None, original, corrected).ratio()
        if similarity < 0.4:
            result.update({"ok": True, "detail": f"纠错结果差异过大，已保留原文（相似度 {similarity:.2f}）"})
            return result

        result.update({
            "ok": True,
            "corrected": corrected,
            "changed": corrected != original,
            "detail": "纠错完成" if corrected != original else "模型认为无需纠错",
        })
        return result
    except Exception as exc:
        logger.warning("ASR correction check failed: %s", exc)
        result["detail"] = _safe_error_detail(exc)
        return result


def correct_asr_errors(
    question: str,
    candidate_answer: str,
) -> str:
    """
    使用 LLM 纠正语音识别错误
    返回纠正后的文本
    """
    if not candidate_answer or len(candidate_answer.strip()) < 10:
        return candidate_answer

    result = run_asr_correction_check(question, candidate_answer)
    if result.get("ok") and result.get("changed"):
        logger.info(
            "ASR correction applied: original_len=%d, corrected_len=%d",
            len(candidate_answer),
            len(str(result.get("corrected") or "")),
        )
        return str(result.get("corrected") or candidate_answer)
    if not result.get("ok"):
        logger.warning("ASR correction skipped: %s", result.get("detail") or "unknown error")
    return candidate_answer


_INTERVIEW_TURN_PROMPT = """你是一位资深技术面试官。结合面试问题、候选人回答、参考答案和代码（如有），客观评估候选人的回答。

列出亮点（例如准确性、深度、实战经验、代码质量）和可改进的风险。默认按准确性、深度、表达评分，每条亮点或风险控制在 30 字以内。
给出下次回答可以直接采用的改进动作、1-2 个可能的面试追问，以及 1-3 个相关技术标签。
如果候选人未回答或回答为空，亮点留空，风险注明“未回答”，各项评分给 0。
根据参考答案和实际回答质量评价，避免编造候选人未提到的项目经历。"""

_EXAM_TURN_PROMPT = """你是一位资深算法与笔试辅导教练，对截图题及本次生成答案进行客观复盘。

关注题意理解、算法正确性、复杂度、代码可提交性和边界用例；分析是否漏读截图条件、复杂度不达标、代码不可运行或缺少假设说明。
默认按正确性、完整性、可提交性评分，每条亮点或风险控制在 30 字以内。
给出下一次答题可采用的改进动作、1-2 个自测变体或边界用例问题，以及 1-3 个算法或工程标签。
如果未生成答案，亮点留空，风险注明“未生成答案”，各项评分给 0。
仅依据题面条件判断。如果答案使用了假设，请评价这些假设是否充分。"""

_INTERVIEW_SUMMARY_PROMPT = """你是一位资深技术面试官。根据本场问答和逐题复盘，对候选人的整场面试表现作出评价。

用 2-3 句话概述整体表现，列出 3-5 条主要亮点和 2-4 条改进方向。
提取 3-5 个高频亮点和 2-4 个高频短板作为关键词。"""

_EXAM_SUMMARY_PROMPT = """你是一位资深算法与笔试辅导教练。根据截图题、生成答案和逐题复盘，总结本次笔试练习。

用 2-3 句话概述答题质量，列出 3-5 条主要亮点和 2-4 条下次截图答题的改进方向。
提取 3-5 个高频亮点和 2-4 个高频风险作为关键词。"""

_TURN_OUTPUT_CONTRACT = """只返回一个 JSON 对象，包含以下字段：
- strengths、risks：字符串数组，可为空。
- scorecard：对象，键是评分维度名称，可自定义维度；值为 0 至 10 的数字。
- evidence：对象，可包含 improvement_advice（改进动作字符串）、follow_up_questions（追问字符串数组）和 tags（标签字符串数组）。"""

_SUMMARY_OUTPUT_CONTRACT = """只返回一个 JSON 对象，包含以下字段：
- summary_markdown：非空 Markdown 字符串，包含整体评价。
- strong_points、weak_points：字符串数组，分别包含亮点和短板关键词，可为空。"""

_REVIEW_PROMPT_SPECS = {
    "review_turn": (_INTERVIEW_TURN_PROMPT, _TURN_OUTPUT_CONTRACT, 1000),
    "review_summary": (_INTERVIEW_SUMMARY_PROMPT, _SUMMARY_OUTPUT_CONTRACT, 1500),
    "exam_review_turn": (_EXAM_TURN_PROMPT, _TURN_OUTPUT_CONTRACT, 1000),
    "exam_review_summary": (_EXAM_SUMMARY_PROMPT, _SUMMARY_OUTPUT_CONTRACT, 1500),
}
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def review_prompt_specs() -> dict[str, tuple[str, str, int]]:
    return _REVIEW_PROMPT_SPECS.copy()


def _request_review_json(scene: str, user_text: str, overrides: Mapping[str, str] | None) -> dict[str, Any]:
    model = get_review_model_cfg()
    if overrides is None:
        configured = getattr(get_config(), "system_prompts", None)
        overrides = configured if isinstance(configured, Mapping) else {}
    default, contract, max_tokens = _REVIEW_PROMPT_SPECS[scene]
    response = _review_chat(
        model,
        [
            {"role": "system", "content": resolve_system_prompt(overrides, scene, default, contract)},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        trace_context={"scene": scene},
    )

    content = response.choices[0].message.content if response.choices else ""
    if not content:
        raise ValueError("复盘模型返回为空")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        # 兼容外层 Markdown 围栏，同时保留 JSON 字符串内部的代码块。
        match = _JSON_FENCE.fullmatch(content.strip())
        if not match:
            raise ValueError("复盘模型未返回有效 JSON") from exc
        try:
            result = json.loads(match.group(1))
        except json.JSONDecodeError as fenced_exc:
            raise ValueError("复盘模型未返回有效 JSON") from fenced_exc
    if not isinstance(result, dict):
        raise ValueError("复盘模型未返回 JSON 对象")
    return result


def _string_list(result: dict, field: str) -> list[str]:
    value = result.get(field)
    if not isinstance(value, list) or not all(isinstance(text, str) for text in value):
        raise ValueError(f"复盘模型字段 {field} 必须是字符串数组")
    return value


def _turn_result(result: dict[str, Any]) -> dict[str, Any]:
    scorecard = result.get("scorecard")
    if not isinstance(scorecard, dict) or not all(
        dimension.strip() and type(score) in (int, float) and 0 <= score <= 10
        for dimension, score in scorecard.items()
    ):
        raise ValueError("复盘模型字段 scorecard 必须是维度名称到 0 至 10 分的映射")

    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("复盘模型字段 evidence 必须是对象")
    action_evidence: dict[str, Any] = {}
    if "improvement_advice" in evidence:
        advice = evidence["improvement_advice"]
        if not isinstance(advice, str):
            raise ValueError("复盘模型字段 improvement_advice 必须是字符串")
        action_evidence["improvement_advice"] = advice
    for field in ("follow_up_questions", "tags"):
        if field in evidence:
            action_evidence[field] = _string_list(evidence, field)

    return {
        "strengths": _string_list(result, "strengths"),
        "risks": _string_list(result, "risks"),
        "scorecard": scorecard,
        "evidence": action_evidence,
    }


def analyze_turn(
    question: str,
    candidate_answer: str,
    reference_answer: str,
    code_text: str = "",
    apply_asr_correction: bool = True,
    review_source: str = "assist",
    *,
    system_prompts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """分析单题并验证模型返回的复盘数据。"""
    is_written_exam = (review_source or "").strip() == "written_exam"
    corrected_answer = candidate_answer
    if not is_written_exam and apply_asr_correction and candidate_answer:
        corrected_answer = correct_asr_errors(question, candidate_answer)

    if is_written_exam:
        scene = "exam_review_turn"
        user_text = f"截图题目/题面：\n{question}\n\n本次生成答案：\n{reference_answer or '(未生成答案)'}"
    else:
        scene = "review_turn"
        user_text = (
            f"问题：\n{question}\n\n候选人回答：\n{corrected_answer or '(未录制到回答)'}"
            f"\n\n参考答案：\n{reference_answer}"
        )
    if code_text:
        user_text += f"\n\n代码：\n{code_text}"

    result = _turn_result(_request_review_json(scene, user_text, system_prompts))
    if is_written_exam:
        result["evidence"]["review_mode"] = "written_exam"
    if corrected_answer != candidate_answer:
        result["evidence"]["asr_correction"] = {
            "original": candidate_answer,
            "corrected": corrected_answer,
        }
    result["corrected_answer"] = corrected_answer if corrected_answer != candidate_answer else None
    return result


def generate_summary(
    turns: list[dict[str, Any]],
    review_source: str = "assist",
    *,
    system_prompts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """根据已经分析的逐题数据生成整体评价。"""
    is_written_exam = (review_source or "").strip() == "written_exam"
    if not turns:
        return {
            "summary_markdown": "本次笔试未录制到有效题目" if is_written_exam else "本场面试未录制到有效问答",
            "strong_points": [],
            "weak_points": [],
        }

    if is_written_exam:
        scene = "exam_review_summary"
        user_parts = [f"本次笔试练习共 {len(turns)} 道截图题："]
    else:
        scene = "review_summary"
        user_parts = [f"本场面试共 {len(turns)} 轮问答："]

    for idx, turn in enumerate(turns, start=1):
        user_parts.append(f"\n### 第 {idx} 题")
        if is_written_exam:
            user_parts.append(f"截图题目/题面：{turn.get('question_text', '')}")
            user_parts.append(f"生成答案：{turn.get('reference_answer_text', '(未生成)')}")
        else:
            user_parts.append(f"问题：{turn.get('question_text', '')}")
            user_parts.append(f"回答：{turn.get('candidate_answer_text', '(未录制)')}")
        for field, label in (("strengths", "亮点"), ("risks", "风险")):
            values = turn.get(field)
            if isinstance(values, list):
                texts = [text for text in values if isinstance(text, str)]
                if texts:
                    user_parts.append(f"{label}：{', '.join(texts)}")

    result = _request_review_json(scene, "\n".join(user_parts), system_prompts)
    summary = result.get("summary_markdown")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("复盘模型字段 summary_markdown 必须是非空字符串")
    return {
        "summary_markdown": summary,
        "strong_points": _string_list(result, "strong_points"),
        "weak_points": _string_list(result, "weak_points"),
    }
