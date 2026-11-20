"""Resume × JD Grounded Agent (P2-agent-resume).

把原本「prompt → 一次性出全文」的简历优化拆成三步:

1) JD 关键能力抽取(JSON 输出,小模型 / low temperature)
2) 命中 / 缺失分析(逐条引用简历行号 L<n>,流式 Markdown)
3) 改写建议(必须引用对应 L<n> 的原文,禁止编造经历)

简历正文在送入步骤 2、3 前预先打编号，默认提示词要求用 L<n> 引用。
失败回退策略:
- 步骤 1 失败 → 直接走 `_legacy_pass`(单次综合分析),保证用户至少拿到基础结果。
- 步骤 2 失败 → 在已输出步骤 1 的基础上追加提示,再走 `_legacy_pass` 续输出,
  避免只输出半截 Markdown。
- 步骤 3 失败 → 在已输出步骤 1、2 的基础上,追加 `_legacy_pass` 的「兜底改写」段。
"""
from __future__ import annotations

import json
import re
import secrets
from collections.abc import Mapping
from typing import Any, Dict, Generator, List, Tuple


def _new_eid() -> str:
    """生成短错误编号,便于用户在 UI 看到 + 服务端日志反查。"""
    return secrets.token_hex(3)

from core.config import get_config
from core.logger import get_logger
from core.prompt_settings import resolve_system_prompt
from services.llm import _add_tokens, get_client
from services.llm import trace

_log = get_logger("resume_optimizer")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JD_EXTRACT_PROMPT = """你是资深技术招聘顾问。请从 JD 中抽取候选人需要具备的关键能力，按硬性必需、加分项和软实力分组。
每条写成简洁的能力词或场景短语，避免照抄整段 JD。"""

_MATCH_PROMPT = """你是简历审核员。根据 JD 关键能力清单和已编号的简历，逐条判断能力是否在简历中有体现。

用 Markdown 输出“命中能力”和“缺失能力”两个小节。
- 命中：用粗体写出能力名称，给出简历行号并附简短原文片段。
- 缺失：用粗体写出能力名称，简短建议在哪个段落补充。
只引用简历中实际存在的行号，使用 L<n> 格式，如 L12。优先说明硬性要求的匹配情况。"""

_REWRITE_PROMPT = """你是资深简历教练。根据 JD、命中与缺失分析以及简历，给出 3-6 条针对性的改写建议。

用 Markdown 列出建议，每条给出一句话标题、简历引用、改写后的版本、对应的 JD 能力和改写理由。引用的原文直接摘自简历。
每条建议用 L<n> 引用要修改的原文行号；涉及插入时注明“在 L<n> 之后插入”。只使用简历中实际存在的行号。
只能基于现有内容加强表达、重排或量化，避免编造候选人的经历或技术栈。"""

_FALLBACK_PROMPT = """你是资深技术招聘顾问。用 Markdown 简略对比 JD 和简历并给出优化建议，包含匹配度评分、命中亮点、缺失关键词和面试重点。"""

_JD_OUTPUT_CONTRACT = """只返回一个 JSON 对象。must_have、nice_to_have、soft_skills 三个字段均为字符串数组，可为空。"""

_RESUME_PROMPT_SPECS = {
    "resume_jd": (_JD_EXTRACT_PROMPT, _JD_OUTPUT_CONTRACT, 700),
    "resume_match": (_MATCH_PROMPT, "", 1600),
    "resume_rewrite": (_REWRITE_PROMPT, "", 1800),
    "resume_fallback": (_FALLBACK_PROMPT, "", 2400),
}


def resume_prompt_specs() -> dict[str, tuple[str, str, int]]:
    return _RESUME_PROMPT_SPECS.copy()


def _resume_request(
    overrides: Mapping[str, str], scene: str, user_text: str,
) -> tuple[list[dict], int]:
    default, contract, max_tokens = _RESUME_PROMPT_SPECS[scene]
    return [
        {"role": "system", "content": resolve_system_prompt(overrides, scene, default, contract)},
        {"role": "user", "content": user_text},
    ], max_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _number_resume(text: str) -> Tuple[str, Dict[int, str]]:
    """给简历每个非空行打 L<n> 前缀,返回 (numbered_text, {n: 原文})."""
    out: List[str] = []
    line_map: Dict[int, str] = {}
    counter = 0
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped.strip():
            out.append("")
            continue
        counter += 1
        line_map[counter] = stripped
        out.append(f"L{counter}: {stripped}")
    return "\n".join(out), line_map


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _try_parse_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    cleaned = _JSON_FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # 退而求其次:尝试匹配第一个 {...} 块
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


_REQUIREMENT_SECTIONS = {
    "must_have": "必需技能 / 经验",
    "nice_to_have": "加分项",
    "soft_skills": "软实力",
}


def _parse_requirements(raw: str) -> dict[str, list[str]]:
    parsed = _try_parse_json(raw)
    requirements = {}
    for key in _REQUIREMENT_SECTIONS:
        items = parsed.get(key)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"JD 抽取模型字段 {key} 必须是字符串数组")
        requirements[key] = [item.strip() for item in items if item.strip()]
    return requirements


def _format_requirements(requirements: dict[str, list[str]]) -> str:
    sections = []
    for key, label in _REQUIREMENT_SECTIONS.items():
        items = requirements[key]
        if items or key == "must_have":
            bullets = "\n".join(f"- {item}" for item in items) if items else "- (无)"
            sections.append(f"**{label}**\n{bullets}")
    return "\n\n".join(sections)


_LINE_REF_RE = re.compile(r"\bL\d+\b")


def _rewrite_needs_line_reference_warning(text: str) -> bool:
    """改写建议必须能追溯到简历行号；缺失时提示人工确认。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return _LINE_REF_RE.search(cleaned) is None


@trace.traced("resume_optimization")
def _stream_chat(
    client,
    model_cfg,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """流式调用,逐 chunk yield 文本。usage 自动累加。api_type 任意协议可用。"""
    from services.llm.protocols import build_chat_payload, get_api_type, post_chat_stream

    api_type = get_api_type(model_cfg)
    if api_type != "openai":
        payload = build_chat_payload(
            api_type,
            model_cfg.model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        response = post_chat_stream(model_cfg.api_base_url, api_type, model_cfg.api_key, payload)
    else:
        response = client.chat.completions.create(
            model=model_cfg.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
    for chunk in response:
        choices = getattr(chunk, "choices", None) or []
        if choices and choices[0].delta and choices[0].delta.content:
            yield choices[0].delta.content
        usage = getattr(chunk, "usage", None)
        if usage:
            _add_tokens(usage.prompt_tokens or 0, usage.completion_tokens or 0)


def _non_stream_chat(
    client,
    model_cfg,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    trace_context: dict | None = None,
) -> str:
    from services.llm.streaming import chat_completion_create

    response = chat_completion_create(
        model_cfg,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        trace_context=trace_context,
    )
    usage = getattr(response, "usage", None)
    if usage:
        _add_tokens(usage.prompt_tokens or 0, usage.completion_tokens or 0)
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_resume_stream(
    jd_text: str,
    *,
    system_prompts: Mapping[str, str] | None = None,
) -> Generator[str, None, None]:
    """三步式 grounded 流式输出。任一步失败自动降级到单次分析。"""
    cfg = get_config()
    resume_text = cfg.resume_text
    if not resume_text:
        yield "[错误: 请先上传简历]"
        return

    model_cfg = cfg.get_active_model()
    if model_cfg is None or not getattr(model_cfg, "enabled", True):
        yield "[错误: 请先选择通用默认模型]"
        return
    client = get_client()

    configured = getattr(cfg, "system_prompts", None) if system_prompts is None else system_prompts
    overrides = dict(configured) if isinstance(configured, Mapping) else {}
    numbered_resume, line_map = _number_resume(resume_text)
    if not line_map:
        yield "[错误: 简历内容为空,无法进行行号引用]"
        return

    # ----- Step 1: 抽取 JD 关键能力 -----
    yield "## 📊 JD 关键能力抽取\n\n"
    jd_input = f"JD：\n{jd_text}"
    fallback_input = f"{jd_input}\n\n简历：\n{resume_text}"
    try:
        messages, max_tokens = _resume_request(overrides, "resume_jd", jd_input)
        raw_extract = _non_stream_chat(
            client,
            model_cfg,
            messages,
            temperature=0.2,
            max_tokens=max_tokens,
            trace_context={"scene": "resume_jd"},
        )
        requirements = _parse_requirements(raw_extract)
    except Exception as e:  # noqa: BLE001
        eid = _new_eid()
        _log.warning("resume agent step1 failed [%s]: %s", eid, e)
        yield (
            f"\n> 抽取步骤失败,已自动降级到单次综合分析。"
            f"(错误编号 `{eid}`,详情见服务端日志)\n\n---\n\n"
        )
        yield from _legacy_pass(client, model_cfg, overrides, fallback_input)
        return

    yield _format_requirements(requirements) + "\n\n---\n\n"

    # ----- Step 2: 命中 / 缺失分析 -----
    yield "## 🎯 命中 / 缺失分析\n\n"
    analysis_chunks: List[str] = []
    try:
        requirements_text = json.dumps(requirements, ensure_ascii=False)
        match_input = f"JD 关键能力：\n{requirements_text}\n\n简历（每行已编号）：\n{numbered_resume}"
        messages, max_tokens = _resume_request(overrides, "resume_match", match_input)
        for piece in _stream_chat(
            client,
            model_cfg,
            messages,
            temperature=0.3,
            max_tokens=max_tokens,
            trace_context={"scene": "resume_match"},
        ):
            analysis_chunks.append(piece)
            yield piece
    except Exception as e:  # noqa: BLE001
        eid = _new_eid()
        _log.warning("resume agent step2 failed [%s]: %s", eid, e)
        yield (
            f"\n\n> 匹配步骤失败,已切换为兜底综合分析。"
            f"(错误编号 `{eid}`)\n\n"
            "---\n\n## 🛟 兜底综合分析(降级)\n\n"
        )
        yield from _legacy_pass(client, model_cfg, overrides, fallback_input)
        return

    analysis_text = "".join(analysis_chunks).strip()
    if not analysis_text:
        yield (
            "\n\n> 模型返回空匹配结果,使用兜底综合分析。\n\n"
            "---\n\n## 🛟 兜底综合分析(降级)\n\n"
        )
        yield from _legacy_pass(client, model_cfg, overrides, fallback_input)
        return
    yield "\n\n---\n\n"

    # ----- Step 3: 改写建议 -----
    yield "## ✏️ 简历改写建议\n\n"
    rewrite_chunks: List[str] = []
    try:
        rewrite_input = (
            f"{jd_input}\n\n命中 / 缺失分析：\n{analysis_text}"
            f"\n\n简历（每行已编号）：\n{numbered_resume}"
        )
        messages, max_tokens = _resume_request(overrides, "resume_rewrite", rewrite_input)
        for piece in _stream_chat(
            client,
            model_cfg,
            messages,
            temperature=0.45,
            max_tokens=max_tokens,
            trace_context={"scene": "resume_rewrite"},
        ):
            rewrite_chunks.append(piece)
            yield piece
        rewrite_text = "".join(rewrite_chunks)
        if "resume_rewrite" not in overrides and _rewrite_needs_line_reference_warning(rewrite_text):
            yield (
                "\n\n> 需人工确认：部分改写建议缺少行号引用，"
                "请回到原简历逐条核对后再采用。"
            )
    except Exception as e:  # noqa: BLE001
        eid = _new_eid()
        _log.warning("resume agent step3 failed [%s]: %s", eid, e)
        yield (
            f"\n\n> 改写步骤失败,已切换为兜底改写建议。"
            f"(错误编号 `{eid}`)\n\n"
            "---\n\n## 🛟 兜底改写建议(降级)\n\n"
        )
        yield from _legacy_pass(client, model_cfg, overrides, fallback_input)


def _legacy_pass(
    client,
    model_cfg,
    overrides: Mapping[str, str],
    user_text: str,
) -> Generator[str, None, None]:
    """三步链路完全不可用时的兜底:旧的一次性 prompt。"""
    try:
        messages, max_tokens = _resume_request(overrides, "resume_fallback", user_text)
        yield from _stream_chat(
            client,
            model_cfg,
            messages,
            temperature=0.6,
            max_tokens=max_tokens,
            trace_context={"scene": "resume_fallback"},
        )
    except Exception as e:  # noqa: BLE001
        eid = _new_eid()
        _log.warning("resume legacy_pass failed [%s]: %s", eid, e)
        yield f"\n\n[兜底分析也失败,错误编号 `{eid}`,详情见服务端日志]"
