"""Default assist prompts and output cleaning."""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from core.config import get_config
from services.llm import build_system_prompt, create_answer_stream_sanitizer, postprocess_answer_for_mode
from services.llm.prompts import build_assist_user_context


@pytest.fixture
def reset_resume():
    cfg = get_config()
    old = cfg.resume_text
    try:
        yield cfg
    finally:
        cfg.resume_text = old


def _asr_prompt(high_churn: bool = False) -> str:
    return build_system_prompt(
        mode="asr_realtime", high_churn_short_answer=high_churn
    )


def _manual_prompt() -> str:
    return build_system_prompt(mode="manual_text")


@pytest.mark.parametrize("short", [False, True])
def test_voice_prompt_uses_real_speaker_labeled_conversation(short):
    prompt = _asr_prompt(high_churn=short)
    assert "真实对话" in prompt
    assert "【面试官】" in prompt and "【候选人】" in prompt
    assert "【面试官｜当前发言】" in prompt
    assert "直接说出口" in prompt
    assert "第一句直接" in prompt
    assert "先按大概率" not in prompt
    assert "我可以换个方向" not in prompt
    assert "追问上下文" not in prompt
    assert "术语" in prompt and "转写" in prompt
    assert len(prompt) < 500


def test_default_prompt_has_resume_two_mode_rule():
    p = _manual_prompt()
    assert "简历使用规则" in p
    assert "简历深挖题" in p
    assert "简历相关题" in p
    assert "简历无关题" in p


def test_user_context_has_no_resume_reference_when_resume_missing(reset_resume):
    cfg = reset_resume
    cfg.resume_text = None
    context = build_assist_user_context(cfg, mode="manual_text")
    assert "<resume_context>" not in context


def test_written_exam_task_never_includes_resume(reset_resume):
    cfg = reset_resume
    cfg.resume_text = "项目A: 后端重构，Kafka, Redis；2022 字节实习。"
    p = build_system_prompt(mode="written_exam")
    assert "项目A" not in p
    assert "Kafka" not in p
    assert "简历使用规则" not in p
    assert "<resume_context>" not in p
    assert "项目A" not in build_assist_user_context(cfg, mode="written_exam")


def test_user_context_can_explicitly_disable_resume_reference(reset_resume):
    cfg = reset_resume
    cfg.resume_text = "项目A: 后端重构，Kafka, Redis；2022 字节实习。"
    context = build_assist_user_context(cfg, mode="asr_realtime", include_resume=False)
    assert "项目A" not in context
    assert "<resume_context>" not in context


def test_written_exam_prompt_handles_incremental_screenshots_and_failures(reset_resume):
    p = build_system_prompt(mode="written_exam")
    assert "连续截图/失败反馈规则" in p
    assert "当前截图优先于上一版答案" in p
    assert "新增约束" in p
    assert "失败用例" in p
    assert "修正后的完整可提交代码" in p


def test_first_sentence_hard_constraint_in_manual_text():
    assert "首句硬约束" in _manual_prompt()


@pytest.mark.parametrize("mode,short", [("asr_realtime", False), ("asr_realtime", True), ("manual_text", False)])
def test_live_prompt_asks_for_candidate_voice_without_markdown(mode, short):
    prompt = build_system_prompt(mode=mode, high_churn_short_answer=short, cfg=SimpleNamespace(system_prompts={}))
    assert "纯文本" in prompt
    assert "不要使用 Markdown 标题、列表、加粗或代码围栏" in prompt
    assert "专注面板输出协议" not in prompt
    assert "`##`" not in prompt


def test_manual_prompt_warns_not_to_emit_template_labels():
    p = _manual_prompt()
    assert "真人候选人口吻" in p
    assert "简单题用几句话说清结论和理由" in p
    assert "复杂题可多写几个短段落" in p
    assert "普通技术题不要强行改写成个人项目故事" in p
    assert "关键机制、方案取舍、验证和风险" in p
    assert "只有题干明确问" in p


def test_structured_manual_use_keeps_the_focus_output_contract():
    prompt = build_system_prompt(
        mode="manual_text", structured_manual=True, cfg=SimpleNamespace(system_prompts={}),
    )
    assert "专注面板输出协议" in prompt
    assert "3-6 个 `##` section" in prompt
    assert "不要使用 Markdown 标题" not in prompt


def test_stream_sanitizer_removes_split_think_tags():
    s = create_answer_stream_sanitizer("manual_text")
    chunks = ["<thi", "nking>内部思考</thinking>用 AOF", " 和 RDB。"]
    out = "".join(s.push(c) for c in chunks) + s.finish()
    assert out == "用 AOF 和 RDB。"


def test_plain_stream_keeps_code_and_suppresses_nested_internal_tags():
    sanitizer = create_answer_stream_sanitizer("manual_text", plain_text=True)
    assert sanitizer.push("可用条件 x <") == "可用条件 x "
    assert sanitizer.push(" 3。\n<ana") == "< 3。\n"
    assert sanitizer.push("lysis>草稿<THINK/>仍是草稿<thinking>内层</thinking>外层") == ""
    assert sanitizer.push("</analysis>    int **p;") == "    int **p;"
    assert sanitizer.finish() == ""


def test_plain_stream_preserves_xml_elements_with_similar_names():
    sanitizer = create_answer_stream_sanitizer("manual_text", plain_text=True)
    assert sanitizer.push("<analysis") == ""
    literal = "<analysis-result>data</analysis-result>"
    assert sanitizer.push("-result>data</analysis-result>") == literal
    assert sanitizer.finish() == ""


def test_stream_sanitizer_cleans_manual_markdown_and_meta_preface():
    s = create_answer_stream_sanitizer("manual_text")
    out = s.push("这是一个经典问题。\n# 回答\n**核心**是先止血。") + s.finish()
    assert "这是一个经典问题" not in out
    assert "#" not in out
    assert "回答" not in out
    assert "**" not in out
    assert "核心是先止血。" in out


def test_stream_sanitizer_keeps_focus_tab_headings_for_manual_text():
    s = create_answer_stream_sanitizer("manual_text")
    out = s.push("## 多元答案\n核心是先止血。") + s.finish()
    assert "## 多元答案" in out
    assert "核心是先止血。" in out


def test_stream_sanitizer_keeps_screen_markdown_but_removes_think():
    s = create_answer_stream_sanitizer("server_screen_code")
    out = s.push("<analysis>草稿</analysis># 题目理解\n正文") + s.finish()
    assert "草稿" not in out
    assert "# 题目理解" in out


def test_custom_answer_keeps_unicode_and_indentation_across_thought_tag_boundary():
    visible = "    İstanbul\n\n# Başlık\n**Kalın**"
    tag_start = '<think reason="' + "x" * 100
    chunks = [visible[:12] + tag_start, '">隐藏内容</thi', "nk>" + visible[12:]]

    sanitizer = create_answer_stream_sanitizer("manual_text", preserve_format=True)
    streamed = "".join(sanitizer.push(chunk) for chunk in chunks) + sanitizer.finish()

    assert streamed == visible
    assert postprocess_answer_for_mode("".join(chunks), "manual_text", preserve_format=True) == visible


def test_custom_answer_preserves_literal_thought_tags_in_markdown_fences():
    literal = "```xml\n<think>示例</think>\n</analysis>\n```"
    visible = "示例：\n\n" + literal + "\n\n" + "以上是代码示例。" * 15
    raw = "<analysis>隐藏内容</analysis>" + visible
    chunks = [raw[index:index + 7] for index in range(0, len(raw), 7)]
    mode = "manual_text"
    sanitizer = create_answer_stream_sanitizer(mode, preserve_format=True)

    assert "".join(sanitizer.push(chunk) for chunk in chunks) + sanitizer.finish() == visible
    assert postprocess_answer_for_mode(raw, mode, preserve_format=True) == visible
