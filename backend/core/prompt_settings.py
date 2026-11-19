from collections.abc import Mapping

# Display order also defines the order in the settings scene selector.
PROMPT_SCENES = {
    "assist_realtime": ("实时辅助", "语音回答", "用于每轮实时语音答题。"),
    "assist_manual": ("实时辅助", "手动提问", "在实时辅助底部输入问题，或粘贴图片后发送。"),
    "assist_screen": ("实时辅助", "电脑截图答题", "通过悬浮窗的截图审题、截图快捷键或连续截图提交。"),
    "assist_exam": ("实时辅助", "笔试答题", "设置 → 模式 → 开启笔试模式后，适用于手动提问及截图。"),
    "review_turn": ("面试复盘", "面试逐题评价", "生成面试复盘时，评价每一轮问答。"),
    "review_summary": ("面试复盘", "面试整场总结", "生成面试复盘时，总结整场面试表现。"),
    "exam_review_turn": ("面试复盘", "笔试逐题评价", "生成笔试复盘时，评价每一道题。"),
    "exam_review_summary": ("面试复盘", "笔试整体总结", "生成笔试复盘时，总结整场答题表现。"),
    "resume_jd": ("简历优化", "JD 能力提取", "开始简历优化后，提取职位描述中的关键能力。"),
    "resume_match": ("简历优化", "匹配分析", "简历优化中，对照职位能力清单和简历。"),
    "resume_rewrite": ("简历优化", "改写建议", "简历优化中，基于匹配结果生成改写建议。"),
    "resume_fallback": ("简历优化", "综合分析", "简历优化的分步分析失败时使用。"),
}


def patch_system_prompts(current: dict[str, str], updates: dict[str, str | None]) -> dict[str, str]:
    if not isinstance(updates, dict):
        raise ValueError("系统提示词设置必须按场景提交")
    result = current.copy()
    for scene, prompt in updates.items():
        if scene not in PROMPT_SCENES:
            raise ValueError("系统提示词包含未知场景")
        if prompt is None:
            result.pop(scene, None)
        elif isinstance(prompt, str):
            result[scene] = prompt
        else:
            raise ValueError("系统提示词必须是文本")
    return result


def resolve_system_prompt(
    overrides: Mapping[str, str],
    scene: str,
    default: str,
    contract: str = "",
) -> str:
    prompt = overrides.get(scene, default)
    if not contract:
        return prompt
    constraint = "程序要求的输出结构（优先遵守）\n" + contract
    return f"{prompt}\n\n{constraint}" if prompt else constraint
