"""会话记忆滚动摘要 Agent (P2-agent-memory)。

会话超过阈值时,把最早的若干条 messages 异步压成一段事实笔记替换掉,
保留最近 N 条原文不动。摘要由活动模型本身完成,失败时维持原样,
保证主对话链路绝不被阻塞。

调用入口由 Session 在 add_*_message 时触发,执行体在守护线程里跑;
最终覆盖动作再次拿 conversation_lock,避免并发写。
"""
from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

from core.logger import get_logger

_log = get_logger("memory_agent")

_SUMMARY_PROMPT = """你是面试会话的记忆压缩助手。请把下面这段更早的「问答历史」凝练成简洁的事实笔记,
保留:候选人提到过的关键技术栈/项目/数字、面试官追问过的方向、已经回答过的问题及结论。
舍弃:寒暄、重复表达、过程性思考。

输出要求:
- 中文,Markdown 列表,不超过 12 条 bullet。
- 每条不超过 60 字,不复述原文。
- 结尾如果之前已有摘要,请将旧摘要与本次合并去重。

之前已有的滚动摘要:
{prior}

需要压缩的新历史:
{chunk}

请只输出新的滚动摘要本身,不要任何前后缀文字。
"""


def _render_messages(messages: Iterable[dict]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: List[str] = []
            for piece in content:
                if isinstance(piece, dict) and piece.get("type") == "text":
                    text_parts.append(str(piece.get("text", "")))
                elif isinstance(piece, dict) and piece.get("type") == "image_url":
                    text_parts.append("[图片]")
            content_str = "\n".join(t for t in text_parts if t).strip()
        else:
            content_str = str(content or "").strip()
        if not content_str:
            continue
        # 单条防止过长
        if len(content_str) > 1500:
            content_str = content_str[:1500] + "…"
        parts.append(f"[{role}] {content_str}")
    return "\n\n".join(parts)


def summarize_messages(messages: Sequence[dict], prior_summary: str = "") -> str:
    """同步调用 LLM 做一次压缩。失败抛异常,调用方负责兜底。"""
    if not messages:
        return prior_summary or ""
    rendered = _render_messages(messages)
    if not rendered.strip():
        return prior_summary or ""

    # 延迟导入,避免循环依赖 (services.llm <-> core.session 路径)
    from core.config import get_config
    from services.llm import _add_tokens
    from services.llm.streaming import _build_think_params, chat_completion_create

    cfg = get_config()
    model_cfg = cfg.get_active_model()
    if model_cfg is None or not getattr(model_cfg, "enabled", True):
        raise ValueError("请先选择通用默认模型")

    prompt = _SUMMARY_PROMPT.format(
        prior=prior_summary.strip() or "(无)",
        chunk=rendered,
    )
    max_tokens = 600
    think_params = _build_think_params(
        model_cfg,
        SimpleNamespace(think_mode=False, think_effort="off", max_tokens=max_tokens),
    )
    response = chat_completion_create(
        model_cfg,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
        think_params=think_params,
        trace_context={"scene": "memory_compaction"},
    )
    usage = getattr(response, "usage", None)
    if usage:
        _add_tokens(usage.prompt_tokens or 0, usage.completion_tokens or 0)
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("memory compaction returned empty choices")
    message = getattr(choices[0], "message", None)
    text = (getattr(message, "content", None) or "").strip()
    if not text:
        raise RuntimeError("memory compaction returned empty summary text")
    return text


def schedule_compaction(target_session) -> Optional[threading.Thread]:
    """供 Session 内部调用:在守护线程中执行一次压缩,然后回写。

    target_session 必须暴露:
      - conversation_history (list)
      - SUMMARY_KEEP_RECENT (int)
      - system_summary (str)
      - _compaction_running (bool)
      - session_version (int, 可选，默认 0)
    """
    from core.session import conversation_lock

    keep = max(0, int(getattr(target_session, "SUMMARY_KEEP_RECENT", 4)))

    with conversation_lock:
        if target_session._compaction_running:
            return None
        history = target_session.conversation_history
        if len(history) <= keep:
            return None
        snapshot_len = len(history) - keep
        snapshot_refs = tuple(history[:snapshot_len])
        snapshot = copy.deepcopy(history[:snapshot_len])
        prior = target_session.system_summary or ""
        captured_version = getattr(target_session, "session_version", 0)
        target_session._compaction_running = True

    def _worker() -> None:
        compaction_ok = False
        new_summary = prior
        try:
            new_summary = summarize_messages(snapshot, prior)
            compaction_ok = True
        except Exception as e:  # noqa: BLE001
            _log.warning("memory compaction failed, keep prior summary and history: %s", e)
        with conversation_lock:
            same_session = getattr(target_session, "session_version", 0) == captured_version
            try:
                if not same_session:
                    _log.debug(
                        "discard stale memory compaction result (captured_v=%s current_v=%s)",
                        captured_version,
                        getattr(target_session, "session_version", 0),
                    )
                    return
                if not compaction_ok:
                    return
                current_history = target_session.conversation_history
                prefix_unchanged = (
                    len(current_history) >= snapshot_len
                    and all(cur is ref for cur, ref in zip(current_history, snapshot_refs))
                    and current_history[:snapshot_len] == snapshot
                )
                summary_unchanged = (target_session.system_summary or "") == prior
                if prefix_unchanged and summary_unchanged:
                    target_session.conversation_history = current_history[snapshot_len:]
                    target_session.system_summary = new_summary
                else:
                    _log.debug(
                        "skip memory compaction writeback because session state drifted "
                        "(len=%d snapshot_len=%d prefix_unchanged=%s summary_unchanged=%s)",
                        len(current_history),
                        snapshot_len,
                        prefix_unchanged,
                        summary_unchanged,
                    )
            finally:
                if same_session:
                    target_session._compaction_running = False

    try:
        thread = threading.Thread(target=_worker, name="memory-compactor", daemon=True)
        thread.start()
    except Exception:
        with conversation_lock:
            if getattr(target_session, "session_version", 0) == captured_version:
                target_session._compaction_running = False
        raise
    return thread
