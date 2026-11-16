"""Fit realtime interview prompts to the model window and retry overflow once."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from uuid import uuid4

from core.config import DEFAULT_CONTEXT_WINDOW_TOKENS, get_config
from core.logger import get_logger
from services.llm.protocols import API_TYPE_RESPONSES, RESPONSES_MIN_OUTPUT_TOKENS, get_api_type
from services.llm.streaming import LLMContextExceeded, chat_stream_single_model


_SEPARATOR = "\n\n"
_MESSAGE_OVERHEAD = 64
_log = get_logger("llm.interview_budget")


class InterviewContextTooLarge(LLMContextExceeded):
    user_msg = "当前问题和任务资料超过模型上下文限制，请调整模型窗口或更换模型。"


@dataclass(frozen=True)
class InterviewPrompt:
    user_text: str
    retained_rounds: int
    context_window_tokens: int
    estimated_input_tokens: int


def _output_limit(model_cfg, requested: int) -> int:
    minimum = RESPONSES_MIN_OUTPUT_TOKENS if get_api_type(model_cfg) == API_TYPE_RESPONSES else 1
    return max(minimum, int(requested))


def prepare_interview_prompt(
    model_cfg,
    rounds: Sequence[Sequence[str]],
    *,
    system_prompt: str | None = None,
    context_prefix: str | None = None,
    override_max_tokens: int,
    context_window_tokens: int | None = None,
) -> InterviewPrompt:
    """Select the largest contiguous suffix that fits, keeping the final round intact."""
    if not rounds or not rounds[-1]:
        raise ValueError("an interview prompt requires a current round")

    window = context_window_tokens
    if window is None:
        window = getattr(model_cfg, "context_window_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS)
    # Each UTF-8 byte can cost a token across providers. Leave space for
    # framing and preserve the full requested completion allocation.
    output_limit = _output_limit(model_cfg, override_max_tokens)
    input_budget = min(window * 9 // 10, window - output_limit)
    remaining = input_budget - _MESSAGE_OVERHEAD - len((system_prompt or "").encode("utf-8"))
    if context_prefix:
        remaining -= len(context_prefix.encode("utf-8")) + len(_SEPARATOR)

    retained: list[str] = []
    for round_parts in reversed(rounds):
        round_text = _SEPARATOR.join(round_parts)
        cost = len(round_text.encode("utf-8")) + (len(_SEPARATOR) if retained else 0)
        if cost > remaining:
            break
        retained.append(round_text)
        remaining -= cost

    if not retained:
        raise InterviewContextTooLarge(
            "The system prompt, task context and current interview round must fit within the model window.",
            context_window_tokens=window,
        )

    retained.reverse()
    user_text = _SEPARATOR.join(([context_prefix] if context_prefix else []) + retained)
    return InterviewPrompt(user_text, len(retained), window, input_budget - remaining)


def stream_interview_answer(
    model_cfg,
    rounds: Sequence[Sequence[str]],
    *,
    system_prompt: str | None = None,
    context_prefix: str | None = None,
    override_max_tokens: int | None = None,
    stream_fn: Callable[..., Iterable[tuple[str, str]]] = chat_stream_single_model,
    abort_check: Callable[[], bool] | None = None,
    mmpg_request=None,
    config_snapshot=None,
    trace_context: dict | None = None,
    **stream_kwargs,
) -> Iterator[tuple[str, str]]:
    """Stream one logical answer with at most one shorter transport attempt."""
    def aborted() -> bool:
        return bool((mmpg_request is not None and mmpg_request.cancelled) or (abort_check and abort_check()))

    if aborted():
        return

    cfg = config_snapshot if config_snapshot is not None else get_config()
    output_limit = _output_limit(model_cfg, cfg.max_tokens if override_max_tokens is None else override_max_tokens)
    context = dict(trace_context or {})
    context["call_id"] = context.get("call_id") or uuid4().hex
    prompt_args = {
        "system_prompt": system_prompt,
        "context_prefix": context_prefix,
        "override_max_tokens": output_limit,
    }

    def stream(prompt: InterviewPrompt, *, retry: bool = False) -> Iterable[tuple[str, str]]:
        _log.info(
            "window=%d estimated_input=%d output_reserved=%d retained_rounds=%d trimmed_rounds=%d retry=%s",
            prompt.context_window_tokens, prompt.estimated_input_tokens, output_limit,
            prompt.retained_rounds, len(rounds) - prompt.retained_rounds, retry,
        )
        return stream_fn(
            model_cfg, [{"role": "user", "content": prompt.user_text}],
            system_prompt=system_prompt, override_max_tokens=output_limit,
            abort_check=abort_check, mmpg_request=mmpg_request, config_snapshot=cfg,
            trace_context=context, protect_output_limit=True, **stream_kwargs,
        )

    prompt = prepare_interview_prompt(model_cfg, rounds, **prompt_args)
    if aborted():
        return
    emitted = False
    try:
        first_attempt = iter(stream(prompt))
        try:
            for chunk_type, chunk_text in first_attempt:
                if chunk_text:
                    emitted = True
                yield chunk_type, chunk_text
        finally:
            close = getattr(first_attempt, "close", None)
            if close is not None:
                close()
    except LLMContextExceeded as error:
        if aborted():
            return
        if emitted or prompt.retained_rounds == 1:
            raise

        provider_window = error.context_window_tokens
        if provider_window is not None and 0 < provider_window < prompt.context_window_tokens:
            retry_rounds = rounds[-(prompt.retained_rounds - 1):]
            retry_window = provider_window
        else:
            retry_rounds = rounds[-1:]
            retry_window = prompt.context_window_tokens
        prompt = prepare_interview_prompt(model_cfg, retry_rounds, context_window_tokens=retry_window, **prompt_args)
        if not aborted():
            yield from stream(prompt, retry=True)
