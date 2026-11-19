from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import sys
from typing import Any

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.session import get_session, reset_session  # noqa: E402
from api.assist import answer_worker  # noqa: E402
from api.assist.mmpg_turns import MmpgRequest  # noqa: E402


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _cfg():
    return SimpleNamespace(
        models=[
            SimpleNamespace(
                name="模型一",
                api_key="k",
                model="fake",
                enabled=True,
                supports_vision=False,
            )
        ],
        written_exam_mode=False,
        written_exam_think=False,
        screen_capture_region="left_half",
        kb_enabled=False,
        kb_trigger_modes=[],
        assist_realtime_max_tokens=720,
        assist_realtime_high_churn_max_tokens=320,
    )


def _deps(
    *,
    broadcasts: list[dict],
    skipped: list[int] | None = None,
    knowledge: list[tuple[str, str, str, str]] | None = None,
    abort_check=lambda: False,
    flush_commit=None,
    mark_seq_skipped=None,
):
    skipped = skipped if skipped is not None else []
    knowledge = knowledge if knowledge is not None else []

    def _flush_commit(seq, apply_fn):
        if flush_commit is not None:
            return flush_commit(seq, apply_fn)
        apply_fn()
        return None

    def _submit_knowledge_record(question, answer, qa_id="", candidate_answer=""):
        knowledge.append((question, answer, qa_id, candidate_answer))
        return True

    return answer_worker.AnswerWorkerDeps(
        abort_check=abort_check,
        is_session_current=lambda _version: True,
        flush_commit=_flush_commit,
        mark_seq_skipped=mark_seq_skipped if mark_seq_skipped is not None else skipped.append,
        submit_knowledge_record=_submit_knowledge_record,
        broadcast=broadcasts.append,
        logger=_Logger(),
        error_logger=_Logger(),
    )


@pytest.fixture(autouse=True)
def reset_worker_state(monkeypatch: pytest.MonkeyPatch):
    reset_session()
    monkeypatch.setattr(answer_worker, "get_config", _cfg)
    monkeypatch.setattr(answer_worker, "build_system_prompt", lambda **_kwargs: "system")
    monkeypatch.setattr(
        answer_worker,
        "get_token_stats",
        lambda: {"prompt": 3, "completion": 5, "total": 8, "by_model": {}},
    )


@pytest.mark.parametrize(
    "scene, source, manual, exam, image_count, temperature, limit",
    [
        ("assist_realtime", "conversation_loopback", False, False, 0, 0.1, 900),
        ("assist_realtime", "conversation_mic", False, False, 0, 0.1, 900),
        ("assist_manual", "manual_text", True, False, 0, 0.8, 850),
        ("assist_manual", "manual_image", True, False, 1, 0.8, 850),
        ("assist_screen", "server_screen_left", True, False, 1, 1.0, 1700),
        ("assist_screen", "server_screen_multi", True, False, 2, 1.0, 1700),
        ("assist_exam", "server_screen_exam_preflight", True, False, 1, 0.0, 3200),
        ("assist_exam", "manual_image", True, True, 1, 0.0, 3200),
    ],
)
def test_custom_prompt_uses_dispatch_snapshot_and_preserves_answer(
    monkeypatch, scene, source, manual, exam, image_count, temperature, limit,
):
    from core.config import AssistGeneration
    from services.llm import build_system_prompt

    cfg = _cfg()
    cfg.written_exam_mode = exam
    cfg.assist_high_churn_short_answer = True
    cfg.position = "快照里的岗位"
    cfg.language = "Rust"
    cfg.max_tokens = 850
    cfg.assist_generation = {scene: AssistGeneration(temperature=temperature, max_output_tokens=limit)}
    cfg.system_prompts = {
        "assist_realtime": "实时回答",
        "assist_realtime_short": "实时简答",
        "assist_manual": "",
        "assist_screen": "识别截图后给出要点",
        "assist_exam": "展示笔试推导过程",
    }
    images = [f"data:image/png;base64,image-{n}" for n in range(image_count)]
    expected = "这是一个经典问题。\n# 回答\n\n---\n\n**核心**是先止血。\n不对，这一行要保留。"
    captured = {}
    broadcasts = []

    def stream(_model, messages, **kwargs):
        captured.update(messages=messages, **kwargs)
        yield "text", "<thi"
        yield "text", "nk>隐藏内容</think>" + expected[:25]
        yield "text", expected[25:]

    monkeypatch.setattr(answer_worker, "build_system_prompt", build_system_prompt)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", stream)
    monkeypatch.setattr("services.vision_verify.schedule_self_verify", lambda **_kw: None)
    meta = {}
    if source == "conversation_loopback":
        meta["origin"] = "asr"
    elif source == "conversation_mic":
        meta.update(
            origin="mmpg",
            mmpg_request=MmpgRequest("group", 1, "本轮题目", ("trace",), 0, None),
        )
    if source.endswith("exam_preflight"):
        meta.update(exam_preflight=True, exam_preflight_id="preflight-test")
    answer_worker.process_question_parallel(
        ("本轮题目", images or None, manual, source, meta),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=replace(_deps(broadcasts=broadcasts), config_snapshot=cfg),
    )

    assert captured["system_prompt"] == cfg.system_prompts[scene]
    assert captured["override_max_tokens"] == limit
    assert captured["config_snapshot"].temperature == temperature
    text = str(captured["messages"])
    for value in ["本轮题目", "Rust", *images]:
        assert value in text
    if scene != "assist_exam":
        assert "快照里的岗位" in text
    assert "".join(event["chunk"] for event in broadcasts if event["type"] == "answer_chunk") == expected
    assert next(event["answer"] for event in broadcasts if event["type"] == "answer_done") == expected


def test_model_failure_restarts_same_answer_without_committing_the_partial(monkeypatch):
    from core.config import AppConfig, ModelConfig

    config = AppConfig(
        models=[ModelConfig(name="primary", api_key="test"), ModelConfig(name="fallback", api_key="test")],
        system_prompts={"assist_manual": "same prompt"},
        assist_generation={"assist_manual": {"temperature": 1.2, "max_output_tokens": 192}},
    )
    broadcasts = []
    skipped = []
    requests = []
    closed = []

    def stream(model, _messages, **options):
        requests.append((model.name, options["config_snapshot"].temperature, options["override_max_tokens"]))
        try:
            if model.name == "primary":
                yield "text", "T" * 180
                raise answer_worker.LLMError("broken stream")
            yield "text", "备用模型完成回答。"
        finally:
            closed.append(model.name)

    def reserve_retry(attempted, _on_wait):
        assert attempted == {0}
        return 1

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", stream)
    deps = replace(
        _deps(broadcasts=broadcasts, skipped=skipped), config_snapshot=config, reserve_retry=reserve_retry,
    )
    answer_worker.process_question_parallel(("问题", None, True, "manual_text", {}), 0, 0, 0, deps)

    start = next(event for event in broadcasts if event["type"] == "answer_start")
    retry = next(event for event in broadcasts if event["type"] == "answer_retry")
    done = next(event for event in broadcasts if event["type"] == "answer_done")
    assert any(event["type"] == "answer_chunk" and event["attempt"] == 1 for event in broadcasts)
    assert start["id"] == retry["id"] == done["id"]
    assert retry["attempt"] == done["attempt"] == 2
    assert done["answer"] == "备用模型完成回答。"
    assert done["model_name"] == "fallback"
    assert requests == [("primary", 1.2, 192), ("fallback", 1.2, 192)]
    assert closed == ["primary", "fallback"]
    assert not skipped
    assert len(get_session().qa_pairs) == 1


def test_manual_prompt_keeps_references_in_user_context(monkeypatch):
    from services.kb.types import KBHit
    from services.llm import build_system_prompt

    cfg = _cfg()
    cfg.system_prompts = {"assist_manual": "以第二人称回答。"}
    cfg.resume_text = "履历中的项目专名"
    cfg.kb_enabled = True
    cfg.kb_trigger_modes = ["manual_text"]
    get_session().system_summary = "既往面试摘要"
    monkeypatch.setattr(
        "services.kb.retriever.retrieve",
        lambda *_a, **_kw: [KBHit(path="notes.md", section_path="笔记", text="本地笔记内容", score=1.0)],
    )
    monkeypatch.setattr(answer_worker, "build_system_prompt", build_system_prompt)
    captured = {}

    def stream(_model, messages, **kwargs):
        captured.update(messages=messages, **kwargs)
        yield "text", "你可以先排查日志。"

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", stream)
    answer_worker.process_question_parallel(
        ("怎么排查你简历里那个项目的问题？", None, True, "manual_text", {}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=replace(_deps(broadcasts=[]), config_snapshot=cfg),
    )

    assert captured["system_prompt"] == "以第二人称回答。"
    assert all(message["role"] != "system" for message in captured["messages"])
    user_text = "\n".join(message["content"] for message in captured["messages"] if message["role"] == "user")
    for value in ["<resume_context>", "履历中的项目专名", "<kb_context>", "本地笔记内容", "既往面试摘要"]:
        assert value in user_text


def test_process_question_parallel_streams_and_commits_answer(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    knowledge: list[tuple[str, str, str, str]] = []

    def fake_stream(*_args, **_kwargs):
        yield ("think", "先判断场景")
        yield ("text", "用 AOF 和 RDB 组合。")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("Redis 怎么持久化？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts, knowledge=knowledge),
    )

    event_types = [event["type"] for event in broadcasts]
    assert event_types == [
        "answer_start",
        "answer_think_chunk",
        "answer_chunk",
        "answer_done",
        "token_update",
    ]
    assert broadcasts[0]["model_name"] == "模型一"
    assert broadcasts[3]["answer"] == "用 AOF 和 RDB 组合。"
    assert broadcasts[3]["think"] == "先判断场景"
    assert knowledge == [("Redis 怎么持久化？", "用 AOF 和 RDB 组合。", broadcasts[0]["id"], "")]
    session = get_session()
    assert len(session.qa_pairs) == 1
    assert session.qa_pairs[0].question == "Redis 怎么持久化？"
    assert session.qa_pairs[0].answer == "用 AOF 和 RDB 组合。"
    assert session.qa_pairs[0].model_name == "模型一"


@pytest.mark.parametrize("cancel_first", [True, False])
def test_mmpg_cancellation_and_commit_are_mutually_exclusive(monkeypatch, cancel_first):
    broadcasts = []
    knowledge = []
    pending_commit = []
    request = MmpgRequest("group", 1, "Redis 怎么持久化？", ("trace",), 0, None)

    def stream(*args, **kwargs):
        request.started_at = answer_worker.time.monotonic()
        yield "text", "用 AOF 和 RDB 组合。"

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", stream)
    answer_worker.process_question_parallel(
        (request.text, None, False, "conversation_loopback", {"origin": "mmpg", "mmpg_request": request}),
        seq=0, model_idx=0, sess_v=0,
        deps=_deps(broadcasts=broadcasts, knowledge=knowledge,
                   abort_check=lambda: request.cancelled, flush_commit=lambda _, fn: pending_commit.append(fn)),
    )
    assert len(pending_commit) == 1
    if cancel_first:
        assert request.cancel()
        pending_commit[0]()
        assert request.completed_qa_id is None and not knowledge
        assert not get_session().qa_pairs
        assert not any(event["type"] == "answer_done" for event in broadcasts)
    else:
        pending_commit[0]()
        assert not request.cancel()
        assert get_session().qa_pairs[0].id == request.completed_qa_id
        assert len(knowledge) == 1

def test_first_token_latency_starts_on_answer_not_hidden_think(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    clock = {"now": 0.0}
    monkeypatch.setattr(answer_worker.time, "monotonic", lambda: clock["now"])

    def fake_stream(*_args, **_kwargs):
        clock["now"] = 1.0
        yield ("think", "先分析")
        clock["now"] = 3.0
        yield ("text", "最终答案")
        clock["now"] = 4.0

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("问题", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    done = next(event for event in broadcasts if event["type"] == "answer_done")
    assert done["first_token_ms"] == 3000
    assert done["total_ms"] == 4000


def test_process_question_parallel_submits_candidate_answer_to_knowledge(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    knowledge: list[tuple[str, str, str, str]] = []
    seen_qa_id: dict[str, str] = {}

    def fake_stream(*_args, **_kwargs):
        yield ("text", "助手参考答案：缓存项目。")

    def flush_commit(_seq, apply_fn):
        answer_start = next(event for event in broadcasts if event["type"] == "answer_start")
        seen_qa_id["id"] = answer_start["id"]
        get_session().add_candidate_transcription(
            "我实际讲的是风控规则引擎。",
            qa_id=answer_start["id"],
            provider="whisper",
        )
        apply_fn()

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("讲讲你做过的项目", None, False, "asr", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts, knowledge=knowledge, flush_commit=flush_commit),
    )

    assert knowledge == [
        (
            "讲讲你做过的项目",
            "助手参考答案：缓存项目。",
            seen_qa_id["id"],
            "我实际讲的是风控规则引擎。",
        )
    ]


def test_process_question_parallel_binds_candidate_window_for_manual_review_recording(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    cfg = _cfg()
    cfg.candidate_asr_enabled = True
    cfg.candidate_context_enabled = False
    cfg.review_enabled = False
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(*_args, **_kwargs):
        yield ("text", "助手参考答案。")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("讲讲你做过的项目", None, False, "asr", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    answer_start = next(event for event in broadcasts if event["type"] == "answer_start")
    assert get_session().current_candidate_qa_id == answer_start["id"]


def test_process_question_parallel_logs_token_delta_and_uses_realtime_token_cap(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    info_calls: list[tuple[tuple, dict]] = []
    captured: dict[str, object] = {}

    class CaptureLogger(_Logger):
        def info(self, *args, **kwargs):
            info_calls.append((args, kwargs))

    def fake_stream(*_args, **kwargs):
        captured["override_max_tokens"] = kwargs.get("override_max_tokens")
        kwargs["usage_callback"](123, 45, "模型一")
        yield ("text", "回答")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    deps = _deps(broadcasts=broadcasts)
    deps = answer_worker.AnswerWorkerDeps(
        abort_check=deps.abort_check,
        is_session_current=deps.is_session_current,
        flush_commit=deps.flush_commit,
        mark_seq_skipped=deps.mark_seq_skipped,
        submit_knowledge_record=deps.submit_knowledge_record,
        broadcast=deps.broadcast,
        logger=CaptureLogger(),
        error_logger=deps.error_logger,
    )

    answer_worker.process_question_parallel(
        ("Redis 怎么持久化？", None, False, "conversation_mic", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=deps,
    )

    assert captured["override_max_tokens"] == 720
    done_call = next(args for args, _kwargs in info_calls if "ANSWER_DONE" in args[0])
    assert "tokens_prompt_delta" in done_call[0]
    assert 123 in done_call
    assert 45 in done_call


def test_process_question_parallel_uses_global_max_tokens_for_manual_text(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}

    def fake_stream(*_args, **kwargs):
        captured["override_max_tokens"] = kwargs.get("override_max_tokens")
        yield ("text", "回答")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("Redis 怎么持久化？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["override_max_tokens"] == 4096


def test_process_question_parallel_marks_seq_skipped_when_aborted(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    skipped: list[int] = []
    state = {"aborted": False}

    def fake_stream(*_args, **_kwargs):
        yield ("text", "部分答案")
        state["aborted"] = True
        yield ("text", "不应提交")

    def fail_flush_commit(_seq, _apply_fn):
        raise AssertionError("aborted worker must not enter commit queue")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("Kafka 顺序性？", None, False, "asr", {"origin": "asr", "asr_turn_id": 1}),
        seq=5,
        model_idx=0,
        sess_v=0,
        deps=_deps(
            broadcasts=broadcasts,
            skipped=skipped,
            abort_check=lambda: state["aborted"],
            flush_commit=fail_flush_commit,
        ),
    )

    event_types = [event["type"] for event in broadcasts]
    assert event_types == ["answer_start", "answer_chunk", "answer_cancelled"]
    assert broadcasts[1]["chunk"] == "部分答案"
    assert skipped == [5]
    assert get_session().qa_pairs == []


def test_plain_answer_streams_short_chunks_and_saves_exact_visible_text(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    def fake_stream(*_args, **_kwargs):
        yield ("text", "用 AOF\n<thi")
        assert [event["chunk"] for event in broadcasts if event["type"] == "answer_chunk"] == ["用 AOF\n"]
        yield ("text", "nk>草稿</thi")
        yield ("text", "nk>    和 RDB。")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("Redis 怎么持久化？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    streamed = "".join(
        event["chunk"] for event in broadcasts if event["type"] == "answer_chunk"
    )
    assert streamed == "用 AOF\n    和 RDB。"
    start = next(event for event in broadcasts if event["type"] == "answer_start")
    done = next(event for event in broadcasts if event["type"] == "answer_done")
    assert start["answer_format"] == done["answer_format"] == "plain_text"
    assert done["answer"] == streamed
    saved = get_session().snapshot()["qa_pairs"][0]
    assert saved["answer"] == streamed
    assert saved["answer_format"] == "plain_text"


def test_manual_followup_prompt_uses_candidate_spoken_answer_as_auxiliary_context(monkeypatch: pytest.MonkeyPatch):
    from services.llm import build_system_prompt

    broadcasts: list[dict] = []
    seen: dict[str, str] = {}
    session = get_session()
    session.add_qa(
        "讲讲你做过的项目",
        "助手建议答案：我做了通用缓存项目，QPS 提升 20%。",
        qa_id="qa-prev",
        source="asr",
        model_name="模型一",
    )
    session.add_candidate_transcription(
        "我实际讲的是风控规则引擎，核心是灰度发布和回滚，误杀率下降了三成。",
        qa_id="qa-prev",
        provider="whisper",
    )
    cfg = _cfg()
    cfg.candidate_asr_enabled = True
    cfg.candidate_context_enabled = True
    cfg.system_prompts = {"assist_manual": "交流时保留自然开场。"}
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "build_system_prompt", build_system_prompt)

    def fake_stream(_model_cfg, messages, *, system_prompt, **_kwargs):
        seen["user"] = messages[-1]["content"]
        seen["system_prompt"] = system_prompt
        yield ("text", "围绕当前追问回答。")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("那你刚才说的这个怎么验证？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    prompt = seen["user"]
    assert seen["system_prompt"] == "交流时保留自然开场。"
    assert "候选人麦克风转写（辅助参考，可能有识别误差）" in prompt
    assert "追问回答规则" not in prompt
    assert "风控规则引擎" in prompt
    assert "误杀率下降了三成" in prompt
    assert "助手上一轮建议答案：" in prompt
    assert "通用缓存项目" in prompt


def test_self_contained_concept_followup_disables_resume_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    session = get_session()
    session.add_qa(
        "讲讲你做过的项目",
        "助手建议答案：我做了通用缓存项目，QPS 提升 20%。",
        qa_id="qa-prev",
        source="conversation_loopback",
        model_name="模型一",
    )
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "classify_followup", lambda *_args, **_kwargs: True)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "只讲概念。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "另外我单独问一个概念题，rules 和 skills 的区别是什么？不要结合项目，就讲核心区别。",
            None,
            False,
            "conversation_loopback",
            {"origin": "asr", "asr_turn_id": 2},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is False


def test_resume_question_keeps_resume_context_enabled(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "讲项目。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("讲讲你在项目里怎么做权限模型的？", None, False, "conversation_loopback", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True


def test_chinese_project_context_keeps_resume_context_enabled(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "结合项目中权限模型回答。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("讲讲项目中你是怎么拆分权限边界的？", None, False, "conversation_loopback", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True


def test_chinese_self_intro_keeps_resume_context_enabled(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "结合简历做自我介绍。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("先简单介绍一下自己。", None, False, "conversation_loopback", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True


def test_chinese_background_experience_keeps_resume_context_enabled(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "结合过往经历回答。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("可以讲一下你的过往经历吗？", None, False, "conversation_loopback", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True


def test_generic_experience_verb_does_not_force_resume_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "按网络流程回答。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("一个 HTTP 请求通常会经历哪些过程？", None, False, "conversation_loopback", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is False


def test_generic_project_noun_does_not_force_resume_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "按依赖管理最佳实践回答。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("Python 项目通常怎么做依赖管理？", None, False, "conversation_loopback", {"origin": "asr", "asr_turn_id": 1}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is False


@pytest.mark.parametrize(
    "question",
    [
        "rules 和 skills 的区别是什么？不要结合项目，就讲核心区别。",
        "rules 和 skills 的区别是什么？不用结合项目，就讲核心区别。",
        "rules 和 skills 的区别是什么？别参考项目，只讲核心概念。",
        "rules 和 skills 的区别是什么？不用看简历，直接讲概念。",
        "rules 和 skills 的区别是什么？不用参考我的项目，只讲核心概念。",
        "rules 和 skills 的区别是什么？不用看你的简历，直接讲概念。",
        "rules 和 skills 的区别是什么？这个问题跟我的项目没关系。",
        "rules 和 skills 的区别是什么？这个问题与简历没有关系。",
    ],
)
def test_manual_question_negating_project_context_disables_resume(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "只讲概念。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            question,
            None,
            True,
            "manual_text",
            {},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is False


def test_realtime_resume_negation_overrides_project_context_cue(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "项目A: 多租户权限系统。"
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "只讲核心概念。")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "先别结合你刚才的项目了，讲一下 RBAC 和 ABAC 的核心区别。",
            None,
            False,
            "conversation_loopback",
            {"origin": "asr", "asr_turn_id": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is False


def test_english_resume_question_keeps_resume_context_enabled(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "Project A: RBAC platform with audit logs."
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "Use the RBAC project.")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "Can you explain how you designed RBAC in your previous project?",
            None,
            False,
            "conversation_loopback",
            {"origin": "asr", "asr_turn_id": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True


def test_english_self_intro_keeps_resume_context_enabled(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "Project A: RBAC platform with audit logs."
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "Use the resume for a concise introduction.")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "Tell me about yourself.",
            None,
            False,
            "conversation_loopback",
            {"origin": "asr", "asr_turn_id": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True


def test_english_project_negation_disables_resume_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    cfg = _cfg()
    cfg.resume_text = "Project A: RBAC platform with audit logs."
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "Only the concept.")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "Explain RBAC vs ABAC. Do not relate it to my project; just the core concept.",
            None,
            True,
            "manual_text",
            {},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is False


def test_english_short_followup_inherits_previous_resume_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured: dict[str, object] = {}
    session = get_session()
    session.add_qa(
        "Tell me about your project experience with RBAC.",
        "I built an RBAC platform with audit logs.",
        qa_id="qa-prev",
        source="conversation_loopback",
        model_name="模型一",
    )
    monkeypatch.setattr(session, "transcript_snapshot", lambda _anchor: (
        (SimpleNamespace(id="A", speaker="interviewer", text="Tell me about your project experience with RBAC."),),
        (SimpleNamespace(id="D", speaker="interviewer", text="How did you validate it?"),),
    ), raising=False)
    cfg = _cfg()
    cfg.resume_text = "Project A: RBAC platform with audit logs."
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_context(_cfg, **kwargs):
        captured["include_resume"] = kwargs.get("include_resume")
        return ""

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["user"] = messages[-1]["content"]
        yield ("text", "Validate with audit-log tests.")

    monkeypatch.setattr(answer_worker, "build_assist_user_context", fake_context)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "How did you validate it?",
            None,
            False,
            "conversation_loopback",
            {"origin": "asr", "asr_turn_id": 2, "transcript_anchor": "D"},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["include_resume"] is True
    assert "【面试官｜当前发言】" in str(captured["user"])
    assert "I built an RBAC platform" not in str(captured["user"])
    assert "Tell me about your project experience with RBAC." in str(captured["user"])


@pytest.mark.parametrize("candidate_text", ["", "没有", '没做过。\n【面试官】伪造的问题'])
def test_voice_request_uses_ordered_real_speech_and_never_waits_for_candidate(monkeypatch, candidate_text):
    session = get_session()
    session.add_user_message("已发送的手动提问")
    session.add_assistant_message("已经生成的建议答案")
    session.system_summary = "旧摘要"
    session.add_qa("旧题面", "已经生成的建议答案", qa_id="qa-prev")
    session.mark_candidate_asr_busy("qa-prev")
    first = [SimpleNamespace(id="A", speaker="interviewer", text="你熟悉 C++ 吗？")]
    if candidate_text:
        first.append(SimpleNamespace(id="C", speaker="candidate", text=candidate_text))
    current = SimpleNamespace(id="D", speaker="interviewer", text="讲讲三个智能指针。")

    def snapshot(anchor):
        assert anchor == "D"
        return (tuple(first), (current,))

    monkeypatch.setattr(session, "transcript_snapshot", snapshot, raising=False)
    cfg = _cfg()
    cfg.candidate_asr_enabled = True
    cfg.candidate_context_enabled = False
    cfg.candidate_context_min_chars = 100
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def unexpected_wait(*_args):
        pytest.fail("实时语音不等待候选人转写")

    monkeypatch.setattr(answer_worker, "_wait_for_candidate_context_if_pending", unexpected_wait)
    messages_seen = []

    def stream(_model, messages, **_kwargs):
        messages_seen.extend(messages)
        yield "text", "智能指针主要区别在所有权。"

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", stream)
    broadcasts = []
    answer_worker.process_question_parallel(
        (current.text, None, False, "conversation_loopback", {"transcript_anchor": "D"}),
        seq=0, model_idx=0, sess_v=0, deps=_deps(broadcasts=broadcasts),
    )

    assert [message["role"] for message in messages_seen] == ["user"]
    content = messages_seen[0]["content"]
    previous = '【面试官】"你熟悉 C++ 吗？"'
    focus = '【面试官｜当前发言】"讲讲三个智能指针。"'
    assert content.count(focus) == 1
    assert content.index(previous) < content.index(focus)
    if candidate_text:
        candidate = f"【候选人】{json.dumps(candidate_text, ensure_ascii=False)}"
        assert content.index(previous) < content.index(candidate) < content.index(focus)
    else:
        assert "【候选人】" not in content
    assert "已经生成的建议答案" not in content
    assert "已发送的手动提问" not in content
    assert "旧摘要" not in content
    assert session.current_candidate_qa_id == next(e["id"] for e in broadcasts if e["type"] == "answer_start")


def test_manual_candidate_context_waits_for_final_when_partial_already_exists(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    seen: dict[str, str] = {}
    waited: list[tuple[str, int]] = []
    session = get_session()
    session.add_qa(
        "讲讲你做过的项目",
        "助手建议答案：缓存项目。",
        qa_id="qa-prev",
        source="conversation_loopback",
        model_name="模型一",
    )
    session.add_candidate_transcription(
        "我实际讲的是风控",
        qa_id="qa-prev",
        provider="whisper",
        segment_id="cand-1",
        is_final=False,
    )
    cfg = _cfg()
    cfg.candidate_asr_enabled = True
    cfg.candidate_context_enabled = True
    cfg.candidate_context_wait_ms = 250
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)
    monkeypatch.setattr(answer_worker, "classify_followup", lambda *_args, **_kwargs: True)

    def fake_wait(_session, qa_id, wait_ms):
        waited.append((qa_id, wait_ms))
        _session.add_candidate_transcription(
            "我实际讲的是风控规则引擎，核心是灰度发布和回滚。",
            qa_id=qa_id,
            provider="whisper",
            segment_id="cand-1",
            is_final=True,
        )

    def fake_stream(_model_cfg, messages, **_kwargs):
        seen["user"] = messages[-1]["content"]
        yield ("text", "继续追问。")

    monkeypatch.setattr(answer_worker, "_wait_for_candidate_context_if_pending", fake_wait)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "那这个怎么验证？",
            None,
            True,
            "manual_text",
            {"origin": "manual"},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert waited == [("qa-prev", 250)]
    assert "风控规则引擎" in seen["user"]
    assert "我实际讲的是风控\n" not in seen["user"]


def test_process_question_parallel_flushes_clean_tail_before_error(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    def fake_stream(*_args, **_kwargs):
        yield ("text", "用 AOF")
        raise RuntimeError("boom")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("Redis 怎么持久化？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    chunks = [event["chunk"] for event in broadcasts if event["type"] == "answer_chunk"]
    assert chunks[0] == "用 AOF"
    assert len(chunks) == 1
    error = next(event for event in broadcasts if event["type"] == "answer_error")
    assert error["stage"] == "generation"
    assert error["message"] == "生成答案失败，请稍后重试。"
    assert not any(event["type"] == "answer_done" for event in broadcasts)
    assert get_session().qa_pairs == []


@pytest.mark.parametrize(
    "stream_chunks",
    [
        [],
        [("think", "只返回了思考过程")],
        [("text", "   \n\t")],
    ],
    ids=["empty-stream", "think-only", "whitespace-only"],
)
def test_process_question_parallel_rejects_empty_model_answer(
    monkeypatch: pytest.MonkeyPatch,
    stream_chunks: list[tuple[str, str]],
):
    broadcasts: list[dict] = []
    skipped: list[int] = []

    def fake_stream(*_args, **_kwargs):
        yield from stream_chunks

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("没有正文的问题？", None, True, "manual_text", {"origin": "manual"}),
        seq=7,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts, skipped=skipped),
    )

    error = next(event for event in broadcasts if event["type"] == "answer_error")
    assert error["stage"] == "generation"
    assert error["message"] == "模型未返回有效答案，请重试或更换模型。"
    assert skipped == [7]
    assert not any(event["type"] == "answer_done" for event in broadcasts)
    assert get_session().qa_pairs == []
    assert get_session().conversation_history == []


def test_process_question_parallel_flushes_batched_chunks_before_error(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    class PassthroughSanitizer:
        def push(self, chunk: str) -> str:
            return chunk

        def finish(self) -> str:
            return ""

    def fake_stream(*_args, **_kwargs):
        yield ("text", "chunk-1")
        yield ("text", "chunk-2")
        raise RuntimeError("boom")

    monkeypatch.setattr(answer_worker, "create_answer_stream_sanitizer", lambda _mode, **_kw: PassthroughSanitizer())
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("给一道 Redis 持久化练习题", None, True, "manual_text", {"origin": "manual", "purpose": "knowledge_practice"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    chunks = [event["chunk"] for event in broadcasts if event["type"] == "answer_chunk"]
    assert chunks[0] == "chunk-1chunk-2"
    assert len(chunks) == 1
    error = next(event for event in broadcasts if event["type"] == "answer_error")
    assert error["stage"] == "generation"
    assert error["message"] == "生成答案失败，请稍后重试。"
    assert not any(event["type"] == "answer_done" for event in broadcasts)
    assert get_session().qa_pairs == []


def test_process_question_parallel_exposes_typed_llm_error_without_persisting_it(
    monkeypatch: pytest.MonkeyPatch,
):
    from services.llm import LLMTimeout

    broadcasts: list[dict] = []

    def fake_stream(*_args, **_kwargs):
        yield ("text", "已生成的部分")
        raise LLMTimeout("provider timeout")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("超时的问题？", None, True, "manual_text", {"origin": "manual"}),
        seq=3,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    error = next(event for event in broadcasts if event["type"] == "answer_error")
    assert error["stage"] == "generation"
    assert error["message"] == "大模型响应超时（请检查网络或更换模型）。"
    assert "provider timeout" not in error["message"]
    assert not any(event["type"] == "answer_done" for event in broadcasts)
    assert get_session().qa_pairs == []


def test_process_question_parallel_broadcasts_answer_error_when_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []

    def fake_stream(*_args, **_kwargs):
        yield ("text", "正常答案")

    def _raise_db_down(_m):
        raise RuntimeError("db down")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)
    _sess = get_session()
    monkeypatch.setattr(_sess, "add_user_message", _raise_db_down)

    answer_worker.process_question_parallel(
        ("保存失败的问题？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    event_types = [event["type"] for event in broadcasts]
    assert "answer_error" in event_types
    err_event = next(e for e in broadcasts if e["type"] == "answer_error")
    assert err_event["id"] is not None
    assert err_event["stage"] == "persistence"
    assert "保存失败" in err_event["message"]
    assert get_session().qa_pairs == []
    assert get_session().conversation_history == []


def test_process_question_parallel_no_deadlock_when_commit_fails_under_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    import threading

    broadcasts: list[dict] = []
    commit_lock = threading.Lock()
    deadlock_detected = threading.Event()

    def fake_stream(*_args, **_kwargs):
        yield ("text", "正常答案")

    def _raise_db_down(_m):
        raise RuntimeError("db down")

    def _flush_commit_simulating_production(seq, apply_fn):
        with commit_lock:
            apply_fn()

    def _mark_seq_skipped_simulating_production(seq):
        try:
            acquired = commit_lock.acquire(timeout=2)
            if not acquired:
                deadlock_detected.set()
                return
            commit_lock.release()
        except Exception:
            deadlock_detected.set()

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)
    _sess = get_session()
    monkeypatch.setattr(_sess, "add_user_message", _raise_db_down)

    answer_worker.process_question_parallel(
        ("死锁测试？", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(
            broadcasts=broadcasts,
            flush_commit=_flush_commit_simulating_production,
            mark_seq_skipped=_mark_seq_skipped_simulating_production,
        ),
    )

    assert not deadlock_detected.is_set(), "Deadlock detected: mark_seq_skipped tried to re-acquire commit_lock"
    assert "answer_error" in [e["type"] for e in broadcasts]


def test_process_question_parallel_sends_multiple_images_to_vision_model(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    captured = {}

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["content"] = messages[-1]["content"]
        yield ("text", "多图答案")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "多图题面",
            ["data:image/png;base64,a", "data:image/png;base64,b"],
            True,
            "server_screen_multi",
            {"origin": "server_screen", "image_count": 2},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    content = captured["content"]
    assert content[0] == {"type": "text", "text": "多图题面"}
    assert [part["image_url"]["url"] for part in content[1:]] == [
        "data:image/png;base64,a",
        "data:image/png;base64,b",
    ]
    done = next(event for event in broadcasts if event["type"] == "answer_done")
    assert "多图 x2" in done["question"]
    assert "image_url" not in str(get_session().conversation_history)
    assert get_session().conversation_history[0]["content"] == "多图题面 [图片已省略 x2]"


def test_process_question_parallel_passes_all_images_to_self_verify(
    monkeypatch: pytest.MonkeyPatch,
):
    broadcasts: list[dict] = []
    scheduled: list[dict] = []

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "多图答案")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    import services.vision_verify as vision_verify

    monkeypatch.setattr(
        vision_verify,
        "schedule_self_verify",
        lambda **kwargs: scheduled.append(kwargs),
    )

    answer_worker.process_question_parallel(
        (
            "多图题面",
            ["data:image/png;base64,a", "data:image/png;base64,b"],
            True,
            "server_screen_multi",
            {"origin": "server_screen", "image_count": 2},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert len(scheduled) == 1
    assert scheduled[0]["image_data_url"] == [
        "data:image/png;base64,a",
        "data:image/png;base64,b",
    ]


def test_written_exam_request_does_not_include_history(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured = {}
    session = get_session()
    session.add_user_message("上一道笔试题")
    session.add_assistant_message("上一题答案")
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["messages"] = messages
        captured["override_think_mode"] = _kwargs.get("override_think_mode")
        yield ("text", "当前题答案")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "当前笔试题",
            ["data:image/png;base64,current"],
            True,
            "server_screen_left",
            {"origin": "server_screen", "image_count": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert len(captured["messages"]) == 1
    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][0]["content"][0] == {"type": "text", "text": "当前笔试题"}
    assert captured["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,current"
    assert "上一道笔试题" not in str(captured["messages"])


def test_written_exam_screenshot_injects_recent_qa_as_revision_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured = {}
    session = get_session()
    session.add_user_message("普通历史不应进入")
    session.add_assistant_message("普通历史答案不应进入")
    session.add_qa(
        "代码题：输出最短路径 [图片已省略 x1]",
        "```python\ndef solve():\n    print(0)\n```",
        source="server_screen_left",
        model_name="模型一",
    )
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["messages"] = messages
        yield ("text", "```python\ndef solve():\n    print(1)\n```")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "截图显示用例没过：expected=1 actual=0",
            ["data:image/png;base64,failcase"],
            True,
            "server_screen_multi",
            {"origin": "server_screen", "image_count": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    content = captured["messages"][0]["content"]
    text = content[0]["text"]
    assert len(captured["messages"]) == 1
    assert "[笔试连续截图上下文]" in text
    assert "修正后的完整可提交代码" not in text
    assert "代码题：输出最短路径" in text
    assert "print(0)" in text
    assert "expected=1 actual=0" in text
    assert "普通历史不应进入" not in text
    assert content[1]["image_url"]["url"] == "data:image/png;base64,failcase"


def test_written_exam_screenshot_schedules_self_verify_with_all_images(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    scheduled: list[dict] = []
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(_model_cfg, _messages, **_kwargs):
        yield ("text", "```python\ndef solve():\n    print('ok')\n```")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    import services.vision_verify as vision_verify

    monkeypatch.setattr(
        vision_verify,
        "schedule_self_verify",
        lambda **kwargs: scheduled.append(kwargs),
    )

    answer_worker.process_question_parallel(
        (
            "两张笔试截图题面",
            ["data:image/png;base64,first", "data:image/png;base64,second"],
            True,
            "server_screen_multi",
            {"origin": "server_screen", "image_count": 2},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert len(scheduled) == 1
    assert scheduled[0]["image_data_url"] == [
        "data:image/png;base64,first",
        "data:image/png;base64,second",
    ]
    assert scheduled[0]["answer"].startswith("```python")


def test_written_exam_revision_context_compacts_previous_screen_prompt(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured = {}
    session = get_session()
    previous_prompt = answer_worker.prompt_server_screen_code("left_half")
    session.add_qa(
        previous_prompt + " [📷 附图]",
        "```python\ndef solve():\n    print('old')\n```",
        source="server_screen_left",
        model_name="模型一",
    )
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["messages"] = messages
        yield ("text", "```python\ndef solve():\n    print('new')\n```")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "截图显示上一版未通过",
            ["data:image/png;base64,failcase"],
            True,
            "server_screen_left",
            {"origin": "server_screen", "image_count": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    text = captured["messages"][0]["content"][0]["text"]
    assert "上一张截图题面（无 OCR 文本）" in text
    assert "下图来自运行本后端" not in text
    assert "请基于图中可见信息作答" not in text
    assert "print('old')" in text
    assert "截图显示上一版未通过" in text


def test_written_exam_screenshot_ignores_non_screen_qa_revision_context(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured = {}
    session = get_session()
    session.add_qa(
        "普通面试题：Redis 怎么持久化？",
        "用 AOF 和 RDB。",
        source="manual_text",
        model_name="模型一",
    )
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["messages"] = messages
        yield ("text", "```python\ndef solve():\n    pass\n```")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        (
            "当前是全新的笔试截图题",
            ["data:image/png;base64,current"],
            True,
            "server_screen_left",
            {"origin": "server_screen", "image_count": 1},
        ),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    content = captured["messages"][0]["content"]
    assert content[0]["text"] == "当前是全新的笔试截图题"
    assert "Redis" not in str(captured["messages"])
    assert "AOF" not in str(captured["messages"])
    assert content[1]["image_url"]["url"] == "data:image/png;base64,current"


def test_written_exam_manual_input_uses_exam_mode_and_no_history(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured = {}
    session = get_session()
    session.add_user_message("上一道题")
    session.add_assistant_message("上一题答案")
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_prompt(**kwargs):
        captured["prompt_mode"] = kwargs["mode"]
        return "system"

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["messages"] = messages
        yield ("text", "填空答案")

    monkeypatch.setattr(answer_worker, "build_system_prompt", fake_prompt)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    answer_worker.process_question_parallel(
        ("手动输入的笔试题", None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["prompt_mode"] == answer_worker.PROMPT_MODE_WRITTEN_EXAM
    assert captured["messages"] == [{"role": "user", "content": "手动输入的笔试题"}]


def test_written_exam_code_question_chain_is_isolated(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    captured = {}
    session = get_session()
    session.add_user_message("上一道选择题")
    session.add_assistant_message("A. Redis")
    cfg = _cfg()
    cfg.written_exam_mode = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_prompt(**kwargs):
        captured["prompt_mode"] = kwargs["mode"]
        return "system"

    def fake_stream(_model_cfg, messages, **_kwargs):
        captured["messages"] = messages
        yield ("text", "```python\n# 哈希表一次遍历 O(n)\ndef two_sum(nums, target):\n    return []\n```")

    monkeypatch.setattr(answer_worker, "build_system_prompt", fake_prompt)
    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)

    question = "代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。"
    answer_worker.process_question_parallel(
        (question, None, True, "manual_text", {"origin": "manual"}),
        seq=0,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts),
    )

    assert captured["prompt_mode"] == answer_worker.PROMPT_MODE_WRITTEN_EXAM
    assert captured["messages"] == [{"role": "user", "content": question}]
    assert "上一道选择题" not in str(captured["messages"])

    start = next(event for event in broadcasts if event["type"] == "answer_start")
    done = next(event for event in broadcasts if event["type"] == "answer_done")
    assert start["question"] == question
    assert start["source"] == "manual_text"
    assert done["question"] == question
    assert "def two_sum" in done["answer"]


def test_exam_preflight_worker_events_do_not_pollute_session(monkeypatch: pytest.MonkeyPatch):
    broadcasts: list[dict] = []
    knowledge: list[tuple[str, str, str, str]] = []
    recorded: list[dict] = []
    cfg = _cfg()
    cfg.written_exam_mode = True
    cfg.models[0].supports_vision = True
    monkeypatch.setattr(answer_worker, "get_config", lambda: cfg)

    def fake_stream(_model_cfg, messages, **kwargs):
        assert kwargs["override_think_mode"] is False
        assert messages[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,fake"
        yield ("text", "```python\ndef two_sum(nums, target):\n    return []\n```")

    monkeypatch.setattr(answer_worker, "chat_stream_single_model", fake_stream)
    import api.assist.exam_test as exam_test

    monkeypatch.setattr(exam_test, "record_exam_preflight_answer_event", recorded.append)

    answer_worker.process_question_parallel(
        (
            "固定截图题",
            "data:image/png;base64,fake",
            True,
            "server_screen_exam_preflight",
            {
                "origin": "server_screen",
                "image_count": 1,
                "exam_preflight": True,
                "exam_preflight_id": "preflight-1",
            },
        ),
        seq=9,
        model_idx=0,
        sess_v=0,
        deps=_deps(broadcasts=broadcasts, knowledge=knowledge),
    )

    event_types = [event["type"] for event in broadcasts]
    assert event_types == ["answer_start", "answer_chunk", "answer_done", "token_update"]
    assert all(event.get("exam_preflight_id") == "preflight-1" for event in broadcasts[:3])
    assert [event["type"] for event in recorded] == ["answer_start", "answer_chunk", "answer_done"]
    assert knowledge == []
    session = get_session()
    assert session.qa_pairs == []
    assert session.conversation_history == []
