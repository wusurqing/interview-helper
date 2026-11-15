"""Run the knowledge-tag extraction requested from the ability analysis page."""

import json
from dataclasses import asdict, dataclass
from threading import Lock, Thread
from types import SimpleNamespace
from uuid import uuid4

from core.config import KnowledgeAnalysisSettings, ModelConfig, get_config, knowledge_model_available
from core.logger import get_logger
from core.prompt_settings import resolve_system_prompt
from services.llm import streaming
from services.storage import knowledge

DEFAULT_SYSTEM_PROMPT = (
    "从单条面试问答中提取技术关键词或知识点标签。"
    "有候选人实际口述时，优先提取口述涉及的知识点和能力点，"
    "助手参考答案只补充题目语境。没有口述时，根据问题和助手参考答案提取。"
)
OUTPUT_CONTRACT = '仅返回含 3–5 个互异非空字符串的 JSON 数组，如 ["Redis", "缓存穿透", "布隆过滤器"]。'

_log = get_logger("knowledge.tags")
_lock = Lock()


@dataclass
class AnalysisJob:
    job_id: str
    total: int
    status: str = "running"
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0


_job: AnalysisJob | None = None


def get_analysis_status() -> dict:
    with _lock:
        job = asdict(_job) if _job else None
    return {"pending_count": len(knowledge.pending_assist_ids()), "job": job}


def start_analysis() -> dict:
    global _job
    with _lock:
        record_ids = knowledge.pending_assist_ids()
        if _job and _job.status == "running":
            return {"pending_count": len(record_ids), "job": asdict(_job)}
        if not record_ids:
            return {"pending_count": 0, "job": None}

        cfg = get_config()
        settings = cfg.knowledge_analysis
        index = settings.model_index
        if index is None or index >= len(cfg.models) or not knowledge_model_available(cfg.models[index]):
            raise ValueError("请在分析设置中选择已启用且配置了 API Key 的模型")
        model = cfg.models[index].model_copy(deep=True)
        think_params = streaming._build_think_params(model, SimpleNamespace(
            think_effort=settings.think_effort,
            think_mode=settings.think_effort != "off",
            max_tokens=settings.max_tokens,
        ))
        if settings.think_effort != "off" and (not model.supports_think or not think_params):
            raise ValueError("所选模型和输出上限无法使用该推理强度，请调整分析设置")

        previous_job = _job
        _job = AnalysisJob(job_id=uuid4().hex, total=len(record_ids))
        worker = Thread(
            target=_run_analysis,
            args=(_job, record_ids, model, settings.model_copy(), think_params),
            name="knowledge-analysis",
            daemon=True,
        )
        try:
            worker.start()
        except RuntimeError:
            _job = previous_job
            raise
        return {"pending_count": len(record_ids), "job": asdict(_job)}


def _run_analysis(
    job: AnalysisJob, record_ids: list[int], model: ModelConfig,
    settings: KnowledgeAnalysisSettings, think_params: dict,
) -> None:
    overrides = {"knowledge_tags": settings.system_prompt} if settings.system_prompt else {}
    system_prompt = resolve_system_prompt(overrides, "knowledge_tags", DEFAULT_SYSTEM_PROMPT, OUTPUT_CONTRACT)
    for record_id in record_ids:
        outcome = "skipped"
        try:
            record = knowledge.get_pending_assist_record(record_id)
            if record is not None:
                tags = _extract_tags(record, model, settings, system_prompt, think_params, job.job_id)
                if knowledge.write_analysis_tags(record, tags):
                    outcome = "succeeded"
        except Exception as exc:
            _log.warning("Knowledge analysis failed for record %d (%s)", record_id, type(exc).__name__)
            outcome = "failed"
        with _lock:
            setattr(job, outcome, getattr(job, outcome) + 1)
        _log.info("KNOWLEDGE_TAGS_RESULT job_id=%s record_id=%d outcome=%s", job.job_id, record_id, outcome)
    with _lock:
        job.status = "completed"


def _extract_tags(
    record: knowledge.AnalysisRecord, model: ModelConfig, settings: KnowledgeAnalysisSettings,
    system_prompt: str, think_params: dict, job_id: str,
) -> list[str]:
    _log.info("KNOWLEDGE_TAGS_REQUEST job_id=%s record_id=%d", job_id, record.id)
    response = streaming.chat_completion_create(
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({
                "问题": (record.question or "")[:300],
                "候选人实际口述": (record.candidate_answer or "")[:800],
                "助手参考答案": (record.answer or "")[:800],
            }, ensure_ascii=False)},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        think_params=think_params,
    )
    raw = response.choices[0].message.content if response.choices else None
    tags = json.loads(raw or "")
    if not isinstance(tags, list) or not 3 <= len(tags) <= 5:
        raise ValueError("expected 3–5 knowledge tags")
    if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError("expected nonblank knowledge tags")
    cleaned = [tag.strip() for tag in tags]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("expected distinct knowledge tags")
    return cleaned
