from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool

from core.config import (
    ConfigCorruptError, KnowledgeAnalysisSettings, ModelListChanged, get_config, get_provider_layout,
    knowledge_model_available, model_health_fingerprint, update_knowledge_analysis_settings,
)
from services import knowledge_analysis
from services.storage.knowledge import get_summary, get_history, reset_all

router = APIRouter()
ABILITY_SESSION_TYPES = ("assist",)


class KnowledgeSettingsUpdate(KnowledgeAnalysisSettings):
    model_fingerprint: str | None = None


def _settings_payload(cfg) -> dict:
    providers, layout = get_provider_layout(cfg)
    names = {provider.id: provider.name for provider in providers}
    provider_names = {model.id: names.get(model.provider_id) for model in layout}
    return {
        "settings": cfg.knowledge_analysis.model_dump(),
        "default_system_prompt": knowledge_analysis.DEFAULT_SYSTEM_PROMPT,
        "output_contract": knowledge_analysis.OUTPUT_CONTRACT,
        "models": [
            {
                "index": index,
                "id": model.id,
                "name": model.name,
                "provider_name": provider_names.get(model.id),
                "model": model.model,
                "fingerprint": model_health_fingerprint(model),
                "available": knowledge_model_available(model),
                "supports_think": model.supports_think,
                "supports_vision": model.supports_vision,
            }
            for index, model in enumerate(cfg.models)
            if knowledge_model_available(model)
        ],
    }


@router.get("/knowledge/settings")
def api_knowledge_settings():
    return _settings_payload(get_config())


@router.put("/knowledge/settings")
def api_update_knowledge_settings(body: KnowledgeSettingsUpdate):
    settings = KnowledgeAnalysisSettings(**body.model_dump(exclude={"model_fingerprint"}, exclude_unset=True))
    settings.system_prompt = (settings.system_prompt or "").strip() or None
    try:
        return _settings_payload(update_knowledge_analysis_settings(settings, body.model_fingerprint))
    except ConfigCorruptError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ModelListChanged as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError:
        raise HTTPException(500, "无法保存分析设置，请检查配置文件目录的写入权限") from None


@router.get("/knowledge/analysis")
def api_knowledge_analysis_status():
    return knowledge_analysis.get_analysis_status()


@router.post("/knowledge/analysis")
def api_start_knowledge_analysis(response: Response):
    try:
        result = knowledge_analysis.start_analysis()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError:
        raise HTTPException(503, "暂时无法启动分析，请重试") from None
    if result["job"]:
        response.status_code = 202
    return result


@router.get("/knowledge/summary")
async def api_knowledge_summary():
    return {"tags": await run_in_threadpool(get_summary, ABILITY_SESSION_TYPES)}


@router.get("/knowledge/history")
async def api_knowledge_history(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
):
    return await run_in_threadpool(get_history, page, page_size, ABILITY_SESSION_TYPES)


@router.delete("/knowledge/reset")
async def api_knowledge_reset():
    await run_in_threadpool(reset_all)
    return {"ok": True}
