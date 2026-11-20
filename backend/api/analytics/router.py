from fastapi import APIRouter, HTTPException, Response

from core.config import (
    KnowledgeAnalysisSettings, ModelListChanged, get_config, knowledge_model_available,
    model_health_fingerprint, update_knowledge_analysis_settings,
)
from services import knowledge_analysis
from services.storage.knowledge import get_summary, get_history, reset_all

router = APIRouter()
ABILITY_SESSION_TYPES = ("assist",)


class KnowledgeSettingsUpdate(KnowledgeAnalysisSettings):
    model_fingerprint: str | None = None


def _settings_payload(cfg) -> dict:
    return {
        "settings": cfg.knowledge_analysis.model_dump(),
        "default_system_prompt": knowledge_analysis.DEFAULT_SYSTEM_PROMPT,
        "output_contract": knowledge_analysis.OUTPUT_CONTRACT,
        "models": [
            {
                "index": index,
                "name": model.name,
                "model": model.model,
                "fingerprint": model_health_fingerprint(model),
                "available": knowledge_model_available(model),
                "supports_think": model.supports_think,
                "supports_vision": model.supports_vision,
            }
            for index, model in enumerate(cfg.models)
        ],
    }


@router.get("/knowledge/settings")
def api_knowledge_settings():
    return _settings_payload(get_config())


@router.put("/knowledge/settings")
def api_update_knowledge_settings(body: KnowledgeSettingsUpdate):
    settings = KnowledgeAnalysisSettings(**body.model_dump(exclude={"model_fingerprint"}))
    settings.system_prompt = (settings.system_prompt or "").strip() or None
    try:
        return _settings_payload(update_knowledge_analysis_settings(settings, body.model_fingerprint))
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
    return {"tags": get_summary(ABILITY_SESSION_TYPES)}


@router.get("/knowledge/history")
async def api_knowledge_history(page: int = 1, page_size: int = 20):
    return get_history(page, page_size, ABILITY_SESSION_TYPES)


@router.delete("/knowledge/reset")
async def api_knowledge_reset():
    reset_all()
    return {"ok": True}
