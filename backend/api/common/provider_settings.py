"""Read and save provider configuration without returning credentials."""

from typing import Any, Literal

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, ValidationError

from core.config import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    AppConfig,
    ConfigCorruptError,
    ModelConfig,
    ModelListChanged,
    ModelProvider,
    get_config,
    get_model_usages,
    get_provider_layout,
    provider_layout_revision,
    provider_url,
    save_provider_layout,
)


router = APIRouter(prefix="/model-providers", tags=["model providers"])


class ProviderInput(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    api_type: Literal["openai", "anthropic", "responses"]
    api_base_url: str
    api_key: str | None = Field(default=None, repr=False)


class ProviderModelInput(BaseModel):
    id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    enabled: bool
    context_window_tokens: int = Field(default=DEFAULT_CONTEXT_WINDOW_TOKENS, strict=True, ge=1, le=2**53 - 1)
    supports_think: bool = False
    supports_vision: bool = False
    think_enabled_params: dict[str, Any] = Field(default_factory=dict)
    think_disabled_params: dict[str, Any] = Field(default_factory=dict)


class ProviderSettingsInput(BaseModel):
    revision: str
    providers: list[ProviderInput]
    models: list[ProviderModelInput]
    confirmed_remove_ids: set[str] = Field(default_factory=set)
    active_model_id: str | None


def _provider_settings_payload(cfg: AppConfig) -> dict:
    providers, models = get_provider_layout(cfg)
    usages = get_model_usages(cfg)
    return {
        "revision": provider_layout_revision(cfg),
        "providers": [
            {
                **provider.model_dump(exclude={"api_key"}),
                "has_api_key": provider.api_key.strip() not in ("", "sk-your-api-key-here"),
            }
            for provider in providers
        ],
        "models": [
            model.model_dump(exclude={"api_key", "api_base_url", "api_type"})
            for model in models
        ],
        "usages": {model.id: usages[model.id] for model in models if model.id in usages},
        "active_model_id": next(
            (model.id for model in models if model.enabled and model.id == cfg.active_model_id),
            None,
        ),
    }


@router.get("")
def get_provider_settings() -> dict:
    return _provider_settings_payload(get_config())


def _resolve_providers(inputs: list[ProviderInput], cfg: AppConfig) -> list[ModelProvider]:
    saved = {provider.id: provider for provider in get_provider_layout(cfg)[0]}
    providers = []
    for item in inputs:
        previous = saved.get(item.id)
        address = provider_url(item.api_base_url)
        connection_changed = previous is not None and (
            provider_url(previous.api_base_url) != address or previous.api_type != item.api_type
        )
        key = (item.api_key or "").strip()
        if connection_changed and not key:
            raise ValueError("更改服务地址或接口类型后，请填写新的 API Key")
        if "api_key" not in item.model_fields_set and previous is not None:
            key = previous.api_key
        if item.api_key is not None and not key:
            raise ValueError("请填写 API Key，或明确清除凭据")
        providers.append(ModelProvider(
            id=item.id, name=item.name.strip(), api_type=item.api_type,
            api_base_url=address, api_key=key,
        ))
    return providers


@router.put("")
def put_provider_settings(body: Any = Body()) -> dict:
    from api.common.provider_catalog import was_verified

    # Use a sanitized error to avoid FastAPI echoing submitted API keys in validation details.
    try:
        parsed = ProviderSettingsInput.model_validate(body)
        cfg = get_config()
        if parsed.revision != provider_layout_revision(cfg):
            raise ModelListChanged("供应商或模型用途已变更，请刷新并重新确认")
        providers = _resolve_providers(parsed.providers, cfg)
        models = [ModelConfig.model_validate(model.model_dump()) for model in parsed.models]
        updated = save_provider_layout(
            parsed.revision, providers, models, parsed.confirmed_remove_ids,
            parsed.active_model_id, was_verified,
        )
    except ValidationError:
        raise HTTPException(422, "供应商或模型配置格式无效") from None
    except ConfigCorruptError as exc:
        raise HTTPException(409, str(exc)) from None
    except ModelListChanged as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except OSError:
        raise HTTPException(500, "无法保存模型供应商配置，请检查配置文件权限") from None
    return _provider_settings_payload(updated)
