import type { AppConfig, ModelInfo } from '@/stores/slices/types'

export function isConfiguredModel(model: ModelInfo) {
  return model.enabled !== false && model.has_api_key !== false
}

export function getAssistModelEntries(config: AppConfig | null | undefined) {
  const catalog = (config?.models ?? []).map((model, index) => ({ model, index }))
  const selectedIds = config?.assist_model_ids
  if (selectedIds === undefined) {
    return catalog.filter(({ model }) => isConfiguredModel(model)).sort((left, right) =>
      Number(right.index === config?.active_model) - Number(left.index === config?.active_model),
    )
  }

  const byId = new Map(catalog.map((entry) => [entry.model.id, entry]))
  return selectedIds.flatMap((id) => {
    const entry = byId.get(id)
    return entry ? [entry] : []
  })
}

export function getAssistScreenshotModel(config: AppConfig | null | undefined) {
  const selectedId = config?.assist_screenshot_model_id
  const catalog = config?.models ?? []
  const model = selectedId === undefined
    ? catalog.find((option) => option.supports_vision && isConfiguredModel(option))
    : catalog.find((option) => option.id === selectedId)
  return model && model.supports_vision && isConfiguredModel(model) ? model : undefined
}
