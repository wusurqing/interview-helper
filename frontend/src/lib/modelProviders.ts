import { request } from './api'

export type ModelProtocol = 'openai' | 'anthropic' | 'responses'

export interface ModelProvider {
  id: string
  name: string
  api_type: ModelProtocol
  api_base_url: string
  has_api_key: boolean
}

export interface ProviderDraft extends ModelProvider {
  api_key?: string | null
}

export interface ProviderModel {
  id: string
  provider_id: string
  name: string
  model: string
  enabled: boolean
  supports_vision: boolean
  supports_think: boolean
  context_window_tokens: number
  think_enabled_params: Record<string, unknown>
  think_disabled_params: Record<string, unknown>
  verified?: boolean
}

export interface ProviderSnapshot {
  revision: string
  providers: ModelProvider[]
  models: ProviderModel[]
  usages: Record<string, string[]>
  active_model_id: string | null
}

export interface ProviderSettingsUpdate extends Pick<ProviderSnapshot, 'revision' | 'active_model_id'> {
  providers: Omit<ProviderDraft, 'has_api_key'>[]
  confirmed_remove_ids: string[]
  models: Omit<ProviderModel, 'verified'>[]
}

export interface CatalogModel {
  id: string
  name?: string | null
  owned_by?: string | null
  supports_vision?: boolean | null
  supports_think?: boolean | null
  context_window_tokens?: number | null
  incompatible_reason?: string | null
}

export interface ModelVerification {
  ok: boolean
  detail?: string
  latency_ms?: number
}

export function normalizeProviderUrl(url: string) {
  return url.trim().replace(/\/+$/, '')
}

export function sameProviderConnection(a: ProviderDraft, b: ProviderDraft) {
  return a.api_type === b.api_type
    && normalizeProviderUrl(a.api_base_url) === normalizeProviderUrl(b.api_base_url)
    && a.api_key === b.api_key
}

export function hasDraftCredential(provider: ProviderDraft, original?: ModelProvider) {
  if (provider.api_key !== undefined) return Boolean(provider.api_key?.trim())
  return !!original?.has_api_key && sameProviderConnection(provider, original)
}

function connection(provider: ProviderDraft) {
  return {
    ...(provider.has_api_key ? { provider_id: provider.id } : {}),
    api_type: provider.api_type,
    api_base_url: normalizeProviderUrl(provider.api_base_url),
    api_key: provider.api_key?.trim() || (provider.api_key === null ? null : undefined),
  }
}

export const modelProviders = {
  get: (signal?: AbortSignal) => request<ProviderSnapshot>('/api/model-providers', { signal }),
  save: (settings: ProviderSettingsUpdate) => request<ProviderSnapshot>('/api/model-providers', {
    method: 'PUT',
    body: JSON.stringify(settings),
  }),
  catalog: (provider: ProviderDraft, signal: AbortSignal) => request<{ models: CatalogModel[] }>('/api/model-providers/catalog', {
    method: 'POST',
    signal,
    body: JSON.stringify(connection(provider)),
  }),
  verify: (provider: ProviderDraft, model: string, signal: AbortSignal) => request<ModelVerification>('/api/model-providers/verify', {
    method: 'POST',
    signal,
    body: JSON.stringify({ ...connection(provider), model }),
  }),
}
