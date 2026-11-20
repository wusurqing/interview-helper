import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { BrainCircuit, Plus, RefreshCw, Save } from 'lucide-react'
import { getErrorMessage } from '@/lib/api'
import { refreshConfig } from '@/lib/configSync'
import {
  hasDraftCredential,
  modelProviders,
  normalizeProviderUrl,
  sameProviderConnection,
  type CatalogModel,
  type ProviderDraft,
  type ProviderModel,
  type ProviderSnapshot,
} from '@/lib/modelProviders'
import ModelProviderCard, { modelDraft, type VerificationByModel, type VerificationState } from './ModelProviderCard'
import { Field, GradientCard, SaveStateBadge, useSettingsDirtyRegistration, type SaveState } from './shared'

function createId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

export default function ModelsTab({ onSavingChange }: { onSavingChange?: (saving: boolean) => void }) {
  const activeInput = useId()
  const [saved, setSaved] = useState<ProviderSnapshot | null>(null)
  const [providers, setProviders] = useState<ProviderDraft[]>([])
  const [models, setModels] = useState<ProviderModel[]>([])
  const [activeModel, setActiveModel] = useState<string | null>(null)
  const [verification, setVerification] = useState<Record<string, VerificationByModel>>({})
  const [confirmed, setConfirmed] = useState<string[]>([])
  const [session, setSession] = useState(0)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [saving, setSaving] = useState(false)
  const saveInFlight = useRef(false)
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [saveError, setSaveError] = useState('')
  const [refreshWarning, setRefreshWarning] = useState('')
  const [conflict, setConflict] = useState(false)
  const [reload, setReload] = useState(0)
  const savedProviderById = useMemo(() => new Map(saved?.providers.map((provider) => [provider.id, provider])), [saved])
  const savedModelById = useMemo(() => new Map(saved?.models.map((model) => [model.id, model])), [saved])
  const knownDirty = !!saved && (
    activeModel !== saved.active_model_id || JSON.stringify(providers) !== JSON.stringify(saved.providers)
    || JSON.stringify(models) !== JSON.stringify(saved.models)
  )
  useSettingsDirtyRegistration('models', knownDirty)

  const applySnapshot = useCallback((snapshot: ProviderSnapshot) => {
    setSaved(snapshot)
    setProviders(snapshot.providers)
    setModels(snapshot.models)
    setActiveModel(snapshot.active_model_id)
    setConfirmed([])
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setLoadError('')
    setSaveError('')
    setRefreshWarning('')
    setSaveState('idle')
    setConflict(false)
    void modelProviders.get(controller.signal).then((data) => {
      if (controller.signal.aborted) return
      applySnapshot(data)
      setVerification({})
      setSession((current) => current + 1)
    }).catch((error) => {
      if (!controller.signal.aborted) setLoadError(getErrorMessage(error, '无法加载供应商配置'))
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [applySnapshot, reload])

  function markChanged() {
    setSaveState('idle')
    setSaveError('')
    setRefreshWarning('')
  }

  function newProvider() {
    setProviders((current) => [...current, {
      id: createId(),
      name: '',
      api_type: 'openai',
      api_base_url: 'https://api.openai.com/v1',
      has_api_key: false,
    }])
    markChanged()
  }

  function changeProvider(id: string, patch: Partial<ProviderDraft>) {
    setProviders((current) => current.map((provider) => provider.id === id ? { ...provider, ...patch } : provider))
    markChanged()
  }

  function changeModel(id: string, patch: Partial<ProviderModel>) {
    setModels((current) => current.map((model) => model.id === id ? { ...model, ...patch } : model))
    markChanged()
  }

  function addModel(providerId: string, remote: CatalogModel, enabled: boolean) {
    if (enabled && verification[providerId]?.[remote.id]?.status !== 'ok') return
    setModels((current) => {
      if (current.some((model) => model.provider_id === providerId && model.model === remote.id)) return current
      return [...current, { ...modelDraft(providerId, createId(), remote), enabled }]
    })
    markChanged()
  }

  function usageLines(usedModels: ProviderModel[]) {
    return usedModels.flatMap((model) => {
      const usages = [...(saved?.usages[model.id] ?? [])]
      if (activeModel === model.id && !usages.includes('通用默认模型')) usages.push('通用默认模型')
      return usages.length > 0 ? [model.name + '：' + usages.join('、')] : []
    })
  }

  function rememberRemoval(ids: string[]) {
    setConfirmed((current) => [...new Set([...current, ...ids])])
    if (activeModel && ids.includes(activeModel)) setActiveModel(null)
  }

  function toggleModel(model: ProviderModel) {
    if (!model.enabled && verification[model.provider_id]?.[model.model]?.status !== 'ok') return
    if (model.enabled) {
      const usages = usageLines([model])
      if (usages.length && !window.confirm('取消启用后，保存时会移除以下配置中的模型：\n' + usages.join('\n'))) return
      rememberRemoval([model.id])
    } else {
      setConfirmed((current) => current.filter((id) => id !== model.id))
    }
    changeModel(model.id, { enabled: !model.enabled })
  }

  function removeModel(model: ProviderModel) {
    const usages = usageLines([model])
    const question = usages.length
      ? `删除「${model.name}」的配置后，保存时会移除以下引用：\n` + usages.join('\n')
      : `删除「${model.name}」的配置？`
    if (!window.confirm(question)) return
    rememberRemoval([model.id])
    setModels((current) => current.filter(({ id }) => id !== model.id))
    markChanged()
  }

  function removeProvider(provider: ProviderDraft) {
    const related = models.filter((model) => model.provider_id === provider.id)
    const usages = usageLines(related)
    const question = usages.length
      ? `删除「${provider.name || '新供应商'}」后，保存时会移除以下配置中的模型：\n` + usages.join('\n')
      : `删除「${provider.name || '新供应商'}」及它的模型？`
    if (!window.confirm(question)) return
    rememberRemoval(related.map((model) => model.id))
    setProviders((current) => current.filter(({ id }) => id !== provider.id))
    setModels((current) => current.filter(({ provider_id }) => provider_id !== provider.id))
    setVerification((current) => {
      const next = { ...current }
      delete next[provider.id]
      return next
    })
    markChanged()
  }

  function setProviderVerification(id: string, statuses: VerificationByModel) {
    setVerification((current) => ({ ...current, [id]: statuses }))
  }

  function updateModelVerification(id: string, model: string, status: VerificationState) {
    setVerification((current) => ({ ...current, [id]: { ...current[id], [model]: status } }))
  }

  function reloadConfiguration() {
    if (knownDirty && !window.confirm('重新载入会丢弃此页面未保存的更改，确定继续？')) return
    setReload((current) => current + 1)
  }

  async function save() {
    if (saveInFlight.current || !saved || conflict) return
    setSaveError('')
    setRefreshWarning('')
    const providerById = new Map(providers.map((provider) => [provider.id, provider]))
    const invalidProviders = providers.find((provider) => {
      const original = savedProviderById.get(provider.id)
      try {
        const url = new URL(normalizeProviderUrl(provider.api_base_url))
        if (!provider.name.trim() || !['http:', 'https:'].includes(url.protocol)) return true
      } catch {
        return true
      }
      return !!original && !provider.api_key?.trim()
        && (provider.api_type !== original.api_type || normalizeProviderUrl(provider.api_base_url) !== normalizeProviderUrl(original.api_base_url))
    })
    if (invalidProviders) {
      setSaveState('error')
      setSaveError(`请核对「${invalidProviders.name || '新供应商'}」的名称、地址和凭据。更改接口类型或地址后，请填写对应的 API Key。`)
      return
    }
    const invalidModel = models.find((model) => !model.name.trim() || !Number.isSafeInteger(model.context_window_tokens) || model.context_window_tokens < 1)
    if (invalidModel) {
      setSaveState('error')
      setSaveError(`请核对模型「${invalidModel.name || invalidModel.model}」的名称和上下文长度。`)
      return
    }
    const unverified = models.find((model) => {
      if (!model.enabled) return false
      const current = providerById.get(model.provider_id)
      const original = savedProviderById.get(model.provider_id)
      const old = savedModelById.get(model.id)
      if (old?.enabled && old.model === model.model && original && current && sameProviderConnection(current, original)) return false
      return verification[model.provider_id]?.[model.model]?.status !== 'ok'
    })
    if (unverified) {
      setSaveState('error')
      setSaveError(`请先用当前连接验证「${unverified.name}」，或取消启用。`)
      return
    }

    saveInFlight.current = true
    setSaving(true)
    onSavingChange?.(true)
    setSaveState('saving')
    try {
      const snapshot = await modelProviders.save({
        revision: saved.revision,
        providers: providers.map(({ id, name, api_type, api_base_url, api_key }) => ({
          id, name: name.trim(), api_type, api_base_url: normalizeProviderUrl(api_base_url),
          api_key: api_key?.trim() || (api_key === null ? null : undefined),
        })),
        models: models.map(({ verified: _verified, ...model }) => ({ ...model, name: model.name.trim() })),
        confirmed_remove_ids: confirmed,
        active_model_id: activeModel,
      })
      applySnapshot(snapshot)
      try {
        await refreshConfig()
      } catch {
        setRefreshWarning('配置已保存。应用状态刷新失败，请重新打开设置。')
      }
      setSaveState('saved')
    } catch (error) {
      if (typeof error === 'object' && error !== null && 'status' in error && error.status === 409) {
        setConflict(true)
        setSaveError('配置或模型用途已变化。请重新载入，再核对需要移除的用途。')
      } else {
        setSaveError(getErrorMessage(error, '保存供应商配置失败'))
      }
      setSaveState('error')
    } finally {
      saveInFlight.current = false
      setSaving(false)
      onSavingChange?.(false)
    }
  }

  const enabled = models.filter((model) => {
    const provider = providers.find(({ id }) => id === model.provider_id)
    return model.enabled && provider && hasDraftCredential(provider, savedProviderById.get(provider.id))
  })
  const defaultValue = enabled.some((model) => model.id === activeModel) ? activeModel ?? '' : ''
  const showReload = !!loadError || conflict
  const displayedSaveState = saveState === 'idle' && knownDirty ? 'dirty' : saveState

  return (
    <div className="space-y-4 p-4 pb-8 sm:p-5">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <h2 className="flex items-center gap-2 text-sm font-semibold"><BrainCircuit className="h-4 w-4 text-accent-blue" aria-hidden />模型供应商配置</h2>
          <p className="mt-1 text-xs text-text-muted">每个供应商只填写一套连接信息。勾选模型后，在各功能里选用。</p>
        </div>
        <SaveStateBadge mode="explicit" state={displayedSaveState} />
      </header>
      <fieldset disabled={saving || loading} className="min-w-0 space-y-3">
        {loading && <p role="status" className="text-xs text-text-muted">正在加载供应商配置…</p>}
        {loadError && <p role="alert" className="text-xs text-accent-red">{loadError}</p>}
        {!loading && saved && (
          <>
            {providers.map((provider, index) => (
              <ModelProviderCard key={`${session}:${provider.id}`} provider={provider} original={savedProviderById.get(provider.id)}
                models={models.filter((model) => model.provider_id === provider.id)}
                savedModels={saved.models.filter((model) => model.provider_id === provider.id)}
                verification={verification[provider.id] ?? {}}
                initiallyOpen={index === 0 || !savedProviderById.has(provider.id)}
                changed={(patch) => changeProvider(provider.id, patch)}
                removeProvider={() => removeProvider(provider)}
                addModel={(remote, checked) => addModel(provider.id, remote, checked)}
                changeModel={changeModel}
                toggleModel={toggleModel}
                removeModel={removeModel}
                updateVerification={(model, status) => updateModelVerification(provider.id, model, status)}
                resetVerification={(statuses) => setProviderVerification(provider.id, statuses)}
              />
            ))}
            {!providers.length && <GradientCard className="p-3"><p className="text-xs text-text-muted">添加供应商后即可获取模型或手填 Model ID。</p></GradientCard>}
            <button type="button" onClick={newProvider} className="inline-flex items-center gap-1.5 rounded-md border border-bg-hover px-3 py-2 text-xs hover:bg-bg-hover">
              <Plus className="h-4 w-4" aria-hidden />添加供应商
            </button>
            <GradientCard className="p-3">
              <Field label="通用默认模型" htmlFor={activeInput} hint="供简历优化等通用功能使用。实时辅助、截图答题和笔试在各自的设置中选择。">
                <select id={activeInput} className="input-field" value={defaultValue} onChange={(event) => {
                  setActiveModel(event.target.value || null)
                  markChanged()
                }}>
                  <option value="">未选择</option>
                  {enabled.map((model) => <option key={model.id} value={model.id}>
                    {model.name} · {providers.find((provider) => provider.id === model.provider_id)?.name}
                  </option>)}
                </select>
              </Field>
            </GradientCard>
          </>
        )}
        {saveError && <p role="alert" className="text-xs text-accent-red">{saveError}</p>}
        {refreshWarning && <p role="status" className="text-xs text-accent-amber">{refreshWarning}</p>}
        <div className="flex flex-wrap justify-end gap-2">
          {showReload && (
            <button type="button" onClick={reloadConfiguration} className="inline-flex items-center gap-1.5 rounded-md border border-bg-hover px-3 py-2 text-xs hover:bg-bg-hover">
              <RefreshCw className="h-3.5 w-3.5" aria-hidden />重新载入
            </button>
          )}
          {saved && (
            <button type="button" disabled={!knownDirty || conflict} onClick={() => void save()}
              className="inline-flex items-center gap-1.5 rounded-md bg-accent-blue/15 px-3 py-2 text-xs font-medium text-accent-blue hover:bg-accent-blue/25 disabled:opacity-50">
              <Save className="h-3.5 w-3.5" aria-hidden />{saving ? '保存中…' : '保存供应商配置'}
            </button>
          )}
        </div>
      </fieldset>
    </div>
  )
}
