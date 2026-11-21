import { useEffect, useId, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Loader2, Plus, RefreshCw, Trash2 } from 'lucide-react'
import VisionBadge from '@/components/VisionBadge'
import { api, getErrorMessage } from '@/lib/api'
import {
  hasDraftCredential,
  modelProviders,
  normalizeProviderUrl,
  sameProviderConnection,
  type CatalogModel,
  type ModelProtocol,
  type ModelProvider,
  type ProviderDraft,
  type ProviderModel,
} from '@/lib/modelProviders'
import { useInterviewStore } from '@/stores/configStore'
import { Field, GradientCard } from './shared'

export type VerificationState =
  | { status: 'checking' }
  | { status: 'ok'; latency?: number }
  | { status: 'error'; detail: string }

export type VerificationByModel = Record<string, VerificationState>

const VERIFY_CONCURRENCY = 4
const DEFAULT_CONTEXT_WINDOW_TOKENS = 262144

export function modelDraft(providerId: string, id: string, catalog: CatalogModel): ProviderModel {
  return {
    id,
    provider_id: providerId,
    model: catalog.id,
    name: catalog.name?.trim() || catalog.id,
    enabled: false,
    supports_vision: catalog.supports_vision === true,
    supports_think: catalog.supports_think === true,
    context_window_tokens: catalog.context_window_tokens && catalog.context_window_tokens > 0
      ? catalog.context_window_tokens
      : DEFAULT_CONTEXT_WINDOW_TOKENS,
    think_enabled_params: {},
    think_disabled_params: {},
  }
}

const compactButton = 'inline-flex items-center justify-center gap-1.5 rounded-md border border-bg-hover px-2.5 py-1.5 text-xs font-medium hover:bg-bg-hover disabled:opacity-50'
const actionButton = 'inline-flex items-center justify-center gap-1.5 rounded-md bg-accent-blue/15 px-2.5 py-1.5 text-xs font-medium text-accent-blue hover:bg-accent-blue/25 disabled:opacity-50'

interface CardProps {
  provider: ProviderDraft
  original?: ModelProvider
  models: ProviderModel[]
  savedModels: ProviderModel[]
  verification: VerificationByModel
  initiallyOpen: boolean
  changed: (patch: Partial<ProviderDraft>) => void
  removeProvider: () => void
  addModel: (model: CatalogModel, enabled: boolean) => void
  changeModel: (id: string, patch: Partial<ProviderModel>) => void
  toggleModel: (model: ProviderModel) => void
  removeModel: (model: ProviderModel) => void
  updateVerification: (model: string, status: VerificationState) => void
  resetVerification: (statuses: VerificationByModel) => void
}

function VerificationLabel({ state, incompatible, saved }: { state?: VerificationState; incompatible?: string | null; saved?: boolean }) {
  if (incompatible) return <span className="text-accent-amber">{incompatible}</span>
  if (state?.status === 'checking') return <span className="text-accent-blue">验证中</span>
  if (state?.status === 'error') return (
    <span className="text-accent-red">
      验证失败 <span className="text-text-muted">{state.detail}</span>
    </span>
  )
  if (state?.status === 'ok') return (
    <span className="text-accent-green">可用{state.latency ? <span className="text-text-muted"> · {state.latency} ms</span> : null}</span>
  )
  if (saved) return <span className="text-accent-green">上次验证可用</span>
  return <span className="text-text-muted">未验证</span>
}

function AdvancedModel({ model, changeModel, removeModel, canProbe, onProbe }: {
  model: ProviderModel
  changeModel: (patch: Partial<ProviderModel>) => void
  removeModel: () => void
  canProbe: boolean
  onProbe: () => Promise<boolean>
}) {
  const id = useId()
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const change = async () => {
    setBusy(true)
    setMessage('')
    try {
      if (await onProbe()) setMessage('探测结果已填入。保存后应用。')
    } catch (error) {
      setMessage(getErrorMessage(error, '能力探测失败'))
    } finally {
      setBusy(false)
    }
  }
  return (
    <details className="rounded-md bg-bg-tertiary/50 px-2.5 py-1.5">
      <summary className="cursor-pointer select-none text-[11px] text-text-secondary">高级设置</summary>
      <div className="mt-2 space-y-2.5 pb-1">
        <div className="grid gap-2 sm:grid-cols-2">
          <Field label="模型显示名称" htmlFor={id + '-name'}>
            <input id={id + '-name'} className="input-field" value={model.name}
              onChange={(event) => changeModel({ name: event.target.value })} />
          </Field>
          <Field label="上下文长度（tokens）" htmlFor={id + '-context'}>
            <input id={id + '-context'} className="input-field" type="number" min="1" step="1"
              value={Number.isNaN(model.context_window_tokens) ? '' : model.context_window_tokens}
              onChange={(event) => changeModel({ context_window_tokens: event.target.valueAsNumber })} />
          </Field>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs">
          <label className="flex items-center gap-1.5">
            <input type="checkbox" className="accent-accent-blue" checked={model.supports_vision}
              onChange={(event) => changeModel({ supports_vision: event.target.checked })} />
            支持识图
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" className="accent-accent-blue" checked={model.supports_think}
              onChange={(event) => changeModel({ supports_think: event.target.checked })} />
            支持 Think
          </label>
        </div>
        {(Object.keys(model.think_enabled_params).length > 0 || Object.keys(model.think_disabled_params).length > 0) && (
          <dl className="space-y-1 break-all text-[11px] text-text-muted">
            <dt>Think 开启参数</dt><dd className="font-mono">{JSON.stringify(model.think_enabled_params)}</dd>
            <dt>Think 关闭参数</dt><dd className="font-mono">{JSON.stringify(model.think_disabled_params)}</dd>
          </dl>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => void change()} className={compactButton} disabled={!canProbe || busy}>
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />}
            {busy ? '探测中…' : '探测识图与 Think'}
          </button>
          <span className="text-[11px] text-text-muted">{canProbe ? '会调用模型，可能产生费用。' : '保存并启用后可以探测。'}</span>
        </div>
        {message && <p role="status" className="text-[11px] text-text-secondary">{message}</p>}
        <button type="button" onClick={removeModel} className="inline-flex items-center gap-1 text-[11px] text-accent-red hover:underline">
          <Trash2 className="h-3 w-3" aria-hidden />删除该模型的配置
        </button>
      </div>
    </details>
  )
}

export default function ModelProviderCard({
  provider, original, models, savedModels, verification, initiallyOpen,
  changed, removeProvider, addModel, changeModel, toggleModel, removeModel, updateVerification, resetVerification,
}: CardProps) {
  const id = useId()
  const [expanded, setExpanded] = useState(initiallyOpen)
  const [catalog, setCatalog] = useState<CatalogModel[]>([])
  const [loaded, setLoaded] = useState(false)
  const [filter, setFilter] = useState('')
  const [manualId, setManualId] = useState('')
  const [busy, setBusy] = useState(false)
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [error, setError] = useState('')
  const operation = useRef<AbortController | null>(null)
  const capabilityVersion = useRef(0)

  useEffect(() => () => {
    operation.current?.abort()
    capabilityVersion.current += 1
  }, [])

  function changeConnection(patch: Partial<ProviderDraft>) {
    if (!sameProviderConnection(provider, { ...provider, ...patch })) {
      operation.current?.abort()
      operation.current = null
      capabilityVersion.current += 1
      setBusy(false)
      setCatalogLoading(false)
      setCatalog([])
      setLoaded(false)
      setError('')
      resetVerification({})
    }
    changed(patch)
  }

  function startOperation() {
    if (operation.current) return null
    const controller = new AbortController()
    operation.current = controller
    setBusy(true)
    return controller
  }

  function finishOperation(controller: AbortController) {
    if (operation.current !== controller) return
    operation.current = null
    setBusy(false)
    setCatalogLoading(false)
  }

  async function verify(modelId: string, signal: AbortSignal) {
    try {
      const result = await modelProviders.verify(provider, modelId, signal)
      if (signal.aborted) return
      updateVerification(modelId, result.ok
        ? { status: 'ok', latency: result.latency_ms }
        : { status: 'error', detail: result.detail || '模型没有返回有效文本。' })
    } catch (error) {
      if (!signal.aborted) updateVerification(modelId, { status: 'error', detail: getErrorMessage(error, '验证失败') })
    }
  }

  async function fetchModels() {
    const controller = startOperation()
    if (!controller) return
    const { signal } = controller
    setCatalogLoading(true)
    setError('')
    try {
      const { models: entries } = await modelProviders.catalog(provider, signal)
      if (signal.aborted) return
      setCatalog(entries)
      setLoaded(true)
      setCatalogLoading(false)
      const supported = entries.filter((model) => !model.incompatible_reason)
      const statuses: VerificationByModel = { ...verification }
      for (const model of entries) {
        if (model.incompatible_reason) delete statuses[model.id]
        else statuses[model.id] = { status: 'checking' }
      }
      resetVerification(statuses)
      let next = 0
      async function verifyNext() {
        while (!signal.aborted && next < supported.length) {
          await verify(supported[next++].id, signal)
        }
      }
      await Promise.all(Array.from({ length: Math.min(supported.length, VERIFY_CONCURRENCY) }, verifyNext))
    } catch (error) {
      if (!signal.aborted) setError(getErrorMessage(error, '获取模型失败'))
    } finally {
      finishOperation(controller)
    }
  }

  async function validateOne(modelId: string) {
    const controller = startOperation()
    if (!controller) return
    updateVerification(modelId, { status: 'checking' })
    try {
      await verify(modelId, controller.signal)
    } finally {
      finishOperation(controller)
    }
  }

  async function probeCapabilities(model: ProviderModel) {
    const version = capabilityVersion.current
    const config = useInterviewStore.getState().config
    const index = config?.models.findIndex((entry) => entry.id === model.id) ?? -1
    if (index < 0) throw new Error('找不到已保存的模型，请重新载入配置。')
    const result = await api.probeModelCapabilities(index, model.id)
    if (version !== capabilityVersion.current) return false
    if (!result.ok) throw new Error(result.detail || '能力探测失败')
    changeModel(model.id, {
      supports_vision: result.supports_vision,
      supports_think: result.supports_think,
      think_enabled_params: result.think_params,
      think_disabled_params: result.think_disabled_params,
    })
    return true
  }

  function addManualModel() {
    const id = manualId.trim()
    if (!id) return
    if (models.some((model) => model.model === id) || catalog.some((entry) => entry.id === id)) {
      setError('该 Model ID 已在列表中。')
      return
    }
    addModel({ id }, false)
    setManualId('')
    setError('')
    setFilter('')
  }

  const canUseService = Boolean(normalizeProviderUrl(provider.api_base_url)) && hasDraftCredential(provider, original)
  const catalogById = new Map(catalog.map((entry) => [entry.id, entry]))
  const configuredIds = new Set(models.map((model) => model.model))
  const rows = [
    ...models.map((model) => ({ model, remote: catalogById.get(model.model), remoteId: model.model, key: model.id })),
    ...catalog.filter((entry) => !configuredIds.has(entry.id)).map((remote) => ({ model: null, remote, remoteId: remote.id, key: remote.id })),
  ].filter(({ model, remoteId, remote }) => {
    const needle = filter.trim().toLowerCase()
    return !needle || [model?.name, remote?.name, remoteId, remote?.owned_by].some((value) => value?.toLowerCase().includes(needle))
  })
  const savedIds = new Set(savedModels.filter((model) => model.enabled).map((model) => model.id))
  const savedConnection = !!original && sameProviderConnection(provider, original)
  const checkedCount = models.filter((model) => model.enabled).length

  return (
    <GradientCard className="overflow-hidden">
      <div className="flex min-w-0 items-center gap-2 px-3 py-3">
        <button type="button" aria-expanded={expanded} aria-controls={id + '-panel'} onClick={() => setExpanded(!expanded)}
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left">
          {expanded ? <ChevronDown className="h-4 w-4 shrink-0" aria-hidden /> : <ChevronRight className="h-4 w-4 shrink-0" aria-hidden />}
          <span className="truncate text-sm font-semibold">{provider.name.trim() || '新供应商'}</span>
          <span className="shrink-0 text-[11px] text-text-muted">{checkedCount} 个已启用</span>
        </button>
        <button type="button" aria-label={`删除供应商 ${provider.name}`} title="删除供应商"
          className="rounded-md p-1.5 text-text-muted hover:bg-accent-red/10 hover:text-accent-red" onClick={removeProvider}>
          <Trash2 className="h-4 w-4" aria-hidden />
        </button>
      </div>
      <div id={id + '-panel'} hidden={!expanded} className="space-y-3 border-t border-bg-hover/60 px-3 pb-3 pt-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="供应商名称" htmlFor={id + '-name'}>
            <input id={id + '-name'} className="input-field" value={provider.name} onChange={(event) => changed({ name: event.target.value })} />
          </Field>
          <Field label="接口类型" htmlFor={id + '-protocol'}>
            <select id={id + '-protocol'} className="input-field" value={provider.api_type}
              onChange={(event) => changeConnection({ api_type: event.target.value as ModelProtocol })}>
              <option value="openai">OpenAI Chat Completions</option>
              <option value="anthropic">Anthropic Messages</option>
              <option value="responses">OpenAI Responses</option>
            </select>
          </Field>
          <Field label="API Base URL" htmlFor={id + '-url'}>
            <input id={id + '-url'} className="input-field" type="url" value={provider.api_base_url}
              placeholder="https://api.openai.com/v1"
              onChange={(event) => changeConnection({ api_base_url: event.target.value })} />
          </Field>
          <div className="space-y-1.5">
            <Field label="API Key" htmlFor={id + '-key'}>
              <input id={id + '-key'} className="input-field" type="password" autoComplete="off" value={provider.api_key ?? ''}
                placeholder={original?.has_api_key && savedConnection ? '已保存，留空保留' : '输入此地址的 API Key'}
                onChange={(event) => changeConnection({ api_key: event.target.value.trim() ? event.target.value : undefined })} />
            </Field>
            {original?.has_api_key && (
              <button type="button" className="text-[11px] text-text-secondary hover:text-accent-blue"
                onClick={() => changeConnection({ api_key: provider.api_key === null ? undefined : null })}>
                {provider.api_key === null ? '撤销清除 API Key' : '清除已保存的 API Key'}
              </button>
            )}
          </div>
        </div>

        <section className="space-y-2 rounded-lg border border-bg-hover/60 p-2.5" aria-labelledby={id + '-models'}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 id={id + '-models'} className="text-xs font-semibold">供应商模型</h3>
            <button type="button" className={actionButton} disabled={!canUseService || busy} onClick={() => void fetchModels()}>
              <RefreshCw className={`h-3.5 w-3.5 ${busy ? 'animate-spin' : ''}`} aria-hidden />
              {catalogLoading ? '获取中…' : busy ? '验证中…' : '获取模型'}
            </button>
          </div>
          <p className="text-[11px] leading-relaxed text-text-muted">获取模型后会逐一验证文本生成。获取和手动验证都可能产生费用。</p>
          {rows.length > 0 || filter ? (
            <input className="input-field" type="search" aria-label="搜索供应商模型"
              placeholder="搜索模型名称或 ID" value={filter} onChange={(event) => setFilter(event.target.value)} />
          ) : null}
          {error && <p role="alert" className="break-words text-xs text-accent-red">{error}</p>}
          {loaded && !catalog.length && !error && <p className="text-xs text-text-muted">目录为空。可以手填 Model ID。</p>}
          {filter && !rows.length && <p className="text-xs text-text-muted">没有匹配的模型。</p>}
          {rows.length > 0 && (
            <ul className="max-h-[26rem] divide-y divide-bg-hover/60 overflow-y-auto">
              {rows.map(({ model, remote, remoteId, key }) => {
                const state = verification[remoteId]
                const label = model?.name || remote?.name || remoteId
                const incompatible = remote?.incompatible_reason
                const checked = model?.enabled ?? false
                return (
                  <li key={key} aria-label={label} className="space-y-2 py-2.5">
                    <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
                      <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2">
                        <input type="checkbox" aria-label={`启用 ${label}`} checked={checked}
                          disabled={!checked && (!!incompatible || state?.status !== 'ok')}
                          className="shrink-0 accent-accent-blue"
                          onChange={() => model ? toggleModel(model) : remote && addModel(remote, true)} />
                        <span className="min-w-0">
                          <span className="flex items-center gap-1 break-all text-xs font-medium">
                            {label}{(model ? model.supports_vision : remote?.supports_vision) && <VisionBadge />}
                          </span>
                          {label !== remoteId && <span className="break-all font-mono text-[11px] text-text-muted">{remoteId}</span>}
                        </span>
                      </label>
                      <span className="break-words text-[11px]"><VerificationLabel state={state} incompatible={incompatible} saved={savedConnection && model?.verified} /></span>
                      <button type="button" className={compactButton} disabled={!canUseService || busy || Boolean(incompatible)}
                        onClick={() => void validateOne(remoteId)}>验证模型</button>
                    </div>
                    {model && <AdvancedModel
                      model={model}
                      changeModel={(patch) => changeModel(model.id, patch)}
                      removeModel={() => removeModel(model)}
                      canProbe={checked && savedIds.has(model.id) && savedConnection}
                      onProbe={() => probeCapabilities(model)}
                    />}
                  </li>
                )
              })}
            </ul>
          )}
          <div className="flex items-end gap-2">
            <div className="min-w-0 flex-1">
              <Field label="手填 Model ID" htmlFor={id + '-manual'}>
                <input id={id + '-manual'} className="input-field" value={manualId}
                  onChange={(event) => setManualId(event.target.value)}
                  placeholder="例如 ep-…"
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      addManualModel()
                    }
                  }} />
              </Field>
            </div>
            <button type="button" disabled={!manualId.trim()} className={compactButton} onClick={addManualModel}>
              <Plus className="h-3.5 w-3.5" aria-hidden />添加 Model ID
            </button>
          </div>
        </section>
      </div>
    </GradientCard>
  )
}
