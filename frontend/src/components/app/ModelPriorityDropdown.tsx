import { useEffect, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'

import VisionBadge from '@/components/VisionBadge'
import { api, getErrorMessage } from '@/lib/api'
import { getAssistModelEntries, isConfiguredModel } from '@/lib/assistModels'
import { useInterviewStore, type AppConfig, type ModelHealthStatus } from '@/stores/configStore'
import type { ModelCatalogStatus } from '@/stores/slices/types'

interface ModelPriorityDropdownProps {
  config: AppConfig
  modelHealth: Record<number, ModelHealthStatus>
  modelHealthDetail: Record<number, string>
  modelHealthLatency: Record<number, number>
  onModelChange: (modelId: string | number) => Promise<void> | void
}

function healthDot(
  config: AppConfig,
  modelHealth: Record<number, ModelHealthStatus>,
  index: number,
  catalog?: ModelCatalogStatus,
) {
  if (config.models[index]?.enabled === false) return 'bg-text-muted/40'
  const status = modelHealth[index]
  if (status === 'ok') return 'bg-accent-green'
  if (status === 'checking') return 'bg-accent-amber animate-pulse'
  if (status === 'error') return 'bg-accent-red'
  if (catalog === 'visible') return 'bg-accent-blue'
  return 'bg-text-muted/30'
}

function healthLabel(
  config: AppConfig,
  modelHealth: Record<number, ModelHealthStatus>,
  modelHealthDetail: Record<number, string>,
  modelHealthLatency: Record<number, number>,
  index: number,
): string {
  if (config.models[index]?.enabled === false) return '已停用，请先在模型设置中启用'
  if (config.models[index]?.has_api_key === false) return '未配置 API Key'
  const status = modelHealth[index]
  const detail = modelHealthDetail[index]?.trim()
  const lat = modelHealthLatency[index]
  if (status === 'ok') return lat ? `生成可用 · ${lat}ms` : '生成可用'
  if (status === 'checking') return '正在验证生成…'
  if (status === 'error') return detail ? `生成验证失败：${detail}` : '生成验证失败'
  return ''
}

function catalogLabel(status?: ModelCatalogStatus) {
  if (status === 'visible') return '目录可见'
  if (status === 'not_listed') return '目录未列出'
  return '目录未验证'
}

export function ModelPriorityDropdown({
  config,
  modelHealth,
  modelHealthDetail,
  modelHealthLatency,
  onModelChange,
}: ModelPriorityDropdownProps) {
  const [open, setOpen] = useState(false)
  const [checkingHealth, setCheckingHealth] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const checkingHealthRef = useRef(false)
  const catalog = useInterviewStore((state) => state.modelCatalog)
  const entries = getAssistModelEntries(config)
  const activeIndex = entries[0]?.index
  const activeModel = entries[0]?.model
  function modelStatus(index: number) {
    return [
      catalogLabel(catalog[index]?.status),
      catalog[index]?.detail,
      healthLabel(config, modelHealth, modelHealthDetail, modelHealthLatency, index),
    ].filter(Boolean).join(' · ')
  }
  const activeLabel = activeIndex === undefined ? '请在实时辅助中勾选模型' : modelStatus(activeIndex)

  useEffect(() => {
    if (!open) return
    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [open])

  const runHealthCheck = async () => {
    if (checkingHealthRef.current) return
    checkingHealthRef.current = true
    setCheckingHealth(true)
    const indexes = entries.filter(({ model }) => isConfiguredModel(model)).map(({ index }) => index)
    try {
      const snapshot = await api.checkModelsHealth(indexes)
      if (snapshot) useInterviewStore.getState().setModelCatalog(snapshot)
    } catch (error) {
      useInterviewStore.getState().pushToast(`模型目录查询失败：${getErrorMessage(error)}`, 'error')
    } finally {
      checkingHealthRef.current = false
      setCheckingHealth(false)
    }
  }

  return (
    <div className="relative" ref={rootRef}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        title={`优先答题模型 · ${activeModel?.name ?? ''}\n状态:${activeLabel}`}
        aria-label={`优先答题模型 ${activeModel?.name ?? ''},${activeLabel}`}
        className="flex items-center gap-1.5 bg-bg-tertiary/50 text-text-primary text-xs rounded-xl px-2 py-1.5 sm:px-2.5 border border-bg-hover/50 hover:border-accent-blue/40 transition-all duration-200 max-w-[110px] md:max-w-[160px]"
      >
        <div
          className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${healthDot(config, modelHealth, activeIndex ?? -1, catalog[activeIndex ?? -1]?.status)}`}
          aria-hidden
        />
        <span className="truncate min-w-0 font-medium hidden sm:inline">
          {activeModel?.name ?? '选择模型'}
        </span>
        {activeModel?.supports_vision && <VisionBadge />}
        <ChevronDown className={`w-3 h-3 flex-shrink-0 text-text-muted transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="app-no-drag absolute right-0 top-full mt-1.5 glass border border-bg-hover/50 rounded-xl shadow-xl shadow-black/20 z-50 min-w-[200px] py-1.5 animate-fade-up">
          {entries.map(({ model, index }) => {
            const disabled = !isConfiguredModel(model)
            const statusLabel = disabled ? '不可用' : index === activeIndex ? '优先' : null
            return (
              <button
                key={model.id ?? index}
                disabled={disabled}
                onClick={async () => {
                  setOpen(false)
                  await onModelChange(model.id ?? index)
                }}
                title={`${model.name} · ${modelStatus(index)}`}
                className={`w-full flex items-center gap-2 px-3 py-2 text-xs text-left transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-55 ${
                  index === activeIndex
                    ? 'text-accent-blue bg-accent-blue/5'
                    : 'text-text-primary hover:bg-bg-tertiary/50'
                }`}
              >
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${healthDot(config, modelHealth, index, catalog[index]?.status)}`} aria-hidden />
                <span className="truncate font-medium">
                  {model.name}
                  {model.provider_name && <span className="ml-1 text-text-muted">{model.provider_name}</span>}
                </span>
                {model.supports_vision && <VisionBadge />}
                {model.api_type && model.api_type !== 'openai' && (
                  <span
                    className="flex-shrink-0 text-[9px] px-1 py-px rounded bg-accent-violet/15 text-accent-violet font-medium"
                    title={`接口类型：${model.api_type === 'anthropic' ? 'Anthropic Messages' : 'OpenAI Responses'}`}
                  >
                    {model.api_type === 'anthropic' ? 'Msg' : 'Rsp'}
                  </span>
                )}
                {modelHealth[index] === 'ok' && modelHealthLatency[index] ? (
                  <span className="ml-auto text-[10px] font-mono text-text-muted tabular-nums">
                    {modelHealthLatency[index]}ms
                  </span>
                ) : statusLabel ? (
                  <span className={`ml-auto text-[10px] font-semibold ${disabled ? 'text-text-muted' : 'text-accent-blue'}`}>
                    {statusLabel}
                  </span>
                ) : null}
              </button>
            )
          })}
          {entries.length === 0 && <p className="p-3 text-xs text-text-muted">请在实时辅助中勾选模型。</p>}
          {entries.some(({ model }) => isConfiguredModel(model)) && <div className="border-t border-bg-hover/40 mt-1 pt-1 px-3 py-1.5">
            <button
              type="button"
              onClick={runHealthCheck}
              disabled={checkingHealth}
              className="text-[10px] text-text-muted hover:text-accent-blue transition-colors font-medium disabled:cursor-wait disabled:opacity-60"
            >
              {checkingHealth ? '检测中…' : '重新检查连接'}
            </button>
          </div>}
        </div>
      )}
    </div>
  )
}
