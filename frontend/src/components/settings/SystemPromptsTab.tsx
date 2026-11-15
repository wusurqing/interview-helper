import { useEffect, useState } from 'react'
import { RotateCcw, Save } from 'lucide-react'

import { api, getErrorMessage, type SystemPromptScene } from '@/lib/api'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'
import { GradientCard, StatusBadge, useSettingsDirtyRegistration } from './shared'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; scenes: SystemPromptScene[]; overrides: Record<string, string> }

type SaveState =
  | { status: 'idle' | 'saving' | 'saved' }
  | { status: 'error'; message: string }

export default function SystemPromptsTab({ onSavingChange }: { onSavingChange: (saving: boolean) => void }) {
  const [load, setLoad] = useState<LoadState>({ status: 'loading' })
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [selectedId, setSelectedId] = useState('')
  const [draft, setDraft] = useState<string | null | undefined>(undefined)
  const [save, setSave] = useState<SaveState>({ status: 'idle' })

  const scene = load.status === 'ready'
    ? load.scenes.find((item) => item.id === selectedId) ?? load.scenes[0]
    : undefined
  const savedOverride = load.status === 'ready' && scene ? load.overrides[scene.id] ?? null : null
  const nextOverride = draft === undefined ? savedOverride : draft
  const dirty = nextOverride !== savedOverride
  const saving = save.status === 'saving'
  useSettingsDirtyRegistration('prompts', dirty)

  useEffect(() => {
    let active = true
    Promise.all([api.getSystemPromptScenes(), api.getConfig()])
      .then(([{ scenes }, config]) => {
        if (active) setLoad({ status: 'ready', scenes, overrides: config.system_prompts ?? {} })
      })
      .catch((error: unknown) => {
        if (active) setLoad({ status: 'error', message: getErrorMessage(error, '读取失败') })
      })
    return () => { active = false }
  }, [loadAttempt])

  if (load.status === 'loading') {
    return <p role="status" className="p-4 text-sm text-text-muted">正在加载系统提示词…</p>
  }

  if (load.status === 'error') {
    return (
      <div className="space-y-3 p-4">
        <p role="alert" className="text-sm text-accent-red">无法加载系统提示词。{load.message}</p>
        <button
          type="button"
          className="rounded-lg border border-bg-hover px-3 py-1.5 text-xs text-text-primary hover:bg-bg-hover"
          onClick={() => {
            setLoad({ status: 'loading' })
            setLoadAttempt((attempt) => attempt + 1)
          }}
        >
          重试
        </button>
      </div>
    )
  }

  if (!scene) {
    return <p className="p-4 text-sm text-text-muted">暂无可编辑的提示词场景。</p>
  }

  const { scenes } = load
  const groups = new Map<string, SystemPromptScene[]>()
  for (const item of scenes) {
    const group = groups.get(item.group)
    if (group) group.push(item)
    else groups.set(item.group, [item])
  }

  function changeScene(id: string) {
    if (dirty && !window.confirm('当前场景有未保存更改，切换后会丢失。')) return
    setSelectedId(id)
    setDraft(undefined)
    setSave({ status: 'idle' })
  }

  function editDraft(value: string | null) {
    setDraft(savedOverride === null && value === scene.default_prompt ? null : value)
    setSave({ status: 'idle' })
  }

  async function savePrompt() {
    if (!dirty || saving) return
    setSave({ status: 'saving' })
    onSavingChange(true)
    try {
      const config: AppConfig = await api.updateConfig({ system_prompts: { [scene.id]: nextOverride } })
      useInterviewStore.getState().setConfig(config)
      setLoad({ status: 'ready', scenes, overrides: config.system_prompts ?? {} })
      setDraft(undefined)
      setSave({ status: 'saved' })
    } catch (error: unknown) {
      setSave({ status: 'error', message: getErrorMessage(error, '保存失败') })
    } finally {
      onSavingChange(false)
    }
  }

  return (
    <div className="p-4">
      <GradientCard className="space-y-4 p-4">
        <div className="space-y-1.5 sm:max-w-sm">
          <label htmlFor="system-prompt-scene" className="block text-xs font-medium text-text-secondary">
            使用场景
          </label>
          <select
            id="system-prompt-scene"
            value={scene.id}
            onChange={(event) => changeScene(event.target.value)}
            disabled={saving}
            className="input-field"
          >
            {Array.from(groups, ([group, items]) => (
              <optgroup key={group} label={group}>
                {items.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
              </optgroup>
            ))}
          </select>
        </div>

        <dl className="grid gap-3 rounded-lg border border-bg-hover/60 bg-bg-primary/35 px-3 py-2.5 text-xs leading-relaxed sm:grid-cols-[1fr_auto] sm:gap-x-6">
          <div className="min-w-0">
            <dt className="font-medium text-text-secondary">生效入口</dt>
            <dd className="text-text-muted">{scene.entry}</dd>
          </div>
          <div>
            <dt className="font-medium text-text-secondary">输出 Token 上限</dt>
            <dd className="text-text-muted">{scene.max_output_tokens} tokens</dd>
          </div>
          {scene.output_contract && (
            <div className="min-w-0 sm:col-span-2">
              <dt className="font-medium text-text-secondary">输出结构要求</dt>
              <dd className="whitespace-pre-wrap break-words text-text-muted">{scene.output_contract}</dd>
            </div>
          )}
        </dl>

        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <label htmlFor="system-prompt-text" className="text-xs font-medium text-text-secondary">
              系统提示词
            </label>
            <StatusBadge status={savedOverride === null ? 'idle' : 'ok'} label={savedOverride === null ? '使用默认' : '已自定义'} />
          </div>
          <textarea
            id="system-prompt-text"
            aria-describedby="system-prompt-hint"
            className="input-field block min-h-56 resize-y leading-relaxed"
            rows={12}
            value={nextOverride ?? scene.default_prompt}
            onChange={(event) => editDraft(event.target.value)}
            readOnly={saving}
          />
          <p id="system-prompt-hint" className="text-xs leading-relaxed text-text-muted">
            应用会按场景自动提供问题、简历等运行时内容。清空文本后保存可清除当前场景的自定义指令。
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div role="status" className="text-xs text-text-muted">
            {saving ? '正在保存当前场景…' : save.status === 'saved' ? '已保存当前场景' : dirty ? '有未保存更改' : null}
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => editDraft(null)}
              disabled={saving || nextOverride === null}
              className="inline-flex items-center gap-1.5 rounded-lg border border-bg-hover px-3 py-2 text-xs font-medium text-text-secondary hover:bg-bg-hover disabled:opacity-50"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              恢复默认
            </button>
            <button
              type="button"
              onClick={() => void savePrompt()}
              disabled={!dirty || saving}
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent-blue px-3 py-2 text-xs font-medium text-white hover:bg-accent-blue/90 disabled:opacity-50"
            >
              <Save className="h-3.5 w-3.5" aria-hidden="true" />
              {saving ? '保存中…' : '保存当前场景'}
            </button>
          </div>
        </div>
        {save.status === 'error' && <p role="alert" className="text-xs text-accent-red">保存失败。{save.message}</p>}
      </GradientCard>
    </div>
  )
}
