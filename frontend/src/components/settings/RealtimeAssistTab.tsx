import { useState } from 'react'
import { AlignVerticalSpaceAround, BrainCircuit, FileText, GripVertical, LayoutGrid, Save } from 'lucide-react'
import { closestCenter, DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors, type DragEndEvent } from '@dnd-kit/core'
import { arrayMove, SortableContext, sortableKeyboardCoordinates, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { getAssistModelEntries, getAssistScreenshotModel, isConfiguredModel } from '@/lib/assistModels'
import { getErrorMessage } from '@/lib/api'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { useInterviewStore } from '@/stores/configStore'
import type { ModelInfo } from '@/stores/slices/types'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'
import { Field, GradientCard, SaveStateBadge, Section, ToggleSwitch, useSettingsDirtyRegistration } from './shared'
import type { SaveState } from './shared'
import SpeechRolesSettings from './SpeechRolesSettings'
import SystemPromptsTab from './SystemPromptsTab'

const answerLayouts = [
  { key: 'cards', Icon: LayoutGrid, label: '卡片', hint: '每题独立滚动' },
  { key: 'stream', Icon: AlignVerticalSpaceAround, label: '流式', hint: '连续阅读所有回答' },
] as const

function AnswerLayout() {
  const layout = useUiPrefsStore((s) => s.answerPanelLayout)
  const setLayout = useUiPrefsStore((s) => s.setAnswerPanelLayout)

  return (
    <Section title="回答样式" icon={<LayoutGrid className="h-4 w-4" />} keywords="卡片 流式 布局 回答">
      <div className="grid grid-cols-2 gap-2">
        {answerLayouts.map(({ key, Icon, label, hint }) => (
          <button key={key} type="button" aria-pressed={layout === key} onClick={() => setLayout(key)}
            className={`flex flex-col items-start gap-1.5 rounded-xl border p-3 text-left transition-colors ${layout === key
              ? 'border-accent-blue/50 bg-accent-blue/10 text-accent-blue'
              : 'border-bg-hover bg-bg-tertiary/30 hover:border-accent-blue/30'}`}>
            <Icon className="h-4 w-4" aria-hidden />
            <span className="text-xs font-medium">{label}</span>
            <span className="text-[11px] text-text-muted">{hint}</span>
          </button>
        ))}
      </div>
    </Section>
  )
}

type CatalogModel = ModelInfo & { id: string }

function ModelChoice({ model, selected, onToggle, disabled }: {
  model: CatalogModel
  selected: boolean
  onToggle: (id: string) => void
  disabled: boolean
}) {
  const status = model.enabled === false ? '已停用' : model.has_api_key === false ? '未配置 API Key' : null
  return (
    <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 py-2.5">
      <input type="checkbox" aria-label={`用于答题 ${model.name}`} checked={selected}
        disabled={disabled} onChange={() => onToggle(model.id)} className="accent-accent-blue" />
      <span className="truncate text-xs font-medium text-text-primary">{model.name}</span>
      {model.supports_vision && <span className="text-[10px] text-text-muted">识图</span>}
      {status && <span className="text-[10px] text-accent-amber">{status}</span>}
    </label>
  )
}

function SortableModel({ model, onToggle, disabled }: {
  model: CatalogModel
  onToggle: (id: string) => void
  disabled: boolean
}) {
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } = useSortable({
    id: model.id,
    disabled,
  })
  return (
    <li ref={setNodeRef} style={{ transform: CSS.Transform.toString(transform), transition }}
      className={`relative flex items-center gap-2 rounded-lg border border-bg-hover/70 bg-bg-secondary px-2.5 ${isDragging ? 'z-10 shadow-lg' : ''}`}>
      <ModelChoice model={model} selected onToggle={onToggle} disabled={disabled} />
      <button {...attributes} {...listeners} ref={setActivatorNodeRef} type="button" disabled={disabled}
        aria-label={`排序 ${model.name}`} title="拖动调整优先级"
        className="cursor-grab touch-none rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary active:cursor-grabbing disabled:opacity-50">
        <GripVertical className="h-4 w-4" aria-hidden />
      </button>
    </li>
  )
}

interface AnswerModelDraft {
  assist_model_ids: string[]
  assist_screenshot_model_id: string | null
  assist_think_mode: boolean
  assist_think_effort: string
  max_parallel_answers: number
}

function AnswerModels({ onOpenModelConfig }: { onOpenModelConfig: () => void }) {
  const config = useInterviewStore((s) => s.config)
  const [draft, setDraft] = useState<Partial<AnswerModelDraft>>({})
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [error, setError] = useState<string | null>(null)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )
  const catalog = (config?.models ?? []).filter((model): model is CatalogModel => Boolean(model.id))
  const modelById = new Map(catalog.map((model) => [model.id, model]))
  const saved: AnswerModelDraft = {
    assist_model_ids: config?.assist_model_ids ?? getAssistModelEntries(config).flatMap(({ model }) => model.id ? [model.id] : []),
    assist_screenshot_model_id: config?.assist_screenshot_model_id === undefined
      ? getAssistScreenshotModel(config)?.id ?? null : config.assist_screenshot_model_id,
    assist_think_mode: config?.assist_think_mode ?? config?.think_mode ?? false,
    assist_think_effort: config?.assist_think_effort ?? config?.think_effort ?? 'off',
    max_parallel_answers: config?.max_parallel_answers ?? 2,
  }
  const current = { ...saved, ...draft }
  const changed = Object.fromEntries(Object.entries(draft).filter(([key, value]) => (
    JSON.stringify(saved[key as keyof AnswerModelDraft]) !== JSON.stringify(value)
  ))) as Partial<AnswerModelDraft>
  const dirty = Object.keys(changed).length > 0
  const saving = saveState === 'saving'
  useSettingsDirtyRegistration('assist-models', dirty)

  function change(update: Partial<AnswerModelDraft>) {
    setDraft((previous) => ({ ...previous, ...update }))
    setSaveState('idle')
    setError(null)
  }

  function toggleModel(id: string) {
    const selected = current.assist_model_ids
    change({ assist_model_ids: selected.includes(id) ? selected.filter((item) => item !== id) : [...selected, id] })
  }

  function reorderModels({ active, over }: DragEndEvent) {
    if (!over || active.id === over.id) return
    const selected = current.assist_model_ids
    const from = selected.indexOf(String(active.id))
    const to = selected.indexOf(String(over.id))
    if (from >= 0 && to >= 0) change({ assist_model_ids: arrayMove(selected, from, to) })
  }

  async function save() {
    if (!dirty || saving) return
    setSaveState('saving')
    setError(null)
    try {
      await updateConfigAndRefresh(changed)
      setDraft({})
      setSaveState('saved')
    } catch (err) {
      setSaveState('error')
      setError(getErrorMessage(err, '保存答题模型失败'))
    }
  }

  const selected = current.assist_model_ids.flatMap((id) => {
    const model = modelById.get(id)
    return model ? [model] : []
  })
  const unselected = catalog.filter((model) => !current.assist_model_ids.includes(model.id))
  const screenshotOptions = catalog.filter((model) => model.supports_vision || model.id === current.assist_screenshot_model_id)

  return (
    <Section title="答题模型" icon={<BrainCircuit className="h-4 w-4" />} keywords="模型 队列 降级 截图 识图 并行 Think">
      <GradientCard className="space-y-4 p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-xs font-semibold text-text-primary">答题可用模型</h3>
            <p className="mt-1 text-[11px] text-text-muted">首项优先。粘图从此队列使用支持识图的模型。</p>
          </div>
          <button type="button" onClick={onOpenModelConfig} className="shrink-0 text-xs text-accent-blue hover:underline">
            管理模型
          </button>
        </div>
        {selected.length > 0 ? (
          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={reorderModels}>
            <SortableContext items={selected.map(({ id }) => id)} strategy={verticalListSortingStrategy}>
              <ol aria-label="答题可用模型" className="space-y-1.5">
                {selected.map((model) => <SortableModel key={model.id} model={model} onToggle={toggleModel} disabled={saving} />)}
              </ol>
            </SortableContext>
          </DndContext>
        ) : <p className="rounded-lg bg-bg-tertiary/40 p-3 text-xs text-text-muted">请勾选要参与答题的模型。</p>}

        {unselected.length > 0 && (
          <div className="space-y-1">
            <h3 className="text-xs font-semibold text-text-primary">未勾选模型</h3>
            <ul className="divide-y divide-bg-hover/60 px-2.5">
              {unselected.map((model) => (
                <li key={model.id}>
                  <ModelChoice model={model} selected={false} onToggle={toggleModel} disabled={saving} />
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="截图答题模型" htmlFor="assist-screenshot-model" hint="用于面试时的电脑截图">
            <select id="assist-screenshot-model" className="input-field" disabled={saving}
              value={current.assist_screenshot_model_id ?? ''}
              onChange={(event) => change({ assist_screenshot_model_id: event.target.value || null })}>
              <option value="">未选择模型</option>
              {screenshotOptions.map((model) => (
                <option key={model.id} value={model.id} disabled={!model.supports_vision || !isConfiguredModel(model)}>
                  {model.name}{!model.supports_vision || !isConfiguredModel(model) ? '（不可用）' : ''}
                </option>
              ))}
            </select>
          </Field>
          <Field label="并行生成路数" htmlFor="assist-parallel-answers">
            <select id="assist-parallel-answers" className="input-field" disabled={saving}
              value={current.max_parallel_answers}
              onChange={(event) => change({ max_parallel_answers: Number(event.target.value) })}>
              {Array.from({ length: 8 }, (_, index) => index + 1).map((count) => <option key={count}>{count}</option>)}
            </select>
          </Field>
        </div>
        <div className="flex items-center justify-between border-t border-bg-hover/60 pt-3">
          <span className="text-xs font-medium text-text-primary">会场 Think</span>
          <ToggleSwitch title="会场 Think" labelOn="会场 Think" checked={current.assist_think_mode} disabled={saving}
            onChange={(enabled) => change({ assist_think_mode: enabled, assist_think_effort: enabled ? 'xhigh' : 'off' })} />
        </div>
        <div className="flex flex-wrap items-center justify-end gap-3">
          <SaveStateBadge mode="explicit" state={dirty && saveState === 'idle' ? 'dirty' : saveState} error={error} />
          <button type="button" onClick={() => void save()} disabled={!dirty || saving}
            className="inline-flex items-center gap-1.5 rounded-lg border border-accent-blue/30 bg-accent-blue/15 px-3 py-2 text-xs font-medium text-accent-blue hover:bg-accent-blue/25 disabled:opacity-50">
            <Save className="h-3.5 w-3.5" aria-hidden />
            {saving ? '保存中…' : '保存答题模型设置'}
          </button>
        </div>
      </GradientCard>
    </Section>
  )
}

export default function RealtimeAssistTab({ onOpenSpeechConfig, onOpenModelConfig, onSavingPromptChange }: {
  onOpenSpeechConfig: () => void
  onOpenModelConfig: () => void
  onSavingPromptChange: (saving: boolean) => void
}) {
  return (
    <div className="space-y-5 p-4 pb-8 sm:p-5">
      <AnswerLayout />
      <AnswerModels onOpenModelConfig={onOpenModelConfig} />
      <Section title="系统提示词与生成参数" icon={<FileText className="h-4 w-4" />}
        keywords="系统 提示词 温度 temperature max token 上限 语音 手动 电脑截图">
        <SystemPromptsTab scope="assist" onSavingChange={onSavingPromptChange} />
      </Section>
      <SpeechRolesSettings onOpenSpeechConfig={onOpenSpeechConfig} />
    </div>
  )
}
