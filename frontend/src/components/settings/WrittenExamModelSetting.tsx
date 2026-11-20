import { useRef, useState } from 'react'
import { BrainCircuit, Save } from 'lucide-react'
import { getErrorMessage } from '@/lib/api'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { useInterviewStore } from '@/stores/configStore'
import type { ModelInfo } from '@/stores/slices/types'
import { Field, GradientCard, SaveStateBadge, Section, useSettingsDirtyRegistration, type SaveState } from './shared'

function unavailableReason(model: ModelInfo): string | undefined {
  if (!model.supports_vision) return '该模型不支持图片作为输入'
  if (model.enabled === false) return '已停用'
  if (model.has_api_key !== true) return '未配置 API Key'
}

export default function WrittenExamModelSetting({ disabled, onSavingChange }: {
  disabled: boolean
  onSavingChange: (saving: boolean) => void
}) {
  const config = useInterviewStore((state) => state.config)
  const [draft, setDraft] = useState<string | null>()
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [error, setError] = useState<string | null>(null)
  const savingRef = useRef(false)
  const savedId = config?.written_exam_model_id ?? null
  const modelId = draft === undefined ? savedId : draft
  const models = (config?.models ?? []).filter((model): model is ModelInfo & { id: string } => Boolean(model.id))
  const selectedModel = models.find((model) => model.id === modelId)
  const dirty = modelId !== savedId
  const saving = saveState === 'saving'
  const canSave = !modelId || (selectedModel && !unavailableReason(selectedModel))
  useSettingsDirtyRegistration('exam-model', dirty)

  async function save() {
    if (disabled || !dirty || !canSave || savingRef.current) return
    savingRef.current = true
    setSaveState('saving')
    setError(null)
    onSavingChange(true)
    try {
      await updateConfigAndRefresh({ written_exam_model_id: modelId })
      setDraft(undefined)
      setSaveState('saved')
    } catch (failure) {
      setSaveState('error')
      setError(getErrorMessage(failure, '保存笔试模型失败'))
    } finally {
      savingRef.current = false
      onSavingChange(false)
    }
  }

  return (
    <Section title="答题模型" icon={<BrainCircuit className="h-4 w-4" />} keywords="笔试答题模型 图片 识图 首选">
      <GradientCard className="space-y-3 p-4">
        <Field label="笔试答题模型" htmlFor="written-exam-model">
          <select
            id="written-exam-model"
            aria-describedby="written-exam-model-hint"
            className="input-field"
            value={modelId ?? ''}
            disabled={!config || saving || disabled}
            onChange={(event) => {
              setDraft(event.target.value || null)
              setSaveState('idle')
              setError(null)
            }}
          >
            <option value="">未选择模型</option>
            {models.map((model) => {
              const reason = unavailableReason(model)
              return <option key={model.id} value={model.id} disabled={Boolean(reason)}>{model.name}{reason && `（${reason}）`}</option>
            })}
          </select>
        </Field>
        <p id="written-exam-model-hint" className="text-[11px] text-text-muted">
          笔试需要先选择支持图片的模型。首选请求失败时，按面试答题队列顺序尝试可用的识图模型。
        </p>
        <div className="flex flex-wrap items-center justify-end gap-3">
          <SaveStateBadge mode="explicit" state={dirty && saveState === 'idle' ? 'dirty' : saveState} error={error} />
          <button
            type="button"
            onClick={() => void save()}
            disabled={!dirty || !canSave || saving || disabled}
            className="inline-flex items-center gap-1.5 rounded-lg border border-accent-blue/30 bg-accent-blue/15 px-3 py-2 text-xs font-medium text-accent-blue hover:bg-accent-blue/25 disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5" aria-hidden />
            {saving ? '保存中…' : '保存笔试模型'}
          </button>
        </div>
      </GradientCard>
    </Section>
  )
}
