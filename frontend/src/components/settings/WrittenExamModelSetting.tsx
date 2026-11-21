import { useRef, useState } from 'react'
import { BrainCircuit, Save } from 'lucide-react'
import { getErrorMessage } from '@/lib/api'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { useInterviewStore } from '@/stores/configStore'
import ModelChecklist, { type CatalogModel } from './ModelChecklist'
import { GradientCard, SaveStateBadge, Section, useSettingsDirtyRegistration, type SaveState } from './shared'

export default function WrittenExamModelSetting({ disabled, onSavingChange }: {
  disabled: boolean
  onSavingChange: (saving: boolean) => void
}) {
  const config = useInterviewStore((state) => state.config)
  const [draft, setDraft] = useState<string[]>()
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [error, setError] = useState<string | null>(null)
  const savingRef = useRef(false)
  const savedIds = config?.written_exam_model_ids ?? (config?.written_exam_model_id ? [config.written_exam_model_id] : [])
  const modelIds = draft ?? savedIds
  const models = (config?.models ?? []).filter(
    (model): model is CatalogModel => Boolean(model.id) && model.enabled !== false
      && model.has_api_key === true && model.supports_vision,
  )
  const dirty = modelIds.length !== savedIds.length || modelIds.some((id, index) => id !== savedIds[index])
  const saving = saveState === 'saving'
  useSettingsDirtyRegistration('exam-model', dirty)

  async function save() {
    if (disabled || !config || !dirty || savingRef.current) return
    savingRef.current = true
    setSaveState('saving')
    setError(null)
    onSavingChange(true)
    try {
      await updateConfigAndRefresh({ written_exam_model_ids: modelIds })
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
        <ModelChecklist
          models={models}
          selectedIds={modelIds}
          onChange={(ids) => {
            setDraft(ids)
            setSaveState('idle')
            setError(null)
          }}
          title="笔试答题模型"
          description="首项优先。请求失败时，按笔试队列顺序尝试其他可用模型。"
          requireVision
          disabled={!config || saving || disabled}
        />
        <div className="flex flex-wrap items-center justify-end gap-3">
          <SaveStateBadge mode="explicit" state={dirty && saveState === 'idle' ? 'dirty' : saveState} error={error} />
          <button
            type="button"
            onClick={() => void save()}
            disabled={!config || !dirty || saving || disabled}
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
