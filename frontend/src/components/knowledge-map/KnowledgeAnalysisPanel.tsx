import { useEffect, useRef, useState, type FormEvent } from 'react'
import ModelPicker from '@/components/ModelPicker'
import { api, type KnowledgeAnalysisState, type KnowledgeSettingsPayload, type KnowledgeThinkEffort } from '@/lib/api'

const EFFORTS: [KnowledgeThinkEffort, string][] = [
  ['off', '关闭'], ['low', '低'], ['medium', '中'], ['high', '高'], ['xhigh', '超高'],
]

function describeError(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback
}

function SettingsForm({
  data, onSaved, onClose,
}: {
  data: KnowledgeSettingsPayload
  onSaved: (data: KnowledgeSettingsPayload) => void
  onClose: () => void
}) {
  const [modelIndex, setModelIndex] = useState(data.settings.model_index)
  const [prompt, setPrompt] = useState(data.settings.system_prompt ?? data.default_system_prompt)
  const [temperature, setTemperature] = useState(String(data.settings.temperature))
  const [maxTokens, setMaxTokens] = useState(String(data.settings.max_tokens))
  const [effort, setEffort] = useState(data.settings.think_effort)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const model = data.models.find(item => item.index === modelIndex)

  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSaving(true)
    setError('')
    try {
      onSaved(await api.saveKnowledgeSettings({
        model_index: modelIndex,
        model_fingerprint: model?.fingerprint ?? null,
        system_prompt: prompt.trim() === data.default_system_prompt ? null : prompt.trim() || null,
        temperature: Number(temperature),
        max_tokens: Number(maxTokens),
        think_effort: model?.supports_think ? effort : 'off',
      }))
    } catch (e) {
      setError(describeError(e, '保存分析设置失败'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <form onSubmit={save} className="border-t border-bg-tertiary pt-3">
      <fieldset disabled={saving} className="space-y-3 text-xs">
        <ModelPicker
          label="分析模型"
          emptyLabel="请选择模型"
          value={modelIndex === null ? '' : String(modelIndex)}
          disabled={saving}
          onChange={(value) => {
            const index = value === '' ? null : Number(value)
            setModelIndex(index)
            if (!data.models.find(item => item.index === index)?.supports_think) setEffort('off')
          }}
          options={data.models.map((item) => ({
            value: String(item.index),
            label: item.model && item.model !== item.name ? `${item.name} · ${item.model}` : item.name,
            supportsVision: item.supports_vision,
            disabledReason: item.available ? undefined : '不可用',
          }))}
        />
        <label className="block space-y-1">
          <span>系统提示词</span>
          <textarea value={prompt} onChange={event => setPrompt(event.target.value)} rows={4} className="input-field w-full" />
        </label>
        <button type="button" onClick={() => setPrompt(data.default_system_prompt)} className="text-accent-blue hover:underline">
          恢复默认提示词
        </button>
        <p className="text-text-muted">{data.output_contract}</p>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="block space-y-1">
            <span>温度</span>
            <input type="number" required min="0" max="2" step="any" value={temperature}
              onChange={event => setTemperature(event.target.value)} className="input-field w-full" />
          </label>
          <label className="block space-y-1">
            <span>最大输出 token</span>
            <input type="number" required min="256" max="32768" step="1" value={maxTokens}
              onChange={event => setMaxTokens(event.target.value)} className="input-field w-full" />
          </label>
          <label className="block space-y-1">
            <span>推理强度</span>
            <select value={model?.supports_think ? effort : 'off'} disabled={!model?.supports_think}
              onChange={event => setEffort(event.target.value as KnowledgeThinkEffort)} className="input-field w-full">
              {EFFORTS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
        </div>
        {error && <p role="alert" className="text-accent-red">{error}</p>}
        <div className="flex gap-2">
          <button type="submit" disabled={modelIndex !== null && !model?.available}
            className="rounded-lg bg-accent-blue px-3 py-1.5 font-medium text-white disabled:opacity-50">
            {saving ? '保存中…' : '保存设置'}
          </button>
          <button type="button" onClick={onClose} className="rounded-lg px-3 py-1.5 hover:bg-bg-tertiary">取消</button>
        </div>
      </fieldset>
    </form>
  )
}

export default function KnowledgeAnalysisPanel({ refresh, onComplete }: { refresh: number; onComplete: () => void }) {
  const [settings, setSettings] = useState<KnowledgeSettingsPayload | null>(null)
  const [analysis, setAnalysis] = useState<KnowledgeAnalysisState | null>(null)
  const [editing, setEditing] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')
  const statusRequest = useRef(0)
  const settingsRequest = useRef(0)
  const wasRunning = useRef(false)
  const job = analysis?.job
  const runningId = job?.status === 'running' ? job.job_id : null
  const ready = settings?.models.some(model => model.index === settings.settings.model_index && model.available)

  useEffect(() => {
    let cancelled = false
    const settingsRead = ++settingsRequest.current
    const statusRead = ++statusRequest.current
    setError('')
    setEditing(false)
    Promise.all([api.knowledgeSettings(), api.knowledgeAnalysisStatus()])
      .then(([data, state]) => {
        if (cancelled) return
        if (settingsRead === settingsRequest.current) setSettings(data)
        if (statusRead === statusRequest.current) setAnalysis(state)
      })
      .catch(e => {
        if (!cancelled) setError(describeError(e, '读取分析设置或进度失败'))
      })
    return () => { cancelled = true }
  }, [refresh])

  useEffect(() => {
    if (!runningId && wasRunning.current) onComplete()
    wasRunning.current = !!runningId
  }, [onComplete, runningId])

  useEffect(() => {
    if (!runningId) return
    let cancelled = false
    let timer: number
    async function poll() {
      const statusRead = ++statusRequest.current
      try {
        const state = await api.knowledgeAnalysisStatus()
        if (cancelled) return
        if (statusRead === statusRequest.current) {
          setError('')
          setAnalysis(state)
          if (state.job?.status !== 'running' || state.job.job_id !== runningId) return
        }
      } catch (e) {
        if (cancelled) return
        if (statusRead === statusRequest.current) setError(describeError(e, '读取分析进度失败，正在重试'))
      }
      timer = window.setTimeout(poll, 750)
    }
    timer = window.setTimeout(poll, 750)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [runningId])

  const start = async () => {
    setStarting(true)
    setError('')
    try {
      const state = await api.startKnowledgeAnalysis()
      ++statusRequest.current
      setAnalysis(state)
      if (!state.job) onComplete()
    } catch (e) {
      setError(describeError(e, '无法启动分析'))
    } finally {
      setStarting(false)
    }
  }

  return (
    <section aria-label="知识点分析" className="space-y-3 rounded-xl border border-bg-tertiary bg-bg-secondary p-3 text-text-primary">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="space-y-1">
          <h3 className="text-xs font-semibold">知识点分析</h3>
          {analysis && <p className="text-xs text-text-muted">待分析 {analysis.pending_count} 条问答</p>}
        </div>
        <div className="flex gap-2">
          <button type="button" aria-expanded={editing} disabled={!settings} onClick={() => setEditing(current => !current)}
            className="rounded-lg border border-bg-tertiary px-3 py-1.5 text-xs hover:bg-bg-tertiary disabled:opacity-50">
            分析设置
          </button>
          <button type="button" onClick={start} disabled={!ready || !analysis?.pending_count || !!runningId || starting || editing}
            className="rounded-lg bg-accent-blue px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50">
            {runningId || starting ? '分析中…' : '开始分析'}
          </button>
        </div>
      </div>
      {settings && !ready && <p className="text-xs text-text-muted">请在分析设置中选择可用模型。</p>}
      {job && (
        <p role="status" className="text-xs text-text-secondary">
          {runningId ? '已处理' : '本批已结束'} {job.succeeded + job.failed + job.skipped}/{job.total} 条
          {' · '}成功 {job.succeeded} 条 · 失败 {job.failed} 条 · 跳过 {job.skipped} 条
          {!runningId && (analysis?.pending_count ?? 0) > 0 && '。可再次点击「开始分析」处理剩余问答。'}
        </p>
      )}
      {error && <p role="alert" className="text-xs text-accent-red">{error}</p>}
      {settings && editing && <SettingsForm data={settings} onClose={() => setEditing(false)} onSaved={data => {
        ++settingsRequest.current
        setSettings(data)
        setEditing(false)
      }} />}
    </section>
  )
}
