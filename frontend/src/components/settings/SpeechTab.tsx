import { useRef, useState } from 'react'
import { AudioWaveform, Cloud, Zap } from 'lucide-react'
import { api } from '@/lib/api'
import { refreshConfig } from '@/lib/configSync'
import { mmpgEngineFormFromConfig, type MmpgEngineForm } from '@/lib/mmpg/settings'
import type { SegmentedSpeechProvider } from '@/lib/speechProviders'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'
import { Field, SaveStateBadge, Section, StatusBadge } from './shared'
import { useSpeechSettingsDraft } from './useSpeechSettingsDraft'
import MmpgSettings from './MmpgSettings'

const doubaoFields = [
  { key: 'doubao_stt_api_key', label: 'API Key（新版控制台）', secret: true, hint: '新版凭据优先，留空时可使用旧版凭据。' },
  { key: 'doubao_stt_app_id', label: 'App ID（旧版控制台）' },
  { key: 'doubao_stt_access_token', label: 'Access Token（旧版控制台）', secret: true },
  { key: 'doubao_stt_resource_id', label: 'Resource ID', defaultValue: 'volc.seedasr.sauc.duration' },
  { key: 'doubao_stt_boosting_table_id', label: '热词表 ID（可选）' },
] as const

const genericFields = [
  { key: 'generic_stt_api_base_url', label: 'Base URL', hint: '填写 OpenAI 兼容接口地址，例如 https://api.example.com/v1。' },
  { key: 'generic_stt_api_key', label: 'API Key', secret: true },
  { key: 'generic_stt_model', label: 'Model' },
  {
    key: 'generic_stt_custom_headers',
    label: '自定义 Header（可选）',
    secret: true,
    multiline: true,
    hint: '使用 JSON 对象，或每行填写一组 Key: Value；留空保留已保存配置。',
  },
] as const

type PublicEngineFieldKey =
  | 'doubao_stt_app_id'
  | 'doubao_stt_resource_id'
  | 'doubao_stt_boosting_table_id'
  | 'generic_stt_api_base_url'
  | 'generic_stt_model'

type SecretEngineFieldKey =
  | 'doubao_stt_api_key'
  | 'doubao_stt_access_token'
  | 'generic_stt_api_key'
  | 'generic_stt_custom_headers'

type EngineField =
  | {
      key: PublicEngineFieldKey
      label: string
      hint?: string
      secret?: false
      multiline?: boolean
      defaultValue?: string
    }
  | {
      key: SecretEngineFieldKey
      label: string
      hint?: string
      secret: true
      multiline?: boolean
      defaultValue?: string
    }

type EngineForm = Partial<Record<PublicEngineFieldKey, string>> & Partial<Record<SecretEngineFieldKey, string | null>>

const saveButton = 'rounded-lg border border-accent-blue/30 bg-accent-blue/15 px-3 py-1.5 text-xs font-medium text-accent-blue disabled:opacity-50'

function ProviderStatus({ configured }: { configured: boolean }) {
  return <StatusBadge status={configured ? 'ok' : 'idle'} label={configured ? '已配置' : '待配置'} />
}

function SegmentedEngineSettings({
  provider, title, fields, config,
}: { provider: SegmentedSpeechProvider; title: string; fields: readonly EngineField[]; config: AppConfig }) {
  const saved = Object.fromEntries(
    fields
      .filter((field): field is Extract<EngineField, { secret?: false }> => !field.secret)
      .map(({ key, defaultValue }) => [key, config[key] ?? defaultValue ?? '']),
  ) as EngineForm
  const draft = useSpeechSettingsDraft<EngineForm>('speech-' + provider, saved)
  const [testResult, setTestResult] = useState<{ ok: boolean; detail?: string; text?: string } | null>(null)
  const [testing, setTesting] = useState(false)
  const testRef = useRef(false)

  async function testEngine() {
    if (testRef.current || draft.saving) return
    testRef.current = true
    setTesting(true)
    setTestResult(null)
    try {
      if (!(await draft.save())) return
      setTestResult(await api.sttTest({ provider }))
    } catch (error) {
      setTestResult({ ok: false, detail: error instanceof Error ? error.message : '测试失败' })
    } finally {
      testRef.current = false
      setTesting(false)
    }
  }

  const busy = draft.saving || testing
  return (
    <Section title={title} icon={<Cloud size={14} />} keywords="云端 语音 识别 分段 STT 凭据 API Key 热词 测试">
      <ProviderStatus configured={Boolean(config.stt_configured?.[provider])} />
      <fieldset disabled={busy} className="grid gap-3 sm:grid-cols-2">
        {fields.map((field) => {
          const { key, label, hint, multiline } = field
          if (field.secret) {
            const configured = Boolean(config[`${field.key}_configured`])
            const cleared = draft.form[field.key] === null
            const value = draft.form[field.key] ?? ''
            const placeholder = cleared ? '保存后清除' : configured ? '已配置 · 留空保留' : '未配置'
            const clearButton = (configured || value || cleared) && (
              <button
                type="button"
                className="text-xs text-accent-blue shrink-0"
                aria-label={cleared ? `撤销清除 ${label}` : `清除 ${label}`}
                onClick={() => {
                  draft.edit({ [field.key]: cleared || !configured ? undefined : null })
                  setTestResult(null)
                }}
              >
                {cleared ? '撤销' : '清除'}
              </button>
            )
            return (
              <Field label={label} htmlFor={key} hint={hint} key={key}>
                {multiline ? (
                  <div className="space-y-1.5">
                    <textarea
                      id={key}
                      className="input-field min-h-20 font-mono"
                      autoComplete="off"
                      value={value}
                      placeholder={placeholder}
                      onChange={(event) => {
                        draft.edit({ [field.key]: event.target.value || undefined })
                        setTestResult(null)
                      }}
                    />
                    {clearButton && <div className="flex justify-end">{clearButton}</div>}
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <input
                      id={key}
                      className="input-field"
                      type="password"
                      autoComplete="off"
                      value={value}
                      placeholder={placeholder}
                      onChange={(event) => {
                        draft.edit({ [field.key]: event.target.value || undefined })
                        setTestResult(null)
                      }}
                    />
                    {clearButton}
                  </div>
                )}
              </Field>
            )
          }
          return (
            <Field label={label} htmlFor={key} hint={hint} key={key}>
              {multiline ? (
                <textarea
                  id={key}
                  className="input-field min-h-20 font-mono"
                  value={draft.form[field.key] ?? ''}
                  onChange={(event) => { draft.edit({ [field.key]: event.target.value }); setTestResult(null) }}
                />
              ) : (
                <input
                  id={key}
                  className="input-field"
                  type="text"
                  value={draft.form[field.key] ?? ''}
                  onChange={(event) => { draft.edit({ [field.key]: event.target.value }); setTestResult(null) }}
                />
              )}
            </Field>
          )
        })}
      </fieldset>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <SaveStateBadge mode="explicit" state={draft.state} error={draft.error} />
        <button type="button" className={saveButton} disabled={!draft.dirty || busy} onClick={() => void draft.save()}>
          {draft.saving ? '保存中…' : '保存'}
        </button>
        <button type="button" className={saveButton} disabled={busy} onClick={() => void testEngine()}>
          {testing ? '测试中…' : '保存并测试'}
        </button>
      </div>
      {testResult && (
        <div role="status" className="flex flex-wrap items-center gap-2 text-xs text-text-secondary">
          <StatusBadge status={testResult.ok ? 'ok' : 'error'} label={testResult.ok ? '连接成功' : '测试失败'} />
          {testResult.text || testResult.detail}
        </div>
      )}
    </Section>
  )
}

function validateMmpg(patch: Partial<MmpgEngineForm>) {
  const { silence_ms, expire_time_ms } = patch
  if (silence_ms !== undefined && (!Number.isInteger(silence_ms) || silence_ms < 500 || silence_ms > 2999)) {
    throw new Error('MMPG 判停静音时间必须为 500–2999 毫秒内的整数。')
  }
  if (expire_time_ms != null && (!Number.isSafeInteger(expire_time_ms) || expire_time_ms < 1)) {
    throw new Error('MMPG 识别过期时间必须为正整数或留空。')
  }
}

function MmpgEngineSettings({ stored, configured }: { stored: AppConfig['mmpg']; configured: boolean }) {
  const saved = mmpgEngineFormFromConfig(stored)
  const draft = useSpeechSettingsDraft('speech-mmpg', saved)

  const save = () => draft.save((changes) => {
    validateMmpg(changes)
    return { mmpg: changes }
  })
  async function importProbe() {
    await api.importMmpgProbe()
    await refreshConfig()
    draft.reset()
  }

  return (
    <Section title="豆包 MMPG" icon={<AudioWaveform size={14} />} keywords="火山 RTC 智能判停 AIVAD 二遍识别 Token 凭据 语音 测试">
      <ProviderStatus configured={configured} />
      <MmpgSettings form={draft.form} stored={stored} onChange={draft.edit} onImport={importProbe} onSave={save} saving={draft.saving} />
      <div className="flex flex-wrap items-center justify-end gap-2">
        <SaveStateBadge mode="explicit" state={draft.state} error={draft.error} />
        <button type="button" className={saveButton} disabled={!draft.dirty || draft.saving} onClick={() => void save()}>
          {draft.saving ? '保存中…' : '保存'}
        </button>
      </div>
    </Section>
  )
}

export default function SpeechTab() {
  const config = useInterviewStore((state) => state.config)
  if (!config) return <p role="status" className="p-5 text-xs text-text-muted">正在加载语音配置…</p>

  return (
    <div className="space-y-4 p-5 pb-8">
      <p className="flex items-center gap-2 text-xs text-text-secondary">
        <Zap size={14} aria-hidden="true" /> 在这里配置和测试引擎，再到「实时辅助」为面试官和我的回答选择引擎。
      </p>
      <SegmentedEngineSettings provider="doubao" title="火山原生豆包" fields={doubaoFields} config={config} />
      <SegmentedEngineSettings provider="generic" title="OpenAI 兼容 ASR" fields={genericFields} config={config} />
      <MmpgEngineSettings stored={config.mmpg} configured={Boolean(config.stt_configured?.doubao_mmpg)} />
    </div>
  )
}
