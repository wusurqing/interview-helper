import { AudioWaveform, Mic, MonitorSpeaker } from 'lucide-react'
import { isSegmentedSpeech, speechProvider, speechProviders, type SpeechProvider } from '@/lib/speechProviders'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'
import { Field, SaveStateBadge, Section, StatusBadge, ToggleSwitch } from './shared'
import { useSpeechSettingsDraft } from './useSpeechSettingsDraft'

type NumberField = { label: string; value: number; min: number; max: number; step: number; hint?: string }

const fields = {
  silence_threshold: { label: '静音阈值', value: 0.01, min: 0.001, max: 0.1, step: 0.001, hint: '环境噪声较大时可以适当调高。' },
  silence_duration: { label: '静音时长（秒）', value: 1.2, min: 0.5, max: 10, step: 0.1 },
  assist_vad_max_speech_sec: { label: 'VAD 单段最长时长（秒）', value: 18, min: 6, max: 60, step: 1 },
  assist_transcription_merge_gap_sec: { label: '转写合并间隔（秒）', value: 2, min: 0, max: 15, step: 0.1, hint: '设为 0 时逐段送出转写，自动答题仍经过问句确认。' },
  assist_transcription_merge_max_sec: { label: '转写合并最长等待（秒）', value: 12, min: 1, max: 120, step: 0.1, hint: '启用转写合并时，从首段起等待的最长时间。' },
  assist_asr_confirm_window_sec: { label: '问句确认窗口（秒）', value: 0.45, min: 0, max: 5, step: 0.05 },
  assist_asr_group_max_wait_sec: { label: '问题组最长等待（秒）', value: 1.2, min: 0.2, max: 8, step: 0.1 },
  transcription_min_sig_chars: { label: '最少有效字', value: 2, min: 1, max: 50, step: 1 },
  candidate_context_wait_ms: { label: '手动提问等待转写（毫秒）', value: 200, min: 0, max: 2000, step: 1, hint: '设为 0 时立即提交。' },
  candidate_context_max_chars: { label: '手动参考最多字数', value: 900, min: 100, max: 4000, step: 1 },
  candidate_context_min_chars: { label: '手动参考最少字数', value: 6, min: 1, max: 100, step: 1 },
  answer_delay_ms: { label: '答题前等待时长（毫秒）', value: 1500, min: 0, max: Number.MAX_SAFE_INTEGER, step: 1, hint: '判停后等待续说，设为 0 时立即提交。' },
  followup_window_ms: { label: '回答启动后的续说窗口（毫秒）', value: 1500, min: 0, max: Number.MAX_SAFE_INTEGER, step: 1, hint: '窗口内续说会合并问题并重新生成，设为 0 时关闭该窗口。' },
} satisfies Record<string, NumberField>

type NumberKey = keyof typeof fields
type RoleSettings = Record<NumberKey, number> & {
  stt_provider: SpeechProvider | ''
  candidate_stt_provider: SpeechProvider | ''
  candidate_asr_enabled: boolean
  candidate_mic_compatibility_mode: boolean
  candidate_context_enabled: boolean
  auto_detect: boolean
}

const numberKeys = Object.keys(fields) as NumberKey[]
const segmentedTiming = [
  'assist_vad_max_speech_sec', 'assist_transcription_merge_gap_sec', 'assist_transcription_merge_max_sec',
  'assist_asr_confirm_window_sec', 'assist_asr_group_max_wait_sec',
] as const

function normalizeNumber(key: NumberKey, value: number | undefined) {
  const field: NumberField = fields[key]
  const requested = value !== undefined && Number.isFinite(value) ? value : field.value
  const bounded = Math.max(field.min, Math.min(field.max, requested))
  return field.step === 1 ? Math.floor(bounded) : bounded
}

function roleSettingsFrom(config: AppConfig): RoleSettings {
  const values = Object.fromEntries(numberKeys.map((key) => {
    const value = key === 'answer_delay_ms' || key === 'followup_window_ms' ? config.mmpg?.[key] : config[key]
    return [key, normalizeNumber(key, value)]
  })) as Record<NumberKey, number>
  const candidate = speechProvider(config.candidate_stt_provider)
  return {
    ...values,
    stt_provider: speechProvider(config.stt_provider),
    candidate_stt_provider: candidate,
    candidate_asr_enabled: Boolean(candidate && config.candidate_asr_enabled),
    candidate_mic_compatibility_mode: config.candidate_mic_compatibility_mode ?? true,
    candidate_context_enabled: config.candidate_context_enabled ?? true,
    auto_detect: config.auto_detect ?? true,
  }
}

function rolePatch(edits: Partial<RoleSettings>) {
  const { answer_delay_ms, followup_window_ms, ...role } = edits
  const body: Record<string, unknown> = { ...role }
  const mmpg: Record<string, number> = {}
  for (const key of numberKeys) {
    if (edits[key] === undefined) continue
    const value = normalizeNumber(key, edits[key])
    if (key === 'answer_delay_ms' || key === 'followup_window_ms') {
      mmpg[key] = value
    } else {
      body[key] = value
    }
  }
  if (Object.keys(mmpg).length) body.mmpg = mmpg
  return body
}

function EngineSelect({ id, label, value, onChange }: {
  id: string
  label: string
  value: SpeechProvider | ''
  onChange: (provider: SpeechProvider | '') => void
}) {
  return (
    <Field label={label} htmlFor={id}>
      <select id={id} className="input-field" value={value} onChange={(event) => onChange(speechProvider(event.target.value))}>
        <option value="">待选择引擎</option>
        {speechProviders.map((provider) => <option key={provider.id} value={provider.id}>{provider.label}</option>)}
      </select>
    </Field>
  )
}

function EngineReadiness({ provider, configured, running, onOpen }: {
  provider: SpeechProvider | ''
  configured: boolean
  running: boolean
  onOpen?: () => void
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {running ? (
        <StatusBadge status="ok" label="识别中" />
      ) : (
        <StatusBadge status={configured ? 'ok' : 'idle'} label={configured ? '引擎已配置' : provider ? '引擎待配置' : '待选择引擎'} />
      )}
      {provider && !configured && onOpen && (
        <button type="button" onClick={onOpen} className="text-xs text-accent-blue">打开语音配置</button>
      )}
    </div>
  )
}

function SpeechSettingsForm({ config, onOpenSpeechConfig }: { config: AppConfig; onOpenSpeechConfig?: () => void }) {
  const draft = useSpeechSettingsDraft('speech-roles', roleSettingsFrom(config))
  const { form, edit } = draft
  const isRecording = useInterviewStore((state) => state.isRecording)
  const interviewerReady = useInterviewStore((state) => state.sttLoaded && state.sttActiveProvider === form.stt_provider)
  const candidateReady = useInterviewStore((state) => state.candidateSttLoaded && state.candidateSttProvider === form.candidate_stt_provider)
  const platformInfo = useInterviewStore((state) => state.platformInfo)
  const interviewer = form.stt_provider
  const candidate = form.candidate_stt_provider
  const interviewerSegmented = isSegmentedSpeech(interviewer)
  const candidateSegmented = form.candidate_asr_enabled && isSegmentedSpeech(candidate)
  const dualMmpgManual = interviewer === 'doubao_mmpg' && form.candidate_asr_enabled && candidate === 'doubao_mmpg' && config.mmpg?.token_mode === 'manual'

  const timing = (key: NumberKey, hint?: string) => {
    const field: NumberField = fields[key]
    return (
      <Field key={key} label={field.label} htmlFor={key} hint={hint ?? field.hint}>
        <input
          id={key}
          type="number"
          className="input-field"
          min={field.min}
          max={field.max}
          step={field.step}
          value={Number.isFinite(form[key]) ? form[key] : ''}
          onChange={(event) => edit({ [key]: event.target.valueAsNumber })}
        />
      </Field>
    )
  }

  return (
    <div className="space-y-4">
      <Section title="面试官 / 会议音频" icon={<MonitorSpeaker size={14} />} keywords="语音 ASR VAD 节奏 问句 确认 自动检测 自动答题 合并 等待 续说 MMPG">
        <fieldset disabled={draft.saving} className="space-y-3">
          <EngineSelect id="interviewer-speech-provider" label="面试官 ASR 引擎" value={interviewer} onChange={(provider) => edit({ stt_provider: provider })} />
          <EngineReadiness provider={interviewer} configured={Boolean(interviewer && config.stt_configured?.[interviewer])} running={isRecording && interviewerReady} onOpen={onOpenSpeechConfig} />
          {platformInfo?.needs_virtual_device && <p className="whitespace-pre-line text-xs text-text-secondary">{platformInfo.instructions}</p>}
          <div className="flex items-center justify-between gap-2 text-xs text-text-secondary">
            <span>自动检测问题并生成答案</span>
            <ToggleSwitch checked={form.auto_detect} onChange={(auto_detect) => edit({ auto_detect })} labelOn="自动检测问题并生成答案" />
          </div>
          {interviewer && (
            <details className="text-xs text-text-secondary">
              <summary className="cursor-pointer">{interviewerSegmented ? '分段高级设置' : 'MMPG 高级设置'}</summary>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                {interviewerSegmented ? segmentedTiming.map((key) => timing(key)) : (
                  <>
                    {timing('answer_delay_ms')}
                    {timing('followup_window_ms')}
                  </>
                )}
                {timing('transcription_min_sig_chars', interviewerSegmented ? '低于该值的分段转写会被过滤。' : '作为 MMPG 最终转写的问题分类阈值。')}
              </div>
            </details>
          )}
        </fieldset>
      </Section>

      <Section title="我的回答" icon={<Mic size={14} />} keywords="候选人 口述 麦克风 ASR 语音 设备 兼容 上下文">
        <fieldset disabled={draft.saving} className="space-y-3">
          <EngineSelect id="candidate-speech-provider" label="我的回答 ASR 引擎" value={candidate} onChange={(provider) => edit({
            candidate_stt_provider: provider,
            ...(provider ? {} : { candidate_asr_enabled: false }),
          })} />
          <EngineReadiness provider={candidate} configured={Boolean(candidate && config.stt_configured?.[candidate])} running={isRecording && candidateReady && form.candidate_asr_enabled} onOpen={onOpenSpeechConfig} />
          <div className="flex items-center justify-between gap-2 text-xs text-text-secondary">
            <span>我的回答记录（麦克风）</span>
            <ToggleSwitch disabled={!candidate} checked={form.candidate_asr_enabled} onChange={(candidate_asr_enabled) => edit({ candidate_asr_enabled })} labelOn="我的回答记录（麦克风）" />
          </div>
          <p className="text-xs text-text-muted">启用后，在会场里选择独立的麦克风设备。口述转写用于复盘和实时语音上下文。</p>
          {dualMmpgManual && <p role="status" className="text-xs text-accent-amber">两路都选用豆包 MMPG 时，请在语音配置启用 RTC Token 自动签发。</p>}
          {form.candidate_asr_enabled && (
            <details className="text-xs text-text-secondary">
              <summary className="cursor-pointer">麦克风与上下文</summary>
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span>麦克风兼容模式</span>
                  <ToggleSwitch checked={form.candidate_mic_compatibility_mode} onChange={(candidate_mic_compatibility_mode) => edit({ candidate_mic_compatibility_mode })} labelOn="麦克风兼容模式" />
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span>手动提问参考我的回答</span>
                  <ToggleSwitch checked={form.candidate_context_enabled} onChange={(candidate_context_enabled) => edit({ candidate_context_enabled })} labelOn="手动提问参考我的回答" />
                </div>
                {form.candidate_context_enabled && (
                  <div className="grid gap-3 sm:grid-cols-2">
                    {timing('candidate_context_wait_ms')}
                    {timing('candidate_context_min_chars')}
                    {timing('candidate_context_max_chars')}
                  </div>
                )}
              </div>
            </details>
          )}
        </fieldset>
      </Section>

      {(interviewerSegmented || candidateSegmented) && (
        <Section title="分段采集" icon={<AudioWaveform size={14} />} keywords="语音 VAD 静音 阈值 切段 分段">
          <p className="text-xs text-text-muted">两路使用分段引擎时共用下列参数。</p>
          <fieldset disabled={draft.saving} className="grid gap-3 sm:grid-cols-2">
            {timing('silence_threshold')}
            {timing('silence_duration')}
          </fieldset>
        </Section>
      )}

      <div className="flex flex-wrap items-center justify-end gap-2">
        {isRecording && <p className="mr-auto text-xs text-text-muted">引擎与采集参数在下次开始采集时生效。</p>}
        <SaveStateBadge mode="explicit" state={draft.state} error={draft.error} />
        <button
          type="button"
          className="rounded-lg border border-accent-blue/30 bg-accent-blue/15 px-3 py-1.5 text-xs font-medium text-accent-blue disabled:opacity-50"
          disabled={!draft.dirty || draft.saving}
          onClick={() => void draft.save(rolePatch)}
        >
          {draft.saving ? '保存中…' : '保存语音使用设置'}
        </button>
      </div>
    </div>
  )
}

export default function SpeechRolesSettings({ onOpenSpeechConfig }: { onOpenSpeechConfig?: () => void }) {
  const config = useInterviewStore((state) => state.config)
  return config ? <SpeechSettingsForm config={config} onOpenSpeechConfig={onOpenSpeechConfig} /> : null
}
