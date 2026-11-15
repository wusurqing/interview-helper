import { useEffect, useState } from 'react'
import { useInterviewStore } from '@/stores/configStore'
import { type MmpgConfig, type MmpgForm, type MmpgSecret } from '@/lib/mmpg/settings'
import { mmpgController, mmpgStateLabels, mmpgUnavailable } from '@/lib/mmpg/rtcController'
import { useMmpgStatus } from '@/lib/mmpg/useMmpgStatus'
import { Field } from './shared'

type Props = {
  form: MmpgForm
  stored?: MmpgConfig
  onChange: (form: MmpgForm) => void
  onImport: () => Promise<void>
  onSave: () => Promise<boolean>
  saving: boolean
  testEnabled: boolean
}

const secretLabels: Record<MmpgSecret, string> = {
  volc_access_key: '火山引擎 Access Key',
  volc_secret_key: '火山引擎 Secret Key',
  rtc_app_key: 'RTC App Key',
  rtc_token: '手动 RTC Token',
  doubao_api_key: '豆包 ASR API Key',
}

const ids = [
  ['rtc_app_id', 'RTC App ID'], ['rtc_room_id', 'RTC 房间 ID'],
  ['rtc_user_id', 'RTC 用户 ID'], ['rtc_bot_id', 'MMPG Bot ID'], ['doubao_resource_id', '豆包 ASR Resource ID'],
] as const
const switches = [
  ['aivad', '智能判停（AIVAD）'], ['twopass', '二遍识别（Two-pass）'],
  ['enable_prefetch', '预取（Prefetch）'], ['force_end', '强制结束（Force end）'],
] as const

// Convert the stored Unix time to the page's local calendar time for editing.
function dateTimeLocal(seconds: number | null) {
  if (seconds == null) return ''
  const value = new Date(seconds * 1000)
  const calendar = new Date(value.getTime() - value.getTimezoneOffset() * 60_000)
  return Number.isFinite(calendar.valueOf()) ? calendar.toISOString().slice(0, 19) : ''
}

export default function MmpgSettings({ form, stored, onChange, onImport, onSave, saving, testEnabled }: Props) {
  const devices = useInterviewStore((state) => state.devices)
  const platform = useInterviewStore((state) => state.platformInfo?.platform)
  const recording = useInterviewStore((state) => state.isRecording)
  const status = useMmpgStatus()
  const [device, setDevice] = useState('')
  const [importing, setImporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [testRequested, setTestRequested] = useState(false)
  const test = status.mode === 'test' && testRequested
  const testRunning = test && !['idle', 'error'].includes(status.state)
  const unavailable = mmpgUnavailable(platform)

  // Closing settings or selecting a different engine ends the settings-only test.
  useEffect(() => mmpgController.stopTest, [])

  const set = <K extends keyof MmpgForm>(key: K, value: MmpgForm[K]) => onChange({ ...form, [key]: value })
  const secret = (key: MmpgSecret) => {
    const label = secretLabels[key]
    const configured = Boolean(stored?.[`${key}_configured`])
    const cleared = form[key] === null
    return (
      <Field label={label}>
        <div className="flex items-center gap-2">
          <input
            aria-label={label}
            autoComplete="off"
            type="password"
            value={form[key] ?? ''}
            placeholder={cleared ? '保存后清除' : configured ? '已配置 · 留空保留' : '未配置'}
            className="input-field"
            onChange={(event) => set(key, event.target.value)}
          />
          {(configured || form[key]) && (
            <button type="button" className="text-xs text-accent-blue shrink-0" aria-label={cleared ? `撤销清除 ${label}` : `清除 ${label}`} onClick={() => set(key, cleared ? '' : null)}>
              {cleared ? '撤销' : '清除'}
            </button>
          )}
        </div>
      </Field>
    )
  }

  const importProbe = async () => {
    setImporting(true)
    setError(null)
    try {
      await onImport()
    } catch (err) {
      setError(err instanceof Error ? err.message : '导入探针配置失败')
    } finally {
      setImporting(false)
    }
  }

  const testConnection = async () => {
    setError(null)
    setTestRequested(true)
    try {
      await mmpgController.start('test', Number(device), null, platform, onSave)
    } catch (err) {
      if (!(err instanceof Error) || err.name !== 'AbortError') {
        setError(err instanceof Error ? err.message : '测试连接失败')
      }
    }
  }

  const importLocal = window.isSecureContext && ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)
  return (
    <div className="space-y-3 rounded-xl border border-bg-hover/60 p-3">
      <p className="text-xs text-text-secondary">豆包 MMPG 通过火山 RTC 识别所选音源并自动判停。开始面试或运行测试会将所选设备音频发送到火山引擎，可能产生费用。</p>
      <button type="button" className="text-xs text-accent-blue disabled:opacity-50" onClick={importProbe} disabled={!importLocal || importing || saving || testRunning}>
        {importing ? '正在导入…' : '导入本机探针配置'}
      </button>
      <fieldset disabled={saving || importing || testRunning} className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {ids.map(([key, label]) => (
          <Field key={key} label={label}>
            <input aria-label={label} className="input-field" value={form[key]} onChange={(event) => set(key, event.target.value)} />
          </Field>
        ))}
        {secret('volc_access_key')}
        {secret('volc_secret_key')}
        {secret('doubao_api_key')}
        <Field label="RTC Token 方式">
          <select aria-label="RTC Token 方式" className="input-field" value={form.token_mode} onChange={(event) => set('token_mode', event.target.value as MmpgForm['token_mode'])}>
            <option value="auto">使用 RTC App Key 自动签发与续期</option>
            <option value="manual">手动填入 RTC Token</option>
          </select>
        </Field>
        {form.token_mode === 'auto' ? secret('rtc_app_key') : (
          <>
            {secret('rtc_token')}
            <Field label="手动 Token 到期时间（可选）" hint="使用本机时区。未填时由后端根据 Token 内容检查有效期。">
              <input aria-label="手动 Token 到期时间" type="datetime-local" step={1} value={dateTimeLocal(form.rtc_token_expires_at)} className="input-field" onChange={(event) => set('rtc_token_expires_at', event.target.value ? Math.floor(new Date(event.target.value).getTime() / 1000) : null)} />
            </Field>
          </>
        )}
        <details className="col-span-full text-xs text-text-secondary">
          <summary className="cursor-pointer">豆包 MMPG 高级设置</summary>
          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="判停静音时间（ms）" hint="允许 500–2999 毫秒。">
              <input aria-label="判停静音时间" className="input-field" type="number" min={500} max={2999} value={Number.isFinite(form.silence_ms) ? form.silence_ms : ''} onChange={(event) => set('silence_ms', event.target.valueAsNumber)} />
            </Field>
            <Field label="答题前等待时长（毫秒）" hint="判停后等待续说。续说结束后合并文字并重新计时。设为 0 可立即提交。">
              <input aria-label="答题前等待时长" className="input-field" type="number" min={0} step={1} value={Number.isFinite(form.answer_delay_ms) ? form.answer_delay_ms : ''} onChange={(event) => set('answer_delay_ms', event.target.valueAsNumber)} />
            </Field>
            <Field label="回答启动后的续说窗口（毫秒）" hint="从发起回答开始计时。窗口内续说会停止正在生成的回答，合并后重新回答。设为 0 可关闭。">
              <input aria-label="回答启动后的续说窗口" className="input-field" type="number" min={0} step={1} value={Number.isFinite(form.followup_window_ms) ? form.followup_window_ms : ''} onChange={(event) => set('followup_window_ms', event.target.valueAsNumber)} />
            </Field>
            <Field label="识别过期时间（ms，可选）">
              <input aria-label="识别过期时间" type="number" min={1} value={form.expire_time_ms ?? ''} className="input-field" onChange={(event) => set('expire_time_ms', event.target.value ? event.target.valueAsNumber : null)} />
            </Field>
            {switches.map(([key, label]) => (
              <label key={key} className="inline-flex items-center gap-2">
                <input type="checkbox" checked={form[key]} onChange={(event) => set(key, event.target.checked)} />
                {label}
              </label>
            ))}
          </div>
        </details>
      </fieldset>

      {testEnabled && (
      <div className="space-y-2 rounded-lg bg-bg-primary/40 p-3 text-xs text-text-secondary">
        <Field label="MMPG 测试音频设备">
          <select aria-label="MMPG 测试音频设备" className="input-field" value={device} disabled={testRunning} onChange={(event) => setDevice(event.target.value)}>
            <option value="">请选择测试设备</option>
            {devices.map(({ id, name }) => <option value={id} key={id}>{name}</option>)}
          </select>
        </Field>
        <p>最多运行 60 秒，仅显示识别结果。</p>
        {testRunning ? (
          <button type="button" className="text-accent-blue" onClick={mmpgController.stopTest}>结束 MMPG 测试</button>
        ) : (
          <button type="button" className="text-accent-blue disabled:opacity-50" disabled={Boolean(unavailable) || recording || saving || importing || !devices.some(({ id }) => String(id) === device)} onClick={testConnection}>保存并测试 MMPG</button>
        )}
        {unavailable && <p>{unavailable}</p>}
        {recording && <p>请结束面试后再运行测试。</p>}
        {test && (
          <div className="space-y-1" aria-live="polite">
            <p>{status.state === 'idle' ? '测试已结束' : mmpgStateLabels[status.state]}</p>
            {status.tokenExpiresAt && <p>RTC Token 到期时间 {new Date(status.tokenExpiresAt).toLocaleString()}</p>}
            <p>后端丢弃 {status.audioDropped} 个采样 · 缓冲欠载 {status.underruns} 次 · 溢出 {status.overrunSamples} 个采样</p>
            {status.notice && <p className="text-accent-amber">{status.notice}</p>}
            <ol aria-label="MMPG 测试事件" className="max-h-40 space-y-1 overflow-y-auto">
              {status.timeline.map(({ offsetMs, label, text }, index) => (
                <li key={index} className="flex items-baseline gap-2 break-words">
                  <span className="shrink-0 tabular-nums text-text-muted">{(offsetMs / 1000).toFixed(2)} 秒</span>
                  <span>{label}{text && <span className="ml-2 text-text-primary">{text}</span>}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
      )}
      {(error || (test && status.error)) && <p role="alert" className="text-xs text-accent-red">{error || status.error}</p>}
    </div>
  )
}
