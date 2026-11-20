import { useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { useInterviewStore } from '@/stores/configStore'

export function ConfigCorruptBanner() {
  const configCorrupt = useInterviewStore((s) => Boolean(s.config?.config_corrupt))
  const corruptInfo = useInterviewStore((s) => s.config?.config_corrupt_info)
  const [confirming, setConfirming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!configCorrupt) return null

  const backupHint = corruptInfo?.backup_filename
    ? `原文件已自动备份为 ${corruptInfo.backup_filename}。`
    : '请先检查 backend/config.json。'

  const handleConfirmOverwrite = async () => {
    if (confirming) return
    const ok = window.confirm(
      '确认以当前默认配置覆盖损坏的 config.json？原文件已保留在 .corrupt 备份中，但当前配置中的模型与密钥需重新填写。',
    )
    if (!ok) return
    setConfirming(true)
    setError(null)
    try {
      await updateConfigAndRefresh({ confirm_overwrite_corrupt: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : '覆盖保存失败，请重试')
    } finally {
      setConfirming(false)
    }
  }

  return (
    <div
      role="alert"
      className="flex flex-wrap items-center justify-between gap-2 border-b border-accent-amber/35 bg-accent-amber/12 px-3 py-2 text-xs text-text-primary flex-shrink-0"
    >
      <div className="flex items-start gap-2 min-w-0">
        <AlertTriangle className="w-4 h-4 text-accent-amber flex-shrink-0 mt-0.5" aria-hidden />
        <div className="min-w-0 leading-relaxed">
          <span className="font-semibold text-accent-amber">配置文件损坏，已进入只读保护模式：</span>
          <span>{corruptInfo?.error || 'config.json 校验失败'}。</span>
          <span className="text-text-secondary">
            {backupHint} 未经确认不会覆盖写入，避免清空您的模型与密钥；可手动修复后重启，或确认以默认配置重置。
          </span>
          {error && <span className="ml-2 text-accent-red">{error}</span>}
        </div>
      </div>
      <button
        type="button"
        disabled={confirming}
        onClick={() => void handleConfirmOverwrite()}
        className="rounded-md border border-accent-amber/40 bg-accent-amber/15 px-2.5 py-1 text-xs font-medium text-accent-amber hover:bg-accent-amber/25 disabled:opacity-50 flex-shrink-0"
      >
        {confirming ? '正在重置…' : '确认覆盖为默认配置'}
      </button>
    </div>
  )
}
