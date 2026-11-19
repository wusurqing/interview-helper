import NetworkQRCode from './NetworkQRCode'
import QuickPromptsEditor from './QuickPromptsEditor'
import GlobalShortcutsEditor from './GlobalShortcutsEditor'
import { useSettingsSearch } from './shared'

export default function ShortcutsTab() {
  const searchQuery = useSettingsSearch()

  return (
    <div className="p-5 space-y-5 pb-8" data-in-search={searchQuery.trim() ? '1' : undefined}>
      <div className="rounded-xl border border-bg-hover/60 bg-bg-primary/35 px-4 py-3">
        <p className="text-sm font-semibold text-text-primary">快捷操作</p>
        <p className="mt-0.5 text-[11px] leading-relaxed text-text-muted">
          快捷提示词即时生效，桌面端可编辑全局快捷键。
        </p>
      </div>
      <NetworkQRCode />
      <QuickPromptsEditor />
      <GlobalShortcutsEditor />
    </div>
  )
}
