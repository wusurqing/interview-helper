import { useState } from 'react'
import { Move, Paintbrush, PanelsTopLeft } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useOverlayLayout } from '@/hooks/useOverlayLayout'
import type { OverlayAnchor, OverlayLayout, OverlayLayoutResult, OverlayMode } from '@/lib/interviewOverlay'
import { getOverlayAppearanceStyles, OVERLAY_FONTS, type OverlayAppearance } from '@/lib/overlayAppearance'
import { getShortcutDisplay } from '@/lib/shortcuts'
import { useShortcutsStore } from '@/stores/shortcutsStore'
import { getOverlayAppearance, useUiPrefsStore } from '@/stores/uiPrefsStore'
import { Field, Section } from './shared'

const MODES = [
  { key: 'glass', label: '磨砂玻璃' },
  { key: 'prompt', label: '提词模式' },
  { key: 'focus', label: '专注面板' },
] as const

const ANCHORS: Array<[OverlayAnchor, string]> = [
  ['top-left', '左上'], ['top-center', '上方居中'], ['top-right', '右上'],
  ['middle-left', '左侧居中'], ['middle-center', '居中'], ['middle-right', '右侧居中'],
  ['bottom-left', '左下'], ['bottom-center', '下方居中'], ['bottom-right', '右下'],
]

const choiceClass = (active: boolean) =>
  `rounded-lg border px-3 py-2 text-xs transition-colors ${active
    ? 'border-accent-blue bg-accent-blue/10 text-accent-blue'
    : 'border-bg-hover bg-bg-tertiary/30 text-text-secondary hover:bg-bg-hover/40'}`

const resetClass = 'text-xs font-medium text-accent-blue hover:underline'

export default function OverlaySettingsTab() {
  const enabled = useUiPrefsStore((state) => state.interviewOverlayEnabled)
  const mode = useUiPrefsStore((state) => state.interviewOverlayMode)
  const setEnabled = useUiPrefsStore((state) => state.setInterviewOverlayEnabled)
  const setMode = useUiPrefsStore((state) => state.setInterviewOverlayMode)
  const appearance = useUiPrefsStore(useShallow((state) => getOverlayAppearance(state, state.interviewOverlayMode)))
  const updateAppearance = useUiPrefsStore((state) => state.updateOverlayAppearance)
  const shortcuts = useShortcutsStore((state) => state.shortcuts)
  const { layout: snapshot, setLayout, layoutError } = useOverlayLayout()
  const [saveError, setSaveError] = useState(false)
  const api = window.electronAPI
  const geometry = snapshot?.layouts[mode]
  const desktop = Boolean(api?.getOverlayLayout && api.updateOverlayLayout && api.resetOverlayLayout)

  async function saveGeometry(result: Promise<OverlayLayoutResult>) {
    setSaveError(false)
    try {
      const saved = await result
      if (saved.ok) {
        if (!api?.onOverlayLayout) setLayout(saved)
      } else {
        setSaveError(true)
      }
    } catch {
      setSaveError(true)
    }
  }

  function updateGeometry(patch: Partial<OverlayLayout>) {
    if (api?.updateOverlayLayout) void saveGeometry(api.updateOverlayLayout(mode, patch))
  }

  const patchAppearance = (patch: Partial<OverlayAppearance>) => updateAppearance(mode, patch)
  const preview = getOverlayAppearanceStyles(mode, appearance)

  return (
    <div className="space-y-4">
      <Section title="悬浮窗设置" icon={<PanelsTopLeft className="h-4 w-4" />} keywords="启用 模式 悬浮 浮窗 overlay 快捷键">
        <Checkbox label="启用悬浮提示窗" checked={enabled} onChange={setEnabled} />
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3" role="group" aria-label="显示模式">
          {MODES.map(({ key, label }) => (
            <button key={key} type="button" className={choiceClass(mode === key)} aria-pressed={mode === key} onClick={() => setMode(key)}>
              {label}
            </button>
          ))}
        </div>
        <p className="text-xs leading-relaxed text-text-muted">
          <kbd>{getShortcutDisplay(shortcuts.toggleInterviewOverlay?.key || 'CommandOrControl+O')}</kbd> 显示或隐藏窗口。
          <kbd className="ml-2">{getShortcutDisplay(shortcuts.moveOverlayToMouse?.key || 'CommandOrControl+M')}</kbd> 临时移动到鼠标附近。
        </p>
      </Section>

      <Section title="外观预览" keywords="示例 样式 字号 颜色 透明度 面板">
        <div className="ov-settings-preview rounded-xl border border-bg-hover/60">
          <div className={preview.shellClass} style={preview.shellStyle}>
            <div className={mode === 'focus' ? 'ov-focus-content' : 'ov-answer'} style={preview.answerStyle}>
              先说明结论，再用一个具体例子说明你的做法。
              <br />
              用数据举例，让对方更容易判断结果。
            </div>
          </div>
        </div>
      </Section>

      <Section title="位置与尺寸" icon={<Move className="h-4 w-4" />} keywords="屏幕 显示器 坐标 大小 宽度 高度 缩放 位置 layout">
        {!desktop ? (
          <p className="text-xs text-text-muted">位置与尺寸仅在桌面端可用。当前可调整外观并查看预览。</p>
        ) : geometry && snapshot ? (
          <div key={mode} className="space-y-3">
            <Field label="显示器" htmlFor="overlay-display">
              <select id="overlay-display" className="input-field w-full" value={geometry.displayId ?? 'primary'}
                onChange={(e) => updateGeometry({ displayId: e.target.value === 'primary' ? null : Number(e.target.value) })}>
                <option value="primary">系统主显示器</option>
                {geometry.displayId !== null && !snapshot.displays.some((item) => item.id === geometry.displayId) && (
                  <option value={geometry.displayId}>已保存的显示器（未连接）</option>
                )}
                {snapshot.displays.map(({ id, label, primary, workArea }) => (
                  <option key={id} value={id}>
                    {label || `显示器 ${id}`} · {workArea.width} × {workArea.height}{primary ? '（主屏）' : ''}
                  </option>
                ))}
              </select>
            </Field>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <span className="text-xs font-medium text-text-secondary">位置预设</span>
                <div className="grid grid-cols-3 gap-1" role="group" aria-label="位置预设">
                  {ANCHORS.map(([anchor, label]) => (
                    <button key={anchor} type="button" className={choiceClass(geometry.anchor === anchor)}
                      aria-pressed={geometry.anchor === anchor} onClick={() => updateGeometry({ anchor, x: 0, y: 0 })}>
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 content-start gap-2">
                <GeometryNumber field="x" label="X 偏移" value={geometry.x} onCommit={updateGeometry} />
                <GeometryNumber field="y" label="Y 偏移" value={geometry.y} onCommit={updateGeometry} />
                <p className="col-span-2 text-xs leading-relaxed text-text-muted">相对于所选位置。向右和向下为正，单位为逻辑像素。</p>
              </div>
            </div>
            {mode !== 'glass' && (
              <Field label="尺寸模式" htmlFor="overlay-size-mode">
                <select id="overlay-size-mode" className="input-field w-full" value={geometry.sizeMode}
                  onChange={(e) => updateGeometry({ sizeMode: e.target.value as OverlayLayout['sizeMode'] })}>
                  <option value={mode === 'prompt' ? 'auto' : 'percent'}>{mode === 'prompt' ? '随内容调整' : '按工作区比例'}</option>
                  <option value="fixed">固定宽高</option>
                </select>
              </Field>
            )}
            <div className="grid grid-cols-2 gap-2">
              {geometry.sizeMode === 'auto' ? (
                <>
                  <GeometryNumber field="maxWidth" label="最大宽度" value={geometry.maxWidth ?? 820} min={180} onCommit={updateGeometry} />
                  <GeometryNumber field="maxHeight" label="最大高度" value={geometry.maxHeight ?? 420} min={72} onCommit={updateGeometry} />
                </>
              ) : geometry.sizeMode === 'percent' ? (
                <>
                  <GeometryNumber field="widthPct" label="宽度比例" value={geometry.widthPct ?? 96} min={50} max={100} unit="%" onCommit={updateGeometry} />
                  <GeometryNumber field="heightPct" label="高度比例" value={geometry.heightPct ?? 90} min={35} max={100} unit="%" onCommit={updateGeometry} />
                </>
              ) : (
                <>
                  <GeometryNumber field="width" label="宽度" value={geometry.width} min={mode === 'focus' ? 480 : mode === 'prompt' ? 180 : 300} onCommit={updateGeometry} />
                  <GeometryNumber field="height" label="高度" value={geometry.height} min={mode === 'focus' ? 240 : mode === 'prompt' ? 72 : 100} onCommit={updateGeometry} />
                </>
              )}
            </div>
            <p className="text-xs text-text-muted">实际位置与尺寸会限制在显示器的可用工作区内。</p>
            <button type="button" className={resetClass} onClick={() => api?.resetOverlayLayout && void saveGeometry(api.resetOverlayLayout(mode))}>
              重置位置与尺寸
            </button>
            {saveError && <p role="alert" className="text-xs text-accent-red">位置与尺寸保存失败，请重试。</p>}
          </div>
        ) : (
          <p className="text-xs text-text-muted" role="status">{layoutError ? '无法读取桌面窗口设置。' : '正在读取显示器和尺寸…'}</p>
        )}
      </Section>

      <Section title="文字样式" icon={<Paintbrush className="h-4 w-4" />} keywords="字体 字号 字体大小 字重 行距 字色 颜色 对齐 阴影 text">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Range id="overlay-font-size" label="字号" value={appearance.fontSize} min={12} max={64} unit="px" onChange={(fontSize) => patchAppearance({ fontSize })} />
          <Range id="overlay-line-height" label="行距" value={appearance.lineHeight} min={1.2} max={2.2} step={0.01} onChange={(lineHeight) => patchAppearance({ lineHeight })} />
          <Field label="字体" htmlFor="overlay-font-family">
            <select id="overlay-font-family" className="input-field w-full" value={appearance.fontFamily}
              onChange={(e) => patchAppearance({ fontFamily: e.target.value as OverlayAppearance['fontFamily'] })}>
              {Object.entries(OVERLAY_FONTS).map(([key, font]) => <option key={key} value={key}>{font.label}</option>)}
            </select>
          </Field>
          <Field label="字重" htmlFor="overlay-font-weight">
            <select id="overlay-font-weight" className="input-field w-full" value={appearance.fontWeight}
              onChange={(e) => patchAppearance({ fontWeight: Number(e.target.value) })}>
              <option value={400}>常规</option>
              <option value={500}>中等</option>
              <option value={600}>半粗</option>
              <option value={700}>粗体</option>
            </select>
          </Field>
          <Field label="对齐方式" htmlFor="overlay-text-align">
            <select id="overlay-text-align" className="input-field w-full" value={appearance.textAlign}
              onChange={(e) => patchAppearance({ textAlign: e.target.value as OverlayAppearance['textAlign'] })}>
              <option value="left">左对齐</option>
              <option value="center">居中</option>
              <option value="right">右对齐</option>
            </select>
          </Field>
          <Color id="overlay-font-color" label="文字颜色" value={appearance.fontColor} onChange={(fontColor) => patchAppearance({ fontColor })} />
        </div>
        <Checkbox label="文字阴影" checked={appearance.textShadow} onChange={(textShadow) => patchAppearance({ textShadow })} />
      </Section>

      <Section title="透明度与面板" keywords="背景 底色 不透明度 透明度 圆角 阴影 opacity">
        <Range id="overlay-opacity" label="整体不透明度" value={Math.round(appearance.opacity * 100)} min={0} max={100} unit="%" onChange={(value) => patchAppearance({ opacity: value / 100 })} />
        {mode !== 'prompt' && (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Color id="overlay-background-color" label="面板底色" value={appearance.backgroundColor} onChange={(backgroundColor) => patchAppearance({ backgroundColor })} />
              <Range id="overlay-background-opacity" label="底色不透明度" value={Math.round(appearance.backgroundOpacity * 100)} min={0} max={100} unit="%" onChange={(value) => patchAppearance({ backgroundOpacity: value / 100 })} />
              <Range id="overlay-radius" label="圆角" value={appearance.radius} min={0} max={40} unit="px" onChange={(radius) => patchAppearance({ radius })} />
              <Checkbox label="面板阴影" checked={appearance.panelShadow} onChange={(panelShadow) => patchAppearance({ panelShadow })} />
            </div>
          </>
        )}
      </Section>

      <Section title="阅读行为" keywords="最大行数 自动跟随 提词 滚动 外观重置">
        <Range id="overlay-max-lines" label="最大显示行数" value={appearance.maxLines} min={0} max={50}
          valueLabel={appearance.maxLines === 0 ? '不限' : undefined} onChange={(maxLines) => patchAppearance({ maxLines })} />
        {mode === 'prompt' && (
          <Checkbox label="靠近底部时自动跟随最新内容" checked={appearance.autoFollow} onChange={(autoFollow) => patchAppearance({ autoFollow })} />
        )}
        <button type="button" className={resetClass} onClick={() => updateAppearance(mode, null)}>重置外观</button>
      </Section>
    </div>
  )
}

function GeometryNumber({ field, label, value, min = -32768, max = 32768, unit = 'px', onCommit }: {
  field: 'x' | 'y' | 'width' | 'height' | 'maxWidth' | 'maxHeight' | 'widthPct' | 'heightPct'
  label: string
  value: number
  min?: number
  max?: number
  unit?: string
  onCommit: (patch: Partial<OverlayLayout>) => void
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const id = `overlay-${field}`
  return (
    <Field label={label} htmlFor={id}>
      <div className="flex items-center gap-1.5">
        <input id={id} type="number" step={1} min={min} max={max} value={draft ?? value} className="input-field w-full min-w-0"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') e.currentTarget.blur()
            if (e.key === 'Escape' && draft !== null) {
              e.stopPropagation()
              setDraft(null)
            }
          }}
          onBlur={(e) => {
            const edited = e.currentTarget.valueAsNumber
            const next = Math.max(min, Math.min(max, Math.round(edited)))
            if (draft !== null && Number.isFinite(next) && next !== value) onCommit({ [field]: next })
            setDraft(null)
          }}
        />
        <span aria-hidden className="text-xs text-text-muted">{unit}</span>
      </div>
    </Field>
  )
}

function Range({ id, label, value, min, max, step = 1, unit, valueLabel, onChange }: {
  id: string
  label: string
  value: number
  min: number
  max: number
  step?: number
  unit?: string
  valueLabel?: string
  onChange: (value: number) => void
}) {
  return (
    <Field label={`${label} ${valueLabel ?? `${value}${unit ?? ''}`}`} htmlFor={id}>
      <input id={id} type="range" className="w-full accent-accent-blue" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  )
}

function Color({ id, label, value, onChange }: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <Field label={label} htmlFor={id}>
      <div className="flex items-center gap-2">
        <input id={id} type="color" className="h-8 w-12 cursor-pointer rounded-md border border-bg-hover bg-transparent" value={value} onChange={(e) => onChange(e.target.value)} />
        <span className="font-mono text-xs text-text-muted">{value}</span>
      </div>
    </Field>
  )
}

function Checkbox({ label, checked, onChange }: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
      <input type="checkbox" className="h-4 w-4 rounded accent-accent-blue" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}
