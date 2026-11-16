export type OverlayMode = 'glass' | 'prompt' | 'focus'
export type PromptWheelModifier = 'alt' | 'ctrl' | 'shift'
export type PromptWheelStatus = 'ready' | 'unavailable' | 'inactive'

export function isPromptWheelModifier(value: unknown): value is PromptWheelModifier {
  return value === 'alt' || value === 'ctrl' || value === 'shift'
}

export const INTERVIEW_OVERLAY_PROMPT_MIN_WIDTH = 200
export const INTERVIEW_OVERLAY_PROMPT_MAX_WIDTH = 820
export const INTERVIEW_OVERLAY_PROMPT_DEFAULT_WIDTH = 820

export interface OverlayStatePayload {
  trigger?: 'shortcut'
  initialized?: boolean
  enabled: boolean
  visible?: boolean
  mode?: OverlayMode
  opacity: number
  fontSize: number
  fontColor: string
  showBg: boolean
  focusWidthPct?: number
  focusHeightPct?: number
  promptMaxWidth?: number
  promptAutoFollow?: boolean
  promptWheelModifier?: PromptWheelModifier
  maxLines: number
}

export const INTERVIEW_OVERLAY_STORAGE_KEYS = {
  enabled: 'ia_overlay_enabled',
  opacity: 'ia_overlay_opacity',
  fontSize: 'ia_overlay_font_size',
  fontColor: 'ia_overlay_font_color',
  showBg: 'ia_overlay_show_bg',
  mode: 'ia_overlay_mode',
  focusWidthPct: 'ia_overlay_focus_width_pct',
  focusHeightPct: 'ia_overlay_focus_height_pct',
  promptMaxWidth: 'ia_overlay_prompt_max_width',
  promptAutoFollow: 'ia_overlay_prompt_auto_follow',
  promptWheelModifier: 'ia_overlay_prompt_wheel_modifier',
  maxLines: 'ia_overlay_max_lines',
  appearance: 'ia_overlay_appearance',
} as const

export type OverlayAnchor = `${'top' | 'middle' | 'bottom'}-${'left' | 'center' | 'right'}`

export interface OverlayLayout {
  displayId: number | null
  anchor: OverlayAnchor
  x: number
  y: number
  sizeMode: 'fixed' | 'auto' | 'percent'
  width: number
  height: number
  maxWidth?: number
  heightLimit?: 'screen' | 'custom'
  maxHeight?: number
  widthPct?: number
  heightPct?: number
}

export interface OverlayLayoutSnapshot {
  layouts: Record<OverlayMode, OverlayLayout>
  displays: Array<{
    id: number
    label: string
    primary: boolean
    workArea: { x: number; y: number; width: number; height: number }
  }>
  promptViewport: {
    sizeMode: OverlayLayout['sizeMode']
    width: number
    height: number
  }
}

export type OverlayLayoutResult = ({ ok: true } & OverlayLayoutSnapshot) | { ok: false }

export const INTERVIEW_OVERLAY_STORAGE_KEY_SET = new Set<string>(Object.values(INTERVIEW_OVERLAY_STORAGE_KEYS))

export function isOverlayMode(value: unknown): value is OverlayMode {
  return value === 'glass' || value === 'prompt' || value === 'focus'
}

export function overlayModeToShowBg(mode: OverlayMode): boolean {
  return mode !== 'prompt'
}

export function showBgToOverlayMode(showBg: boolean): OverlayMode {
  return showBg ? 'glass' : 'prompt'
}

export function isInterviewOverlayStorageKey(key: string | null | undefined): key is string {
  return typeof key === 'string' && INTERVIEW_OVERLAY_STORAGE_KEY_SET.has(key)
}

export function warnInterviewOverlaySyncIssue(context: string, error: unknown) {
  const isDevRuntime =
    typeof window !== 'undefined' && /^(localhost|127(?:\.\d{1,3}){3})$/.test(window.location.hostname)
  if (!isDevRuntime) return
  console.warn(`[interview-overlay] ${context}`, error)
}
