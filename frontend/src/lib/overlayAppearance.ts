import type { CSSProperties } from 'react'
import type { OverlayMode } from './interviewOverlay'

export const OVERLAY_FONTS = {
  default: { label: '默认', family: undefined },
  sans: { label: '系统无衬线', family: 'system-ui, "Microsoft YaHei UI", "PingFang SC", sans-serif' },
  serif: { label: '衬线', family: 'Georgia, SimSun, "Songti SC", serif' },
  mono: { label: '等宽', family: '"JetBrains Mono", Consolas, "Microsoft YaHei UI", monospace' },
} as const

export interface OverlayAppearance {
  fontSize: number
  fontFamily: keyof typeof OVERLAY_FONTS
  fontWeight: number
  lineHeight: number
  fontColor: string
  textAlign: 'left' | 'center' | 'right'
  textShadow: boolean
  opacity: number
  backgroundColor: string
  backgroundOpacity: number
  radius: number
  panelShadow: boolean
  maxLines: number
  autoFollow: boolean
}

export type OverlayAppearances = Partial<Record<OverlayMode, OverlayAppearance>>

export function defaultOverlayAppearance(mode: OverlayMode, legacy?: {
  fontSize: number
  fontColor: string
  opacity: number
  maxLines: number
  promptAutoFollow: boolean
}): OverlayAppearance {
  const focus = mode === 'focus'
  const prompt = mode === 'prompt'
  return {
    fontSize: Math.max(12, legacy?.fontSize ?? 14),
    fontFamily: 'default',
    fontWeight: focus || prompt ? 600 : 400,
    lineHeight: focus ? 1.58 : prompt ? 1.5 : 1.65,
    fontColor: focus ? '#263241' : legacy?.fontColor ?? '#e2e8f0',
    textAlign: 'left',
    textShadow: prompt,
    opacity: focus ? 1 : legacy?.opacity ?? 0.88,
    backgroundColor: focus ? '#ffffff' : '#0a0c14',
    backgroundOpacity: focus ? Math.min(0.95, Math.max(0.1, legacy?.opacity ?? 0.88)) : 0.78,
    radius: focus ? 10 : 14,
    panelShadow: true,
    maxLines: legacy?.maxLines ?? 0,
    autoFollow: legacy?.promptAutoFollow ?? false,
  }
}

function normalizeNumber(value: unknown, fallback: number, min: number, max: number, decimals = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
  const scale = 10 ** decimals
  return Math.round(Math.min(max, Math.max(min, value)) * scale) / scale
}

function normalizeColor(value: unknown, fallback: string) {
  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value) ? value : fallback
}

export function normalizeOverlayAppearance(value: unknown, base: OverlayAppearance): OverlayAppearance {
  if (!value || typeof value !== 'object') return base
  const input = value as Partial<OverlayAppearance>
  const fontFamily = input.fontFamily && Object.prototype.hasOwnProperty.call(OVERLAY_FONTS, input.fontFamily)
    ? input.fontFamily
    : base.fontFamily
  const textAlign = input.textAlign && ['left', 'center', 'right'].includes(input.textAlign) ? input.textAlign : base.textAlign
  return {
    fontFamily,
    textAlign,
    fontSize: normalizeNumber(input.fontSize, base.fontSize, 12, 64),
    fontWeight: [400, 500, 600, 700].includes(input.fontWeight ?? 0) ? input.fontWeight! : base.fontWeight,
    lineHeight: normalizeNumber(input.lineHeight, base.lineHeight, 1.2, 2.2, 2),
    fontColor: normalizeColor(input.fontColor, base.fontColor),
    textShadow: typeof input.textShadow === 'boolean' ? input.textShadow : base.textShadow,
    opacity: normalizeNumber(input.opacity, base.opacity, 0, 1, 2),
    backgroundColor: normalizeColor(input.backgroundColor, base.backgroundColor),
    backgroundOpacity: normalizeNumber(input.backgroundOpacity, base.backgroundOpacity, 0, 1, 2),
    radius: normalizeNumber(input.radius, base.radius, 0, 40),
    panelShadow: typeof input.panelShadow === 'boolean' ? input.panelShadow : base.panelShadow,
    maxLines: normalizeNumber(input.maxLines, base.maxLines, 0, 50),
    autoFollow: typeof input.autoFollow === 'boolean' ? input.autoFollow : base.autoFollow,
  }
}

export function getOverlayAppearanceStyles(mode: OverlayMode, appearance: OverlayAppearance) {
  const { backgroundColor, backgroundOpacity } = appearance
  const rgb = [1, 3, 5].map((offset) => parseInt(backgroundColor.slice(offset, offset + 2), 16))
  const boundedAlpha = (scale: number, min: number, max: number) =>
    String(Math.max(min, Math.min(max, backgroundOpacity * scale)))

  return {
    shellClass: `ov-shell ${mode === 'focus' ? 'ov-shell--focus' : mode === 'prompt' ? 'ov-shell--nobg' : 'ov-shell--bg'}`,
    shellStyle: {
      opacity: appearance.opacity,
      color: appearance.fontColor,
      ...(mode !== 'prompt' && {
        backgroundColor: `rgba(${rgb.join(', ')}, ${backgroundOpacity})`,
        borderRadius: appearance.radius,
        boxShadow: appearance.panelShadow ? undefined : 'none',
      }),
      ...(mode === 'focus' && {
        '--ov-focus-toolbar-alpha': boundedAlpha(0.48, 0.08, 0.42),
        '--ov-focus-tabs-alpha': boundedAlpha(0.7, 0.12, 0.58),
        '--ov-focus-key-alpha': boundedAlpha(0.86, 0.2, 0.76),
      }),
    } as CSSProperties,
    answerStyle: {
      color: appearance.fontColor,
      fontSize: appearance.fontSize,
      fontFamily: OVERLAY_FONTS[appearance.fontFamily].family,
      fontWeight: appearance.fontWeight,
      lineHeight: appearance.lineHeight,
      textAlign: appearance.textAlign,
      textShadow: appearance.textShadow
        ? '0 0 1px #000f, 0 1px 3px #000e, 0 2px 12px #0009'
        : 'none',
    } satisfies CSSProperties,
  }
}
