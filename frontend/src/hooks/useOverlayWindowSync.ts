import { useEffect, useRef } from 'react'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'

export function useOverlayWindowSync(_isRecording: boolean, _appMode: string) {
  void _isRecording
  void _appMode
  const interviewOverlayEnabled = useUiPrefsStore((s) => s.interviewOverlayEnabled)
  const interviewOverlayOpacity = useUiPrefsStore((s) => s.interviewOverlayOpacity)
  const interviewOverlayFontSize = useUiPrefsStore((s) => s.interviewOverlayFontSize)
  const interviewOverlayFontColor = useUiPrefsStore((s) => s.interviewOverlayFontColor)
  const interviewOverlayShowBg = useUiPrefsStore((s) => s.interviewOverlayShowBg)
  const interviewOverlayMode = useUiPrefsStore((s) => s.interviewOverlayMode)
  const interviewOverlayFocusWidthPct = useUiPrefsStore((s) => s.interviewOverlayFocusWidthPct)
  const interviewOverlayFocusHeightPct = useUiPrefsStore((s) => s.interviewOverlayFocusHeightPct)
  const interviewOverlayPromptMaxWidth = useUiPrefsStore((s) => s.interviewOverlayPromptMaxWidth)
  const interviewOverlayPromptAutoFollow = useUiPrefsStore((s) => s.interviewOverlayPromptAutoFollow)
  const promptWheelModifier = useUiPrefsStore((s) => s.promptWheelModifier)
  const interviewOverlayMaxLines = useUiPrefsStore((s) => s.interviewOverlayMaxLines)
  const prevEnabledRef = useRef<boolean | undefined>(undefined)

  useEffect(() => {
    if (!window.electronAPI?.syncOverlayWindow) return
    const payload: Record<string, unknown> = {
      opacity: interviewOverlayOpacity,
      fontSize: interviewOverlayFontSize,
      fontColor: interviewOverlayFontColor,
      showBg: interviewOverlayShowBg,
      mode: interviewOverlayMode,
      focusWidthPct: interviewOverlayFocusWidthPct,
      focusHeightPct: interviewOverlayFocusHeightPct,
      promptMaxWidth: interviewOverlayPromptMaxWidth,
      promptAutoFollow: interviewOverlayPromptAutoFollow,
      promptWheelModifier,
      maxLines: interviewOverlayMaxLines,
    }

    if (prevEnabledRef.current !== interviewOverlayEnabled) {
      payload.enabled = interviewOverlayEnabled
    }
    // enabled OFF → force hide overlay + show main window
    if (prevEnabledRef.current && !interviewOverlayEnabled) {
      payload.visible = false
    }
    prevEnabledRef.current = interviewOverlayEnabled

    window.electronAPI.syncOverlayWindow(payload).catch(() => {})
  }, [
    interviewOverlayEnabled,
    interviewOverlayOpacity,
    interviewOverlayFontSize,
    interviewOverlayFontColor,
    interviewOverlayShowBg,
    interviewOverlayMode,
    interviewOverlayFocusWidthPct,
    interviewOverlayFocusHeightPct,
    interviewOverlayPromptMaxWidth,
    interviewOverlayPromptAutoFollow,
    promptWheelModifier,
    interviewOverlayMaxLines,
  ])

  useEffect(() => {
    return () => {
      const destroyOverlay = window.electronAPI?.destroyOverlay
      if (destroyOverlay) {
        destroyOverlay().catch(() => {})
      }
    }
  }, [])

  useEffect(() => {
    if (!window.electronAPI?.onOverlayState) return
    const removeOverlayListener = window.electronAPI.onOverlayState((payload) => {
      if (typeof payload.visible === 'boolean') {
        useUiPrefsStore.setState({ interviewOverlayVisible: payload.visible })
      }
      if (payload.trigger === 'shortcut' && payload.enabled && !useUiPrefsStore.getState().interviewOverlayEnabled) {
        useUiPrefsStore.getState().setInterviewOverlayEnabled(true)
      }
    })
    return () => removeOverlayListener?.()
  }, [])
}
