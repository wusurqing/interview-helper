import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useShallow } from 'zustand/react/shallow'
import ReactMarkdown, { type Components } from 'react-markdown'
import { useInterviewWS } from '@/hooks/useInterviewWS'
import { useOverlayLayout } from '@/hooks/useOverlayLayout'
import { applyStoredColorSchemeToDocument, COLOR_SCHEME_STORAGE_KEY } from '@/lib/colorScheme'
import { api, getErrorMessage } from '@/lib/api'
import { mmpgStateLabels } from '@/lib/mmpg/rtcController'
import {
  isInterviewOverlayStorageKey,
  warnInterviewOverlaySyncIssue,
} from '@/lib/interviewOverlay'
import { getShortcutDisplay } from '@/lib/shortcuts'
import { getOverlayAppearanceStyles } from '@/lib/overlayAppearance'
import { useInterviewStore, type QAPair } from '@/stores/configStore'
import { useShortcutsStore } from '@/stores/shortcutsStore'
import { getOverlayAppearance, useUiPrefsStore } from '@/stores/uiPrefsStore'
import PlainAnswerContent, { plainAnswerStatus } from './PlainAnswerContent'

type FocusTabPane = {
  key: string
  label: string
  content: string
  isGenerating?: boolean
}

type VisionVerifyState = {
  verdict: 'PASS' | 'FAIL' | 'UNKNOWN'
  reason: string
}

const FOCUS_TAB_CACHE_LIMIT = 20
const PROMPT_WIDTH_INSET = 20
const PROMPT_HEIGHT_INSET = 14

const OVERLAY_MARKDOWN_COMPONENTS: Components = {
  a({ children }) {
    return <span>{children}</span>
  },
  pre({ children }) {
    return <pre className="ov-code-block">{children}</pre>
  },
  code({ className, children }) {
    return <code className={className ? `ov-code ${className}` : 'ov-code'}>{children}</code>
  },
}

export default function InterviewOverlay() {
  const qaPairs = useInterviewStore((s) => s.qaPairs)
  const streamingIds = useInterviewStore((s) => s.streamingIds)
  const isRecording = useInterviewStore((s) => s.isRecording)
  const isPaused = useInterviewStore((s) => s.isPaused)
  const interviewerStreaming = useInterviewStore((s) => s.interviewerStreaming)
  const mmpgPhase = useInterviewStore((s) => s.mmpgPhase)
  const mmpgAnswerLabel = useInterviewStore((s) => s.mmpgAnswerLabel)
  const clearSession = useInterviewStore((s) => s.clearSession)
  const toastMessage = useInterviewStore((s) => s.toastMessage)
  const setToastMessage = useInterviewStore((s) => s.setToastMessage)
  const config = useInterviewStore((s) => s.config)
  const setConfig = useInterviewStore((s) => s.setConfig)
  const shortcuts = useShortcutsStore((s) => s.shortcuts)
  const setShortcuts = useShortcutsStore((s) => s.setShortcuts)
  const isExamMode = !interviewerStreaming && config?.written_exam_mode === true
  const standbyHint = interviewerStreaming
    ? (isPaused && mmpgPhase === 'idle' ? '录音已暂停'
        : mmpgPhase === 'listening' && mmpgAnswerLabel ? mmpgAnswerLabel : mmpgStateLabels[mmpgPhase ?? 'connecting'])
    : isRecording ? (isExamMode ? '笔试中…' : '正在聆听…') : (isExamMode ? '点击开始笔试' : '等待面试开始')
  const enabled = useUiPrefsStore((s) => s.interviewOverlayEnabled)
  const overlayMode = useUiPrefsStore((s) => s.interviewOverlayMode)
  const appearance = useUiPrefsStore(useShallow((state) => getOverlayAppearance(state, state.interviewOverlayMode)))
  const { fontSize, fontFamily, fontWeight, lineHeight, maxLines, autoFollow: overlayPromptAutoFollow } = appearance
  const legacyPromptWidth = useUiPrefsStore((s) => s.interviewOverlayPromptMaxWidth)
  const { layout, layoutError } = useOverlayLayout()
  const promptViewport = layout?.promptViewport
  const overlayPromptMaxWidth = promptViewport?.width ?? legacyPromptWidth
  const overlayPromptMaxHeight = promptViewport?.height ?? 420
  const fixedPromptSize = promptViewport?.sizeMode === 'fixed'
  const waitingForPromptViewport = Boolean(window.electronAPI?.getOverlayLayout && !promptViewport && !layoutError)
  const overlayVisible = useUiPrefsStore((s) => s.interviewOverlayVisible)
  const syncPrefs = useUiPrefsStore((s) => s.syncInterviewOverlayPrefs)
  const applyState = useUiPrefsStore((s) => s.applyInterviewOverlayState)
  const hasOverlayVisibilityIpc = Boolean(window.electronAPI?.getOverlayState || window.electronAPI?.onOverlayState)
  useInterviewWS(enabled && (overlayVisible || !hasOverlayVisibilityIpc))
  const [activeFocusTabsByQaId, setActiveFocusTabsByQaId] = useState<Record<string, string>>({})
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [questionView, setQuestionView] = useState<{
    reviewQaId: string | null
    liveFocusQaId: string | null
  }>({ reviewQaId: null, liveFocusQaId: null })
  const { reviewQaId, liveFocusQaId } = questionView
  const busyActionRef = useRef<string | null>(null)
  const pinnedFocusTabsByQaIdRef = useRef<Record<string, boolean>>({})

  // The overlay is a standalone entry point (it does not mount App's
  // useAppBootstrap hook), so it must hydrate config itself. Without this,
  // a second Electron window stays in interview mode even while the main
  // window has switched to written-exam mode.
  useEffect(() => {
    let active = true
    api.getConfig()
      .then((nextConfig) => {
        if (active) setConfig(nextConfig)
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [setConfig])

  const latestQa = useMemo(() => {
    if (streamingIds.length > 0) {
      const currentId = streamingIds[streamingIds.length - 1]
      const active = qaPairs.find((item) => item.id === currentId)
      if (active) return active
    }
    return qaPairs[qaPairs.length - 1] ?? null
  }, [qaPairs, streamingIds])
  const liveFocusQa = useMemo(
    () => (liveFocusQaId ? qaPairs.find((item) => item.id === liveFocusQaId) : null),
    [liveFocusQaId, qaPairs],
  )
  const displayedQa = useMemo(
    () => (reviewQaId ? qaPairs.find((item) => item.id === reviewQaId) : null) ?? liveFocusQa ?? latestQa,
    [latestQa, liveFocusQa, qaPairs, reviewQaId],
  )
  const displayedQaIndex = displayedQa ? qaPairs.findIndex((item) => item.id === displayedQa.id) : -1
  const isReviewingHistory = Boolean(reviewQaId && displayedQa)
  const plain = displayedQa?.answerFormat === 'plain_text'

  const answerText =
    plain ? displayedQa.answer
    : displayedQa?.status === 'cancelled'
      ? (displayedQa.cancelReason === 'continuation'
          ? [displayedQa.answer, '已因续说停止'].filter(Boolean).join('\n\n')
          : '上一条回答已取消')
      : displayedQa?.status === 'error'
        ? `保存失败: ${displayedQa.errorMessage || '未知原因'}`
        : displayedQa?.retrying
          ? plainAnswerStatus(displayedQa, true)
          : displayedQa?.answer?.trim() || (displayedQa ? (displayedQa.isThinking ? '思考中…' : '正在组织回答…') : '')
  const activeVisionVerify =
    displayedQa?.status !== 'cancelled' && displayedQa?.status !== 'error'
      ? displayedQa?.visionVerify
      : undefined
  const activeVisionVerifyKey = activeVisionVerify
    ? `${activeVisionVerify.verdict}:${activeVisionVerify.reason}`
    : ''
  const isStreaming = displayedQa ? streamingIds.includes(displayedQa.id) : false
  const hasContent = Boolean(displayedQa)
  const liveGeneratingQa = useMemo(
    () => qaPairs.find((item) => item.id !== displayedQa?.id && streamingIds.includes(item.id)) ?? null,
    [displayedQa?.id, qaPairs, streamingIds],
  )
  const focusTabs = useMemo(
    () => plain
      ? [{ key: 'answer', label: '回答', content: answerText, isGenerating: false }]
      : buildFocusTabs(answerText, displayedQa?.question ?? '', isStreaming),
    [answerText, displayedQa?.question, isStreaming, plain],
  )
  const displayedQaKey = displayedQa?.id ?? '__empty__'
  const retainedFocusQaIds = useMemo(
    () => getRetainedFocusQaIds(qaPairs.map((item) => item.id), displayedQa?.id ?? null, streamingIds),
    [displayedQa?.id, qaPairs, streamingIds],
  )
  const activeFocusTab = activeFocusTabsByQaId[displayedQaKey] ?? focusTabs[0]?.key ?? 'answer'
  const activeSection = focusTabs.find((tab) => tab.key === activeFocusTab) ?? focusTabs[0]

  const overlayAnswerSlice = useMemo(() => sliceMaxLines(answerText, maxLines, !plain), [answerText, maxLines, plain])

  const answerScrollRef = useRef<HTMLDivElement | null>(null)
  const promptScrollableRef = useRef<boolean | null>(null)
  const promptWheelRemainder = useRef(0)
  const lastPromptSize = useRef<{ width: number; height: number } | null>(null)
  const answerAutoFollowRef = useRef(true)
  const previousAnswer = useRef({ id: '', text: '' })
  const previousNewestQaId = useRef<string | null>(null)
  const updateAnswerAutoFollow = useCallback(() => {
    if (overlayMode === 'prompt' && !overlayPromptAutoFollow) {
      answerAutoFollowRef.current = false
      return
    }
    const el = answerScrollRef.current
    if (!el) return
    const threshold = answerAutoFollowRef.current ? 28 : 1
    answerAutoFollowRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= threshold
  }, [overlayMode, overlayPromptAutoFollow])

  const wheelOptionsRef = useRef({ fontSize, updateAnswerAutoFollow })
  useLayoutEffect(() => {
    wheelOptionsRef.current = { fontSize, updateAnswerAutoFollow }
  }, [fontSize, updateAnswerAutoFollow])

  const reportPromptScrollability = useCallback(() => {
    const report = window.electronAPI?.reportPromptScrollable
    if (!report) return
    const el = answerScrollRef.current
    const scrollable = Boolean(enabled && overlayVisible && overlayMode === 'prompt' && el && el.scrollHeight > el.clientHeight)
    if (promptScrollableRef.current === scrollable) return
    promptScrollableRef.current = scrollable
    report(scrollable).then((result) => {
      if (!result.ok && promptScrollableRef.current === scrollable) promptScrollableRef.current = null
    }).catch(() => { promptScrollableRef.current = null })
  }, [enabled, overlayVisible, overlayMode])

  useLayoutEffect(() => {
    reportPromptScrollability()
  }, [reportPromptScrollability, answerText, activeVisionVerifyKey, displayedQaKey, fontSize, fontFamily, fontWeight, lineHeight, maxLines, overlayPromptMaxWidth, overlayPromptMaxHeight, fixedPromptSize])

  useEffect(() => {
    const api = window.electronAPI
    if (!enabled || !overlayVisible || overlayMode !== 'prompt' || !api?.reportPromptScrollable) return
    const answer = answerScrollRef.current
    const content = answer?.querySelector<HTMLElement>('.ov-answer-stack, .ov-standby-hint')
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(reportPromptScrollability)
    if (answer) observer?.observe(answer)
    if (content) observer?.observe(content)
    window.addEventListener('resize', reportPromptScrollability)
    document.fonts?.addEventListener('loadingdone', reportPromptScrollability)
    reportPromptScrollability()
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', reportPromptScrollability)
      document.fonts?.removeEventListener('loadingdone', reportPromptScrollability)
    }
  }, [enabled, overlayVisible, overlayMode, hasContent, reportPromptScrollability])

  useEffect(() => {
    const api = window.electronAPI
    if (!enabled || !overlayVisible || overlayMode !== 'prompt' || !api?.reportPromptScrollable) return
    const unsubscribe = api.onPromptWheel?.((delta) => {
      const el = answerScrollRef.current
      if (!el || !Number.isFinite(delta) || el.scrollHeight <= el.clientHeight) return
      const lineHeight = Number.parseFloat(window.getComputedStyle(el).lineHeight)
      const linePixels = Number.isFinite(lineHeight) ? lineHeight : wheelOptionsRef.current.fontSize * 1.5
      const previousTop = el.scrollTop
      const wanted = promptWheelRemainder.current - delta * linePixels * 3 / 120
      const wholePixels = Math.trunc(wanted)
      promptWheelRemainder.current = wanted - wholePixels
      const requestedTop = previousTop + wholePixels
      el.scrollTop = Math.max(0, Math.min(el.scrollHeight - el.clientHeight, requestedTop))
      if (requestedTop < 0 || requestedTop > el.scrollHeight - el.clientHeight) promptWheelRemainder.current = 0
      if (delta > 0 && el.scrollTop < previousTop) {
        answerAutoFollowRef.current = false
      } else {
        wheelOptionsRef.current.updateAnswerAutoFollow()
      }
    })
    reportPromptScrollability()
    return () => {
      unsubscribe?.()
      promptWheelRemainder.current = 0
      promptScrollableRef.current = null
      api.reportPromptScrollable?.(false).catch(() => {})
    }
  }, [enabled, overlayVisible, overlayMode, reportPromptScrollability])

  useEffect(() => {
    answerAutoFollowRef.current = overlayMode !== 'prompt' || overlayPromptAutoFollow
  }, [displayedQaKey, overlayMode, activeFocusTab, overlayPromptAutoFollow])

  useLayoutEffect(() => {
    const before = previousAnswer.current
    previousAnswer.current = { id: displayedQaKey, text: answerText }
    if (plain ? before.id === displayedQaKey && before.text === answerText : !isStreaming) return
    if (overlayMode === 'prompt' && !overlayPromptAutoFollow) return
    const el = answerScrollRef.current
    if (!el) return
    if (!answerAutoFollowRef.current) return
    el.scrollTop = el.scrollHeight
  }, [answerText, activeFocusTab, displayedQaKey, isStreaming, overlayMode, overlayPromptAutoFollow, plain])

  useEffect(() => {
    if (!enabled || !overlayVisible || overlayMode !== 'prompt' || !overlayPromptAutoFollow) return
    const followOnResize = () => {
      const answer = answerScrollRef.current
      if (answer && answerAutoFollowRef.current) answer.scrollTop = answer.scrollHeight
    }
    window.addEventListener('resize', followOnResize)
    return () => window.removeEventListener('resize', followOnResize)
  }, [enabled, overlayVisible, overlayMode, overlayPromptAutoFollow])

  useLayoutEffect(() => {
    const resize = window.electronAPI?.resizeOverlayWindow
    if (!enabled || !overlayVisible || overlayMode !== 'prompt' || fixedPromptSize || waitingForPromptViewport || !resize) {
      lastPromptSize.current = null
      return
    }
    const el = answerScrollRef.current
    const shell = el?.parentElement
    const root = shell?.parentElement
    if (!el || !shell || !root) return
    const content = el.querySelector<HTMLElement>('.ov-answer-stack, .ov-standby-hint') ?? el

    const measure = () => {
      const visibleWidth = root.style.width
      const scrollTop = el.scrollTop
      let contentWidth: number
      let contentHeight: number
      // Measure at the configured width, then restore the visible layout before painting.
      root.style.width = `${overlayPromptMaxWidth}px`
      try {
        contentWidth = Math.ceil(shell.getBoundingClientRect().width)
        contentHeight = Math.ceil(Math.max(content.scrollHeight, content.getBoundingClientRect().height))
      } finally {
        root.style.width = visibleWidth
        if (el.scrollTop !== scrollTop) el.scrollTop = scrollTop
      }

      const size = {
        width: hasContent ? Math.min(overlayPromptMaxWidth, Math.max(180, contentWidth + PROMPT_WIDTH_INSET)) : overlayPromptMaxWidth,
        height: Math.min(overlayPromptMaxHeight, Math.max(72, contentHeight + PROMPT_HEIGHT_INSET)),
      }
      const previous = lastPromptSize.current
      if (previous?.width === size.width && previous.height === size.height) return
      lastPromptSize.current = size
      resize(size).catch(() => {
        if (lastPromptSize.current === size) lastPromptSize.current = null
      })
    }
    measure()
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(measure)
    observer?.observe(content)
    document.fonts?.addEventListener('loadingdone', measure)
    return () => {
      observer?.disconnect()
      document.fonts?.removeEventListener('loadingdone', measure)
    }
  }, [activeVisionVerifyKey, answerText, enabled, fixedPromptSize, fontSize, fontFamily, fontWeight, lineHeight, hasContent, maxLines, overlayMode, overlayAnswerSlice.text, overlayPromptMaxWidth, overlayPromptMaxHeight, overlayVisible, waitingForPromptViewport])

  const refreshShortcuts = useCallback(() => {
    window.electronAPI?.getShortcuts?.()
      .then((nextShortcuts) => { setShortcuts(nextShortcuts) })
      .catch(() => {})
  }, [setShortcuts])

  useEffect(() => {
    document.documentElement.classList.add('overlay-window')
    document.body.classList.add('overlay-window')
    applyStoredColorSchemeToDocument()

    const fontHref = 'https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600&family=JetBrains+Mono:wght@400;500&display=swap'
    let link = document.querySelector<HTMLLinkElement>(`link[href="${fontHref}"]`)
    if (!link) {
      link = document.createElement('link')
      link.rel = 'stylesheet'
      link.href = fontHref
      document.head.appendChild(link)
    }

    return () => {
      document.documentElement.classList.remove('overlay-window')
      document.body.classList.remove('overlay-window')
    }
  }, [])

  useEffect(() => {
    syncPrefs()
    window.electronAPI?.getOverlayState?.()
      .then((payload) => { if (payload) applyState(payload) })
      .catch((error) => { warnInterviewOverlaySyncIssue('bootstrap overlay state', error) })
    refreshShortcuts()

    const onStorage = (event: StorageEvent) => {
      if (!event.key || event.key === COLOR_SCHEME_STORAGE_KEY) applyStoredColorSchemeToDocument()
      if (!event.key || isInterviewOverlayStorageKey(event.key)) syncPrefs()
    }
    window.addEventListener('storage', onStorage)
    const removeOverlayListener = window.electronAPI?.onOverlayState?.((payload) => { applyState(payload) })
    const removeShortcutsListener = window.electronAPI?.onShortcuts?.((nextShortcuts) => { setShortcuts(nextShortcuts) })
    return () => {
      window.removeEventListener('storage', onStorage)
      removeOverlayListener?.()
      removeShortcutsListener?.()
    }
  }, [applyState, refreshShortcuts, setShortcuts, syncPrefs])

  const suppressMouseSelection = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault()
    const selection = window.getSelection?.()
    if (selection && selection.rangeCount > 0) {
      selection.removeAllRanges()
    }
  }, [])

  const suppressToolbarMouseDown = useCallback((event: React.MouseEvent<HTMLElement>) => {
    event.preventDefault()
  }, [])

  useEffect(() => {
    if (!toastMessage) return
    const timer = window.setTimeout(() => setToastMessage(null), 2500)
    return () => window.clearTimeout(timer)
  }, [setToastMessage, toastMessage])

  const runOverlayAction = useCallback(async (action: 'screen' | 'cancel' | 'clear' | 'hide') => {
    if (busyActionRef.current) return
    busyActionRef.current = action
    setBusyAction(action)
    try {
      if (action === 'screen') {
        await api.askFromServerScreen()
        setQuestionView({ reviewQaId: null, liveFocusQaId: null })
        setToastMessage('已提交截图审题')
      } else if (action === 'cancel') {
        await api.cancelAsk()
        setToastMessage('已取消生成')
      } else if (action === 'clear') {
        await api.clear()
        clearSession()
        setToastMessage('已清空')
      } else {
        await window.electronAPI?.syncOverlayWindow?.({ visible: false })
      }
    } catch (error) {
      setToastMessage(getErrorMessage(error, '操作失败'))
    } finally {
      busyActionRef.current = null
      setBusyAction(null)
    }
  }, [clearSession, setToastMessage])

  useEffect(() => {
    if (reviewQaId && !qaPairs.some((item) => item.id === reviewQaId)) {
      setQuestionView((current) => (current.reviewQaId ? { ...current, reviewQaId: null } : current))
    }
  }, [qaPairs, reviewQaId])

  useEffect(() => {
    const newest = qaPairs[qaPairs.length - 1]?.id ?? null
    const before = previousNewestQaId.current
    previousNewestQaId.current = newest
    if (reviewQaId) return
    setQuestionView((current) => {
      if (current.reviewQaId) return current
      const currentQa = qaPairs.find((item) => item.id === current.liveFocusQaId)
      let nextLiveId: string | null
      if (currentQa && streamingIds.includes(currentQa.id)) {
        nextLiveId = current.liveFocusQaId
      } else if (currentQa?.answerFormat === 'plain_text') {
        nextLiveId = newest !== before ? newest : current.liveFocusQaId
      } else {
        nextLiveId = latestQa?.id ?? null
      }
      return nextLiveId === current.liveFocusQaId
        ? current
        : { ...current, liveFocusQaId: nextLiveId }
    })
  }, [latestQa?.id, qaPairs, reviewQaId, streamingIds])

  useEffect(() => {
    setActiveFocusTabsByQaId((current) => {
      const next = pruneFocusTabCache(current, retainedFocusQaIds)
      return next === current ? current : next
    })

    const pinnedTabs = pinnedFocusTabsByQaIdRef.current
    for (const qaId of Object.keys(pinnedTabs)) {
      if (!retainedFocusQaIds.has(qaId)) delete pinnedTabs[qaId]
    }
  }, [retainedFocusQaIds])

  useEffect(() => {
    if (!focusTabs.length) return
    if (!focusTabs.some((tab) => tab.key === activeFocusTab)) {
      pinnedFocusTabsByQaIdRef.current[displayedQaKey] = false
      setActiveFocusTabsByQaId((current) => {
        if (current[displayedQaKey] === focusTabs[0].key) return current
        return pruneFocusTabCache({ ...current, [displayedQaKey]: focusTabs[0].key }, retainedFocusQaIds)
      })
      return
    }
    if (isStreaming && !pinnedFocusTabsByQaIdRef.current[displayedQaKey]) {
      const latestGeneratingTab = [...focusTabs].reverse().find((tab) => tab.isGenerating)
      if (latestGeneratingTab && latestGeneratingTab.key !== activeFocusTab) {
        setActiveFocusTabsByQaId((current) => ({
          ...pruneFocusTabCache(current, retainedFocusQaIds),
          ...(retainedFocusQaIds.has(displayedQaKey) ? { [displayedQaKey]: latestGeneratingTab.key } : {}),
        }))
      }
    }
  }, [activeFocusTab, displayedQaKey, focusTabs, isStreaming, retainedFocusQaIds])

  const moveFocusTab = useCallback((direction: 'prev' | 'next') => {
    pinnedFocusTabsByQaIdRef.current[displayedQaKey] = true
    setActiveFocusTabsByQaId((currentTabs) => {
      if (!focusTabs.length) return currentTabs
      const current = currentTabs[displayedQaKey] ?? focusTabs[0]?.key ?? 'answer'
      const currentIndex = Math.max(0, focusTabs.findIndex((tab) => tab.key === current))
      const delta = direction === 'next' ? 1 : -1
      const nextIndex = (currentIndex + delta + focusTabs.length) % focusTabs.length
      const nextKey = focusTabs[nextIndex]?.key ?? current
      return pruneFocusTabCache({ ...currentTabs, [displayedQaKey]: nextKey }, retainedFocusQaIds)
    })
  }, [displayedQaKey, focusTabs, retainedFocusQaIds])

  const moveOverlayQuestion = useCallback((direction: 'prev' | 'next') => {
    if (qaPairs.length < 2) return
    // Pure functional update over combined questionView state so rapid successive
    // next/prev (one act / one key-repeat burst) advances cleanly without side effects.
    setQuestionView((current) => {
      const currentReview = current.reviewQaId
        ? qaPairs.find((item) => item.id === current.reviewQaId)
        : null
      const currentLive = current.liveFocusQaId
        ? qaPairs.find((item) => item.id === current.liveFocusQaId)
        : null
      const baseId =
        currentReview?.id ??
        currentLive?.id ??
        latestQa?.id ??
        qaPairs[qaPairs.length - 1]?.id
      const currentIndex = Math.max(0, qaPairs.findIndex((item) => item.id === baseId))
      const delta = direction === 'next' ? 1 : -1
      const nextIndex = Math.max(0, Math.min(qaPairs.length - 1, currentIndex + delta))
      const nextId = qaPairs[nextIndex]?.id ?? baseId
      // Clear pin when landing on the true latest so live auto-follow resumes.
      // Also retarget liveFocus atomically: an older still-streaming liveFocus would
      // otherwise keep currentStillLive and snap the view away from the latest item.
      const latestId = latestQa?.id ?? qaPairs[qaPairs.length - 1]?.id
      if (nextId === latestId) {
        return {
          reviewQaId: null,
          liveFocusQaId: latestId ?? null,
        }
      }
      return {
        ...current,
        reviewQaId: nextId,
      }
    })
  }, [latestQa?.id, qaPairs])

  useEffect(() => {
    return window.electronAPI?.onFocusTabCommand?.((direction) => { moveFocusTab(direction) })
  }, [moveFocusTab])

  useEffect(() => {
    return window.electronAPI?.onOverlayQuestionCommand?.((direction) => { moveOverlayQuestion(direction) })
  }, [moveOverlayQuestion])

  if (!enabled) {
    return <div className="h-screen w-screen bg-transparent" />
  }

  const { shellClass, shellStyle, answerStyle } = getOverlayAppearanceStyles(overlayMode, appearance)
  const promptShellStyle = {
    ...shellStyle,
    // 与自适应量宽的 margin（root padding 12 + 安全边 8）对齐，避免触顶时右侧被窗口裁剪
    '--ov-prompt-max-width': `${Math.max(164, overlayPromptMaxWidth - PROMPT_WIDTH_INSET)}px`,
  } as CSSProperties
  const focusTools = [
    { key: 'screen' as const, label: '截图审题', shortcut: shortcuts.askFromServerScreen?.key },
    { key: 'cancel' as const, label: '取消生成', shortcut: shortcuts.cancelAnswer?.key },
    { key: 'clear' as const, label: '清空重来', shortcut: shortcuts.hardClearSession?.key },
    { key: 'hide' as const, label: '隐藏面板', shortcut: shortcuts.toggleInterviewOverlay?.key },
  ]

  const renderedAnswer = hasContent ? (
    <div className="ov-answer-stack">
      {displayedQa && qaPairs.length > 1 && (
        <div className="ov-review-line">
          {displayedQa.updatesQaId ? '补充后的回答' : isReviewingHistory ? '回看' : '当前'} {displayedQaIndex + 1}/{qaPairs.length} · {displayedQa.question}
        </div>
      )}
      {activeVisionVerify && <OverlayVisionVerify verify={activeVisionVerify} />}
      {plain
        ? <OverlayPlainAnswer qa={displayedQa} text={overlayAnswerSlice.text} isStreaming={isStreaming} />
        : <OverlayMarkdown content={overlayAnswerSlice.text} />}
      {overlayMode !== 'prompt' && !plain && isStreaming && <span className="ov-caret" />}
    </div>
  ) : (
    <span className="ov-standby-hint" style={{ fontSize }}>
      {standbyHint}
    </span>
  )

  const overlayToast = toastMessage ? (
    <div
      role="status"
      aria-live="polite"
      className="ov-toast pointer-events-none fixed bottom-3 left-1/2 z-50 max-w-[calc(100%-16px)] -translate-x-1/2 break-words rounded-lg border border-white/15 bg-slate-900/90 px-3 py-1.5 text-center text-xs font-medium text-white shadow-lg backdrop-blur-sm"
    >
      {toastMessage}
    </div>
  ) : null

  if (overlayMode === 'focus') {
    return (
      <div className="ov-root ov-root--focus">
        <div className={shellClass} style={shellStyle} onMouseDown={suppressMouseSelection}>
          <div className="ov-grip" aria-hidden />
          <div className="ov-focus-toolbar" aria-label="专注面板工具">
            {focusTools.map((tool) => (
              <button
                key={tool.key}
                type="button"
                disabled={Boolean(busyAction)}
                className={`ov-focus-tool ${busyAction === tool.key ? 'ov-focus-tool--busy' : ''}`}
                onMouseDown={suppressToolbarMouseDown}
                onClick={() => runOverlayAction(tool.key)}
              >
                <span>{busyAction === tool.key ? '处理中…' : tool.label}</span>
                {tool.shortcut && <kbd>{getShortcutDisplay(tool.shortcut)}</kbd>}
              </button>
            ))}
          </div>

          <div className="ov-focus-tabs" aria-label="答案分区">
            {focusTabs.map((tab) => (
              <button
                key={tab.key}
                type="button"
                className={`ov-focus-tab ${tab.key === activeFocusTab ? 'ov-focus-tab--active' : ''} ${tab.isGenerating ? 'ov-focus-tab--forming' : ''}`}
                aria-pressed={tab.key === activeFocusTab}
                onMouseDown={suppressToolbarMouseDown}
                onClick={() => {
                  pinnedFocusTabsByQaIdRef.current[displayedQaKey] = true
                  setActiveFocusTabsByQaId((current) => (
                    pruneFocusTabCache({ ...current, [displayedQaKey]: tab.key }, retainedFocusQaIds)
                  ))
                }}
              >
                {tab.label}
              </button>
            ))}
            {!plain && (
              <span className="ov-focus-tab-keys" aria-hidden>
                <span>切换分区</span>
                <kbd>{getShortcutDisplay(shortcuts.focusPrevTab?.key ?? 'CommandOrControl+Left')}</kbd>
                <kbd>{getShortcutDisplay(shortcuts.focusNextTab?.key ?? 'CommandOrControl+Right')}</kbd>
              </span>
            )}
          </div>
          {displayedQa && (
            <div className="ov-focus-question-strip" aria-label="题目导航">
              <span>{isReviewingHistory ? '回看' : '当前'} {displayedQaIndex + 1}/{qaPairs.length || 1}</span>
              <strong>{displayedQa.updatesQaId ? '补充后的回答 · ' : ''}{displayedQa.question}</strong>
              {(liveGeneratingQa || plain) && (
                <span className={`ov-focus-live-hint ${liveGeneratingQa ? '' : 'invisible'}`} title={liveGeneratingQa?.question}>
                  新答案生成中
                </span>
              )}
              <span aria-hidden>
                <span className="ov-focus-key-label">切换题目</span>
                <kbd>{getShortcutDisplay(shortcuts.overlayPrevQuestion?.key ?? 'CommandOrControl+Up')}</kbd>
                <kbd>{getShortcutDisplay(shortcuts.overlayNextQuestion?.key ?? 'CommandOrControl+Down')}</kbd>
              </span>
            </div>
          )}

          <div
            ref={answerScrollRef}
            className="ov-focus-content"
            style={answerStyle}
            onScroll={updateAnswerAutoFollow}
          >
            {hasContent ? (
              <div className="ov-focus-active-pane" key={`${displayedQaKey}:${activeSection.key}`}>
                {activeVisionVerify && <OverlayVisionVerify verify={activeVisionVerify} />}
                {plain ? (
                  <OverlayPlainAnswer qa={displayedQa} text={overlayAnswerSlice.text} isStreaming={isStreaming} />
                ) : (
                  <FocusSection
                    title={activeSection.label}
                    section={sliceMaxLines(activeSection.content, maxLines)}
                    muted={!activeSection.content.trim()}
                    emptyHint={activeSection.isGenerating ? '正在生成这个分区…' : '当前回答还没有拆出这个分区。'}
                  />
                )}
                {!plain && isStreaming && <span className="ov-caret" />}
              </div>
            ) : (
              <span className="ov-standby-hint" style={{ fontSize }}>
                {standbyHint}
              </span>
            )}
          </div>
        </div>
        {overlayToast}
      </div>
    )
  }

  return (
    <div
      className={`ov-root ${overlayMode === 'prompt' ? 'ov-root--prompt' : ''}${overlayMode === 'prompt' && fixedPromptSize ? ' ov-root--prompt-fixed' : ''}`}
    >
      <div className={shellClass} style={overlayMode === 'prompt' ? promptShellStyle : shellStyle}>
        <div className="ov-grip" aria-hidden />

        <div
          ref={answerScrollRef}
          className="ov-content ov-answer"
          style={answerStyle}
          onMouseDown={suppressMouseSelection}
          onScroll={updateAnswerAutoFollow}
        >
          {renderedAnswer}
        </div>
      </div>
      {overlayToast}
    </div>
  )
}

function getRetainedFocusQaIds(qaIds: string[], displayedQaId: string | null, streamingQaIds: string[]) {
  const liveQaIds = new Set(qaIds)
  const retained: string[] = []
  const seen = new Set<string>()
  const add = (qaId: string | null | undefined) => {
    if (!qaId || seen.has(qaId) || !liveQaIds.has(qaId) || retained.length >= FOCUS_TAB_CACHE_LIMIT) return
    seen.add(qaId)
    retained.push(qaId)
  }

  add(displayedQaId)
  for (let i = streamingQaIds.length - 1; i >= 0 && retained.length < FOCUS_TAB_CACHE_LIMIT; i -= 1) {
    add(streamingQaIds[i])
  }
  for (let i = qaIds.length - 1; i >= 0 && retained.length < FOCUS_TAB_CACHE_LIMIT; i -= 1) {
    add(qaIds[i])
  }
  return seen
}

function pruneFocusTabCache<T>(cache: Record<string, T>, retainedQaIds: Set<string>) {
  let changed = false
  const next: Record<string, T> = {}
  for (const [qaId, value] of Object.entries(cache)) {
    if (retainedQaIds.has(qaId)) {
      next[qaId] = value
    } else {
      changed = true
    }
  }
  return changed ? next : cache
}

function sliceMaxLines(text: string, maxLines: number, markdown = true): { text: string; omitted: boolean } {
  if (maxLines <= 0) return { text, omitted: false }
  const lines = text.split('\n')
  if (lines.length <= maxLines) return { text, omitted: false }
  const start = Math.max(0, lines.length - maxLines)
  const tail = lines.slice(start)
  const openFence = markdown ? getOpenMarkdownFenceAt(lines.slice(0, start)) : ''
  if (openFence) tail.unshift(openFence)
  return { text: tail.join('\n'), omitted: true }
}

function getOpenMarkdownFenceAt(lines: string[]) {
  let openFence = ''
  let openFenceMarker = ''
  for (const line of lines) {
    const fence = parseMarkdownFenceLine(line)
    if (!fence) continue
    const marker = fence.marker
    if (!openFence) {
      openFence = fence.opener
      openFenceMarker = marker
    } else if (isClosingMarkdownFence(marker, openFenceMarker)) {
      openFence = ''
      openFenceMarker = ''
    }
  }
  return openFence
}

function parseMarkdownFenceLine(line: string): { marker: string; opener: string } | null {
  const match = line.match(/^\s{0,3}(```+|~~~+)(.*)$/)
  if (!match) return null
  return {
    marker: match[1],
    opener: `${match[1]}${match[2] ?? ''}`.trimEnd(),
  }
}

function isClosingMarkdownFence(marker: string, openMarker: string) {
  return marker[0] === openMarker[0] && marker.length >= openMarker.length
}

function buildFocusTabs(answerText: string, question: string, isStreaming: boolean): FocusTabPane[] {
  const sections = parseMarkdownFocusSections(answerText)
  if (sections.length) {
    return sections.map((section, index) => ({
      key: `section-${index}`,
      label: section.label,
      content: section.content,
      isGenerating: isStreaming && index === sections.length - 1,
    }))
  }

  const body = answerText.trim()
  if (body) {
    return [{
      key: 'answer',
      label: isStreaming ? '生成中' : '回答',
      content: body,
      isGenerating: isStreaming,
    }]
  }

  const questionText = question.trim()
  if (questionText) {
    return [{
      key: 'question',
      label: '问题',
      content: questionText,
      isGenerating: isStreaming,
    }]
  }

  return [{
    key: 'waiting',
    label: isStreaming ? '生成中' : '等待',
    content: isStreaming ? '正在生成答案结构…' : '等待新的回答…',
    isGenerating: isStreaming,
  }]
}

function parseMarkdownFocusSections(answerText: string): Array<{ label: string; content: string }> {
  const sections: Array<{ label: string; content: string }> = []
  let current: { label: string; content: string } | null = null
  let codeFenceMarker = ''
  let preamble = ''

  for (const rawLine of answerText.split('\n')) {
    const line = rawLine.trimEnd()
    const fence = parseMarkdownFenceLine(line)
    if (fence) {
      if (!codeFenceMarker) {
        codeFenceMarker = fence.marker
      } else if (isClosingMarkdownFence(fence.marker, codeFenceMarker)) {
        codeFenceMarker = ''
      }
      if (current) {
        current.content = appendFocusLine(current.content, rawLine)
      } else {
        preamble = appendFocusLine(preamble, rawLine)
      }
      continue
    }

    const label = codeFenceMarker ? null : getFocusHeadingLabel(line)
    if (label) {
      current = { label, content: '' }
      sections.push(current)
      continue
    }

    if (current) {
      current.content = appendFocusLine(current.content, rawLine)
    } else {
      preamble = appendFocusLine(preamble, rawLine)
    }
  }

  const cleanPreamble = preamble.trim()
  const withPreamble = cleanPreamble ? [{ label: getOpeningTabLabel(cleanPreamble), content: cleanPreamble }, ...sections] : sections

  return withPreamble
    .map((tab) => ({ ...tab, content: tab.content.trim() }))
    .filter((tab) => tab.label)
}

function getOpeningTabLabel(text: string) {
  if (/```|class\s+\w+|def\s+\w+|function\s+\w+|SELECT\s+/i.test(text)) return '代码'
  if (/[。！？.!?]\s*$/.test(text) && text.length <= 90) return '结论'
  return '概览'
}

function getFocusHeadingLabel(line: string) {
  const markdown = line.match(/^\s{0,3}#{2,4}\s+(.+?)\s*#*\s*$/)
  const bracket = line.match(/^\s*【([^】]{1,28})】\s*$/)
  const bold = line.match(/^\s*(?:[-*+]\s+)?(?:[\d一二三四五六七八九十]+[.)、\s-]+)?(?:\*\*|__)([^*_]{1,32}?)(?:\*\*|__)\s*[：:]?\s*$/)
  const raw = markdown?.[1] ?? bracket?.[1] ?? bold?.[1]
  if (!raw) return null
  const label = raw
    .replace(/^[\d一二三四五六七八九十]+[.)、\s-]+/, '')
    .replace(/^(\*\*|__)(.+)\1$/, '$2')
    .replace(/[`*_]/g, '')
    .replace(/[：:]\s*$/, '')
    .trim()
  if (!label || label.length > 24) return null
  return label
}

function appendFocusLine(content: string, line: string) {
  return `${content}${content ? '\n' : ''}${line}`
}

function OverlayMarkdown({ content }: { content: string }) {
  return (
    <div className="ov-markdown markdown-body">
      <ReactMarkdown components={OVERLAY_MARKDOWN_COMPONENTS}>{content}</ReactMarkdown>
    </div>
  )
}

function OverlayPlainAnswer({ qa, text, isStreaming }: { qa: QAPair; text: string; isStreaming: boolean }) {
  const status = plainAnswerStatus(qa, isStreaming)
  return (
    <div className="ov-plain-answer max-w-full min-w-0">
      <div className="ov-plain-meta" title={status}>
        <span>{status}</span>
        {qa.firstTokenMs != null && <span>首字 {qa.firstTokenMs}ms</span>}
        {qa.totalMs != null && <span>总 {qa.totalMs}ms</span>}
      </div>
      <PlainAnswerContent text={text} />
    </div>
  )
}

function OverlayVisionVerify({ verify }: { verify: VisionVerifyState }) {
  const label =
    verify.verdict === 'PASS'
      ? '自检通过'
      : verify.verdict === 'FAIL'
        ? '自检不一致 · 人工复核'
        : '自检无定论'
  const reason = verify.reason.trim()
  const tone = verify.verdict.toLowerCase()
  return (
    <div
      className={`ov-verify ov-verify--${tone}`}
      role={verify.verdict === 'FAIL' ? 'alert' : 'status'}
      aria-live={verify.verdict === 'FAIL' ? 'assertive' : 'polite'}
    >
      <span className="ov-verify-dot" aria-hidden />
      <span className="ov-verify-label">{label}</span>
      {reason && <span className="ov-verify-reason">{reason}</span>}
    </div>
  )
}

function FocusSection({
  title,
  section,
  muted = false,
  emptyHint = '等待内容…',
}: {
  title: string
  section: { text: string; omitted: boolean }
  muted?: boolean
  emptyHint?: string
}) {
  const body = section.text.trim()
  return (
    <section className="ov-focus-section">
      <h2>{title}</h2>
      <div className={muted ? 'ov-focus-muted' : ''}>
        {section.omitted && <div className="ov-focus-omitted">…以上内容已省略</div>}
        {body ? <OverlayMarkdown content={body} /> : emptyHint}
      </div>
    </section>
  )
}
