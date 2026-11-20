import { useCallback, useEffect, useState, useRef, lazy, Suspense } from 'react'
import { Settings, SlidersHorizontal, MonitorSmartphone, PanelLeftClose, PanelLeftOpen, Minus, X, ChevronDown } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'
import { getAssistModelEntries, getAssistScreenshotModel, getWrittenExamModel, isConfiguredModel, missingWrittenExamModel } from '@/lib/assistModels'
import { speechProvider } from '@/lib/speechProviders'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'
import { useShortcutsStore } from '@/stores/shortcutsStore'
import { useInterviewWS } from '@/hooks/useInterviewWS'
import { useAppBootstrap } from '@/hooks/useAppBootstrap'
import { useOverlayWindowSync } from '@/hooks/useOverlayWindowSync'
import { useAssistSplit } from '@/hooks/useAssistSplit'
import { api } from '@/lib/api'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { mmpgErrorMessage, mmpgStateLabels } from '@/lib/mmpg/rtcController'
import { useMmpgStatus } from '@/lib/mmpg/useMmpgStatus'
import TranscriptionPanel from '@/components/TranscriptionPanel'
import AnswerPanel from '@/components/AnswerPanel'
import ControlBar from '@/components/ControlBar'
import SettingsDrawer from '@/components/SettingsDrawer'
import SessionSettingsPopover from '@/components/SessionSettingsPopover'
import KnowledgeButton from '@/components/kb/KnowledgeButton'
import KnowledgeDrawer from '@/components/kb/KnowledgeDrawer'
import { AppToastStack } from '@/components/app/AppToastStack'
import { ConfigCorruptBanner } from '@/components/app/ConfigCorruptBanner'
import { InitErrorScreen } from '@/components/app/InitErrorScreen'
import VisionBadge from '@/components/VisionBadge'
import { ModelPriorityDropdown } from '@/components/app/ModelPriorityDropdown'
import { WindowOpacitySlider } from '@/components/app/WindowOpacitySlider'
const ReviewMode = lazy(() => import('@/components/ReviewMode'))
const KnowledgeMap = lazy(() => import('@/components/KnowledgeMap'))
const ResumeOptimizer = lazy(() => import('@/components/ResumeOptimizer'))
const JobTracker = lazy(() => import('@/components/JobTracker'))

const APP_MODE_TABS = [
  ['assist', '面试辅助'],
  ['exam', '笔试辅助'],
  ['review', '面试复盘'],
  ['knowledge', '能力分析'],
  ['resume-opt', '简历优化'],
  ['job-tracker', '求职看板'],
] as const
type WorkspaceTab = typeof APP_MODE_TABS[number][0]

export default function App() {
  useInterviewWS()
  // 精确订阅, 避免 store 任意字段变化(LLM token 流式 / toast 等)触发 App 重渲染
  const { config, toggleSettings, openModelsDrawer } = useInterviewStore(
    useShallow((s) => ({
      config: s.config,
      toggleSettings: s.toggleSettings,
      openModelsDrawer: s.openModelsDrawer,
    })),
  )
  const sttLoaded = useInterviewStore((s) => s.sttLoaded)
  const sttLoading = useInterviewStore((s) => s.sttLoading)
  const isRecording = useInterviewStore((s) => s.isRecording)
  const interviewerStreaming = useInterviewStore((s) => s.interviewerStreaming)
  const isPaused = useInterviewStore((s) => s.isPaused)
  const backendMmpgPhase = useInterviewStore((s) => s.mmpgPhase)
  const mmpgAnswerLabel = useInterviewStore((s) => s.mmpgAnswerLabel)
  const backendMmpgError = useInterviewStore((s) => s.mmpgError)
  const mmpg = useMmpgStatus()
  const isExamMode = !interviewerStreaming && config?.written_exam_mode === true
  const usesMmpg = !isExamMode && (isRecording ? interviewerStreaming : config?.stt_provider === 'doubao_mmpg')
  const localMmpgActive = mmpg.mode && mmpg.mode !== 'test' && !['idle', 'error'].includes(mmpg.state)
  const localMmpgError = mmpg.mode && mmpg.mode !== 'test' && !['connecting', 'reconnecting', 'listening', 'muted'].includes(backendMmpgPhase) ? mmpg.error : null
  const mmpgPhase = localMmpgActive ? mmpg.state : localMmpgError ? 'error' : backendMmpgPhase ?? 'idle'
  const mmpgLabel = isPaused && mmpgPhase === 'idle' ? '已暂停'
    : mmpgPhase === 'listening' && mmpgAnswerLabel ? mmpgAnswerLabel : mmpgStateLabels[mmpgPhase]
  const mmpgError = (localMmpgActive ? mmpg.error : localMmpgError) ?? (backendMmpgError ? mmpgErrorMessage(backendMmpgError) : null)
  const mmpgNotice = localMmpgActive ? mmpg.notice : null
  const selectedSpeechProvider = speechProvider(config?.stt_provider)
  const speechConfigured = config?.stt_configured
    ? Boolean(selectedSpeechProvider && config.stt_configured[selectedSpeechProvider])
    : sttLoaded
  const speechStatus = sttLoading ? '连接中' : sttLoaded ? '就绪' : '未就绪'
  const [mobileTab, setMobileTab] = useState<'transcript' | 'answer'>('transcript')
  const appMode = useUiPrefsStore((s) => s.appMode)
  const setAppMode = useUiPrefsStore((s) => s.setAppMode)
  const activeTab = appMode === 'assist' && isExamMode ? 'exam' : appMode
  const examModel = getWrittenExamModel(config)
  const assistTranscriptCollapsed = useUiPrefsStore((s) => s.assistTranscriptCollapsed)
  const toggleAssistTranscriptCollapsed = useUiPrefsStore((s) => s.toggleAssistTranscriptCollapsed)
  useOverlayWindowSync(isRecording, appMode)

  const [serverScreenLoading, setServerScreenLoading] = useState(false)
  const [sessionPopoverOpen, setSessionPopoverOpen] = useState(false)
  const [moduleMenuOpen, setModuleMenuOpen] = useState(false)
  const [workspaceSwitching, setWorkspaceSwitching] = useState(false)
  const workspaceSwitchRef = useRef(false)
  const serverScreenAskRef = useRef(false)
  const modelChangeSavingRef = useRef(false)
  const pendingModelChangeRef = useRef<string | number | null>(null)
  const sessionAnchorRef = useRef<HTMLButtonElement | null>(null)
  const moduleMenuRef = useRef<HTMLDivElement | null>(null)

  const {
    assistSplitContainerRef,
    assistSplitDragging,
    assistSplitPct,
    assistSplitPctRef,
    persistAssistSplitPct,
  } = useAssistSplit()

  const { initError } = useAppBootstrap()

  const selectWorkspace = useCallback(async (tab: WorkspaceTab) => {
    if (workspaceSwitchRef.current) return
    setModuleMenuOpen(false)
    if (tab !== 'assist' && tab !== 'exam') {
      setAppMode(tab)
      return
    }
    const state = useInterviewStore.getState()
    if (!state.config) return
    const exam = tab === 'exam'
    if (Boolean(state.config.written_exam_mode) === exam) {
      setAppMode('assist')
      return
    }
    if (state.isRecording || localMmpgActive) {
      state.setToastMessage(exam ? '请先结束当前面试，再切换到笔试辅助。' : '请先结束当前笔试，再切换到面试辅助。')
      return
    }
    workspaceSwitchRef.current = true
    setWorkspaceSwitching(true)
    try {
      await updateConfigAndRefresh({ written_exam_mode: exam })
      setAppMode('assist')
    } catch (error) {
      state.setToastMessage(error instanceof Error ? error.message : '切换失败，请重试。')
    } finally {
      workspaceSwitchRef.current = false
      setWorkspaceSwitching(false)
    }
  }, [localMmpgActive, setAppMode])

  useEffect(() => {
    if (!moduleMenuOpen) return
    const onPointerDown = (event: MouseEvent) => {
      if (!moduleMenuRef.current?.contains(event.target as Node)) {
        setModuleMenuOpen(false)
      }
    }
    window.addEventListener('mousedown', onPointerDown)
    return () => window.removeEventListener('mousedown', onPointerDown)
  }, [moduleMenuOpen])

  useEffect(() => {
    setModuleMenuOpen(false)
  }, [appMode])

  useEffect(() => {
    if (!window.electronAPI?.getShortcuts) return
    window.electronAPI.getShortcuts()
      .then((shortcuts) => useShortcutsStore.getState().setShortcuts(shortcuts))
      .catch(() => {})
  }, [])

  // Desktop main-process shortcuts (think toggle / force think) push toast + config
  // updates over IPC because POST /api/config does not broadcast over WS.
  useEffect(() => {
    const unsubToast = window.electronAPI?.onDesktopToast?.((payload) => {
      const msg = typeof payload === 'string' ? payload : payload?.message
      if (!msg) return
      const level = typeof payload === 'object' && payload ? payload.level : undefined
      if (level) {
        useInterviewStore.getState().pushToast(msg, level)
      } else {
        useInterviewStore.getState().setToastMessage(msg)
      }
    })
    const unsubConfig = window.electronAPI?.onConfigUpdated?.((nextConfig) => {
      if (!nextConfig || typeof nextConfig !== 'object') return
      useInterviewStore.getState().setConfig(nextConfig as unknown as AppConfig)
    })
    return () => {
      unsubToast?.()
      unsubConfig?.()
    }
  }, [])

  const hasGuided = useRef(false)
  useEffect(() => {
    if (!config || hasGuided.current) return
    const hasConfiguredModel = config.assist_model_ids === undefined
      ? config.api_key_set
      : config.models.some(isConfiguredModel)
    if (!hasConfiguredModel) {
      hasGuided.current = true
      openModelsDrawer()
    }
  }, [config, openModelsDrawer])

  // Cmd+Shift+J / Ctrl+Shift+J: 切换实时转录面板显隐
  // 注: Chrome 等浏览器把 Cmd+J / Ctrl+J 保留给「下载」, 故叠加 Shift 降低冲突.
  // 仅桌面端 assist 模式下生效; 输入框/contenteditable 内按键忽略.
  useEffect(() => {
    if (activeTab !== 'assist') return
    const onKeyDown = (e: KeyboardEvent) => {
      const isToggle =
        (e.metaKey || e.ctrlKey) &&
        e.shiftKey &&
        !e.altKey &&
        (e.key === 'j' || e.key === 'J')
      if (!isToggle) return
      const target = e.target as HTMLElement | null
      const tag = target?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || target?.isContentEditable) return
      e.preventDefault()
      toggleAssistTranscriptCollapsed()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activeTab, toggleAssistTranscriptCollapsed])

  const handleModelChange = useCallback(async (modelId: string | number) => {
    pendingModelChangeRef.current = modelId
    if (modelChangeSavingRef.current) return

    modelChangeSavingRef.current = true
    try {
      while (pendingModelChangeRef.current !== null) {
        const nextActiveModel = pendingModelChangeRef.current
        pendingModelChangeRef.current = null
        const current = useInterviewStore.getState().config
        const entry = getAssistModelEntries(current).find(({ model, index }) => (
          typeof nextActiveModel === 'string' ? model.id === nextActiveModel : index === nextActiveModel
        ))
        if (!entry) {
          useInterviewStore.getState().setToastMessage('请先在实时辅助中勾选该模型')
          continue
        }
        const { model: targetModel, index } = entry
        if (!isConfiguredModel(targetModel)) {
          useInterviewStore.getState().setToastMessage('请先在模型供应商配置中启用该模型')
          continue
        }
        try {
          const selected = current?.assist_model_ids
          if (selected && targetModel.id) {
            await updateConfigAndRefresh({ assist_model_ids: [
              targetModel.id,
              ...selected.filter((id) => id !== targetModel.id),
            ] })
          } else {
            await updateConfigAndRefresh({ active_model: index })
          }
          useInterviewStore.getState().setToastMessage(`已设为优先答题模型：${targetModel.name}`)
        } catch (error) {
          useInterviewStore.getState().setToastMessage(error instanceof Error ? error.message : '设置优先模型失败')
        }
      }
    } finally {
      modelChangeSavingRef.current = false
    }
  }, [])

  const handleServerScreenAsk = useCallback(async () => {
    if (serverScreenAskRef.current) return
    const current = useInterviewStore.getState().config
    if (current && !(current.written_exam_mode ? getWrittenExamModel(current) : getAssistScreenshotModel(current))) {
      useInterviewStore.getState().setToastMessage(current.written_exam_mode ? missingWrittenExamModel : '请先在实时辅助中配置截图答题模型')
      return
    }
    serverScreenAskRef.current = true
    setServerScreenLoading(true)
    try {
      await api.askFromServerScreen()
      useInterviewStore.getState().setToastMessage('已按当前截图区域配置提交服务端截图审题，请在答案区查看')
    } catch (e: unknown) {
      useInterviewStore.getState().setToastMessage(e instanceof Error ? e.message : '提交失败')
    } finally {
      serverScreenAskRef.current = false
      setServerScreenLoading(false)
    }
  }, [])

  const modelHealth = useInterviewStore((s) => s.modelHealth)
  const modelHealthDetail = useInterviewStore((s) => s.modelHealthDetail)
  const modelHealthLatency = useInterviewStore((s) => s.modelHealthLatency)
  const fallbackToast = useInterviewStore((s) => s.fallbackToast)
  const toastMessage = useInterviewStore((s) => s.toastMessage)
  const toasts = useInterviewStore((s) => s.toasts)
  const dismissToast = useInterviewStore((s) => s.dismissToast)
  const wsIsLeader = useInterviewStore((s) => s.wsIsLeader)
  const currentAppModeLabel = APP_MODE_TABS.find(([key]) => key === activeTab)?.[1] ?? '模块'

  useEffect(() => {
    if (!fallbackToast) return
    const timer = setTimeout(() => useInterviewStore.getState().setFallbackToast(null), 4000)
    return () => clearTimeout(timer)
  }, [fallbackToast])
  useEffect(() => {
    if (!toastMessage) return
    const timer = setTimeout(() => useInterviewStore.getState().setToastMessage(null), 2000)
    return () => clearTimeout(timer)
  }, [toastMessage])

  const toastTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  useEffect(() => {
    const timers = toastTimersRef.current
    const currentIds = new Set(toasts.map((t) => t.id))
    for (const [id, timer] of timers) {
      if (!currentIds.has(id)) { clearTimeout(timer); timers.delete(id) }
    }
    for (const t of toasts) {
      if (!timers.has(t.id)) {
        const timer = setTimeout(() => useInterviewStore.getState().dismissToast(t.id), t.ttlMs)
        timers.set(t.id, timer)
      }
    }
  }, [toasts])
  useEffect(() => () => {
    for (const timer of toastTimersRef.current.values()) clearTimeout(timer)
    toastTimersRef.current.clear()
  }, [])
  if (initError) {
    return <InitErrorScreen initError={initError} />
  }

  return (
    <div className="h-screen flex flex-col bg-bg-primary overflow-hidden noise-bg">
      {/* Header — 强制单行不换行, 各按钮文字按宽度阶梯隐藏, 实在不够再让左区横向滚动 */}
      <header className="app-drag-region header-gradient flex flex-row items-center justify-between gap-2 px-3 md:px-5 py-2.5 flex-shrink-0 min-w-0">
        <div className="flex items-center gap-2 md:gap-2.5 flex-shrink min-w-0 overflow-hidden">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-blue/20 to-accent-blue/5 flex items-center justify-center border border-accent-blue/10">
              <span className="text-sm">🎙️</span>
            </div>
            <h1 className="text-sm font-bold hidden lg:block flex-shrink-0 tracking-tight">学习助手</h1>
          </div>

          <div className="relative ml-1 md:hidden" ref={moduleMenuRef}>
            <button
              type="button"
              onClick={() => setModuleMenuOpen((prev) => !prev)}
              aria-haspopup="menu"
              aria-expanded={moduleMenuOpen}
              aria-label="切换功能模块"
              className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                moduleMenuOpen
                  ? 'border-accent-blue/40 bg-accent-blue/10 text-accent-blue'
                  : 'border-bg-hover/30 bg-bg-tertiary/60 text-text-primary'
              }`}
            >
              <MonitorSmartphone className="h-3.5 w-3.5" />
              <span className="max-w-[88px] truncate">{currentAppModeLabel}</span>
              <ChevronDown className={`h-3.5 w-3.5 transition-transform ${moduleMenuOpen ? 'rotate-180' : ''}`} />
            </button>
            {moduleMenuOpen ? (
              <div
                role="menu"
                aria-label="功能模块"
                className="absolute left-0 top-[calc(100%+0.5rem)] z-40 min-w-[180px] rounded-2xl border border-bg-hover/50 bg-bg-secondary/95 p-2 shadow-2xl shadow-black/10 backdrop-blur"
              >
                <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-muted">模块</div>
                <div className="mt-1 space-y-1">
                  {APP_MODE_TABS.map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      role="menuitem"
                      disabled={workspaceSwitching}
                      onClick={() => void selectWorkspace(key)}
                      className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-sm transition-colors ${
                        activeTab === key
                          ? 'bg-accent-blue text-white'
                          : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
                      }`}
                    >
                      <span>{label}</span>
                      {activeTab === key ? <span className="text-[11px] font-semibold">当前</span> : null}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="hidden overflow-x-auto bg-bg-tertiary/60 rounded-xl p-0.5 ml-1 border border-bg-hover/30 scrollbar-none md:flex" role="tablist" aria-label="功能模块">
            {APP_MODE_TABS.map(([key, label]) => (
              <button key={key} role="tab" aria-selected={activeTab === key} disabled={workspaceSwitching} onClick={() => void selectWorkspace(key)}
                className={`px-2.5 md:px-3 py-1.5 text-xs rounded-[10px] transition-all duration-200 whitespace-nowrap flex-shrink-0 font-medium ${activeTab === key ? 'bg-accent-blue text-white shadow-sm shadow-accent-blue/20' : 'text-text-muted hover:text-text-primary hover:bg-bg-hover/50'}`}>
                {label}
              </button>
            ))}
          </div>

          {/* 状态 chip:合并 STT + REC,录音中优先显示 REC,平时显示 STT 状态 */}
          {usesMmpg ? (
            <div role="status" title={mmpgError ?? mmpgNotice ?? `豆包 MMPG · ${mmpgLabel}`} className={`ml-1.5 flex items-center gap-1.5 rounded-lg border border-bg-hover/30 px-2 py-1 text-[10px] font-medium ${mmpgPhase === 'error' ? 'text-accent-red' : mmpgPhase === 'listening' ? 'text-accent-green' : 'text-text-muted'}`}>
              <span>{`MMPG ${mmpgLabel}`}</span>
            </div>
          ) : isRecording ? (
            <div
              className={`flex items-center gap-1.5 ml-1.5 flex-shrink-0 rounded-lg px-2 py-1 border animate-fade-up ${
                isPaused
                  ? 'bg-accent-amber/10 border-accent-amber/30'
                  : 'bg-accent-red/10 border-accent-red/30'
              }`}
              role="status"
              aria-live="polite"
              title={
                isPaused
                  ? isExamMode ? '笔试已暂停' : `录音已暂停 · ASR ${speechStatus}`
                  : isExamMode ? '笔试进行中' : `正在录音中 · ASR ${speechStatus}`
              }
            >
              <span className="relative inline-flex w-1.5 h-1.5 flex-shrink-0">
                {!isPaused && (
                  <span className="absolute inset-0 rounded-full bg-accent-red opacity-75 motion-safe:animate-ping" aria-hidden />
                )}
                <span
                  className={`relative inline-flex w-1.5 h-1.5 rounded-full ${isPaused ? 'bg-accent-amber' : 'bg-accent-red'}`}
                  aria-hidden
                />
              </span>
              <span
                className={`text-[10px] font-semibold leading-none hidden sm:inline ${isPaused ? 'text-accent-amber' : 'text-accent-red'}`}
              >
                {isPaused ? 'PAUSED' : isExamMode ? 'EXAM' : 'REC'}
              </span>
            </div>
          ) : (
            <div
              className="flex items-center gap-1.5 ml-1.5 flex-shrink-0 bg-bg-tertiary/30 rounded-lg px-2 py-1 border border-bg-hover/20"
              title={
                isExamMode
                  ? '笔试模式 · 等待提问'
                  : speechConfigured ? '语音引擎已配置 · 等待录音' : '请在实时辅助中选择已配置的语音引擎'
              }
            >
              <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${speechConfigured ? 'bg-accent-green' : 'bg-accent-amber'}`} />
              <span className="text-[10px] text-text-muted hidden md:inline font-medium">
                {isExamMode ? '笔试待开始' : speechConfigured ? 'ASR 已配置' : 'ASR 待配置'}
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-1.5 md:gap-2 flex-shrink-0 flex-nowrap justify-end">
          <WindowOpacitySlider />
          {activeTab === 'exam' ? (
            <button type="button" aria-label={`笔试模型 ${examModel?.name ?? '未配置'}`}
              title="打开笔试配置"
              className="inline-flex max-w-[160px] items-center gap-1.5 rounded-xl border border-bg-hover/50 px-2.5 py-1.5 text-xs text-text-primary hover:bg-bg-hover"
              onClick={() => useInterviewStore.setState({ settingsOpen: true, settingsDrawerTab: 'mode' })}>
              <span className="truncate">{examModel?.name ?? '配置笔试模型'}</span>
              {examModel?.supports_vision && <VisionBadge />}
            </button>
          ) : config?.models && config.models.length > 0 && (
            <ModelPriorityDropdown
              config={config}
              modelHealth={modelHealth}
              modelHealthDetail={modelHealthDetail}
              modelHealthLatency={modelHealthLatency}
              onModelChange={handleModelChange}
            />
          )}

          {/* 会场设置:聚合 Think + 岗位 + 语言 + Token */}
          <div className="relative">
            <button
              ref={sessionAnchorRef}
              type="button"
              onClick={() => setSessionPopoverOpen((v) => !v)}
              title="会场设置:Think 思考模式 / 岗位 / 语言 / Token 用量"
              aria-haspopup="dialog"
              aria-expanded={sessionPopoverOpen}
              aria-label="打开会场设置"
              className={`relative inline-flex items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-xs border transition-all duration-200 flex-shrink-0
                ${sessionPopoverOpen
                  ? 'border-accent-blue/50 bg-accent-blue/10 text-accent-blue shadow-sm shadow-accent-blue/10'
                  : 'border-bg-hover/50 bg-bg-tertiary/50 text-text-secondary hover:border-accent-blue/40 hover:text-text-primary'}`}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              <span className="font-medium hidden sm:inline">会场</span>
              {(config?.assist_think_mode ?? config?.think_mode) && (
                <span
                  className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-accent-green ring-2 ring-bg-primary shadow-[0_0_6px] shadow-accent-green/60"
                  aria-hidden
                  title="Think 已开启"
                />
              )}
            </button>
            <SessionSettingsPopover
              open={sessionPopoverOpen}
              onClose={() => setSessionPopoverOpen(false)}
              anchorRef={sessionAnchorRef}
            />
          </div>

          {activeTab === 'assist' && (
            <button
              type="button"
              onClick={toggleAssistTranscriptCollapsed}
              className="hidden md:inline-flex items-center justify-center min-h-[32px] min-w-[32px] p-1.5 rounded-xl hover:bg-bg-tertiary/60 text-text-muted hover:text-accent-blue transition-all duration-200 border border-transparent hover:border-accent-blue/20 flex-shrink-0"
              title={assistTranscriptCollapsed ? '显示实时转录面板 (⌘⇧J / Ctrl+⇧+J)' : '隐藏实时转录面板 (⌘⇧J / Ctrl+⇧+J)'}
              aria-label={assistTranscriptCollapsed ? '显示实时转录面板' : '隐藏实时转录面板'}
              aria-expanded={!assistTranscriptCollapsed}
            >
              {assistTranscriptCollapsed ? (
                <PanelLeftOpen className="w-4 h-4" />
              ) : (
                <PanelLeftClose className="w-4 h-4" />
              )}
            </button>
          )}
          <KnowledgeButton />
        <button
          type="button"
          onClick={toggleSettings}
          className="inline-flex items-center justify-center min-h-[32px] min-w-[32px] p-1.5 rounded-xl hover:bg-bg-tertiary/60 text-text-muted hover:text-text-primary transition-all duration-200 border border-transparent hover:border-bg-hover/40"
          title="设置中心 (实时辅助 / 语音配置 / 模型供应商配置 / 外观 / 快捷操作)"
          aria-label="打开设置"
        >
          <Settings className="w-4 h-4" />
        </button>
        {window.electronAPI && (
          <>
            <button
              type="button"
              onClick={() => window.electronAPI?.minimizeWindow()}
              className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-bg-hover/60 text-text-muted hover:text-text-primary transition-colors duration-150"
              title="最小化"
              aria-label="最小化窗口"
            >
              <Minus className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => window.electronAPI?.quitApp()}
              className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-accent-red/80 text-text-muted hover:text-white transition-colors duration-150"
              title="退出"
              aria-label="退出应用"
            >
              <X className="w-4 h-4" />
            </button>
          </>
        )}
        </div>
      </header>

      <ConfigCorruptBanner />

      {/* ── Assist Mode ── */}
      {appMode === 'assist' && (
        <>
          {/* Mobile tab switcher */}
          {!isExamMode && <div className="flex md:hidden border-b border-bg-tertiary flex-shrink-0">
            <button onClick={() => setMobileTab('transcript')}
              className={`flex-1 py-2 text-xs font-medium text-center transition-colors ${mobileTab === 'transcript' ? 'text-accent-blue border-b-2 border-accent-blue' : 'text-text-muted'}`}>
              实时转录
            </button>
            <button onClick={() => setMobileTab('answer')}
              className={`flex-1 py-2 text-xs font-medium text-center transition-colors ${mobileTab === 'answer' ? 'text-accent-blue border-b-2 border-accent-blue' : 'text-text-muted'}`}>
              AI 答案
            </button>
          </div>}

          <div
            ref={assistSplitContainerRef}
            className={`flex-1 ${isExamMode ? 'flex' : 'hidden md:flex'} overflow-hidden min-h-0`}
          >
            {!isExamMode && !assistTranscriptCollapsed && (
              <>
            <div
              className="flex flex-col min-w-0 flex-shrink-0 border-r border-bg-tertiary"
              style={{
                width: `${assistSplitPct}%`,
                minWidth: '220px',
                maxWidth: '70%',
              }}
            >
              <TranscriptionPanel />
            </div>
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="拖动调节转录区与答案区宽度"
              aria-valuemin={24}
              aria-valuemax={62}
              aria-valuenow={Math.round(assistSplitPct)}
              tabIndex={0}
              className="w-1 flex-shrink-0 cursor-col-resize group relative z-10 outline-none focus-visible:ring-2 focus-visible:ring-accent-blue/50 focus-visible:ring-inset bg-bg-hover/30 hover:bg-accent-blue/20 active:bg-accent-blue/40 transition-all duration-150"
              title="拖动调节左右宽度；双击恢复默认比例"
              onMouseDown={(e) => {
                e.preventDefault()
                assistSplitDragging.current = true
                document.body.style.cursor = 'col-resize'
                document.body.style.userSelect = 'none'
              }}
              onDoubleClick={(e) => {
                e.preventDefault()
                const c = 32
                assistSplitPctRef.current = c
                persistAssistSplitPct(c)
              }}
              onKeyDown={(e) => {
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                  e.preventDefault()
                  const delta = e.key === 'ArrowLeft' ? -2 : 2
                  const c = Math.min(62, Math.max(24, assistSplitPctRef.current + delta))
                  assistSplitPctRef.current = c
                  persistAssistSplitPct(c)
                }
                if (e.key === 'Home' || e.key === 'End') {
                  e.preventDefault()
                  const c = e.key === 'Home' ? 24 : 62
                  assistSplitPctRef.current = c
                  persistAssistSplitPct(c)
                }
              }}
            >
              <span
                className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-bg-hover group-hover:bg-accent-blue/50 pointer-events-none"
                aria-hidden
              />
            </div>
              </>
            )}
            <div className="flex-1 flex flex-col min-w-0 min-h-0">
              <AnswerPanel />
            </div>
          </div>

          {!isExamMode && (
            <div className="flex-1 flex md:hidden overflow-hidden min-h-0">
              {mobileTab === 'transcript' ? <TranscriptionPanel /> : <AnswerPanel />}
            </div>
          )}

          {/* 当前截图区域由服务端配置。 */}
          {(isExamMode || mobileTab === 'answer') && (
            <div className={`shrink-0 border-t border-bg-tertiary bg-bg-secondary/95 px-3 py-3 ${isExamMode ? 'md:px-5 md:py-2' : 'md:hidden'}`}>
              <button
                type="button"
                disabled={serverScreenLoading || workspaceSwitching}
                onClick={handleServerScreenAsk}
                className="flex min-h-[42px] w-full items-center justify-center gap-2 rounded-xl bg-accent-blue px-4 py-2 text-sm font-semibold text-white disabled:opacity-50 md:w-auto"
              >
                <MonitorSmartphone className="w-5 h-5 flex-shrink-0" />
                {serverScreenLoading ? '截图审题提交中…' : isExamMode ? '截图答题' : '服务端截图审题'}
              </button>
            </div>
          )}

          <fieldset disabled={workspaceSwitching} className="min-w-0 shrink-0">
            <ControlBar />
          </fieldset>
        </>
      )}

      {/* ── Review Mode ── */}
      {appMode === 'review' && (
        <Suspense fallback={<div className="flex-1 flex items-center justify-center text-sm text-text-muted">加载面试复盘中…</div>}>
          <ReviewMode />
        </Suspense>
      )}

      {/* ── Knowledge Map ── */}
      {appMode === 'knowledge' && (
        <Suspense fallback={<div className="flex-1 flex items-center justify-center text-sm text-text-muted">加载能力分析中…</div>}>
          <KnowledgeMap />
        </Suspense>
      )}

      {/* ── Resume Optimizer ── */}
      {appMode === 'resume-opt' && (
        <Suspense fallback={<div className="flex-1 flex items-center justify-center text-sm text-text-muted">加载简历优化中…</div>}>
          <ResumeOptimizer />
        </Suspense>
      )}

      {/* ── Job tracker ── */}
      {appMode === 'job-tracker' && (
        <Suspense fallback={<div className="flex-1 flex items-center justify-center text-sm text-text-muted">加载求职看板中…</div>}>
          <JobTracker />
        </Suspense>
      )}

      <AppToastStack
        wsIsLeader={wsIsLeader}
        fallbackToast={fallbackToast}
        toasts={toasts}
        dismissToast={dismissToast}
      />

      {usesMmpg && (mmpgError || mmpgNotice) && <div role="status" className="px-4 py-1.5 text-xs text-accent-amber">{mmpgError ?? mmpgNotice}</div>}
      <SettingsDrawer />
      <KnowledgeDrawer />
    </div>
  )
}

