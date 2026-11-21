import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { useInterviewStore } from '@/stores/configStore'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'


const apiMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getDevices: vi.fn(),
  getOptions: vi.fn(),
  checkModelsHealth: vi.fn(),
  kbStatus: vi.fn(),
  updateConfig: vi.fn(),
  askFromServerScreen: vi.fn(),
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

vi.mock('@/lib/api', () => ({
  api: apiMock,
}))

vi.mock('@/hooks/useInterviewWS', () => ({
  useInterviewWS: () => undefined,
}))

vi.mock('@/components/TranscriptionPanel', () => ({ default: () => <div>transcript</div> }))
vi.mock('@/components/AnswerPanel', () => ({ default: () => <div>answer</div> }))
vi.mock('@/components/ControlBar', () => ({ default: () => <div>controls</div> }))
vi.mock('@/components/SettingsDrawer', () => ({ default: () => null }))
vi.mock('@/components/KnowledgeMap', () => ({ default: () => <div>knowledge</div> }))
vi.mock('@/components/ResumeOptimizer', () => ({ default: () => <div>resume</div> }))
vi.mock('@/components/JobTracker', () => ({ default: () => <div>jobs</div> }))


describe('App bootstrap', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useInterviewStore.setState({
      config: null,
      devices: [],
      options: null,
      sttLoaded: false,
      sttLoading: true,
      sttActiveProvider: '',
      sttFallbackLoaded: false,
      modelHealth: {},
      modelHealthDetail: {},
      modelHealthLatency: {},
      tokenUsage: { prompt: 0, completion: 0, total: 0, byModel: {} },
      fallbackToast: null,
      toastMessage: null,
      settingsOpen: false,
      qaPairs: [],
      streamingIds: [],
      currentStreamingId: null,
      transcriptions: [],
      isPaused: false,
      isRecording: false,
      wsConnected: true,
    } as any)
    useUiPrefsStore.setState({ appMode: 'assist' })
    apiMock.getConfig.mockResolvedValue({
      models: [{ name: 'demo', supports_vision: false }],
      active_model: 0,
      api_key_set: true,
      think_mode: false,
      think_effort: 'off',
      stt_provider: 'whisper',
    })
    apiMock.getDevices.mockResolvedValue({ devices: [], platform: null })
    apiMock.getOptions.mockResolvedValue({ positions: [], languages: [] })
    apiMock.checkModelsHealth.mockResolvedValue(undefined)
    apiMock.updateConfig.mockResolvedValue({ ok: true })
    apiMock.askFromServerScreen.mockResolvedValue({ ok: true })
    apiMock.kbStatus.mockResolvedValue({
      enabled: false,
      total_docs: 0,
      total_chunks: 0,
      deadline_ms: 150,
      asr_deadline_ms: 80,
      deps: { docx: false, pdf: false, ocr: false, vision: false },
    })
  })

  it('renders the init error view when config bootstrap fails', async () => {
    apiMock.getConfig.mockRejectedValue(new Error('backend down'))

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('连接后端失败')).toBeInTheDocument()
    })
    expect(screen.getByText('backend down')).toBeInTheDocument()
    expect(screen.getByText(/正在请求 \/api\/config/)).toBeInTheDocument()
    expect(screen.getByText(/确认后端服务已启动/)).toBeInTheDocument()
  })

  it('keeps rendering the app when non-critical bootstrap requests fail', async () => {
    apiMock.getDevices.mockRejectedValue(new Error('devices down'))
    apiMock.getOptions.mockRejectedValue(new Error('options down'))

    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('tab', { name: '面试辅助' })).toBeInTheDocument()
    })

    expect(screen.queryByText('连接后端失败')).not.toBeInTheDocument()
  })

  it('switches between dedicated interview and exam tabs while keeping manual and screenshot tools', async () => {
    const current = await apiMock.getConfig()
    apiMock.getConfig.mockImplementation(async () => ({ ...current }))
    apiMock.updateConfig.mockImplementation(async (patch) => {
      Object.assign(current, patch)
      return { ok: true }
    })
    render(<App />)
    await screen.findByRole('tab', { name: '面试辅助', selected: true })
    expect(screen.getAllByRole('tab').slice(0, 3).map((item) => item.textContent)).toEqual([
      '面试辅助', '笔试辅助', '面试复盘',
    ])
    fireEvent.click(screen.getByRole('tab', { name: '笔试辅助' }))
    await screen.findByRole('tab', { name: '笔试辅助', selected: true })
    expect(apiMock.updateConfig).toHaveBeenCalledWith({ written_exam_mode: true })
    expect(screen.queryByText('transcript')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^截图答题$/ })).toBeInTheDocument()
    expect(screen.getByText('controls')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: '面试辅助' }))
    await screen.findByRole('tab', { name: '面试辅助', selected: true })
    expect(apiMock.updateConfig).toHaveBeenCalledWith({ written_exam_mode: false })
    expect(screen.getAllByText('transcript').length).toBeGreaterThan(0)
  })

  it('keeps the current workflow if an interview is still recording', async () => {
    useInterviewStore.setState({ isRecording: true })
    render(<App />)
    await screen.findByRole('tab', { name: '面试辅助', selected: true })
    fireEvent.click(screen.getByRole('tab', { name: '笔试辅助' }))
    expect(await screen.findByText('请先结束当前面试，再切换到笔试辅助。')).toBeInTheDocument()
    expect(apiMock.updateConfig).not.toHaveBeenCalled()
    expect(screen.getByRole('tab', { name: '面试辅助', selected: true })).toBeInTheDocument()
  })

  it('does not allow disabled models to be picked as the priority answer model', async () => {
    apiMock.getConfig.mockResolvedValue({
      models: [
        { id: 'enabled', name: 'Enabled Model', supports_vision: false, enabled: true },
        { id: 'disabled', name: 'Disabled Model', supports_vision: true, enabled: false },
        { id: 'other', name: 'Unselected Model', supports_vision: true, enabled: true },
      ],
      assist_model_ids: ['enabled', 'disabled'],
      active_model: 0,
      api_key_set: true,
      think_mode: false,
      think_effort: 'off',
      stt_provider: 'whisper',
    })

    render(<App />)

    const trigger = await screen.findByRole('button', { name: /优先答题模型 Enabled Model/ })
    fireEvent.click(trigger)

    expect(screen.queryByRole('button', { name: /Disabled Model/ })).not.toBeInTheDocument()
    expect(trigger).not.toHaveTextContent('识图')
    expect(screen.queryByRole('button', { name: 'Unselected Model' })).not.toBeInTheDocument()

    expect(apiMock.updateConfig).not.toHaveBeenCalled()
  })

  it('serializes priority model changes so the latest selection wins', async () => {
    const firstSave = deferred<{ ok: boolean }>()
    const config = {
      models: [
        { id: 'first', name: 'First Model', supports_vision: false, enabled: true },
        { id: 'second', name: 'Second Model', supports_vision: false, enabled: true },
        { id: 'third', name: 'Third Model', supports_vision: true, enabled: true },
      ],
      assist_model_ids: ['first', 'second', 'third'],
      active_model: 0,
      api_key_set: true,
      think_mode: false,
      think_effort: 'off',
      stt_provider: 'whisper',
    }
    apiMock.getConfig.mockImplementation(async () => ({ ...config }))
    apiMock.updateConfig
      .mockImplementationOnce(async (patch) => {
        await firstSave.promise
        Object.assign(config, patch)
        return { ok: true }
      })
      .mockImplementation(async (patch) => {
        Object.assign(config, patch)
        return { ok: true }
      })

    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: /优先答题模型 First Model/ }))
    fireEvent.click(screen.getByRole('button', { name: /Second Model/ }))

    await waitFor(() => {
      expect(apiMock.updateConfig).toHaveBeenCalledWith({ assist_model_ids: ['second', 'first', 'third'] })
    })
    expect(apiMock.updateConfig).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole('button', { name: /优先答题模型 First Model/ }))
    fireEvent.click(screen.getByRole('button', { name: /Third Model/ }))

    expect(apiMock.updateConfig).toHaveBeenCalledTimes(1)

    await act(async () => {
      firstSave.resolve({ ok: true })
      await firstSave.promise
    })

    await waitFor(() => {
      expect(apiMock.updateConfig).toHaveBeenCalledTimes(2)
    })
    expect(apiMock.updateConfig).toHaveBeenNthCalledWith(2, { assist_model_ids: ['third', 'second', 'first'] })
    await waitFor(() => {
      expect(useInterviewStore.getState().toastMessage).toBe('已设为优先答题模型：Third Model')
    })
  })

  it('surfaces model health detail in the priority model tooltip', async () => {
    useInterviewStore.setState({
      modelHealth: { 0: 'error' },
      modelHealthDetail: { 0: '401 unauthorized' },
      modelHealthLatency: {},
    } as any)

    render(<App />)

    const trigger = await screen.findByRole('button', { name: /生成验证失败：401 unauthorized/ })
    expect(trigger).toHaveAttribute('title', expect.stringContaining('401 unauthorized'))
  })

  it('prevents duplicate mobile server screen asks while one is in flight', async () => {
    apiMock.getConfig.mockResolvedValue({
      models: [{ id: 'screen', name: 'Screenshot model', supports_vision: true, enabled: true }],
      active_model: 0,
      assist_model_ids: [],
      assist_screenshot_model_id: 'screen',
      api_key_set: true,
    })
    let resolveAsk: ((value: unknown) => void) | null = null
    const pendingAsk = new Promise((resolve) => {
      resolveAsk = resolve
    })
    apiMock.askFromServerScreen.mockReturnValueOnce(pendingAsk)

    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: 'AI 答案' }))
    const screenAsk = screen.getByRole('button', { name: '服务端截图审题' })

    act(() => {
      screenAsk.click()
      screenAsk.click()
    })

    expect(apiMock.askFromServerScreen).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: '截图审题提交中…' })).toBeDisabled()

    await act(async () => {
      if (!resolveAsk) throw new Error('server screen ask resolver was not captured')
      resolveAsk({ ok: true })
      await pendingAsk
    })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '服务端截图审题' })).not.toBeDisabled()
    })
  })

  it('shows a single answer panel and submits screenshots in the mobile written-exam workspace', async () => {
    const originalInnerWidth = window.innerWidth
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 390 })
    apiMock.getConfig.mockResolvedValue({
      models: [
        { id: 'other', name: 'Second Vision', supports_vision: true, has_api_key: true, enabled: true },
        { id: 'off', name: 'Disabled Vision', supports_vision: true, has_api_key: true, enabled: false },
        { id: 'exam', name: 'Exam Vision', supports_vision: true, has_api_key: true, enabled: true },
      ],
      assist_screenshot_model_id: null,
      assist_model_ids: [],
      written_exam_model_ids: ['off', 'exam', 'other'],
      active_model: 0,
      api_key_set: true,
      written_exam_mode: true,
      think_mode: false,
      think_effort: 'off',
      stt_provider: 'whisper',
    })

    try {
      render(<App />)

      const screenshotButton = await screen.findByRole('button', { name: '截图答题' })
      expect(screen.getByRole('button', { name: '笔试模型 Exam Vision' })).toHaveTextContent('识图')
      expect(screen.queryByText('transcript')).not.toBeInTheDocument()
      expect(screen.getByText('answer')).toBeInTheDocument()
      fireEvent.click(screenshotButton)
      await waitFor(() => expect(apiMock.askFromServerScreen).toHaveBeenCalledTimes(1))
    } finally {
      Object.defineProperty(window, 'innerWidth', { configurable: true, value: originalInnerWidth })
    }
  })

  it('mounts a single TranscriptionPanel/AnswerPanel tree per matchMedia viewport and switches on resize', async () => {
    const originalMatchMedia = window.matchMedia
    let matches = true
    const listeners = new Set<(e: MediaQueryListEvent) => void>()
    const matchMediaMock = vi.fn().mockImplementation((query: string) => ({
      get matches() {
        return matches
      },
      media: query,
      onchange: null,
      addEventListener: (_type: string, cb: (e: MediaQueryListEvent) => void) => {
        listeners.add(cb)
      },
      removeEventListener: (_type: string, cb: (e: MediaQueryListEvent) => void) => {
        listeners.delete(cb)
      },
      addListener: (cb: (e: MediaQueryListEvent) => void) => {
        listeners.add(cb)
      },
      removeListener: (cb: (e: MediaQueryListEvent) => void) => {
        listeners.delete(cb)
      },
      dispatchEvent: () => true,
    }))
    Object.defineProperty(window, 'matchMedia', { configurable: true, value: matchMediaMock })

    try {
      render(<App />)
      await screen.findByRole('tab', { name: '面试辅助', selected: true })

      expect(screen.getAllByText('transcript')).toHaveLength(1)
      expect(screen.getAllByText('answer')).toHaveLength(1)

      act(() => {
        matches = false
        for (const cb of listeners) {
          cb({ matches: false, media: '(min-width: 768px)' } as MediaQueryListEvent)
        }
      })

      expect(screen.getAllByText('transcript')).toHaveLength(1)
      expect(screen.queryByText('answer')).not.toBeInTheDocument()

      fireEvent.click(screen.getByRole('button', { name: 'AI 答案' }))
      expect(screen.queryByText('transcript')).not.toBeInTheDocument()
      expect(screen.getAllByText('answer')).toHaveLength(1)

      act(() => {
        matches = true
        for (const cb of listeners) {
          cb({ matches: true, media: '(min-width: 768px)' } as MediaQueryListEvent)
        }
      })

      expect(screen.getAllByText('transcript')).toHaveLength(1)
      expect(screen.getAllByText('answer')).toHaveLength(1)
    } finally {
      Object.defineProperty(window, 'matchMedia', { configurable: true, value: originalMatchMedia })
    }
  })
})

describe('Window control buttons', () => {
  const minimizeSpy = vi.fn()
  const quitSpy = vi.fn()
  const originalElectronAPI = window.electronAPI

  beforeEach(() => {
    vi.clearAllMocks()
    useInterviewStore.setState({
      config: null,
      devices: [],
      options: null,
      sttLoaded: false,
      sttLoading: true,
      sttActiveProvider: '',
      sttFallbackLoaded: false,
      modelHealth: {},
      modelHealthDetail: {},
      modelHealthLatency: {},
      tokenUsage: { prompt: 0, completion: 0, total: 0, byModel: {} },
      fallbackToast: null,
      toastMessage: null,
      settingsOpen: false,
      qaPairs: [],
      streamingIds: [],
      currentStreamingId: null,
      transcriptions: [],
      isPaused: false,
      wsConnected: true,
    } as any)
    useUiPrefsStore.setState({ appMode: 'assist' })
    apiMock.getConfig.mockResolvedValue({
      models: [{ name: 'demo', supports_vision: false }],
      active_model: 0,
      api_key_set: true,
      think_mode: false,
      think_effort: 'off',
      stt_provider: 'whisper',
    })
    apiMock.getDevices.mockResolvedValue({ devices: [], platform: null })
    apiMock.getOptions.mockResolvedValue({ positions: [], languages: [] })
    apiMock.checkModelsHealth.mockResolvedValue(undefined)
    apiMock.updateConfig.mockResolvedValue({ ok: true })
    apiMock.askFromServerScreen.mockResolvedValue({ ok: true })
    apiMock.kbStatus.mockResolvedValue({
      enabled: false,
      total_docs: 0,
      total_chunks: 0,
      deadline_ms: 150,
      asr_deadline_ms: 80,
      deps: { docx: false, pdf: false, ocr: false, vision: false },
    })
    minimizeSpy.mockReset()
    quitSpy.mockReset()
  })

  afterEach(() => {
    window.electronAPI = originalElectronAPI
  })

  it('does not show window control buttons when electronAPI is absent', async () => {
    delete (window as any).electronAPI

    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('tab', { name: '面试辅助' })).toBeInTheDocument()
    })

    expect(screen.queryByRole('button', { name: '最小化窗口' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '退出应用' })).not.toBeInTheDocument()
  })

  it('shows minimize and quit buttons when electronAPI is present', async () => {
    window.electronAPI = {
      minimizeWindow: minimizeSpy,
      quitApp: quitSpy,
    } as any

    render(<App />)

    expect(await screen.findByRole('button', { name: '最小化窗口' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '退出应用' })).toBeInTheDocument()
  })

  it('calls electronAPI.minimizeWindow when minimize button is clicked', async () => {
    window.electronAPI = {
      minimizeWindow: minimizeSpy,
      quitApp: quitSpy,
    } as any

    render(<App />)

    const btn = await screen.findByRole('button', { name: '最小化窗口' })
    fireEvent.click(btn)

    expect(minimizeSpy).toHaveBeenCalledTimes(1)
    expect(quitSpy).not.toHaveBeenCalled()
  })

  it('calls electronAPI.quitApp when quit button is clicked', async () => {
    window.electronAPI = {
      minimizeWindow: minimizeSpy,
      quitApp: quitSpy,
    } as any

    render(<App />)

    const btn = await screen.findByRole('button', { name: '退出应用' })
    fireEvent.click(btn)

    expect(quitSpy).toHaveBeenCalledTimes(1)
    expect(minimizeSpy).not.toHaveBeenCalled()
  })
})
