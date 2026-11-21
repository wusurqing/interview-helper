import { Profiler } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ControlBar from './ControlBar'
import { useInterviewStore } from '@/stores/configStore'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'
import { candidateMmpgController, mmpgController } from '@/lib/mmpg/rtcController'


const apiMock = vi.hoisted(() => ({
  ask: vi.fn(),
  getDevices: vi.fn(),
  start: vi.fn(),
  stop: vi.fn(),
  pause: vi.fn(),
  resume: vi.fn(),
  switchSingleMicRole: vi.fn(),
  clear: vi.fn(),
  cancelAsk: vi.fn(),
  audioOutputTest: vi.fn(),
  audioInputTest: vi.fn(),
  audioInputMonitorStart: vi.fn(),
  audioInputMonitorStatus: vi.fn(),
  audioInputMonitorStop: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  api: {
    ...apiMock,
  },
  getErrorMessage: (error: unknown, fallback = '操作失败') =>
    error instanceof Error ? error.message : fallback,
}))

vi.mock('@/components/ResumeHistory', () => ({
  ResumeHistoryPopover: () => <div>resume-history</div>,
}))

class MockFileReader {
  result: string | null = 'data:image/png;base64,xxx'
  onload: ((event: { target: { result: string } }) => void) | null = null

  readAsDataURL() {
    this.onload?.({ target: { result: this.result! } })
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe('ControlBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    vi.stubGlobal('FileReader', MockFileReader as any)
    ;(window as any).electronAPI = {
      syncOverlayWindow: vi.fn().mockResolvedValue(undefined),
    }
    apiMock.ask.mockResolvedValue({ ok: true })
    apiMock.getDevices.mockResolvedValue({ devices: [], platform: null })
    apiMock.start.mockResolvedValue({ ok: true })
    apiMock.stop.mockResolvedValue({ ok: true })
    apiMock.pause.mockResolvedValue({ ok: true })
    apiMock.resume.mockResolvedValue({ ok: true })
    apiMock.switchSingleMicRole.mockResolvedValue({ ok: true })
    apiMock.clear.mockResolvedValue({ ok: true })
    apiMock.cancelAsk.mockResolvedValue({ ok: true })
    apiMock.audioOutputTest.mockResolvedValue({ ok: true, elapsed_sec: 1.0 })
    apiMock.audioInputTest.mockResolvedValue({
      ok: true,
      device_id: 11,
      elapsed_sec: 1.2,
      rms: 0.02,
      peak: 0.1,
      has_signal: true,
      detail: '已捕获输入信号',
    })
    apiMock.audioInputMonitorStart.mockResolvedValue({
      running: true,
      device_id: 11,
      rms: 0.02,
      peak: 0.1,
      level_pct: 50,
      has_signal: true,
      error: null,
    })
    apiMock.audioInputMonitorStatus.mockResolvedValue({
      running: true,
      device_id: 11,
      rms: 0.02,
      peak: 0.1,
      level_pct: 50,
      has_signal: true,
      error: null,
    })
    apiMock.audioInputMonitorStop.mockResolvedValue({ running: false })
    useInterviewStore.setState({
      config: {
        models: [{ name: 'TextOnly', supports_think: false, supports_vision: false, enabled: true }],
        active_model: 0,
        model_name: 'TextOnly',
        temperature: 0.5,
        max_tokens: 4096,
        think_mode: false,
        think_effort: 'off',
        stt_provider: 'whisper',
        whisper_model: 'base',
        whisper_language: 'auto',
        doubao_stt_app_id: '',
        doubao_stt_access_token_configured: false,
        doubao_stt_api_key_configured: false,
        doubao_stt_resource_id: '',
        doubao_stt_boosting_table_id: '',
        generic_stt_api_base_url: '',
        generic_stt_api_key_configured: false,
        generic_stt_model: '',
        generic_stt_custom_headers_configured: false,
        position: '后端开发',
        language: 'Python',
        auto_detect: true,
        silence_threshold: 0.01,
        silence_duration: 1.2,
        api_key_set: true,
        has_resume: false,
      },
      devices: [{ id: 1, name: 'loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' }],
      platformInfo: null,
      streamingIds: [],
      qaPairs: [],
      transcriptions: [],
      wsConnected: true,
      modelHealth: { 0: 'ok' },
      modelHealthDetail: {},
      modelHealthLatency: {},
      sttLoaded: true,
      sttLoading: false,
      sttActiveProvider: '',
      sttFallbackLoaded: false,
      isRecording: false,
      singleMic: null,
      interviewerStreaming: false,
      candidateCaptureProvider: null,
      mmpgPhase: null,
      isPaused: false,
      lastWSError: null,
      toastMessage: null,
    } as any)
    useUiPrefsStore.setState({
      interviewOverlayEnabled: false,
      interviewOverlayMode: 'glass',
      interviewOverlayShowBg: true,
      interviewOverlayOpacity: 0.88,
      interviewOverlayFontSize: 14,
      interviewOverlayFontColor: '#e2e8f0',
      interviewOverlayFocusWidthPct: 96,
      interviewOverlayFocusHeightPct: 90,
      interviewOverlayMaxLines: 0,
    })
  })

  it.each([
    { written_exam_model_ids: [] },
    { written_exam_model_ids: ['missing'] },
    { written_exam_model_ids: ['disabled'] },
    { written_exam_model_ids: ['no-key'] },
    { written_exam_model_ids: ['text'] },
  ])('requires a configured vision primary even when an interview backup is ready ($written_exam_model_ids)', (selection) => {
    useInterviewStore.setState(({ config }) => ({
      config: config && {
        ...config, written_exam_mode: true, ...selection, assist_model_ids: ['backup'], assist_screenshot_model_id: 'backup',
        models: [
          { id: 'backup', name: 'Backup', has_api_key: true, enabled: true, supports_vision: true, supports_think: false },
          { id: 'disabled', name: 'Off', has_api_key: true, enabled: false, supports_vision: true, supports_think: false },
          { id: 'no-key', name: 'No Key', has_api_key: false, enabled: true, supports_vision: true, supports_think: false },
          { id: 'text', name: 'Text', has_api_key: true, enabled: true, supports_vision: false, supports_think: false },
        ],
      },
    }))
    render(<ControlBar />)
    fireEvent.change(screen.getByPlaceholderText('输入问题，Enter 发送…'), { target: { value: '题面' } })
    expect(screen.getByRole('button', { name: '发送问题' })).toBeDisabled()
    expect(screen.getByText(/请先在笔试配置中.*识图模型/)).toBeInTheDocument()
    expect(apiMock.ask).not.toHaveBeenCalled()
  })

  it('sends an exam question with a pasted image using an exam-only model', async () => {
    useInterviewStore.setState(({ config }) => ({
      config: config && {
        ...config, written_exam_mode: true, written_exam_model_ids: ['exam'], assist_model_ids: [],
        models: [{ id: 'exam', name: 'Exam', supports_think: false, supports_vision: true, enabled: true, has_api_key: true }],
      },
    }))
    render(<ControlBar />)
    const input = screen.getByPlaceholderText('输入问题，Enter 发送…')
    const file = new File(['image'], 'shot.png', { type: 'image/png' })
    fireEvent.change(input, { target: { value: '分析题面' } })
    expect(screen.getByRole('button', { name: '发送问题' })).toBeEnabled()
    fireEvent.paste(input, { clipboardData: { items: [{ type: 'image/png', getAsFile: () => file }] } })
    await waitFor(() => expect(screen.getByAltText('screenshot')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: '发送问题' }))
    await waitFor(() => expect(apiMock.ask).toHaveBeenCalledWith('分析题面', expect.stringMatching(/^data:image/)))
  })

  it('blocks pasted screenshots when no enabled model supports vision', async () => {
    useInterviewStore.setState(({ config }) => ({
      config: config && {
        ...config,
        models: [
          { id: 'text', name: 'TextOnly', supports_think: false, supports_vision: false, enabled: true },
          { id: 'screen', name: 'ScreenshotModel', supports_think: false, supports_vision: true, enabled: true },
        ],
        assist_model_ids: ['text'],
        assist_screenshot_model_id: 'screen',
      },
    }))
    render(<ControlBar />)

    const input = screen.getByPlaceholderText('输入问题，Enter 发送…')
    const file = new File(['fake'], 'shot.png', { type: 'image/png' })
    fireEvent.paste(input, {
      clipboardData: {
        items: [
          {
            type: 'image/png',
            getAsFile: () => file,
          },
        ],
      },
    })

    fireEvent.click(screen.getByRole('button', { name: '发送问题' }))

    expect(apiMock.ask).not.toHaveBeenCalled()
    expect(screen.getByText('请先在实时辅助中勾选支持识图的答题模型，再粘贴图片')).toBeInTheDocument()
  })

  it('allows pasted screenshots to fall back to an enabled vision model', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        models: [
          { name: 'TextOnly', supports_think: false, supports_vision: false, enabled: true },
          { name: 'VisionBackup', supports_think: false, supports_vision: true, enabled: true },
        ],
      },
      modelHealth: { 0: 'ok', 1: 'ok' },
    } as any)

    render(<ControlBar />)

    const input = screen.getByPlaceholderText('输入问题，Enter 发送…')
    const file = new File(['fake'], 'shot.png', { type: 'image/png' })
    fireEvent.paste(input, {
      clipboardData: {
        items: [
          {
            type: 'image/png',
            getAsFile: () => file,
          },
        ],
      },
    })

    expect(screen.getByText(/自动使用「VisionBackup」/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '发送问题' }))

    await waitFor(() => {
      expect(apiMock.ask).toHaveBeenCalledWith('', 'data:image/png;base64,xxx')
    })
  })

  it('forces the prompt overlay on when starting written exam mode', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        written_exam_mode: true,
      },
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: /开始笔试/ }))

    await waitFor(() => expect(apiMock.start).toHaveBeenCalledWith(null, null))
    expect(window.electronAPI?.syncOverlayWindow).toHaveBeenCalledWith(expect.objectContaining({
      enabled: true,
      visible: true,
      mode: 'prompt',
      showBg: false,
    }))
    expect(useUiPrefsStore.getState().interviewOverlayEnabled).toBe(true)
    expect(useUiPrefsStore.getState().interviewOverlayMode).toBe('prompt')
  })

  it('resumes written exam mode without passing selected audio devices', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        written_exam_mode: true,
      },
      isRecording: true,
      isPaused: true,
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '继续' }))

    await waitFor(() => expect(apiMock.resume).toHaveBeenCalledWith())
  })

  it('blocks an unassigned interviewer engine while allowing manual questions', async () => {
    useInterviewStore.setState((state) => ({ config: { ...state.config!, stt_provider: '', stt_configured: {} } }))
    render(<ControlBar />)
    expect(screen.getByRole('button', { name: '开始面试' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '配置面试官 ASR' }))
    expect(useInterviewStore.getState().settingsDrawerTab).toBe('realtime')
    expect(screen.getByPlaceholderText(/输入问题/)).not.toBeDisabled()
  })

  it('resumes the captured microphone route after changing the next session engine', async () => {
    const candidateStart = vi.spyOn(candidateMmpgController, 'start').mockResolvedValue(undefined)
    try {
      useInterviewStore.setState((state) => ({
        config: { ...state.config!, stt_provider: 'doubao', candidate_asr_enabled: false, candidate_stt_provider: 'generic' },
        candidateCaptureProvider: 'doubao_mmpg',
        devices: [{ id: 12, name: 'Microphone', channels: 1, is_loopback: false, host_api: 'WASAPI' }],
        singleMic: { id: 'paused-route', device_id: 12, role: 'interviewer', switching: false },
        isRecording: true,
        isPaused: true,
      }))
      render(<ControlBar />)
      fireEvent.click(screen.getByRole('button', { name: '继续' }))
      await waitFor(() => expect(candidateStart).toHaveBeenCalledWith('resume', 12, null, undefined, expect.any(Function), 'paused-route'))
    } finally {
      candidateStart.mockRestore()
    }
  })

  it('resumes a paused MMPG owner after the next session is configured for written exam', async () => {
    const start = vi.spyOn(mmpgController, 'start').mockResolvedValue(undefined)
    try {
      useInterviewStore.setState((state) => ({
        config: { ...state.config, written_exam_mode: true, stt_provider: 'whisper' },
        isRecording: true,
        isPaused: true,
        interviewerStreaming: true,
      }))
      render(<ControlBar />)
      fireEvent.click(screen.getByRole('button', { name: '继续' }))
      await waitFor(() => expect(start).toHaveBeenCalledWith('resume', 1, null, undefined))
      expect(apiMock.resume).not.toHaveBeenCalled()
    } finally {
      start.mockRestore()
    }
  })

  it('blocks interview resume when no audio device is selected', async () => {
    useInterviewStore.setState({
      devices: [],
      isRecording: true,
      isPaused: true,
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '继续' }))

    expect(apiMock.resume).not.toHaveBeenCalled()
    expect(screen.getByText('请先选择音频设备')).toBeInTheDocument()
  })

  it('hides the resume mount in written exam mode', () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        written_exam_mode: true,
      },
    } as any)

    render(<ControlBar />)

    expect(screen.queryByTestId('resume-mount-inline')).not.toBeInTheDocument()
  })

  it('keeps the resume mount available in interview mode', () => {
    render(<ControlBar />)

    expect(screen.getByTestId('resume-mount-inline')).toBeInTheDocument()
  })

  it('hides noisy software audio devices by default with a show all escape hatch', () => {
    useInterviewStore.setState({
      devices: [
        { id: 1, name: 'BlackHole 2ch', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 2, name: 'ZoomAudioDevice', channels: 2, is_loopback: false, host_api: 'Core Audio' },
        { id: 3, name: 'Microsoft Teams Audio', channels: 2, is_loopback: false, host_api: 'Core Audio' },
        { id: 4, name: 'MacBook Pro Microphone', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '选择会议音频设备' }))

    expect(screen.getAllByText(/BlackHole 2ch/).length).toBeGreaterThan(0)
    expect(screen.getByText('MacBook Pro Microphone')).toBeInTheDocument()
    expect(screen.queryByText('ZoomAudioDevice')).not.toBeInTheDocument()
    expect(screen.queryByText('Microsoft Teams Audio')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /显示全部设备/ }))

    expect(screen.getByText('ZoomAudioDevice')).toBeInTheDocument()
    expect(screen.getByText('Microsoft Teams Audio')).toBeInTheDocument()
  })

  it('allows selecting a hidden audio device after showing all devices', () => {
    useInterviewStore.setState({
      devices: [
        { id: 1, name: 'MacBook Pro Microphone', channels: 1, is_loopback: false, host_api: 'Core Audio' },
        { id: 2, name: 'ZoomAudioDevice', channels: 2, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '选择会议音频设备' }))
    fireEvent.click(screen.getByRole('button', { name: /显示全部设备/ }))
    fireEvent.click(screen.getByText('ZoomAudioDevice'))

    expect(screen.getByRole('button', { name: '选择会议音频设备' })).toHaveTextContent('当前：ZoomAudioDevice')
  })

  it('refreshes audio devices from the picker', async () => {
    apiMock.getDevices.mockResolvedValue({
      devices: [
        { id: 9, name: 'USB Headset Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
      platform: { platform: 'Darwin', needs_virtual_device: false, instructions: '' },
    })

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '选择会议音频设备' }))
    fireEvent.click(screen.getByRole('button', { name: '刷新设备列表' }))

    expect(apiMock.getDevices).toHaveBeenCalledTimes(1)
    expect(await screen.findByText('USB Headset Mic')).toBeInTheDocument()
  })

  it('uses fresh interview copy when refreshing devices after mode changes', async () => {
    apiMock.getDevices.mockResolvedValue({
      devices: [
        { id: 9, name: 'USB Headset Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
      platform: { platform: 'Darwin', needs_virtual_device: false, instructions: '' },
    })
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        written_exam_mode: true,
      },
    } as any)

    render(<ControlBar />)

    act(() => {
      useInterviewStore.setState({
        config: {
          ...(useInterviewStore.getState().config as object),
          written_exam_mode: false,
        },
      } as any)
    })

    fireEvent.click(screen.getByRole('button', { name: '选择会议音频设备' }))
    fireEvent.click(screen.getByRole('button', { name: '刷新设备列表' }))

    await waitFor(() => {
      expect(useInterviewStore.getState().toastMessage).toBe('音频设备已刷新')
    })
  })

  it('ignores rapid duplicate audio device refreshes while refresh is pending', async () => {
    const refresh = deferred<{ devices: any[]; platform: null }>()
    apiMock.getDevices.mockReturnValueOnce(refresh.promise)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '选择会议音频设备' }))
    const refreshButton = screen.getByRole('button', { name: '刷新设备列表' })
    act(() => {
      refreshButton.click()
      refreshButton.click()
    })

    expect(apiMock.getDevices).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: '刷新设备列表' })).toBeDisabled()

    await act(async () => {
      refresh.resolve({
        devices: [{ id: 9, name: 'USB Headset Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' }],
        platform: null,
      })
      await refresh.promise
    })

    expect(await screen.findByText('USB Headset Mic')).toBeInTheDocument()
  })

  it('warns when no model is enabled', () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        models: [
          { name: 'Disabled A', supports_think: false, supports_vision: false, enabled: false },
          { name: 'Disabled B', supports_think: false, supports_vision: false, enabled: false },
        ],
      },
      modelHealth: { 0: 'error', 1: 'error' },
    } as any)

    render(<ControlBar />)

    expect(screen.getByText('请先在实时辅助中勾选可用的答题模型。')).toBeInTheDocument()
  })

  it('blocks manual questions when no answer model is enabled', () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        models: [
          { name: 'Disabled A', supports_think: false, supports_vision: false, enabled: false },
        ],
      },
      modelHealth: { 0: 'error' },
    } as any)

    render(<ControlBar />)

    fireEvent.change(screen.getByPlaceholderText('输入问题，Enter 发送…'), {
      target: { value: '解释一下 B 树和 B+ 树' },
    })

    const send = screen.getByRole('button', { name: '发送问题' })
    expect(send).toBeDisabled()
    expect(send).toHaveAttribute('title', '请先在实时辅助中勾选可用的答题模型')
    fireEvent.click(send)
    expect(apiMock.ask).not.toHaveBeenCalled()
  })

  it('prevents duplicate manual submissions while a question is in flight', async () => {
    let resolveAsk: ((value: unknown) => void) | null = null
    const pendingAsk = new Promise((resolve) => {
      resolveAsk = resolve
    })
    apiMock.ask.mockReturnValueOnce(pendingAsk)

    render(<ControlBar />)

    const input = screen.getByPlaceholderText('输入问题，Enter 发送…')
    fireEvent.change(input, {
      target: { value: '解释一下 Redis 持久化' },
    })

    const send = screen.getByRole('button', { name: '发送问题' })
    fireEvent.click(send)

    await waitFor(() => {
      expect(apiMock.ask).toHaveBeenCalledWith('解释一下 Redis 持久化', undefined)
    })
    await waitFor(() => {
      expect(send).toBeDisabled()
    })
    expect(send).toHaveAttribute('title', '正在提交问题')

    fireEvent.click(send)
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    expect(apiMock.ask).toHaveBeenCalledTimes(1)

    await act(async () => {
      if (!resolveAsk) throw new Error('ask resolver was not captured')
      resolveAsk({ ok: true })
      await pendingAsk
    })

    await waitFor(() => {
      expect(input).toHaveValue('')
    })
  })

  it('surfaces stop failures instead of swallowing them', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    apiMock.stop.mockRejectedValue(new Error('stop down'))
    useInterviewStore.setState({ isRecording: true, isPaused: false } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: /结束面试/ }))

    expect(await screen.findByText('结束面试失败：stop down')).toBeInTheDocument()
  })

  it('uses fresh written-exam stop copy after mode changes while recording', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    apiMock.stop.mockRejectedValue(new Error('stop down'))
    useInterviewStore.setState({ isRecording: true, isPaused: false } as any)

    render(<ControlBar />)

    act(() => {
      useInterviewStore.setState({
        config: {
          ...(useInterviewStore.getState().config as object),
          written_exam_mode: true,
        },
      } as any)
    })
    fireEvent.click(screen.getByRole('button', { name: /^结束$/ }))

    expect(confirmSpy).toHaveBeenCalledWith('结束本次笔试？当前答案会保留在页面上。')
    expect(await screen.findByText('结束笔试失败：stop down')).toBeInTheDocument()
  })

  it('surfaces cancel-generation failures instead of swallowing them', async () => {
    apiMock.cancelAsk.mockRejectedValue(new Error('cancel down'))
    useInterviewStore.setState({ streamingIds: ['qa-1'] } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '取消正在生成的回答' }))

    expect(await screen.findByText('取消生成失败：cancel down')).toBeInTheDocument()
  })

  it('cleans up the cancel-generation reset timer on unmount', async () => {
    vi.useFakeTimers()
    const clearTimeoutSpy = vi.spyOn(window, 'clearTimeout')
    try {
      useInterviewStore.setState({ streamingIds: ['qa-1'] } as any)
      const { unmount } = render(<ControlBar />)

      fireEvent.click(screen.getByRole('button', { name: '取消正在生成的回答' }))
      await act(async () => {})

      unmount()

      expect(clearTimeoutSpy).toHaveBeenCalled()
    } finally {
      clearTimeoutSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it('starts assist mode with meeting audio and candidate microphone devices', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 10, name: 'System Loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '开始面试' }))

    await waitFor(() => {
      expect(apiMock.start).toHaveBeenCalledWith(10, 11)
    })
  })

  it('starts a single-mic test with one selected physical input', async () => {
    useInterviewStore.setState((state) => ({
      config: { ...state.config!, candidate_asr_enabled: true },
      devices: [
        { id: 10, name: 'Meeting Audio', channels: 2, is_loopback: true, host_api: 'WASAPI' },
        { id: 11, name: 'Microphone', channels: 1, is_loopback: false, host_api: 'WASAPI' },
      ],
    }))
    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '单人测试' }))
    expect(screen.getByRole('button', { name: '选择测试麦克风' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '选择会议音频设备' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '选择我的麦克风' })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '开始单人测试' }))
    await waitFor(() => expect(apiMock.start).toHaveBeenCalledWith(11, 11, expect.stringMatching(/^[0-9a-f]{32}$/)))
  })

  it('locks role controls until the backend confirms a switch and keeps end available', async () => {
    const initial = { id: 'test-session', device_id: 11, role: 'interviewer', switching: false } as const
    useInterviewStore.setState((state) => ({
      config: { ...state.config!, candidate_asr_enabled: true },
      singleMic: initial,
      isRecording: true,
    }))
    const switchRequest = deferred<{ ok: true }>()
    apiMock.switchSingleMicRole.mockReturnValueOnce(switchRequest.promise)
    render(<ControlBar />)

    const candidate = screen.getByRole('button', { name: '候选人' })
    const interviewer = screen.getByRole('button', { name: '面试官' })
    fireEvent.click(candidate)
    fireEvent.click(candidate)
    expect(apiMock.switchSingleMicRole).toHaveBeenCalledExactlyOnceWith(initial.id, 'candidate')
    expect(candidate).toBeDisabled()
    expect(interviewer).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('切换中，等待上一句转写完成')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '结束单人测试' })).toBeEnabled()

    act(() => useInterviewStore.setState({ singleMic: { ...initial, switching: true } }))
    await act(async () => switchRequest.resolve({ ok: true }))
    expect(candidate).toBeDisabled()
    expect(candidate).toHaveAttribute('aria-pressed', 'false')

    act(() => useInterviewStore.setState({ singleMic: { ...initial, role: 'candidate' } }))
    expect(interviewer).toBeEnabled()
    expect(candidate).toHaveAttribute('aria-pressed', 'true')
  })

  it('resumes a reconnected test with its original microphone and session id', async () => {
    useInterviewStore.setState((state) => ({
      config: { ...state.config!, candidate_asr_enabled: true },
      devices: [
        { id: 11, name: 'Default Mic', channels: 1, is_loopback: false, host_api: 'WASAPI' },
        { id: 12, name: 'Test Mic', channels: 1, is_loopback: false, host_api: 'WASAPI' },
      ],
      singleMic: { id: 'existing', device_id: 12, role: 'candidate', switching: false },
      isRecording: true,
      isPaused: true,
    }))
    render(<ControlBar />)

    expect(screen.getByRole('button', { name: '选择测试麦克风' })).toHaveTextContent('Test Mic')
    expect(screen.getByRole('button', { name: '候选人' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: '继续' }))
    await waitFor(() => expect(apiMock.resume).toHaveBeenCalledWith(12, 12, 'existing'))
  })

  it('configures both MMPG transports with the same microphone and test session', async () => {
    const interviewer = deferred<void>()
    const primaryStart = vi.spyOn(mmpgController, 'start').mockReturnValue(interviewer.promise)
    const prepareCandidate = vi.fn()
    const candidateStart = vi.spyOn(candidateMmpgController, 'start').mockImplementation(async (_mode, _device, _mic, _platform, prepare) => {
      prepareCandidate(await prepare?.())
    })
    try {
      useInterviewStore.setState((state) => ({
        config: { ...state.config!, stt_provider: 'doubao_mmpg', candidate_asr_enabled: true, candidate_stt_provider: 'doubao_mmpg' },
        devices: [
          { id: 11, name: 'Microphone', channels: 1, is_loopback: false, host_api: 'WASAPI' },
        ],
      }))
      render(<ControlBar />)
      fireEvent.click(screen.getByRole('button', { name: '单人测试' }))
      fireEvent.click(screen.getByRole('button', { name: '开始单人测试' }))

      const sessionId = primaryStart.mock.calls[0][5]
      expect(sessionId).toMatch(/^[0-9a-f]{32}$/)
      expect(primaryStart).toHaveBeenCalledWith('start', 11, 11, undefined, undefined, sessionId)
      expect(candidateStart).toHaveBeenCalledWith('start', 11, null, undefined, expect.any(Function), sessionId)
      expect(prepareCandidate).not.toHaveBeenCalled()
      await act(async () => interviewer.resolve())
      expect(prepareCandidate).toHaveBeenCalledWith(true)
    } finally {
      candidateStart.mockRestore()
      primaryStart.mockRestore()
    }
  })

  it('shows a role-switch failure and allows retry from the original role', async () => {
    useInterviewStore.setState({
      singleMic: { id: 'same-session', device_id: 11, role: 'interviewer', switching: false },
      isRecording: true,
    })
    apiMock.switchSingleMicRole.mockRejectedValueOnce(new Error('上一句转写超时，请重试'))
    render(<ControlBar />)

    const candidate = screen.getByRole('button', { name: '候选人' })
    fireEvent.click(candidate)
    expect(await screen.findByText('上一句转写超时，请重试')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '面试官' })).toHaveAttribute('aria-pressed', 'true')
    expect(candidate).toBeEnabled()
    fireEvent.click(candidate)
    expect(apiMock.switchSingleMicRole).toHaveBeenCalledTimes(2)
  })

  it('uses the server response when the final role WebSocket message is missed', async () => {
    const state = { id: 'current', device_id: 11, role: 'interviewer', switching: false } as const
    useInterviewStore.setState({ singleMic: state, isRecording: true })
    apiMock.switchSingleMicRole.mockResolvedValueOnce({
      ok: true,
      single_mic: { ...state, role: 'candidate' },
    })
    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '候选人' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '候选人' })).toHaveAttribute('aria-pressed', 'true'))
    expect(screen.getByRole('button', { name: '面试官' })).toBeEnabled()
  })

  it('pauses the matching single-mic session when candidate MMPG cannot open', async () => {
    const candidateStart = vi.spyOn(candidateMmpgController, 'start').mockRejectedValueOnce(new Error('无法初始化页面音频'))
    try {
      useInterviewStore.setState((state) => ({
        config: { ...state.config!, candidate_asr_enabled: true, candidate_stt_provider: 'doubao_mmpg' },
        devices: [{ id: 11, name: 'Microphone', channels: 1, is_loopback: false, host_api: 'WASAPI' }],
      }))
      render(<ControlBar />)
      fireEvent.click(screen.getByRole('button', { name: '单人测试' }))
      fireEvent.click(screen.getByRole('button', { name: '开始单人测试' }))
      await waitFor(() => {
        const sessionId = apiMock.start.mock.calls[0][2]
        expect(apiMock.pause).toHaveBeenCalledWith(sessionId)
      })
      expect(screen.getByText('我的麦克风 · 无法初始化页面音频')).toBeInTheDocument()
    } finally {
      candidateStart.mockRestore()
    }
  })

  it('ignores a late candidate MMPG startup failure after ending the test', async () => {
    const candidate = deferred<void>()
    const candidateStart = vi.spyOn(candidateMmpgController, 'start').mockReturnValueOnce(candidate.promise)
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    try {
      useInterviewStore.setState((state) => ({
        config: { ...state.config!, candidate_asr_enabled: true, candidate_stt_provider: 'doubao_mmpg' },
        devices: [{ id: 11, name: 'Microphone', channels: 1, is_loopback: false, host_api: 'WASAPI' }],
      }))
      render(<ControlBar />)
      fireEvent.click(screen.getByRole('button', { name: '单人测试' }))
      fireEvent.click(screen.getByRole('button', { name: '开始单人测试' }))
      const sessionId = apiMock.start.mock.calls[0][2]
      act(() => useInterviewStore.setState({
        isRecording: true,
        singleMic: { id: sessionId, device_id: 11, role: 'interviewer', switching: false },
      }))
      const end = screen.getByRole('button', { name: '结束单人测试' })
      await waitFor(() => expect(end).toBeEnabled())
      fireEvent.click(end)
      await waitFor(() => expect(apiMock.stop).toHaveBeenCalledTimes(1))
      await act(async () => candidate.reject(new Error('旧尝试失败')))

      expect(apiMock.pause).not.toHaveBeenCalled()
      expect(screen.queryByText(/旧尝试失败/)).not.toBeInTheDocument()
    } finally {
      candidateStart.mockRestore()
      confirm.mockRestore()
    }
  })

  it.each(['start', 'resume'] as const)('gates candidate MMPG %s on a separate interviewer connection', async (mode) => {
    const interviewer = deferred<void>()
    const primaryStart = vi.spyOn(mmpgController, 'start').mockReturnValue(interviewer.promise)
    const micReady = vi.fn()
    const micStart = vi.spyOn(candidateMmpgController, 'start').mockImplementation(async (_mode, _device, _mic, _platform, prepare) => {
      micReady(await prepare?.())
    })
    try {
      useInterviewStore.setState((state) => ({
        config: { ...state.config!, stt_provider: 'doubao_mmpg', candidate_asr_enabled: true, candidate_stt_provider: 'doubao_mmpg' },
        devices: [
          { id: 10, name: 'Meeting Audio', channels: 2, is_loopback: true, host_api: 'WASAPI' },
          { id: 11, name: 'Microphone', channels: 1, is_loopback: false, host_api: 'WASAPI' },
        ],
        isRecording: mode === 'resume',
        isPaused: mode === 'resume',
        interviewerStreaming: mode === 'resume',
      }))
      render(<ControlBar />)
      fireEvent.click(screen.getByRole('button', { name: mode === 'resume' ? '继续' : '开始面试' }))
      expect(primaryStart).toHaveBeenCalledWith(mode, 10, 11, undefined)
      expect(micStart).toHaveBeenCalledWith(mode, 11, null, undefined, expect.any(Function))
      expect(micReady).not.toHaveBeenCalled()
      await act(async () => interviewer.resolve())
      expect(micReady).toHaveBeenCalledWith(true)
    } finally {
      micStart.mockRestore()
      primaryStart.mockRestore()
    }
  })

  it('prevents duplicate start requests while the interview is starting', async () => {
    let resolveStart: ((value: unknown) => void) | null = null
    const pendingStart = new Promise((resolve) => {
      resolveStart = resolve
    })
    apiMock.start.mockReturnValueOnce(pendingStart)

    render(<ControlBar />)

    const start = screen.getByRole('button', { name: '开始面试' })
    fireEvent.click(start)
    fireEvent.click(start)

    await waitFor(() => {
      expect(apiMock.start).toHaveBeenCalledWith(1, null)
    })
    expect(apiMock.start).toHaveBeenCalledTimes(1)
    expect(start).toBeDisabled()

    await act(async () => {
      if (!resolveStart) throw new Error('start resolver was not captured')
      resolveStart({ ok: true })
      await pendingStart
    })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始面试' })).not.toBeDisabled()
    })
  })

  it('prefers a candidate microphone different from the meeting audio device', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 11, name: 'Built-in Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
        { id: 12, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '选择我的麦克风' })).toHaveTextContent('当前：USB Mic')
    })

    fireEvent.click(screen.getByRole('button', { name: '开始面试' }))

    await waitFor(() => {
      expect(apiMock.start).toHaveBeenCalledWith(11, 12)
    })
  })

  it('warns and skips candidate ASR when candidate microphone matches meeting audio', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    expect(await screen.findByText(/本次不会单独记录你的回答/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '开始面试' }))

    await waitFor(() => {
      expect(apiMock.start).toHaveBeenCalledWith(11, null)
    })
  })

  it('does not pass candidate microphone when candidate ASR is disabled', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: false,
      },
      devices: [
        { id: 10, name: 'System Loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    expect(screen.getByRole('status', { name: '我的回答记录状态' })).toHaveTextContent('我的回答记录已关闭')
    fireEvent.click(screen.getByRole('button', { name: '开始面试' }))

    await waitFor(() => {
      expect(apiMock.start).toHaveBeenCalledWith(10, null)
    })
  })

  it('labels the candidate microphone as answer recording when candidate ASR is enabled', () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 10, name: 'System Loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    expect(screen.getByText('我的麦克风 · 记录我的回答')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '选择我的麦克风' })).toBeInTheDocument()
  })

  it('plays a speaker test sound from the meeting audio picker', async () => {
    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '测试音频输出' }))

    await waitFor(() => {
      expect(apiMock.audioOutputTest).toHaveBeenCalledTimes(1)
    })
  })

  it('ignores rapid duplicate speaker tests while playback test is pending', async () => {
    const output = deferred<{ ok: boolean; elapsed_sec: number }>()
    apiMock.audioOutputTest.mockReturnValueOnce(output.promise)

    render(<ControlBar />)

    const outputButton = screen.getByRole('button', { name: '测试音频输出' })
    act(() => {
      outputButton.click()
      outputButton.click()
    })

    expect(apiMock.audioOutputTest).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: '测试音频输出' })).toBeDisabled()

    await act(async () => {
      output.resolve({ ok: true, elapsed_sec: 1 })
      await output.promise
    })
  })

  it('opens a live candidate microphone meter and renders the input level', async () => {
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 10, name: 'System Loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    fireEvent.click(screen.getByRole('button', { name: '测试麦克风输入' }))

    await waitFor(() => {
      expect(apiMock.audioInputMonitorStart).toHaveBeenCalledWith(11)
    })
    expect(await screen.findByText('50%')).toBeInTheDocument()
    expect(screen.getByRole('dialog', { name: '麦克风输入测试' })).toBeInTheDocument()
  })

  it('ignores rapid duplicate microphone meter starts while input test is pending', async () => {
    const input = deferred<{
      running: boolean
      device_id: number
      rms: number
      peak: number
      level_pct: number
      has_signal: boolean
      error: null
    }>()
    apiMock.audioInputMonitorStart.mockReturnValueOnce(input.promise)
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 10, name: 'System Loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    const inputButton = screen.getByRole('button', { name: '测试麦克风输入' })
    act(() => {
      inputButton.click()
      inputButton.click()
    })

    expect(apiMock.audioInputMonitorStart).toHaveBeenCalledTimes(1)
    expect(apiMock.audioInputMonitorStart).toHaveBeenCalledWith(11)
    expect(screen.getByRole('button', { name: '测试麦克风输入' })).toBeDisabled()

    await act(async () => {
      input.resolve({
        running: true,
        device_id: 11,
        rms: 0.02,
        peak: 0.1,
        level_pct: 50,
        has_signal: true,
        error: null,
      })
      await input.promise
    })

    expect(await screen.findByText('50%')).toBeInTheDocument()
  })

  it('ignores stale microphone meter starts after switching candidate microphones', async () => {
    const input = deferred<{
      running: boolean
      device_id: number
      rms: number
      peak: number
      level_pct: number
      has_signal: boolean
      error: null
    }>()
    apiMock.audioInputMonitorStart.mockReturnValueOnce(input.promise)
    useInterviewStore.setState({
      config: {
        ...(useInterviewStore.getState().config as object),
        candidate_asr_enabled: true,
      },
      devices: [
        { id: 10, name: 'System Loopback', channels: 2, is_loopback: true, host_api: 'Core Audio' },
        { id: 11, name: 'USB Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
        { id: 12, name: 'Desk Mic', channels: 1, is_loopback: false, host_api: 'Core Audio' },
      ],
    } as any)

    render(<ControlBar />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '选择我的麦克风' })).toHaveTextContent('当前：USB Mic')
    })
    fireEvent.click(screen.getByRole('button', { name: '测试麦克风输入' }))
    expect(apiMock.audioInputMonitorStart).toHaveBeenCalledWith(11)

    fireEvent.click(screen.getByRole('button', { name: '选择我的麦克风' }))
    fireEvent.click(screen.getByText('Desk Mic'))
    expect(screen.getByRole('button', { name: '选择我的麦克风' })).toHaveTextContent('当前：Desk Mic')

    await act(async () => {
      input.resolve({
        running: true,
        device_id: 11,
        rms: 0.04,
        peak: 0.2,
        level_pct: 80,
        has_signal: true,
        error: null,
      })
      await input.promise
    })

    expect(apiMock.audioInputMonitorStop).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('dialog', { name: '麦克风输入测试' })).not.toBeInTheDocument()
    expect(screen.queryByText('80%')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '测试麦克风输入' })).not.toBeDisabled()
  })

  it('skips re-rendering during streaming answer chunks and updates on latest candidate transcription', () => {
    const onRender = vi.fn()
    useInterviewStore.setState({
      qaPairs: [{
        id: 'qa-1',
        question: '什么是 MVCC？',
        answer: '初始',
        thinkContent: '',
        isThinking: false,
        timestamp: 1710000000,
        status: 'streaming',
      }],
      streamingIds: ['qa-1'],
      transcriptions: ['第一句面试官转写'],
      candidateTranscriptions: [],
    } as any)

    render(
      <Profiler id="control-bar" onRender={onRender}>
        <ControlBar />
      </Profiler>,
    )
    const rendersAfterMount = onRender.mock.calls.length

    act(() => {
      for (let i = 1; i <= 5; i++) {
        useInterviewStore.setState((state) => ({
          qaPairs: state.qaPairs.map((qa) => (
            qa.id === 'qa-1' ? { ...qa, answer: `${qa.answer} chunk-${i}` } : qa
          )),
          transcriptions: [...state.transcriptions, `后续面试官转写 ${i}`],
        }))
      }
    })

    expect(onRender).toHaveBeenCalledTimes(rendersAfterMount)

    act(() => {
      useInterviewStore.setState({ candidateTranscriptions: ['我的最新回答'] })
    })

    expect(onRender.mock.calls.length).toBeGreaterThan(rendersAfterMount)
    expect(screen.getByText('我的回答记录：我的最新回答')).toBeInTheDocument()
  })
})
