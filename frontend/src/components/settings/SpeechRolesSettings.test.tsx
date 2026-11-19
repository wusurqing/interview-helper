import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { mmpgFormFromConfig } from '@/lib/mmpg/settings'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'
import SpeechRolesSettings from './SpeechRolesSettings'

vi.mock('@/lib/configSync', () => ({ updateConfigAndRefresh: vi.fn() }))

function seed(config: Partial<AppConfig> = {}) {
  useInterviewStore.setState({
    config: {
      stt_provider: 'doubao',
      candidate_stt_provider: 'generic',
      candidate_asr_enabled: true,
      auto_detect: true,
      stt_configured: { doubao: true, generic: false, doubao_mmpg: true },
      mmpg: mmpgFormFromConfig(),
      ...config,
    } as AppConfig,
    platformInfo: null,
    isRecording: false,
    sttLoaded: false,
    candidateSttLoaded: false,
  })
}

describe('SpeechRolesSettings', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    seed()
    vi.mocked(updateConfigAndRefresh).mockImplementation(async (patch) => {
      const prior = useInterviewStore.getState().config!
      const config = { ...prior, ...patch, mmpg: { ...prior.mmpg, ...patch.mmpg as object | undefined } } as AppConfig
      useInterviewStore.setState({ config })
      return config
    })
  })

  it('shows segmented interviewer and common capture settings, along with distinct readiness', () => {
    const openConfig = vi.fn()
    render(<SpeechRolesSettings onOpenSpeechConfig={openConfig} />)
    expect(screen.getByLabelText('面试官 ASR 引擎')).toHaveValue('doubao')
    expect(screen.getByLabelText('我的回答 ASR 引擎')).toHaveValue('generic')
    expect(screen.getByText('分段采集')).toBeInTheDocument()
    expect(screen.getByLabelText('转写合并间隔（秒）')).toBeInTheDocument()
    expect(screen.getByLabelText('问题组最长等待（秒）')).toBeInTheDocument()
    expect(screen.queryByLabelText('答题前等待时长（毫秒）')).not.toBeInTheDocument()
    expect(screen.queryByText(/Whisper|FunASR|延迟预设|回答长度/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '打开语音配置' }))
    expect(openConfig).toHaveBeenCalledOnce()
  })

  it('writes only changed role controls, including clearing an interviewer assignment', async () => {
    render(<SpeechRolesSettings />)
    fireEvent.change(screen.getByLabelText('面试官 ASR 引擎'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('switch', { name: '自动检测问题并生成答案' }))
    fireEvent.click(screen.getByRole('button', { name: '保存语音使用设置' }))
    await waitFor(() => expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ stt_provider: '', auto_detect: false }))
  })

  it('shows MMPG answer timing but keeps shared capture for a segmented microphone', async () => {
    seed({ stt_provider: 'doubao_mmpg' })
    render(<SpeechRolesSettings />)
    expect(screen.getByText('分段采集')).toBeInTheDocument()
    expect(screen.queryByLabelText('转写合并间隔（秒）')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('判停静音时间（ms）')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('答题前等待时长（毫秒）'), { target: { value: '2400' } })
    fireEvent.click(screen.getByRole('button', { name: '保存语音使用设置' }))
    await waitFor(() => expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ mmpg: { answer_delay_ms: 2400 } }))
  })

  it('leaves microphone selection disabled until a removed local engine is replaced', () => {
    seed({ stt_provider: 'whisper', candidate_stt_provider: 'funasr', candidate_asr_enabled: true })
    render(<SpeechRolesSettings />)
    expect(screen.getByLabelText('面试官 ASR 引擎')).toHaveValue('')
    expect(screen.getByLabelText('我的回答 ASR 引擎')).toHaveValue('')
    expect(screen.getByRole('switch', { name: '我的回答记录（麦克风）' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('switch', { name: '我的回答记录（麦克风）' })).toBeDisabled()
  })

  it('keeps unsaved role changes after a provider credential refresh', () => {
    render(<SpeechRolesSettings />)
    fireEvent.change(screen.getByLabelText('我的回答 ASR 引擎'), { target: { value: 'doubao_mmpg' } })
    act(() => useInterviewStore.setState((state) => ({ config: { ...state.config!, generic_stt_api_key: 'refreshed' } })))
    expect(screen.getByLabelText('我的回答 ASR 引擎')).toHaveValue('doubao_mmpg')
    expect(screen.getByText('有未保存更改')).toBeInTheDocument()
  })
})
