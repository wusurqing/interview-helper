import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

  it('offers only configured engines for both roles', () => {
    seed({ candidate_stt_provider: '', candidate_asr_enabled: false })
    render(<SpeechRolesSettings />)
    for (const label of ['面试官 ASR 引擎', '我的回答 ASR 引擎']) {
      const select = within(screen.getByLabelText(label))
      expect(select.getAllByRole('option').map((option) => option.textContent)).toEqual([
        '待选择引擎', '火山原生豆包', '豆包 MMPG',
      ])
    }
  })

  it('keeps an existing unconfigured assignment without offering or enabling it', async () => {
    seed({ stt_provider: 'generic', candidate_asr_enabled: false })
    render(<SpeechRolesSettings />)
    for (const label of ['面试官 ASR 引擎', '我的回答 ASR 引擎']) {
      const select = screen.getByLabelText<HTMLSelectElement>(label)
      expect(select).toHaveValue('generic')
      expect(select.selectedOptions[0]).toBeDisabled()
      expect(select.selectedOptions[0]).toHaveTextContent('OpenAI 兼容 ASR（待配置）')
    }
    expect(screen.getByRole('switch', { name: '我的回答记录（麦克风）' })).toBeDisabled()
    fireEvent.click(screen.getByRole('switch', { name: '自动检测问题并生成答案' }))
    fireEvent.click(screen.getByRole('button', { name: '保存语音使用设置' }))
    await waitFor(() => expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ auto_detect: false }))
  })

  it('opens voice settings from either unassigned role when no engines are ready', () => {
    seed({ stt_provider: '', candidate_stt_provider: '', candidate_asr_enabled: false, stt_configured: {} })
    const openConfig = vi.fn()
    render(<SpeechRolesSettings onOpenSpeechConfig={openConfig} />)
    const links = screen.getAllByRole('button', { name: '打开语音配置' })
    expect(links).toHaveLength(2)
    links.forEach((link) => fireEvent.click(link))
    expect(openConfig).toHaveBeenCalledTimes(2)
  })

  it('blocks saving a new role engine if its credentials became unavailable during editing', () => {
    render(<SpeechRolesSettings />)
    fireEvent.change(screen.getByLabelText('面试官 ASR 引擎'), { target: { value: 'doubao_mmpg' } })
    act(() => useInterviewStore.setState((state) => ({
      config: { ...state.config!, stt_configured: { doubao: true, doubao_mmpg: false } },
    })))
    fireEvent.click(screen.getByRole('button', { name: '保存语音使用设置' }))
    expect(updateConfigAndRefresh).not.toHaveBeenCalled()
    expect(screen.getByText(/引擎尚未配置/)).toBeInTheDocument()
  })

  it('blocks saving a newly enabled microphone if its engine became unavailable', () => {
    seed({ candidate_asr_enabled: false, candidate_stt_provider: 'doubao' })
    render(<SpeechRolesSettings />)
    fireEvent.click(screen.getByRole('switch', { name: '我的回答记录（麦克风）' }))
    act(() => useInterviewStore.setState((state) => ({
      config: { ...state.config!, stt_configured: { doubao: false } },
    })))
    fireEvent.click(screen.getByRole('button', { name: '保存语音使用设置' }))
    expect(updateConfigAndRefresh).not.toHaveBeenCalled()
    expect(screen.getByText(/引擎尚未配置/)).toBeInTheDocument()
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
    act(() => useInterviewStore.setState((state) => ({ config: { ...state.config!, generic_stt_api_key_configured: true } })))
    expect(screen.getByLabelText('我的回答 ASR 引擎')).toHaveValue('doubao_mmpg')
    expect(screen.getByText('有未保存更改')).toBeInTheDocument()
  })
})
