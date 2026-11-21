import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import SpeechTab from './SpeechTab'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { mmpgFormFromConfig } from '@/lib/mmpg/settings'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'

const apiMock = vi.hoisted(() => ({ sttTest: vi.fn() }))
vi.mock('@/lib/api', () => ({ api: apiMock, getErrorMessage: (_error: unknown, fallback: string) => fallback }))
vi.mock('@/lib/configSync', () => ({ updateConfigAndRefresh: vi.fn(), refreshConfig: vi.fn() }))

function engine(name: string) {
  const section = screen.getByRole('heading', { name }).closest('section')
  if (!section) throw new Error('Engine section missing')
  return within(section)
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

describe('SpeechTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.sttTest.mockResolvedValue({ ok: true, text: '识别结果' })
    useInterviewStore.setState({
      config: {
        stt_provider: 'doubao',
        candidate_stt_provider: 'generic',
        candidate_asr_enabled: true,
        mmpg: { ...mmpgFormFromConfig(), answer_delay_ms: 2600, volc_secret_key_configured: true },
      } as AppConfig,
      isRecording: false,
    })
    vi.mocked(updateConfigAndRefresh).mockImplementation(async (patch) => {
      const prior = useInterviewStore.getState().config!
      const config = { ...prior, ...patch, mmpg: { ...prior.mmpg, ...patch.mmpg as object | undefined } } as AppConfig
      useInterviewStore.setState({ config })
      return config
    })
  })

  it('shows cloud engine configuration separately from role and timing controls', () => {
    render(<SpeechTab />)
    expect(screen.getByRole('heading', { name: '火山原生豆包' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'OpenAI 兼容 ASR' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '豆包 MMPG' })).toBeInTheDocument()
    expect(screen.queryByText(/Whisper|FunASR|千问|延迟预设|回答长度/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText('面试官 ASR 引擎')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('答题前等待时长')).not.toBeInTheDocument()
    expect(screen.getByLabelText('判停静音时间')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存并测试 MMPG' })).toBeInTheDocument()
  })

  it('saves only the changed engine fields and explicitly tests that engine', async () => {
    render(<SpeechTab />)
    const doubao = engine('火山原生豆包')
    const generic = engine('OpenAI 兼容 ASR')
    fireEvent.change(doubao.getByLabelText('API Key（新版控制台）'), { target: { value: 'draft-doubao' } })
    fireEvent.change(generic.getByLabelText('Base URL'), { target: { value: 'https://speech.example.test/v1' } })
    fireEvent.click(generic.getByRole('button', { name: '保存并测试' }))
    await waitFor(() => expect(apiMock.sttTest).toHaveBeenCalledWith({ provider: 'generic' }))
    expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ generic_stt_api_base_url: 'https://speech.example.test/v1' })
    expect(doubao.getByLabelText('API Key（新版控制台）')).toHaveValue('draft-doubao')
    expect(generic.getByText('连接成功')).toBeInTheDocument()
  })

  it('saves a cleared MMPG secret without writing role timing or engine selection', async () => {
    render(<SpeechTab />)
    const mmpg = engine('豆包 MMPG')
    expect(mmpg.getByLabelText('火山引擎 Secret Key')).toHaveAttribute('placeholder', '已配置 · 留空保留')
    fireEvent.click(mmpg.getByRole('button', { name: '清除 火山引擎 Secret Key' }))
    fireEvent.click(mmpg.getByRole('button', { name: '保存' }))
    await waitFor(() => expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ mmpg: { volc_secret_key: null } }))
  })

  it('coalesces double clicks while saving and testing the same engine', async () => {
    const test = deferred<{ ok: boolean; text: string }>()
    apiMock.sttTest.mockReturnValue(test.promise)
    render(<SpeechTab />)
    const generic = engine('OpenAI 兼容 ASR')
    const button = generic.getByRole('button', { name: '保存并测试' })
    act(() => {
      button.click()
      button.click()
    })
    await waitFor(() => expect(apiMock.sttTest).toHaveBeenCalledTimes(1))
    expect(generic.getByRole('button', { name: '测试中…' })).toBeDisabled()
    await act(async () => { test.resolve({ ok: true, text: '识别结果' }) })
  })

  it('masks configured segmented STT secrets and sends null only when explicitly cleared', async () => {
    useInterviewStore.setState({
      config: {
        ...useInterviewStore.getState().config!,
        doubao_stt_api_key_configured: true,
        doubao_stt_access_token_configured: true,
        generic_stt_api_key_configured: true,
        generic_stt_custom_headers_configured: true,
      },
    })
    render(<SpeechTab />)
    const doubao = engine('火山原生豆包')
    const generic = engine('OpenAI 兼容 ASR')

    expect(doubao.getByLabelText('API Key（新版控制台）')).toHaveValue('')
    expect(doubao.getByLabelText('API Key（新版控制台）')).toHaveAttribute('placeholder', '已配置 · 留空保留')
    expect(generic.getByLabelText('自定义 Header（可选）')).toHaveValue('')
    expect(generic.getByLabelText('自定义 Header（可选）')).toHaveAttribute('placeholder', '已配置 · 留空保留')

    // Clear and undo leaves the section clean.
    fireEvent.click(doubao.getByRole('button', { name: '清除 API Key（新版控制台）' }))
    expect(doubao.getByLabelText('API Key（新版控制台）')).toHaveAttribute('placeholder', '保存后清除')
    fireEvent.click(doubao.getByRole('button', { name: '撤销清除 API Key（新版控制台）' }))
    expect(doubao.getByRole('button', { name: '保存' })).toBeDisabled()

    // Clearing custom headers sends explicit null without touching API key.
    fireEvent.click(generic.getByRole('button', { name: '清除 自定义 Header（可选）' }))
    fireEvent.click(generic.getByRole('button', { name: '保存' }))
    await waitFor(() =>
      expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ generic_stt_custom_headers: null }),
    )
  })
})
