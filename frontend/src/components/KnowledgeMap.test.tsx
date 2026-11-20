import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import KnowledgeMap from './KnowledgeMap'
import { useInterviewStore } from '@/stores/configStore'

const apiMock = vi.hoisted(() => ({
  knowledgeSummary: vi.fn(),
  knowledgeHistory: vi.fn(),
  knowledgeReset: vi.fn(),
  knowledgeSettings: vi.fn(),
  saveKnowledgeSettings: vi.fn(),
  knowledgeAnalysisStatus: vi.fn(),
  startKnowledgeAnalysis: vi.fn(),
  ask: vi.fn(),
  audioInputMonitorStatus: vi.fn(),
}))

vi.mock('@/lib/api', () => ({ api: apiMock }))

describe('KnowledgeMap', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.knowledgeSettings.mockResolvedValue({
      settings: { model_index: 1, system_prompt: null, temperature: 0.3, max_tokens: 1024, think_effort: 'off' },
      default_system_prompt: '默认知识点提示词',
      output_contract: 'JSON 数组',
      models: [
        { index: 1, name: '知识模型', fingerprint: 'model-1', available: true, supports_think: true, supports_vision: true },
        { index: 2, name: '纯文本', fingerprint: 'model-2', available: true, supports_think: false },
        { index: 3, name: '停用模型', fingerprint: 'model-3', available: false, supports_think: false },
      ],
    })
    apiMock.knowledgeAnalysisStatus.mockResolvedValue({ pending_count: 2, job: null })
    useInterviewStore.setState({
      setToastMessage: vi.fn(),
      candidateTranscriptions: [
        '我刚才说的是先用监控定位慢查询，再用索引和缓存一起优化。',
      ],
      candidateSttLoaded: true,
      candidateSttLoading: false,
      candidateSttProvider: 'whisper',
      config: { candidate_asr_enabled: true },
    } as any)
    apiMock.knowledgeSummary.mockResolvedValue({
      tags: [
        { tag: 'React', count: 2, avg_score: 3, trend: 'down' },
        { tag: 'TS', count: 2, avg_score: 6, trend: 'stable' },
        { tag: 'CSS', count: 2, avg_score: 8, trend: 'up' },
      ],
    })
    apiMock.knowledgeHistory.mockResolvedValue({
      total: 2,
      records: [
        {
          id: 1,
          qa_id: 'qa-1',
          session_type: 'assist',
          question: 'Q1',
          answer: 'A1',
          candidate_answer: '候选人实际回答 A1',
          score: 5,
          tags: ['React'],
          created_at: 100,
        },
        {
          id: 2,
          qa_id: 'qa-2',
          session_type: 'assist',
          question: 'Q2',
          answer: 'A2',
          candidate_answer: '',
          score: 5,
          tags: ['TS'],
          created_at: 120,
        },
      ],
    })
    apiMock.audioInputMonitorStatus.mockResolvedValue({
      running: true,
      device_id: 1002,
      rms: 0.04,
      peak: 0.18,
      level_pct: 64,
      has_signal: true,
      error: null,
    })
  })

  it('renders merged history summary', async () => {
    render(<KnowledgeMap />)
    await waitFor(() => expect(screen.getByText(/历史记录/)).toBeInTheDocument())
    expect(screen.getByText(/×2/)).toBeInTheDocument()
  })

  it('saves independent analysis settings from the ability page', async () => {
    apiMock.saveKnowledgeSettings.mockResolvedValue({
      ...(await apiMock.knowledgeSettings()),
      settings: { model_index: 1, system_prompt: '自定义知识点提示词', temperature: 0.7, max_tokens: 2048, think_effort: 'low' },
    })
    render(<KnowledgeMap />)
    fireEvent.click(await screen.findByRole('button', { name: '分析设置' }))
    const picker = screen.getByRole('button', { name: '分析模型' })
    expect(picker).toHaveTextContent('识图')
    fireEvent.click(picker)
    expect(screen.getByRole('radio', { name: '停用模型' })).toBeDisabled()
    fireEvent.change(screen.getByLabelText('推理强度'), { target: { value: 'high' } })
    fireEvent.click(screen.getByRole('radio', { name: '纯文本' }))
    expect(picker).not.toHaveTextContent('识图')
    expect(screen.getByLabelText('推理强度')).toBeDisabled()
    expect(screen.getByLabelText('推理强度')).toHaveValue('off')
    fireEvent.click(screen.getByRole('radio', { name: '知识模型' }))
    fireEvent.change(screen.getByLabelText('系统提示词'), { target: { value: '自定义知识点提示词' } })
    fireEvent.change(screen.getByLabelText('温度'), { target: { value: '0.7' } })
    fireEvent.change(screen.getByLabelText('最大输出 token'), { target: { value: '2048' } })
    fireEvent.change(screen.getByLabelText('推理强度'), { target: { value: 'low' } })
    fireEvent.click(screen.getByRole('button', { name: '保存设置' }))
    await waitFor(() => expect(apiMock.saveKnowledgeSettings).toHaveBeenCalledWith({
      model_index: 1,
      model_fingerprint: 'model-1',
      system_prompt: '自定义知识点提示词',
      temperature: 0.7,
      max_tokens: 2048,
      think_effort: 'low',
    }))
    expect(apiMock.startKnowledgeAnalysis).not.toHaveBeenCalled()
  })

  it('starts analysis only on click and refreshes results after a partial failure', async () => {
    apiMock.startKnowledgeAnalysis.mockResolvedValue({
      pending_count: 2,
      job: { job_id: 'job', status: 'running', total: 2, succeeded: 0, failed: 0, skipped: 0 },
    })
    apiMock.knowledgeAnalysisStatus
      .mockResolvedValueOnce({ pending_count: 2, job: null })
      .mockResolvedValueOnce({
        pending_count: 1,
        job: { job_id: 'job', status: 'completed', total: 2, succeeded: 1, failed: 1, skipped: 0 },
      })
    render(<KnowledgeMap />)
    const start = await screen.findByRole('button', { name: '开始分析' })
    await waitFor(() => expect(start).toBeEnabled())
    expect(apiMock.startKnowledgeAnalysis).not.toHaveBeenCalled()
    fireEvent.click(start)
    await waitFor(() => expect(screen.getByRole('button', { name: /分析中/ })).toBeDisabled())
    await screen.findByText(/失败 1 条/, {}, { timeout: 2500 })
    expect(apiMock.knowledgeSummary).toHaveBeenCalledTimes(2)
    expect(apiMock.knowledgeHistory).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('button', { name: '开始分析' })).toBeEnabled()
  })

  it('keeps polling if a refresh requested before start returns an older status', async () => {
    let releaseRefresh: (state: { pending_count: number; job: null }) => void = () => {}
    const refreshResult = new Promise(resolve => { releaseRefresh = resolve })
    apiMock.knowledgeAnalysisStatus
      .mockResolvedValueOnce({ pending_count: 2, job: null })
      .mockReturnValueOnce(refreshResult)
    apiMock.startKnowledgeAnalysis.mockResolvedValue({
      pending_count: 2,
      job: { job_id: 'new', status: 'running', total: 2, succeeded: 0, failed: 0, skipped: 0 },
    })
    render(<KnowledgeMap />)
    const start = await screen.findByRole('button', { name: '开始分析' })
    await waitFor(() => expect(start).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '刷新能力分析' }))
    await waitFor(() => expect(apiMock.knowledgeAnalysisStatus).toHaveBeenCalledTimes(2))
    fireEvent.click(start)
    await screen.findByRole('button', { name: '分析中…' })
    await act(async () => releaseRefresh({ pending_count: 2, job: null }))
    expect(screen.getByRole('button', { name: '分析中…' })).toBeDisabled()
  })

  it('requests targeted review questions', async () => {
    render(<KnowledgeMap />)
    await waitFor(() => expect(screen.getByRole('button', { name: '生成针对性复习题' })).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: '生成针对性复习题' }))
    await waitFor(() => expect(apiMock.ask).toHaveBeenCalledWith(
      '请针对以下薄弱知识点出 3 道面试题：React、TS、CSS', undefined, 'knowledge_practice',
    ))
  })

  it('falls back to frequent unscored tags for review questions', async () => {
    apiMock.knowledgeSummary.mockResolvedValueOnce({
      tags: [
        { tag: 'Redis', count: 4, avg_score: null, trend: 'stable' },
        { tag: 'MySQL', count: 2, avg_score: null, trend: 'stable' },
      ],
    })
    apiMock.knowledgeHistory.mockResolvedValueOnce({
      total: 1,
      records: [
        {
          id: 4,
          session_type: 'assist',
          question: 'Redis 怎么治理热 key？',
          answer: '可以用拆分、缓存预热和限流。',
          score: null,
          tags: ['Redis'],
          created_at: 400,
        },
      ],
    })

    render(<KnowledgeMap />)
    await waitFor(() => expect(screen.getByText('还没有评分样本')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: '生成针对性复习题' }))

    await waitFor(() => expect(apiMock.ask).toHaveBeenCalledWith(
      '请针对以下高频但尚未评分的知识点出 3 道面试题，并给出评分要点：Redis、MySQL',
      undefined,
      'knowledge_practice',
    ))
  })

  it('links candidate mic input with ability analysis context', async () => {
    render(<KnowledgeMap />)
    await waitFor(() => expect(screen.getByText('口述输入联动')).toBeInTheDocument())
    expect(screen.getByText('64%')).toBeInTheDocument()
    expect(screen.getByText('whisper 已就绪')).toBeInTheDocument()
    expect(screen.getByText(/先用监控定位慢查询/)).toBeInTheDocument()
  })

  it('shows candidate spoken answer in expanded ability history when present', async () => {
    render(<KnowledgeMap />)
    await waitFor(() => expect(screen.getByText(/历史记录/)).toBeInTheDocument())

    expect(screen.getByText('含口述')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Q1 Q2'))

    expect(screen.getByText('候选人实际回答（多段按顺序拼接）')).toBeInTheDocument()
    expect(screen.getByText('候选人实际回答 A1')).toBeInTheDocument()
    expect(screen.getByText('助手参考答案（多段按顺序拼接）')).toBeInTheDocument()
  })

  it('keeps old history records compatible when candidate spoken answer is absent', async () => {
    apiMock.knowledgeHistory.mockResolvedValueOnce({
      total: 1,
      records: [
        {
          id: 3,
          session_type: 'assist',
          question: '旧问题',
          answer: '旧助手答案',
          score: 5,
          tags: ['Legacy'],
          created_at: 300,
        },
      ],
    })

    render(<KnowledgeMap />)
    await waitFor(() => expect(screen.getByText(/历史记录/)).toBeInTheDocument())
    fireEvent.click(screen.getByText('旧问题'))

    expect(screen.queryByText('候选人实际回答')).not.toBeInTheDocument()
    expect(screen.getByText('旧助手答案')).toBeInTheDocument()
  })

  it('shows a retryable error state when ability data fails to load', async () => {
    const toast = vi.fn()
    useInterviewStore.setState({ setToastMessage: toast } as any)
    apiMock.knowledgeSummary.mockRejectedValueOnce(new Error('DB locked'))

    render(<KnowledgeMap />)

    await waitFor(() => expect(screen.getByText('能力数据暂时读不到')).toBeInTheDocument())
    expect(screen.getByText(/能力分析数据读取失败：DB locked/)).toBeInTheDocument()
    expect(screen.queryByText('还没有能力画像')).not.toBeInTheDocument()
    expect(toast).toHaveBeenCalledWith('DB locked')
  })
})
