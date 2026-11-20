import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useInterviewStore } from '@/stores/configStore'
import SettingsDrawer from './index'

const scenes = [
  {
    id: 'assist_manual',
    group: '实时辅助',
    label: '手动提问',
    entry: '发送手动问题时生效。',
    default_prompt: '回答面试问题。',
    output_contract: '',
    max_output_tokens: 4096,
  },
  {
    id: 'assist_exam',
    group: '实时辅助',
    label: '笔试答题',
    entry: '发送笔试题目时生效。',
    default_prompt: '按笔试要求作答。',
    output_contract: '',
    max_output_tokens: 2048,
  },
  {
    id: 'answer',
    group: '面试辅助',
    label: '面试回答',
    entry: '用于面试助手的回答生成',
    default_prompt: '生成简洁的面试回答。',
    output_contract: '',
    max_output_tokens: 600,
  },
  {
    id: 'review',
    group: '面试复盘',
    label: '复盘分析',
    entry: '用于面试结束后的复盘分析',
    default_prompt: '评估面试表现。',
    output_contract: '返回包含评分字段的 JSON 对象。',
    max_output_tokens: 1400,
  },
]

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

let savedPrompts: Record<string, string>
let savedGeneration: Record<string, { temperature: number; max_output_tokens: number }>
let savedExamModelIds: string[]
let configRequests: string[]
let loadFailure: boolean
let saveFailure: boolean
const examModels = [
  { id: 'vision', name: '识图模型', supports_vision: true, supports_think: true, has_api_key: true },
  { id: 'backup', name: '备用视觉模型', supports_vision: true, supports_think: false, has_api_key: true },
  { id: 'text', name: '文本模型', supports_vision: false, supports_think: false, has_api_key: true },
  { id: 'disabled', name: '停用模型', supports_vision: true, supports_think: false, has_api_key: true, enabled: false },
  { id: 'unconfigured', name: '未配置模型', supports_vision: true, supports_think: false, has_api_key: false },
]

function openPrompts() {
  render(<SettingsDrawer />)
  const nav = within(screen.getByRole('navigation', { name: '设置分区' }))
  fireEvent.click(nav.getByRole('button', { name: /系统提示词/ }))
  return nav
}

function savePrompt() {
  fireEvent.click(screen.getByRole('button', { name: '保存当前场景' }))
}

describe('系统提示词设置', () => {
  beforeEach(() => {
    savedPrompts = { review: '已有复盘规则。' }
    savedGeneration = {
      assist_manual: { temperature: 0.25, max_output_tokens: 1400 },
      assist_exam: { temperature: 0.3, max_output_tokens: 2048 },
    }
    savedExamModelIds = []
    configRequests = []
    loadFailure = false
    saveFailure = false
    useInterviewStore.setState({ config: null, settingsOpen: true, settingsDrawerTab: 'prompts' })
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/prompts') {
        return loadFailure ? jsonResponse({ detail: '目录暂不可用' }, 503) : jsonResponse({ scenes })
      }
      if (url === '/api/config') {
        if (init?.method === 'POST') {
          const body = init.body as string
          configRequests.push(body)
          if (saveFailure) return jsonResponse({ detail: '配置无法写入' }, 500)
          const updates = JSON.parse(body)
          if ('written_exam_model_ids' in updates) savedExamModelIds = updates.written_exam_model_ids
          Object.assign(savedGeneration, updates.assist_generation ?? {})
          for (const [id, text] of Object.entries(updates.system_prompts ?? {}) as [string, string | null][]) {
            if (text === null) delete savedPrompts[id]
            else savedPrompts[id] = text
          }
        }
        return jsonResponse({
          models: examModels,
          system_prompts: savedPrompts,
          assist_generation: savedGeneration,
          written_exam_model_ids: savedExamModelIds,
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('选择场景后展示该场景的系统文本、实际入口、输出限制和已保存状态', async () => {
    openPrompts()

    const editor = await screen.findByRole('textbox', { name: '系统提示词' })
    const selector = screen.getByRole('combobox', { name: '使用场景' })
    expect(within(selector).getByRole('group', { name: '面试辅助' })).toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: '手动提问' })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: '笔试答题' })).not.toBeInTheDocument()
    expect(editor).toHaveValue('生成简洁的面试回答。')
    expect(screen.getByText('使用默认')).toBeInTheDocument()
    expect(screen.getByText(/600\s*tokens/i)).toBeInTheDocument()

    fireEvent.change(selector, { target: { value: 'review' } })

    expect(editor).toHaveValue('已有复盘规则。')
    expect(screen.getByText('已自定义')).toBeInTheDocument()
    expect(screen.getByText('用于面试结束后的复盘分析')).toBeInTheDocument()
    expect(screen.getByText('返回包含评分字段的 JSON 对象。')).toBeInTheDocument()
    expect(screen.getByText(/1400\s*tokens/i)).toBeInTheDocument()
  })

  it('将空文本作为当前场景的覆盖保存，恢复默认只在保存后清除覆盖', async () => {
    openPrompts()

    const editor = await screen.findByRole('textbox', { name: '系统提示词' })
    fireEvent.change(editor, { target: { value: '' } })
    savePrompt()

    expect(await screen.findByText('已自定义')).toBeInTheDocument()
    expect(configRequests.map((body) => JSON.parse(body))).toEqual([{ system_prompts: { answer: '' } }])
    expect(editor).toHaveValue('')

    fireEvent.click(screen.getByRole('button', { name: '恢复默认' }))
    expect(editor).toHaveValue('生成简洁的面试回答。')
    expect(configRequests).toHaveLength(1)

    savePrompt()
    expect(await screen.findByText('使用默认')).toBeInTheDocument()
    expect(configRequests.map((body) => JSON.parse(body))).toEqual([
      { system_prompts: { answer: '' } },
      { system_prompts: { answer: null } },
    ])
    fireEvent.change(screen.getByRole('combobox', { name: '使用场景' }), { target: { value: 'review' } })
    expect(editor).toHaveValue('已有复盘规则。')
  })

  it('保留保存失败的草稿，并在离开当前场景或设置页前确认', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    saveFailure = true
    const nav = openPrompts()

    const editor = await screen.findByRole('textbox', { name: '系统提示词' })
    const selector = screen.getByRole('combobox', { name: '使用场景' })
    fireEvent.change(editor, { target: { value: '未保存的面试回答规则' } })
    savePrompt()
    expect(await screen.findByRole('alert')).toHaveTextContent('配置无法写入')
    expect(editor).toHaveValue('未保存的面试回答规则')

    fireEvent.change(selector, { target: { value: 'review' } })
    fireEvent.click(nav.getByRole('button', { name: /实时辅助/ }))
    fireEvent.click(screen.getByRole('button', { name: '关闭' }))
    expect(confirm).toHaveBeenCalledTimes(3)
    expect(selector).toHaveValue('answer')
    expect(editor).toHaveValue('未保存的面试回答规则')
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    confirm.mockReturnValue(true)
    fireEvent.change(selector, { target: { value: 'review' } })
    expect(editor).toHaveValue('已有复盘规则。')
    fireEvent.change(selector, { target: { value: 'answer' } })
    expect(editor).toHaveValue('生成简洁的面试回答。')
  })

  it('目录加载失败后可以重新读取', async () => {
    loadFailure = true
    openPrompts()

    expect(await screen.findByRole('alert')).toHaveTextContent('目录暂不可用')
    expect(screen.queryByRole('textbox', { name: '系统提示词' })).not.toBeInTheDocument()
    loadFailure = false
    fireEvent.click(screen.getByRole('button', { name: '重试' }))

    await waitFor(() => expect(screen.getByRole('textbox', { name: '系统提示词' })).toHaveValue('生成简洁的面试回答。'))
  })

  it('在实时辅助单独保存当前答题场景的温度、输出上限和提示词', async () => {
    useInterviewStore.setState({ settingsDrawerTab: 'realtime' })
    render(<SettingsDrawer />)

    const temperature = await screen.findByRole('spinbutton', { name: '温度' })
    const maxTokens = screen.getByRole('spinbutton', { name: '最大输出 Token' })
    expect(temperature).toHaveValue(0.25)
    expect(maxTokens).toHaveValue(1400)
    const selector = screen.getByRole('combobox', { name: '使用场景' })
    expect(within(selector).getByRole('option', { name: '手动提问' })).toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: '复盘分析' })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: '笔试答题' })).not.toBeInTheDocument()

    fireEvent.change(temperature, { target: { value: '0.8' } })
    fireEvent.change(maxTokens, { target: { value: '127' } })
    expect(screen.getByRole('button', { name: '保存当前场景' })).toBeDisabled()
    fireEvent.change(maxTokens, { target: { value: '128' } })
    fireEvent.change(screen.getByRole('textbox', { name: '系统提示词' }), { target: { value: '新的答题方式。' } })
    savePrompt()

    expect(await screen.findByText('已保存当前场景')).toBeInTheDocument()
    expect(configRequests.map((body) => JSON.parse(body))).toEqual([{
      system_prompts: { assist_manual: '新的答题方式。' },
      assist_generation: { assist_manual: { temperature: 0.8, max_output_tokens: 128 } },
    }])
    expect(savedPrompts.review).toBe('已有复盘规则。')
  })

  it('只在笔试配置保存笔试的提示词和生成参数', async () => {
    useInterviewStore.setState({ settingsDrawerTab: 'mode' })
    render(<SettingsDrawer />)

    const editor = await screen.findByRole('textbox', { name: '系统提示词' })
    expect(editor).toHaveValue('按笔试要求作答。')
    expect(screen.queryByRole('combobox', { name: '使用场景' })).not.toBeInTheDocument()
    const temperature = screen.getByRole('spinbutton', { name: '温度' })
    const maxTokens = screen.getByRole('spinbutton', { name: '最大输出 Token' })
    expect(temperature).toHaveValue(0.3)
    expect(maxTokens).toHaveValue(2048)

    fireEvent.change(editor, { target: { value: '只写答案。' } })
    fireEvent.change(temperature, { target: { value: '0.1' } })
    fireEvent.change(maxTokens, { target: { value: '2000' } })
    savePrompt()

    expect(await screen.findByText('已保存当前场景')).toBeInTheDocument()
    expect(configRequests.map((body) => JSON.parse(body))).toEqual([{
      system_prompts: { assist_exam: '只写答案。' },
      assist_generation: { assist_exam: { temperature: 0.1, max_output_tokens: 2000 } },
    }])
    expect(savedGeneration.assist_manual).toEqual({ temperature: 0.25, max_output_tokens: 1400 })
  })

  it('勾选笔试模型进入队列，保留禁用原因和识图标签，并显式保存和清空队列', async () => {
    useInterviewStore.setState({
      settingsDrawerTab: 'mode',
      config: { models: examModels, written_exam_model_ids: [] } as ReturnType<typeof useInterviewStore.getState>['config'],
    })
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<SettingsDrawer />)

    const unchecked = within(await screen.findByRole('list', { name: '未勾选模型' }))
    expect(unchecked.getByRole('checkbox', { name: '用于答题 文本模型' })).toBeDisabled()
    expect(unchecked.getByRole('checkbox', { name: '用于答题 停用模型' })).toBeDisabled()
    expect(unchecked.getByRole('checkbox', { name: '用于答题 未配置模型' })).toBeDisabled()
    expect(unchecked.getByText('该模型不支持图片作为输入')).toBeInTheDocument()
    expect(unchecked.getByText('已停用')).toBeInTheDocument()
    expect(unchecked.getByText('未配置 API Key')).toBeInTheDocument()
    const vision = unchecked.getByRole('checkbox', { name: '用于答题 识图模型' })
    expect(vision).toBeEnabled()
    expect(within(vision.closest('li')!).getByText('识图', { exact: true })).toHaveClass('text-accent-blue')
    const save = screen.getByRole('button', { name: '保存笔试模型' })
    expect(save).toBeDisabled()

    fireEvent.click(vision)
    fireEvent.click(unchecked.getByRole('checkbox', { name: '用于答题 备用视觉模型' }))
    const selected = within(await screen.findByRole('list', { name: '笔试答题模型' }))
    expect(selected.getAllByRole('checkbox').map((model) => model.getAttribute('aria-label'))).toEqual([
      '用于答题 识图模型',
      '用于答题 备用视觉模型',
    ])
    expect(selected.getByRole('button', { name: '排序 备用视觉模型' })).toBeEnabled()

    await waitFor(() => expect(screen.getAllByText('未保存：笔试配置')).not.toHaveLength(0))
    expect(configRequests).toHaveLength(0)
    fireEvent.click(within(screen.getByRole('navigation')).getByRole('button', { name: /实时辅助/ }))
    expect(confirm).toHaveBeenCalledWith('有未保存更改，切换后会丢失。')
    expect(selected.getByRole('checkbox', { name: '用于答题 识图模型' })).toBeChecked()

    fireEvent.click(save)
    await waitFor(() => expect(save).toBeDisabled())
    fireEvent.click(selected.getByRole('checkbox', { name: '用于答题 识图模型' }))
    fireEvent.click(selected.getByRole('checkbox', { name: '用于答题 备用视觉模型' }))
    fireEvent.click(save)
    await waitFor(() => expect(configRequests.map((body) => JSON.parse(body))).toEqual([
      { written_exam_model_ids: ['vision', 'backup'] },
      { written_exam_model_ids: [] },
    ]))
    await waitFor(() => expect(save).toBeDisabled())
  })
})
