import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useInterviewStore } from '@/stores/configStore'
import SettingsDrawer from './index'

const scenes = [
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
let configRequests: string[]
let loadFailure: boolean
let saveFailure: boolean

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
    configRequests = []
    loadFailure = false
    saveFailure = false
    useInterviewStore.setState({ config: null, settingsOpen: true, settingsDrawerTab: 'general' })
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/prompts') {
        return loadFailure ? jsonResponse({ detail: '目录暂不可用' }, 503) : jsonResponse({ scenes })
      }
      if (url === '/api/config') {
        if (init?.method === 'POST') {
          const body = init.body as string
          configRequests.push(body)
          if (saveFailure) return jsonResponse({ detail: '配置无法写入' }, 500)
          const updates = JSON.parse(body).system_prompts as Record<string, string | null>
          for (const [id, text] of Object.entries(updates)) {
            if (text === null) delete savedPrompts[id]
            else savedPrompts[id] = text
          }
        }
        return jsonResponse({ models: [], system_prompts: savedPrompts })
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
    fireEvent.click(nav.getByRole('button', { name: /常用/ }))
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
})
