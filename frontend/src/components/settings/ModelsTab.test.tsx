import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ModelsTab from './ModelsTab'

const provider = {
  id: 'ark', name: '火山方舟', api_type: 'openai',
  api_base_url: 'https://ark.example/api/v3', has_api_key: true,
}
const savedModel = {
  id: 'saved', provider_id: 'ark', model: 'ep-old', name: '私有模型', enabled: true,
  supports_vision: true, supports_think: false, context_window_tokens: 32768,
  think_enabled_params: {}, think_disabled_params: {},
}
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((settle) => { resolve = settle })
  return { promise, resolve }
}

function json(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('model provider settings', () => {
  const snapshot = { revision: 'r1', providers: [provider], models: [savedModel], usages: { saved: ['实时辅助', '笔试'] }, active_model_id: null }
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    fetchMock = vi.fn(async (path: string) => {
      if (path === '/api/model-providers') return json(snapshot)
      if (path === '/api/config') return json({ models: [] })
      throw new Error('unexpected request')
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  async function openProvider() {
    render(<ModelsTab />)
    await screen.findByText('火山方舟')
  }

  it('starts four validations at a time and enables only models whose text check passed', async () => {
    const results = new Map(Array.from({ length: 5 }, (_, n) => ['model-' + n, deferred<Response>()]))
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === '/api/model-providers') return json(snapshot)
      if (path.endsWith('/catalog')) return json({ models: Array.from(results.keys(), (id) => ({ id })) })
      return results.get(JSON.parse(String(init?.body)).model)?.promise
    })
    await openProvider()
    fireEvent.click(screen.getByRole('button', { name: '获取模型' }))
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path.endsWith('/verify'))).toHaveLength(4))
    const first = screen.getByRole('checkbox', { name: '启用 model-0' })
    expect(first).toBeDisabled()
    await act(async () => results.get('model-0')?.resolve(json({ ok: true, latency_ms: 12 })))
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path.endsWith('/verify'))).toHaveLength(5))
    expect(first).toBeEnabled()
    await act(async () => results.get('model-1')?.resolve(json({ ok: false, detail: '模型不支持文本生成' })))
    expect(screen.getByRole('checkbox', { name: '启用 model-1' })).toBeDisabled()
    expect(screen.getByText('模型不支持文本生成')).toBeVisible()
  })

  it('aborts a late validation and invalidates the result when the connection is edited', async () => {
    const pending = deferred<Response>()
    fetchMock.mockImplementation(async (path: string) => {
      if (path === '/api/model-providers') return json(snapshot)
      if (path.endsWith('/catalog')) return json({ models: [{ id: 'candidate' }] })
      return pending.promise
    })
    await openProvider()
    fireEvent.click(screen.getByRole('button', { name: '获取模型' }))
    await screen.findByRole('checkbox', { name: '启用 candidate' })
    const request = fetchMock.mock.calls.find(([path]) => path.endsWith('/verify'))
    fireEvent.change(screen.getByLabelText('API Base URL'), { target: { value: 'https://other.example/v1' } })
    expect(request?.[1].signal.aborted).toBe(true)
    await act(async () => pending.resolve(json({ ok: true })))
    expect(screen.queryByText('可用')).not.toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: '启用 私有模型' })).toBeChecked()
  })

  it('allows manual IDs only after explicit validation and omits the stored credential from requests', async () => {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === '/api/model-providers') return json(snapshot)
      if (path.endsWith('/verify')) {
        expect(JSON.parse(String(init?.body))).toEqual({
          provider_id: 'ark', model: 'ep-manual', api_type: 'openai', api_base_url: provider.api_base_url,
        })
        return json({ ok: true, latency_ms: 18 })
      }
      throw new Error('unexpected request')
    })
    await openProvider()
    fireEvent.change(screen.getByLabelText('手填 Model ID'), { target: { value: 'ep-manual' } })
    fireEvent.click(screen.getByRole('button', { name: '添加 Model ID' }))
    expect(screen.getByRole('checkbox', { name: '启用 ep-manual' })).toBeDisabled()
    expect(fetchMock.mock.calls.some(([path]) => path.endsWith('/verify'))).toBe(false)
    fireEvent.click(within(screen.getByRole('listitem', { name: 'ep-manual' })).getByRole('button', { name: '验证模型' }))
    await waitFor(() => expect(screen.getByRole('checkbox', { name: '启用 ep-manual' })).toBeEnabled())
  })

  it('keeps an in-use model selected on cancel and sends confirmed removal with the revision on save', async () => {
    const confirmation = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === '/api/config') return json({ models: [] })
      if (init?.method === 'PUT') {
        expect(JSON.parse(String(init.body))).toMatchObject({ revision: 'r1', confirmed_remove_ids: ['saved'], models: [{ id: 'saved', enabled: false }] })
        return json({ ...snapshot, revision: 'r2', models: [{ ...savedModel, enabled: false }], usages: {} })
      }
      if (path === '/api/model-providers') return json(snapshot)
      throw new Error('unexpected request')
    })
    await openProvider()
    const checkbox = screen.getByRole('checkbox', { name: '启用 私有模型' })
    fireEvent.click(checkbox)
    expect(checkbox).toBeChecked()
    expect(confirmation.mock.calls[0][0]).toMatch(/实时辅助[\s\S]*笔试/)
    fireEvent.click(checkbox)
    expect(checkbox).not.toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: '保存供应商配置' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'PUT')).toBe(true))
    await waitFor(() => expect(screen.getByText('已保存')).toBeVisible())
  })
})
