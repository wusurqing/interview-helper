import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ModelPriorityDropdown } from './ModelPriorityDropdown'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'

const apiMock = vi.hoisted(() => ({
  checkModelsHealth: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  api: apiMock,
  getErrorMessage: (error: Error) => error.message,
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

function renderDropdown() {
  const config = {
    models: [
      { name: 'Primary Model', supports_think: true, supports_vision: false, enabled: true },
      { name: 'Disabled Model', supports_think: false, supports_vision: true, enabled: false },
    ],
    active_model: 0,
  } as AppConfig
  useInterviewStore.setState({ config })

  return render(
    <ModelPriorityDropdown
      config={config}
      modelHealth={useInterviewStore.getState().modelHealth}
      modelHealthDetail={useInterviewStore.getState().modelHealthDetail}
      modelHealthLatency={useInterviewStore.getState().modelHealthLatency}
      onModelChange={vi.fn()}
    />,
  )
}

describe('ModelPriorityDropdown', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.checkModelsHealth.mockResolvedValue({ ok: true, catalog: {}, detail: {}, fingerprint: {} })
    useInterviewStore.setState({
      modelHealth: {},
      modelHealthDetail: {},
      modelHealthLatency: {},
      modelCatalog: {},
      toastMessage: null,
      toasts: [],
    } as any)
  })

  it('checks the directory once for the enabled queue and preserves generation health', async () => {
    const healthCheck = deferred<{ ok: boolean; catalog: Record<number, 'visible'> }>()
    apiMock.checkModelsHealth.mockReturnValueOnce(healthCheck.promise)

    renderDropdown()

    fireEvent.click(screen.getByRole('button', { name: /优先答题模型 Primary Model/ }))
    const recheck = screen.getByRole('button', { name: '重新检查连接' })
    fireEvent.click(recheck)
    fireEvent.click(recheck)

    expect(apiMock.checkModelsHealth).toHaveBeenCalledExactlyOnceWith([0])
    expect(useInterviewStore.getState().modelHealth).toEqual({})
    expect(screen.getByRole('button', { name: '检测中…' })).toBeDisabled()

    await act(async () => {
      healthCheck.resolve({ ok: true, catalog: { 0: 'visible' } })
      await healthCheck.promise
    })

    expect(await screen.findByRole('button', { name: /优先答题模型 Primary Model.*目录可见/ })).toBeInTheDocument()
    expect(useInterviewStore.getState().modelHealth).toEqual({})
  })

  it('surfaces a directory error without marking the model as failed for scheduling', async () => {
    apiMock.checkModelsHealth.mockRejectedValueOnce(new Error('queue busy'))

    renderDropdown()

    fireEvent.click(screen.getByRole('button', { name: /优先答题模型 Primary Model/ }))
    fireEvent.click(screen.getByRole('button', { name: '重新检查连接' }))

    await waitFor(() => {
      expect(useInterviewStore.getState().toastMessage).toContain('queue busy')
    })
    expect(useInterviewStore.getState().modelHealth).toEqual({})
    expect(screen.getByRole('button', { name: '重新检查连接' })).toBeEnabled()
  })
})
