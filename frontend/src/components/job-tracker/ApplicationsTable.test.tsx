import { act, fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import ApplicationsTable from './ApplicationsTable'
import type { Application } from './types'

function app(overrides: Partial<Application> = {}): Application {
  return {
    id: 1,
    company: 'ByteDance',
    position: 'Frontend Engineer',
    city: 'Shanghai',
    stage: 'interview1',
    applied_at: 1710000000,
    next_followup_at: null,
    interviewer_info: '',
    feedback: '',
    todos: [],
    notes: '',
    created_at: 1710000000,
    updated_at: 1710000000,
    sort_order: 0,
    review_summary: {
      review_count: 0,
      latest_review_id: null,
      latest_avg_score: null,
      latest_review_at: null,
      latest_status: null,
      linked_review_count: 0,
      latest_linked_review_id: null,
      latest_linked_avg_score: null,
      latest_linked_review_at: null,
      latest_linked_status: null,
    },
    ...overrides,
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

describe('ApplicationsTable', () => {
  it('ignores rapid duplicate progress saves while a patch is pending', async () => {
    const patch = deferred<boolean>()
    const onPatch = vi.fn(() => patch.promise)

    render(
      <ApplicationsTable
        applications={[app()]}
        offerByAppId={new Map()}
        onPatch={onPatch}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={vi.fn()}
        search=""
        selectedId={1}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )

    fireEvent.change(screen.getByLabelText('当前阶段'), { target: { value: 'interview2' } })

    const saveButton = screen.getByRole('button', { name: '保存进度' })
    act(() => {
      saveButton.click()
      saveButton.click()
    })

    expect(onPatch).toHaveBeenCalledTimes(1)
    expect(onPatch).toHaveBeenCalledWith(1, expect.objectContaining({ stage: 'interview2' }))
    for (const button of screen.getAllByRole('button', { name: '保存中...' })) {
      expect(button).toBeDisabled()
    }

    await act(async () => {
      patch.resolve(true)
      await patch.promise
    })

    expect(screen.getByText('已保存核心信息')).toBeInTheDocument()
  })

  it('shows an inline failure notice when a progress save is rejected by the parent', async () => {
    const onPatch = vi.fn().mockResolvedValue(false)

    render(
      <ApplicationsTable
        applications={[app()]}
        offerByAppId={new Map()}
        onPatch={onPatch}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={vi.fn()}
        search=""
        selectedId={1}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )

    fireEvent.change(screen.getByLabelText('当前阶段'), { target: { value: 'interview2' } })
    fireEvent.click(screen.getByRole('button', { name: '保存进度' }))

    expect(onPatch).toHaveBeenCalledWith(1, expect.objectContaining({ stage: 'interview2' }))
    expect(await screen.findByText('保存失败，请重试')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存进度' })).toBeEnabled()
  })

  it('shows thrown save errors inline near the active application', async () => {
    const onPatch = vi.fn().mockRejectedValue(new Error('后端保存失败'))

    render(
      <ApplicationsTable
        applications={[app()]}
        offerByAppId={new Map()}
        onPatch={onPatch}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={vi.fn()}
        search=""
        selectedId={1}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )

    fireEvent.change(screen.getByLabelText('当前阶段'), { target: { value: 'interview2' } })
    fireEvent.click(screen.getByRole('button', { name: '保存进度' }))

    expect(await screen.findByText('后端保存失败')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存进度' })).toBeEnabled()
  })

  it('keeps short linked reviews reachable without promoting them to formal summaries', () => {
    const item = app({
      review_summary: {
        review_count: 0,
        latest_review_id: null,
        latest_avg_score: null,
        latest_review_at: null,
        latest_status: null,
        linked_review_count: 1,
        latest_linked_review_id: 42,
        latest_linked_avg_score: null,
        latest_linked_review_at: 1710003600,
        latest_linked_status: 'completed',
      },
    })
    const openReviews = vi.fn()

    render(
      <ApplicationsTable
        applications={[item]}
        offerByAppId={new Map()}
        onPatch={vi.fn()}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={openReviews}
        search=""
        selectedId={1}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )

    expect(screen.getAllByText('短样本').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: '查看 ByteDance 的 1 场关联复盘' }))

    expect(openReviews).toHaveBeenCalledWith(expect.objectContaining({ id: 1 }))
  })

  it('edits and saves new duplicate-title todos even when crypto.randomUUID is unavailable', async () => {
    const onPatch = vi.fn().mockResolvedValue(true)
    const originalCrypto = globalThis.crypto
    let counter = 0
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: {
        getRandomValues: (arr: Uint8Array) => {
          counter += 1
          for (let i = 0; i < arr.length; i += 1) arr[i] = (i * 37 + counter * 13) & 0xff
          return arr
        },
      },
    })

    try {
      render(
        <ApplicationsTable
          applications={[
            app({
              todos: [{ id: 'existing-1', title: '跟进 HR', done: true }],
            }),
          ]}
          offerByAppId={new Map()}
          onPatch={onPatch}
          onDelete={vi.fn()}
          onOpenOffer={vi.fn()}
          onOpenReviews={vi.fn()}
          search=""
          selectedId={1}
          onSelect={vi.fn()}
          compactDetailLayout={false}
        />,
      )

      fireEvent.click(screen.getByRole('button', { name: /补充信息/ }))
      fireEvent.change(screen.getByLabelText('待办清单'), {
        target: { value: '跟进 HR\n跟进 HR\n准备二面' },
      })

      await act(async () => {
        fireEvent.click(screen.getAllByRole('button', { name: '保存补充信息' })[0])
      })

      expect(onPatch).toHaveBeenCalledTimes(1)
      const [, patchPayload] = onPatch.mock.calls[0]
      expect(patchPayload.todos).toHaveLength(3)
      expect(patchPayload.todos[0]).toMatchObject({ id: 'existing-1', title: '跟进 HR', done: true })
      expect(patchPayload.todos[1].id).toBeTruthy()
      expect(patchPayload.todos[1].id).not.toBe('existing-1')
      expect(patchPayload.todos[2].id).toBeTruthy()
      expect(patchPayload.todos[2].id).not.toBe(patchPayload.todos[1].id)
    } finally {
      Object.defineProperty(globalThis, 'crypto', {
        configurable: true,
        value: originalCrypto,
      })
    }
  })

  it('preserves unsaved core edits, saveNotice, and open panels when saving extras updates updated_at', async () => {
    const initial = app({
      id: 1,
      company: 'ByteDance',
      notes: 'old note',
      todos: [],
      updated_at: 1710000000,
    })
    const second = app({
      id: 2,
      company: 'Alibaba',
      notes: 'other note',
      updated_at: 1710000005,
    })

    let currentApps = [initial, second]
    let rerenderFn!: (ui: React.ReactElement) => void
    const onPatch = vi.fn(async (id: number, patch: Partial<Application>) => {
      currentApps = currentApps.map((item) => (
        item.id === id
          ? { ...item, ...patch, updated_at: item.updated_at + 100 }
          : item
      ))
      rerenderFn(
        <ApplicationsTable
          applications={currentApps}
          offerByAppId={new Map()}
          onPatch={onPatch}
          onDelete={vi.fn()}
          onOpenOffer={vi.fn()}
          onOpenReviews={vi.fn()}
          search=""
          selectedId={1}
          onSelect={vi.fn()}
          compactDetailLayout={false}
        />,
      )
      return true
    })

    const { rerender } = render(
      <ApplicationsTable
        applications={currentApps}
        offerByAppId={new Map()}
        onPatch={onPatch}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={vi.fn()}
        search=""
        selectedId={1}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )
    rerenderFn = rerender

    fireEvent.click(screen.getByRole('button', { name: '编辑核心信息' }))
    fireEvent.change(screen.getByLabelText('公司名称'), { target: { value: 'ByteDance AI' } })

    fireEvent.click(screen.getByRole('button', { name: /补充信息/ }))
    fireEvent.change(screen.getByLabelText('备注'), { target: { value: 'new extra note' } })
    fireEvent.change(screen.getByLabelText('待办清单'), { target: { value: '准备系统设计' } })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '保存补充信息' }))
    })

    expect(onPatch).toHaveBeenCalledTimes(1)
    expect(screen.getByLabelText('公司名称')).toHaveValue('ByteDance AI')
    expect(screen.getByLabelText('备注')).toHaveValue('new extra note')
    expect(screen.getByText('已保存补充信息')).toBeInTheDocument()
    expect(screen.getByText('核心待保存')).toBeInTheDocument()

    // Switching to another application resets draft and closes edit panels
    rerender(
      <ApplicationsTable
        applications={currentApps}
        offerByAppId={new Map()}
        onPatch={onPatch}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={vi.fn()}
        search=""
        selectedId={2}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )
    expect(screen.queryByLabelText('公司名称')).not.toBeInTheDocument()
    expect(screen.queryByText('已保存补充信息')).not.toBeInTheDocument()
  })

  it('reacts immediately in quick progress section when draft.stage switches between ongoing and terminal', () => {
    render(
      <ApplicationsTable
        applications={[app({ stage: 'interview1' })]}
        offerByAppId={new Map()}
        onPatch={vi.fn()}
        onDelete={vi.fn()}
        onOpenOffer={vi.fn()}
        onOpenReviews={vi.fn()}
        search=""
        selectedId={1}
        onSelect={vi.fn()}
        compactDetailLayout={false}
      />,
    )

    expect(screen.getByText('更新进度')).toBeInTheDocument()
    expect(screen.getByLabelText('下次跟进')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('当前阶段'), { target: { value: 'interview1_rejected' } })
    expect(screen.getByText('改结果')).toBeInTheDocument()
    expect(screen.queryByLabelText('下次跟进')).not.toBeInTheDocument()
    expect(screen.getByText('已结束 · 不进入待跟进')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('当前阶段'), { target: { value: 'interview2' } })
    expect(screen.getByText('更新进度')).toBeInTheDocument()
    expect(screen.getByLabelText('下次跟进')).toBeInTheDocument()
  })

  it('uses non-blocking two-step delete confirmation with double-click lock instead of window.confirm', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const pendingDelete = deferred<boolean>()
    const onDelete = vi.fn(() => pendingDelete.promise)

    try {
      render(
        <ApplicationsTable
          applications={[app()]}
          offerByAppId={new Map()}
          onPatch={vi.fn()}
          onDelete={onDelete}
          onOpenOffer={vi.fn()}
          onOpenReviews={vi.fn()}
          search=""
          selectedId={1}
          onSelect={vi.fn()}
          compactDetailLayout={false}
        />,
      )

      fireEvent.click(screen.getByRole('button', { name: /补充信息/ }))
      fireEvent.click(screen.getByRole('button', { name: '删除这条记录' }))

      expect(confirmSpy).not.toHaveBeenCalled()
      expect(onDelete).not.toHaveBeenCalled()
      expect(screen.getByText(/确认删除「ByteDance」/)).toBeInTheDocument()

      fireEvent.click(screen.getByRole('button', { name: '取消' }))
      expect(screen.queryByRole('button', { name: '确认删除' })).not.toBeInTheDocument()

      fireEvent.click(screen.getByRole('button', { name: '删除这条记录' }))
      const confirmBtn = screen.getByRole('button', { name: '确认删除' })
      act(() => {
        confirmBtn.click()
        confirmBtn.click()
      })

      expect(onDelete).toHaveBeenCalledTimes(1)
      expect(onDelete).toHaveBeenCalledWith(1)
      expect(screen.getByRole('button', { name: '删除中...' })).toBeDisabled()

      await act(async () => {
        pendingDelete.resolve(true)
        await pendingDelete.promise
      })

      expect(screen.queryByRole('button', { name: '确认删除' })).not.toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })
})
