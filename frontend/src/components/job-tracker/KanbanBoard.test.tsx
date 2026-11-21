import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import KanbanBoard from './KanbanBoard'
import type { Application } from './types'

function app(id: number, stage: string): Application {
  return {
    id,
    company: `Company ${id}`,
    position: 'Frontend',
    city: 'Shanghai',
    notes: '',
    stage,
    updated_at: 1710000000 + id,
    created_at: 1710000000 + id,
    applied_at: null,
    next_followup_at: null,
    interviewer_info: '',
    feedback: '',
    todos: [],
    sort_order: id,
    review_summary: {
      review_count: 0,
      latest_review_id: null,
      latest_avg_score: null,
      latest_review_at: null,
      latest_status: null,
    },
  }
}

describe('KanbanBoard', () => {
  it('maps vertical wheel movement to horizontal board scrolling', () => {
    const { container } = render(
      <KanbanBoard
        applications={[
          app(1, 'applied'),
          app(2, 'written'),
          app(3, 'interview1'),
          app(4, 'interview2'),
        ]}
        onStageChange={vi.fn()}
        onReorderInStage={vi.fn()}
        search=""
        showTerminalStages={false}
        onShowTerminalStagesChange={vi.fn()}
        terminalApplicationsCount={0}
      />,
    )

    const board = container.querySelector('[data-kanban-board-scroll]') as HTMLDivElement
    expect(board).not.toBeNull()

    Object.defineProperty(board, 'clientWidth', { configurable: true, value: 320 })
    Object.defineProperty(board, 'scrollWidth', { configurable: true, value: 960 })
    board.scrollLeft = 0

    fireEvent.wheel(board, { deltaY: 180, deltaX: 0 })

    expect(board.scrollLeft).toBe(180)
  })

  it('shows short linked reviews on cards without a formal review count', () => {
    render(
      <KanbanBoard
        applications={[
          {
            ...app(1, 'interview1'),
            review_summary: {
              review_count: 0,
              latest_review_id: null,
              latest_avg_score: null,
              latest_review_at: null,
              latest_status: null,
              linked_review_count: 1,
              latest_linked_review_id: 12,
              latest_linked_avg_score: null,
              latest_linked_review_at: 1710003600,
              latest_linked_status: 'completed',
            },
          },
        ]}
        onStageChange={vi.fn()}
        onReorderInStage={vi.fn()}
        search=""
        showTerminalStages={false}
        onShowTerminalStagesChange={vi.fn()}
        terminalApplicationsCount={0}
      />,
    )

    expect(screen.getByText('复盘 短样本')).toBeInTheDocument()
  })

  it('registers native wheel listener with passive: false and prevents default scrolling', () => {
    const addListenerSpy = vi.spyOn(HTMLDivElement.prototype, 'addEventListener')
    try {
      const { container } = render(
        <KanbanBoard
          applications={[app(1, 'applied'), app(2, 'written')]}
          onStageChange={vi.fn()}
          onReorderInStage={vi.fn()}
          search=""
          showTerminalStages={false}
          onShowTerminalStagesChange={vi.fn()}
          terminalApplicationsCount={0}
        />,
      )

      const board = container.querySelector('[data-kanban-board-scroll]') as HTMLDivElement
      const wheelCallIndex = addListenerSpy.mock.calls.findIndex(
        ([type], idx) => type === 'wheel' && addListenerSpy.mock.instances[idx] === board,
      )
      expect(wheelCallIndex).toBeGreaterThanOrEqual(0)
      expect(addListenerSpy.mock.calls[wheelCallIndex]?.[2]).toEqual({ passive: false })

      Object.defineProperty(board, 'clientWidth', { configurable: true, value: 300 })
      Object.defineProperty(board, 'scrollWidth', { configurable: true, value: 900 })
      board.scrollLeft = 0

      const wheelEvent = new WheelEvent('wheel', { deltaY: 120, deltaX: 0, cancelable: true })
      board.dispatchEvent(wheelEvent)
      expect(wheelEvent.defaultPrevented).toBe(true)
      expect(board.scrollLeft).toBe(120)
    } finally {
      addListenerSpy.mockRestore()
    }
  })

  it('disables drag handles for merged rejected column and filtered-subset columns', () => {
    const { container } = render(
      <KanbanBoard
        applications={[
          app(1, 'applied'),
          app(2, 'applied'),
          app(3, 'interview1'),
          app(4, 'interview1'),
          app(5, 'interview1_rejected'),
          app(6, 'written_rejected'),
        ]}
        onStageChange={vi.fn()}
        onReorderInStage={vi.fn()}
        search=""
        showTerminalStages={true}
        onShowTerminalStagesChange={vi.fn()}
        terminalApplicationsCount={2}
        disabledReorderStages={new Set(['interview1'])}
      />,
    )

    const appliedCol = container.querySelector('[data-kanban-stage="applied"]') as HTMLElement
    const interview1Col = container.querySelector('[data-kanban-stage="interview1"]') as HTMLElement
    const rejectedCol = container.querySelector('[data-kanban-stage="rejected"]') as HTMLElement

    const appliedHandles = appliedCol.querySelectorAll('button[aria-label="拖动排序"]')
    const interview1Handles = interview1Col.querySelectorAll('button[aria-label="拖动排序"]')
    const rejectedHandles = rejectedCol.querySelectorAll('button[aria-label="拖动排序"]')

    expect(appliedHandles).toHaveLength(2)
    appliedHandles.forEach((btn) => expect(btn).not.toBeDisabled())

    expect(interview1Handles).toHaveLength(2)
    interview1Handles.forEach((btn) => expect(btn).toBeDisabled())

    expect(rejectedHandles).toHaveLength(2)
    rejectedHandles.forEach((btn) => expect(btn).toBeDisabled())
  })
})
