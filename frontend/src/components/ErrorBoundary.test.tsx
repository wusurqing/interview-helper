import { render, screen, fireEvent } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ErrorBoundary } from './ErrorBoundary'

function Bomb({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error('render exploded')
  }
  return <div>healthy child</div>
}

describe('ErrorBoundary', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('catches child render errors and allows recovery reset', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    let shouldThrow = true
    function DynamicChild() {
      return <Bomb shouldThrow={shouldThrow} />
    }

    render(
      <ErrorBoundary>
        <DynamicChild />
      </ErrorBoundary>,
    )

    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('render exploded')).toBeInTheDocument()
    expect(consoleSpy).toHaveBeenCalled()

    shouldThrow = false
    fireEvent.click(screen.getByRole('button', { name: '尝试恢复' }))
    expect(screen.getByText('healthy child')).toBeInTheDocument()
  })
})
