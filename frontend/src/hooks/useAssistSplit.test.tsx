import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { useAssistSplit } from './useAssistSplit'

describe('useAssistSplit', () => {
  it('restores document.body cursor and user-select if unmounted while dragging', () => {
    let draggingRef: { current: boolean } | null = null

    function Harness() {
      const { assistSplitDragging } = useAssistSplit()
      draggingRef = assistSplitDragging
      return null
    }

    const { unmount } = render(<Harness />)
    expect(draggingRef).not.toBeNull()

    draggingRef!.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    unmount()

    expect(draggingRef!.current).toBe(false)
    expect(document.body.style.cursor).toBe('')
    expect(document.body.style.userSelect).toBe('')
  })
})
