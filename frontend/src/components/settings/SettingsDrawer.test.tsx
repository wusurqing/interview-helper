import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import SettingsDrawer from './index'
import { useInterviewStore } from '@/stores/configStore'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'

describe('SettingsDrawer', () => {
  beforeEach(() => {
    localStorage.clear()
    useUiPrefsStore.getState().syncInterviewOverlayPrefs()
    useInterviewStore.setState({
      settingsOpen: true,
      settingsDrawerTab: 'appearance',
    })
  })

  afterEach(() => {
    cleanup()
    useInterviewStore.setState({
      settingsOpen: false,
      settingsDrawerTab: 'realtime',
    })
  })

  it('marks both backdrop and dialog as app-no-drag and closes cleanly from the close button', () => {
    const { container } = render(<SettingsDrawer />)

    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveClass('app-no-drag')

    const backdrop = container.querySelector('.z-40')
    expect(backdrop).not.toBeNull()
    expect(backdrop).toHaveClass('app-no-drag')

    const closeBtn = screen.getByRole('button', { name: '关闭' })
    expect(closeBtn).toHaveClass('inline-flex')

    fireEvent.click(closeBtn)
    expect(useInterviewStore.getState().settingsOpen).toBe(false)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('also closes when clicking the backdrop', () => {
    const { container } = render(<SettingsDrawer />)
    const backdrop = container.querySelector('.z-40')
    expect(backdrop).not.toBeNull()

    fireEvent.click(backdrop!)
    expect(useInterviewStore.getState().settingsOpen).toBe(false)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})
