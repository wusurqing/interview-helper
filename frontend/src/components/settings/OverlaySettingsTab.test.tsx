import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'
import OverlaySettingsTab from './OverlaySettingsTab'

beforeEach(() => {
  localStorage.clear()
  useUiPrefsStore.getState().syncInterviewOverlayPrefs()
})

afterEach(() => {
  cleanup()
  delete window.electronAPI
})

describe('OverlaySettingsTab', () => {
  it('previews and persists mode-specific appearance without a native window', () => {
    render(<OverlaySettingsTab />)
    expect(screen.getByText(/位置与尺寸仅在桌面端可用/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText(/^字号/), { target: { value: '28' } })
    expect(screen.getByText(/先说明结论/)).toHaveStyle({ fontSize: '28px' })
    fireEvent.click(screen.getByRole('button', { name: '提词模式' }))
    expect(screen.getByLabelText(/^字号/)).toHaveValue('14')
    fireEvent.click(screen.getByLabelText('靠近底部时自动跟随最新内容'))
    expect(screen.getByLabelText('靠近底部时自动跟随最新内容')).toBeChecked()
    cleanup()
    useUiPrefsStore.getState().syncInterviewOverlayPrefs()
    render(<OverlaySettingsTab />)
    expect(screen.getByLabelText('靠近底部时自动跟随最新内容')).toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: '磨砂玻璃' }))
    expect(screen.getByLabelText(/^字号/)).toHaveValue('28')
    fireEvent.click(screen.getByRole('button', { name: '重置外观' }))
    expect(screen.getByLabelText(/^字号/)).toHaveValue('14')
  })

  it('commits width for the selected mode and keeps the main window open', async () => {
    const snapshot = {
      layouts: { glass: { displayId: null, anchor: 'top-center', x: 0, y: 24, sizeMode: 'fixed', width: 480, height: 320 } },
      displays: [{ id: 1, label: '主屏', primary: true, workArea: { x: 0, y: 0, width: 1600, height: 900 } }],
    }
    const updateOverlayLayout = vi.fn(async () => ({ ok: true, ...snapshot }))
    const syncOverlayWindow = vi.fn()
    window.electronAPI = {
      getOverlayLayout: async () => snapshot,
      updateOverlayLayout,
      resetOverlayLayout: vi.fn(),
      syncOverlayWindow,
    } as unknown as Window['electronAPI']

    render(<OverlaySettingsTab />)
    const width = await screen.findByLabelText('宽度')
    fireEvent.change(width, { target: { value: '760' } })
    fireEvent.blur(width)
    await waitFor(() => expect(updateOverlayLayout).toHaveBeenCalledWith('glass', { width: 760 }))
    expect(syncOverlayWindow).not.toHaveBeenCalled()
  })
})
