import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { WindowOpacitySlider } from './WindowOpacitySlider'
import { useInterviewStore } from '@/stores/configStore'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

describe('main window opacity control', () => {
  const originalAPI = window.electronAPI
  let setOpacity: ReturnType<typeof vi.fn>

  beforeEach(() => {
    setOpacity = vi.fn((opacity: number) => Promise.resolve(opacity))
    window.electronAPI = {
      getWindowState: vi.fn().mockResolvedValue({ opacity: 1, opacitySupported: true }),
      setWindowOpacity: setOpacity,
    } as any
  })

  afterEach(() => {
    window.electronAPI = originalAPI
    vi.clearAllMocks()
  })

  it('stays hidden without desktop support', async () => {
    delete window.electronAPI
    const { unmount } = render(<WindowOpacitySlider />)
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    unmount()
    window.electronAPI = {
      getWindowState: vi.fn().mockResolvedValue({ opacity: 1, opacitySupported: false }),
      setWindowOpacity: setOpacity,
    } as any
    render(<WindowOpacitySlider />)
    await waitFor(() => expect(window.electronAPI!.getWindowState).toHaveBeenCalled())
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
  })

  it('restores the native value and changes whole-window opacity by dragging', async () => {
    window.electronAPI!.getWindowState = vi.fn().mockResolvedValue({ opacity: 0.65, opacitySupported: true })
    render(<WindowOpacitySlider />)
    const slider = await screen.findByRole('slider', { name: '窗口透明度' })
    expect(slider).toHaveValue('35')
    fireEvent.change(slider, { target: { value: '70' } })
    await waitFor(() => expect(setOpacity).toHaveBeenCalledWith(0.3))
    expect(screen.getByText('70%')).toBeInTheDocument()
  })

  it('uses two-point wheel steps over the whole region and respects both bounds', async () => {
    render(<WindowOpacitySlider />)
    const slider = await screen.findByRole('slider', { name: '窗口透明度' })
    const region = screen.getByRole('group', { name: '窗口透明度调节' })
    const wheel = new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true })
    expect(region.dispatchEvent(wheel)).toBe(false)
    await waitFor(() => expect(slider).toHaveValue('2'))
    fireEvent.wheel(region, { deltaY: 100 })
    expect(slider).toHaveValue('0')
    fireEvent.change(slider, { target: { value: '90' } })
    fireEvent.wheel(region, { deltaY: -100 })
    expect(slider).toHaveValue('90')
  })

  it('coalesces rapid changes to the newest requested value', async () => {
    const first = deferred<number>()
    setOpacity.mockReturnValueOnce(first.promise)
    render(<WindowOpacitySlider />)
    const slider = await screen.findByRole('slider', { name: '窗口透明度' })
    fireEvent.change(slider, { target: { value: '20' } })
    fireEvent.change(slider, { target: { value: '80' } })
    fireEvent.change(slider, { target: { value: '60' } })
    expect(setOpacity).toHaveBeenCalledTimes(1)
    first.resolve(0.8)
    await waitFor(() => expect(setOpacity).toHaveBeenCalledTimes(2))
    expect(setOpacity).toHaveBeenLastCalledWith(0.4)
  })

  it('restores the last confirmed value and reports a failed update', async () => {
    setOpacity.mockRejectedValueOnce(new Error('native failure'))
    render(<WindowOpacitySlider />)
    const slider = await screen.findByRole('slider', { name: '窗口透明度' })
    fireEvent.change(slider, { target: { value: '50' } })
    await waitFor(() => expect(slider).toHaveValue('0'))
    expect(useInterviewStore.getState().toastMessage).toContain('透明度调节失败')
  })
})
