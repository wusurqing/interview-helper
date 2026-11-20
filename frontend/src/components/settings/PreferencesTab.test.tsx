import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PreferencesTab from './PreferencesTab'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { useInterviewStore } from '@/stores/configStore'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'

vi.mock('@/lib/configSync', () => ({
  updateConfigAndRefresh: vi.fn().mockResolvedValue({ ok: true }),
}))

describe('PreferencesTab', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  beforeEach(() => {
    localStorage.clear()
    vi.mocked(updateConfigAndRefresh).mockReset()
    vi.mocked(updateConfigAndRefresh).mockResolvedValue({ ok: true } as any)
    useInterviewStore.setState({
      config: {
        stt_provider: 'whisper',
        screen_capture_region: 'left_half',
        screen_capture_max_long_edge: 1600,
        multi_screen_capture_idle_sec: 10,
        written_exam_mode: false,
        written_exam_think: false,
        kb_top_k: 4,
      },
      options: {
        screen_capture_regions: ['full', 'left_half', 'right_half'],
      },
      platformInfo: null,
      sttLoaded: true,
      sttLoading: false,
      sttActiveProvider: 'whisper',
      sttFallbackLoaded: false,
    } as any)
    useUiPrefsStore.setState({
      interviewOverlayEnabled: false,
      interviewOverlayMode: 'glass',
      interviewOverlayShowBg: true,
      interviewOverlayPromptMaxWidth: 820,
      interviewOverlayPromptAutoFollow: false,
    })
  })

  it('exposes the screen capture long-edge limit', async () => {
    render(<PreferencesTab variant="mode" />)

    expect(screen.getByText('自动保存')).toBeInTheDocument()
    expect(screen.getByText('截图最长边: 1600px')).toBeInTheDocument()
    const input = screen.getByDisplayValue('1600')

    fireEvent.change(input, { target: { value: '2400' } })

    await waitFor(() => {
      expect(updateConfigAndRefresh).toHaveBeenCalledWith({ screen_capture_max_long_edge: 2400 })
    })
  })

  it('keeps appearance controls separate from quick prompts', () => {
    render(<PreferencesTab variant="appearance" />)

    expect(screen.getByText('配色方案')).toBeInTheDocument()
    expect(screen.getByText('流式跟滚阈值（像素）')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '快捷提示词' })).not.toBeInTheDocument()
  })

  it('allows zero to disable screen capture scaling', async () => {
    render(<PreferencesTab variant="mode" />)

    fireEvent.change(screen.getByDisplayValue('1600'), { target: { value: '0' } })

    await waitFor(() => {
      expect(updateConfigAndRefresh).toHaveBeenCalledWith({ screen_capture_max_long_edge: 0 })
    })
  })

  it('merges multiple debounced preference edits into one save payload', async () => {
    vi.useFakeTimers()
    render(<PreferencesTab variant="mode" />)

    fireEvent.change(screen.getByDisplayValue('1600'), { target: { value: '2400' } })
    fireEvent.change(screen.getByDisplayValue('10'), { target: { value: '12' } })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(600)
    })

    expect(updateConfigAndRefresh).toHaveBeenCalledWith({
      screen_capture_max_long_edge: 2400,
      multi_screen_capture_idle_sec: 12,
    })
    expect(updateConfigAndRefresh).toHaveBeenCalledTimes(1)
  })

  it('auto-saves knowledge retrieval deadlines from preferences', async () => {
    vi.useFakeTimers()
    render(<PreferencesTab variant="kb" />)

    fireEvent.click(screen.getByText('知识库与引用'))
    fireEvent.change(screen.getByDisplayValue('150'), { target: { value: '220' } })
    fireEvent.change(screen.getByDisplayValue('80'), { target: { value: '60' } })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(600)
    })

    expect(updateConfigAndRefresh).toHaveBeenCalledWith({
      kb_deadline_ms: 220,
      kb_asr_deadline_ms: 60,
    })
  })

  it('flushes pending auto-save edits when unmounted before debounce fires', async () => {
    vi.useFakeTimers()
    const { unmount } = render(<PreferencesTab variant="mode" />)

    fireEvent.change(screen.getByDisplayValue('1600'), { target: { value: '2400' } })
    unmount()

    await act(async () => {
      await Promise.resolve()
    })

    expect(updateConfigAndRefresh).toHaveBeenCalledWith({ screen_capture_max_long_edge: 2400 })
  })

  it('edits exam Think while the exam is inactive and keeps module switching out of settings', async () => {
    render(<PreferencesTab variant="mode" />)

    const think = screen.getByRole('checkbox', { name: '深度思考 (Think)' })
    expect(think).not.toBeChecked()
    expect(screen.queryByText('工作模式')).not.toBeInTheDocument()
    expect(screen.queryByText('笔试模式')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /悬浮窗设置/ })).not.toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /^截图区域/ })).toHaveValue('left_half')
    expect(screen.getByText('多图截图等待: 10 秒')).toBeInTheDocument()

    fireEvent.click(think)

    await waitFor(() => expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ written_exam_think: true }))
  })
})
