import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import SettingsDrawer from './index'
import { useInterviewStore } from '@/stores/configStore'

describe('SettingsDrawer', () => {
  beforeAll(async () => {
    // Prime the lazy default tab before testing focus and dirty-state events.
    await import('./RealtimeAssistTab')
  })

  beforeEach(() => {
    vi.restoreAllMocks()
    Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
      configurable: true,
      get: () => 120,
    })
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
      configurable: true,
      get: () => 32,
    })
    HTMLElement.prototype.getClientRects = vi.fn(() => [{ width: 120, height: 32 }] as any)

    useInterviewStore.setState({
      settingsOpen: true,
      settingsDrawerTab: 'realtime',
      config: {
        stt_provider: 'generic',
        stt_configured: { generic: true, doubao: true },
        whisper_model: 'base',
        whisper_language: 'auto',
        whisper_preload: false,
        doubao_stt_app_id: '',
        doubao_stt_access_token: '',
        doubao_stt_api_key: '',
        doubao_stt_resource_id: '',
        doubao_stt_boosting_table_id: '',
        generic_stt_api_base_url: '',
        generic_stt_api_key: '',
        generic_stt_model: '',
        generic_stt_custom_headers: '',
        candidate_asr_enabled: false,
        candidate_stt_provider: 'generic',
        candidate_whisper_model: '',
        candidate_whisper_language: '',
        candidate_remote_stt_enabled: false,
        candidate_context_enabled: true,
        candidate_context_wait_ms: 200,
        candidate_context_max_chars: 900,
        candidate_context_min_chars: 6,
        candidate_streaming_asr_enabled: true,
        candidate_streaming_asr_interval_ms: 1500,
        candidate_mic_compatibility_mode: true,
        silence_threshold: 0.01,
        silence_duration: 1.2,
        transcription_min_sig_chars: 2,
        assist_transcription_merge_gap_sec: 2,
        assist_transcription_merge_max_sec: 12,
        assist_high_churn_short_answer: false,
        auto_detect: true,
      },
      options: {
        stt_providers: ['whisper', 'doubao', 'generic'],
        whisper_models: ['tiny', 'base'],
      },
    } as any)
  })

  it('keeps focus on the active field when dirty state changes', async () => {
    const outsideButton = document.createElement('button')
    outsideButton.textContent = 'outside'
    document.body.appendChild(outsideButton)
    outsideButton.focus()

    render(<SettingsDrawer />)

    const modelSelect = await screen.findByRole('combobox', { name: '面试官 ASR 引擎' })
    modelSelect.focus()
    fireEvent.change(modelSelect, { target: { value: 'doubao' } })

    await waitFor(() => {
      expect(screen.getAllByText('未保存：实时辅助').length).toBeGreaterThan(0)
    })
    expect(document.activeElement).not.toBe(outsideButton)
    expect(screen.getByRole('dialog').contains(document.activeElement)).toBe(true)

    outsideButton.remove()
  })

  it('restores focus to the opener after dirty state changes and close is confirmed', async () => {
    const outsideButton = document.createElement('button')
    outsideButton.textContent = 'outside'
    document.body.appendChild(outsideButton)
    outsideButton.focus()
    vi.spyOn(window, 'confirm').mockReturnValue(true)

    render(<SettingsDrawer />)

    const modelSelect = await screen.findByRole('combobox', { name: '面试官 ASR 引擎' })
    fireEvent.change(modelSelect, { target: { value: 'doubao' } })
    await waitFor(() => {
      expect(screen.getAllByText('未保存：实时辅助').length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getByRole('button', { name: '关闭' }))

    await waitFor(() => {
      expect(document.activeElement).toBe(outsideButton)
    })
    outsideButton.remove()
  })

  it('asks before switching tabs with unsaved explicit settings', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<SettingsDrawer />)

    const modelSelect = await screen.findByRole('combobox', { name: '面试官 ASR 引擎' })
    fireEvent.change(modelSelect, { target: { value: 'doubao' } })
    await waitFor(() => {
      expect(screen.getAllByText('未保存：实时辅助').length).toBeGreaterThan(0)
    })

    fireEvent.click(within(screen.getByRole('navigation')).getByRole('button', { name: /^模型供应商配置/ }))

    expect(confirmSpy).toHaveBeenCalledWith('有未保存更改，切换后会丢失。')
    expect(screen.getByText('面试官 / 会议音频')).toBeInTheDocument()
  })

  it('shows explicit unsaved sections in the drawer header', async () => {
    render(<SettingsDrawer />)

    const modelSelect = await screen.findByRole('combobox', { name: '面试官 ASR 引擎' })
    fireEvent.change(modelSelect, { target: { value: 'doubao' } })

    // 桌面导航与移动头部各有一处提示
    const hints = await screen.findAllByText('未保存：实时辅助')
    expect(hints.length).toBeGreaterThanOrEqual(1)
  })

  it('opens realtime assist by default with a separate speech configuration tab', async () => {
    useInterviewStore.getState().toggleSettings()
    useInterviewStore.getState().toggleSettings()
    render(<SettingsDrawer />)

    const navigation = within(screen.getByRole('navigation'))
    expect(navigation.getByRole('button', { name: /实时辅助/ })).toHaveAttribute('aria-current', 'page')
    expect(navigation.getByRole('button', { name: /语音配置/ })).toBeInTheDocument()
    expect(navigation.queryByRole('button', { name: /^常用|^节奏|^热词/ })).not.toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /卡片/ })).toBeInTheDocument()
    expect(screen.queryByText('简短回答')).not.toBeInTheDocument()
  })

  it('opens 笔试配置 and keeps the separate overlay settings entry', async () => {
    render(<SettingsDrawer />)

    const navigation = within(screen.getByRole('navigation'))
    expect(navigation.queryByRole('button', { name: /^模式/ })).not.toBeInTheDocument()
    expect(navigation.getByRole('button', { name: /悬浮窗设置/ })).toBeInTheDocument()
    fireEvent.click(navigation.getByRole('button', { name: /笔试配置/ }))

    expect(navigation.getByRole('button', { name: /笔试配置/ })).toHaveAttribute('aria-current', 'page')
    expect(await screen.findByRole('checkbox', { name: '深度思考 (Think)' })).toBeInTheDocument()
  })
})
