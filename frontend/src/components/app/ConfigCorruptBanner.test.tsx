import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { updateConfigAndRefresh } from '@/lib/configSync'
import { useInterviewStore, type AppConfig } from '@/stores/configStore'
import { ConfigCorruptBanner } from './ConfigCorruptBanner'

vi.mock('@/lib/configSync', () => ({ updateConfigAndRefresh: vi.fn() }))

describe('ConfigCorruptBanner', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useInterviewStore.setState({
      config: {
        config_corrupt: false,
        config_corrupt_info: null,
      } as AppConfig,
    })
  })

  it('renders nothing when config is healthy', () => {
    const { container } = render(<ConfigCorruptBanner />)
    expect(container.firstChild).toBeNull()
  })

  it('shows warning with backup filename and confirms before overwriting', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
    vi.mocked(updateConfigAndRefresh).mockResolvedValue({ config_corrupt: false } as AppConfig)

    useInterviewStore.setState({
      config: {
        config_corrupt: true,
        config_corrupt_info: {
          error: '配置校验失败（1 项无效：temperature）',
          backup_filename: 'config.json.corrupt.20261121-002500',
        },
      } as AppConfig,
    })

    render(<ConfigCorruptBanner />)
    expect(screen.getByRole('alert')).toHaveTextContent('config.json.corrupt.20261121-002500')
    expect(screen.getByRole('alert')).toHaveTextContent('temperature')

    // Canceling confirm does not write.
    fireEvent.click(screen.getByRole('button', { name: '确认覆盖为默认配置' }))
    expect(confirmSpy).toHaveBeenCalledTimes(1)
    expect(updateConfigAndRefresh).not.toHaveBeenCalled()

    // Confirming calls updateConfigAndRefresh with explicit flag.
    fireEvent.click(screen.getByRole('button', { name: '确认覆盖为默认配置' }))
    await waitFor(() =>
      expect(updateConfigAndRefresh).toHaveBeenCalledExactlyOnceWith({ confirm_overwrite_corrupt: true }),
    )
  })
})
