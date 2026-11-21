import { afterEach, describe, expect, it, vi } from 'vitest'
import { mmpgController, mmpgErrorMessage } from './rtcController'

describe('MmpgController connect timeout', () => {
  afterEach(() => {
    mmpgController.cancelConnection()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('keeps connectTimer active until prepare() resolves so hung prepare times out', async () => {
    vi.useFakeTimers()

    Object.defineProperty(window, 'isSecureContext', {
      value: true,
      configurable: true,
    })

    class FakeAudioContext {
      state = 'running'
      audioWorklet = {
        addModule: vi.fn().mockResolvedValue(undefined),
      }
      resume = vi.fn().mockResolvedValue(undefined)
      close = vi.fn().mockResolvedValue(undefined)
    }

    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('AudioWorkletNode', function FakeAudioWorkletNode() {})
    vi.stubGlobal('WebSocket', vi.fn())

    const neverResolvingPrepare = () => new Promise<boolean>(() => {})
    const startPromise = mmpgController.start('start', 0, null, 'win32', neverResolvingPrepare)
    const expectation = expect(startPromise).rejects.toThrow(mmpgErrorMessage('connect_timeout'))

    // Let audioReady microtasks settle; connectTimer must still fire at 15s.
    await vi.advanceTimersByTimeAsync(50)
    expect(mmpgController.getSnapshot().state).toBe('connecting')

    await vi.advanceTimersByTimeAsync(15_000)
    await expectation
    expect(mmpgController.getSnapshot().state).toBe('error')
  })
})
