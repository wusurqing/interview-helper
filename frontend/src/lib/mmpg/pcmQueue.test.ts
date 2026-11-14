import { describe, expect, it } from 'vitest'
import { PcmQueue } from './pcmQueue'

describe('MMPG PCM queue', () => {
  it('waits for the jitter target, then interpolates across input chunks', () => {
    const queue = new PcmQueue(16000, 32000, 4, 8, 0)
    const output = new Float32Array(6)
    queue.push(new Float32Array([0, 1]))
    expect(queue.render(output)).toEqual({ underrun: false })
    expect([...output]).toEqual([0, 0, 0, 0, 0, 0])
    queue.push(new Float32Array([2, 3]))
    expect(queue.render(output)).toEqual({ underrun: false })
    expect([...output]).toEqual([0, 0.5, 1, 1.5, 2, 2.5])
  })

  it('drops the oldest samples at capacity and re-buffers after an underrun', () => {
    const queue = new PcmQueue(16000, 16000, 2, 4, 0)
    const output = new Float32Array(5)
    expect(queue.push(new Float32Array([0, 1, 2, 3, 4, 5]))).toBe(2)
    expect(queue.render(output)).toEqual({ underrun: true })
    expect([...output]).toEqual([2, 3, 4, 5, 0])
    queue.push(new Float32Array([6]))
    expect(queue.render(output)).toEqual({ underrun: false })
    expect([...output]).toEqual([0, 0, 0, 0, 0])
  })

  it('retains resampler position across non-integer output blocks', () => {
    const queue = new PcmQueue(16000, 24000, 2, 12, 0)
    const output = new Float32Array(2)
    queue.push(Float32Array.from({ length: 8 }, (_, index) => index))
    queue.render(output)
    expect(output[1]).toBeCloseTo(2 / 3)
    queue.render(output)
    expect(output[0]).toBeCloseTo(4 / 3)
    expect(output[1]).toBeCloseTo(2)
  })

  it('absorbs a sustained difference between input and output clocks', () => {
    for (const drift of [-0.003, 0.003]) {
      const queue = new PcmQueue(16000, 48000, 1600, 16000)
      const output = new Float32Array(480)
      const samples = new Float32Array(161).fill(0.5)
      queue.push(new Float32Array(1600).fill(0.5))
      let produced = 0
      for (let frame = 1; frame <= 12000; frame++) {
        const next = Math.floor(frame * 160 * (1 + drift))
        if (queue.push(samples.subarray(0, next - produced))) throw new Error(`Overflow at frame ${frame}`)
        produced = next
        if (queue.render(output).underrun) throw new Error(`Underrun at frame ${frame}`)
        if (output[0] !== 0.5) throw new Error(`Silence at frame ${frame}`)
      }
    }
  })
})
