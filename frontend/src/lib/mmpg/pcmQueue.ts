/** Fixed-size PCM queue. Drop the oldest input when the producer gets too far ahead. */
export class PcmQueue {
  private readonly samples: Float32Array
  private readonly step: number
  private head = 0
  private size = 0
  private phase = 0
  private primed = false
  private correction = 0

  constructor(inputRate: number, outputRate: number, private readonly target: number, capacity: number, private readonly maxDrift = 0.005) {
    this.samples = new Float32Array(capacity)
    this.step = inputRate / outputRate
  }

  get bufferedSamples() {
    return this.size
  }

  clear() {
    this.head = 0
    this.size = 0
    this.phase = 0
    this.primed = false
    this.correction = 0
  }

  push(chunk: Float32Array): number {
    const dropped = Math.max(0, this.size + chunk.length - this.samples.length)
    const bufferedDrop = Math.min(this.size, dropped)
    this.consume(bufferedDrop)
    if (dropped) this.phase = 0
    const input = chunk.subarray(dropped - bufferedDrop)
    const tail = (this.head + this.size) % this.samples.length
    const firstCount = Math.min(input.length, this.samples.length - tail)
    this.samples.set(input.subarray(0, firstCount), tail)
    this.samples.set(input.subarray(firstCount), 0)
    this.size += input.length
    return dropped
  }

  render(output: Float32Array): { underrun: boolean } {
    output.fill(0)
    if (!this.primed && this.size < this.target) return { underrun: false }
    this.primed = true
    // Average packet-size jitter before adjusting for clock-rate mismatch.
    const desired = Math.max(-this.maxDrift, Math.min(this.maxDrift, (this.size / this.target - 1) * 0.025))
    this.correction += (desired - this.correction) * 0.005
    const step = this.step * (1 + this.correction)
    for (let index = 0; index < output.length; index++) {
      const nextPhase = this.phase + step
      const consumed = Math.floor(nextPhase)
      if (this.size < Math.max(this.phase > 0 ? 2 : 1, consumed)) {
        this.clear()
        return { underrun: true }
      }
      const first = this.samples[this.head]
      const second = this.samples[(this.head + 1) % this.samples.length]
      output[index] = first + (second - first) * this.phase
      this.consume(consumed)
      this.phase = nextPhase - consumed
    }
    return { underrun: false }
  }

  private consume(count: number) {
    this.head = (this.head + count) % this.samples.length
    this.size -= count
  }
}
