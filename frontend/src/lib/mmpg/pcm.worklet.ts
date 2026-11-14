import { PcmQueue } from './pcmQueue'

declare const sampleRate: number
declare const AudioWorkletProcessor: { new(): { port: MessagePort } }
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void

const PCM_RATE = 16000

class MmpgPcmSource extends AudioWorkletProcessor {
  private readonly queue = new PcmQueue(PCM_RATE, sampleRate, PCM_RATE / 10, PCM_RATE)
  private underruns = 0
  private overrunSamples = 0
  private lastAudio = 0
  private waitingForInput = false
  private rendered = 0

  constructor() {
    super()
    this.port.onmessage = ({ data }: MessageEvent<ArrayBuffer | 'clear' | 'listening'>) => {
      if (data === 'clear' || data === 'listening') {
        this.queue.clear()
        this.waitingForInput = data === 'listening'
        this.lastAudio = 0
        this.port.postMessage({ buffered_samples: 0 })
      } else {
        const chunk = new Float32Array(data)
        this.overrunSamples += this.queue.push(chunk)
        this.lastAudio = 0
        this.port.postMessage({ samples_received: chunk.length, buffered_samples: this.queue.bufferedSamples })
      }
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]) {
    const output = outputs[0]?.[0]
    if (!output) return true
    if (this.waitingForInput && this.queue.render(output).underrun) this.underruns++
    if (!this.waitingForInput) output.fill(0)
    if (this.waitingForInput) {
      this.lastAudio += output.length
      if (this.lastAudio >= sampleRate / 2) {
        this.underruns++
        this.lastAudio = 0
      }
    }
    this.rendered += output.length
    if (this.rendered >= sampleRate / 2) {
      this.rendered = 0
      this.port.postMessage({ underruns: this.underruns, overrun_samples: this.overrunSamples, buffered_samples: this.queue.bufferedSamples })
      this.underruns = 0
      this.overrunSamples = 0
    }
    return true
  }
}

registerProcessor('mmpg-pcm-source', MmpgPcmSource)
