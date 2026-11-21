export interface PreflightWsMessage {
  type: string
  [key: string]: unknown
}

type PreflightListener = (msg: PreflightWsMessage) => void

const soundPreflightListeners = new Set<PreflightListener>()
const examPreflightListeners = new Set<PreflightListener>()

export function subscribeSoundPreflight(listener: PreflightListener): () => void {
  soundPreflightListeners.add(listener)
  return () => {
    soundPreflightListeners.delete(listener)
  }
}

export function emitSoundPreflight(msg: PreflightWsMessage): void {
  for (const listener of soundPreflightListeners) {
    try {
      listener(msg)
    } catch (err) {
      console.warn('[ws] sound preflight listener failed', err)
    }
  }
}

export function subscribeExamPreflight(listener: PreflightListener): () => void {
  examPreflightListeners.add(listener)
  return () => {
    examPreflightListeners.delete(listener)
  }
}

export function emitExamPreflight(msg: PreflightWsMessage): void {
  for (const listener of examPreflightListeners) {
    try {
      listener(msg)
    } catch (err) {
      console.warn('[ws] exam preflight listener failed', err)
    }
  }
}
