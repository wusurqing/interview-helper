import { useEffect, useRef, useState } from 'react'
import { Pause, Volume2 } from 'lucide-react'
import { api, getErrorMessage } from '@/lib/api'
import type { ArchiveClip } from './types'

let pauseActivePlayer: (() => void) | undefined

export function locateClip(clips: ArchiveClip[], position: number) {
  let start = 0
  for (const [index, clip] of clips.entries()) {
    if (position < start + clip.duration) return { index, offset: position - start, start }
    start += clip.duration
  }
  return null
}

function clock(seconds: number) {
  const value = Math.max(0, Math.floor(seconds))
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`
}

interface Playback {
  context: AudioContext
  abort: AbortController
  source?: AudioBufferSourceNode
  frame: number
}

export default function ArchiveAudioPlayer({ clips, sessionId, turnId, label }: {
  clips: ArchiveClip[]; sessionId: number; turnId: number; label: string
}) {
  const duration = clips.reduce((sum, clip) => sum + clip.duration, 0)
  const [position, setPosition] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [error, setError] = useState('')
  const state = useRef<Playback | null>(null)
  const currentPosition = useRef(0)
  const resumeAfterDrag = useRef(false)

  const stop = () => {
    const playback = state.current
    state.current = null
    if (!playback) return
    playback.abort.abort()
    cancelAnimationFrame(playback.frame)
    if (playback.source) {
      playback.source.onended = null
      playback.source.stop()
      playback.source.disconnect()
    }
    void playback.context.close().catch(() => {})
  }

  const pause = () => {
    stop()
    if (pauseActivePlayer === pause) pauseActivePlayer = undefined
    setPlaying(false)
  }

  useEffect(() => () => {
    if (state.current) pauseActivePlayer = undefined
    stop()
  }, [sessionId, turnId])

  const updatePosition = (value: number) => {
    currentPosition.current = value
    setPosition(value)
  }

  const play = async (from: number) => {
    pauseActivePlayer?.()
    stop()
    pauseActivePlayer = pause
    setError('')
    setPlaying(true)
    let playback: Playback | null = null
    try {
      playback = { context: new AudioContext(), abort: new AbortController(), frame: 0 }
      state.current = playback
      const { context, abort } = playback
      const buffers = new Map<string, Promise<AudioBuffer>>()
      const load = (clip: ArchiveClip) => {
        let pending = buffers.get(clip.asset_id)
        if (!pending) {
          pending = api.reviewAudio(sessionId, turnId, clip.asset_id, abort.signal)
            .then((bytes) => context.decodeAudioData(bytes))
          buffers.set(clip.asset_id, pending)
        }
        return pending
      }
      await context.resume()
      const first = locateClip(clips, from)
      if (!first) {
        pause()
        return
      }
      for (let index = first.index; index < clips.length && !abort.signal.aborted; index++) {
        const clip = clips[index]
        const range = clips.slice(index, index + 3)
        for (const assetId of buffers.keys()) {
          if (!range.some((item) => item.asset_id === assetId)) buffers.delete(assetId)
        }
        for (const next of range) void load(next).catch(() => {})
        const buffer = await load(clip)
        if (abort.signal.aborted) return
        const offset = index === first.index ? first.offset : 0
        const left = Math.min(clip.duration - offset, buffer.duration - clip.offset - offset)
        if (left <= 0) continue
        const base = index === first.index ? from : currentPosition.current
        await new Promise<void>((resolve) => {
          const source = context.createBufferSource()
          playback!.source = source
          source.buffer = buffer
          source.connect(context.destination)
          const started = context.currentTime
          const tick = () => {
            updatePosition(Math.min(base + left, base + context.currentTime - started))
            playback!.frame = requestAnimationFrame(tick)
          }
          const complete = () => {
            abort.signal.removeEventListener('abort', complete)
            cancelAnimationFrame(playback!.frame)
            source.disconnect()
            playback!.source = undefined
            if (!abort.signal.aborted) updatePosition(base + clip.duration - offset)
            resolve()
          }
          abort.signal.addEventListener('abort', complete, { once: true })
          source.onended = complete
          source.start(0, clip.offset + offset, left)
          tick()
        })
      }
      if (!abort.signal.aborted) {
        updatePosition(duration)
        pause()
      }
    } catch (cause) {
      if (!playback?.abort.signal.aborted && (!playback || state.current === playback)) {
        setError(getErrorMessage(cause, '录音无法播放'))
        pause()
      }
    }
  }

  if (!duration) return null
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
      <button type="button" aria-label={`${playing ? '暂停' : '播放'}${label}录音`} className="rounded p-1.5 hover:bg-bg-hover"
        onClick={() => playing ? pause() : void play(position >= duration ? 0 : position)}>
        {playing ? <Pause className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
      </button>
      <input type="range" min={0} max={duration} step={0.01} value={position} aria-label={`${label}播放进度`}
        className="min-w-16 max-w-64 flex-1 accent-accent-blue" onPointerDown={(event) => {
          resumeAfterDrag.current = playing
          if (playing) pause()
          event.currentTarget.setPointerCapture?.(event.pointerId)
        }} onPointerUp={() => {
          if (resumeAfterDrag.current && currentPosition.current < duration) void play(currentPosition.current)
          resumeAfterDrag.current = false
        }} onPointerCancel={() => { resumeAfterDrag.current = false }} onChange={(event) => {
          const next = Number(event.currentTarget.value)
          updatePosition(next)
          if (playing && !resumeAfterDrag.current && next < duration) void play(next)
          else if (playing) pause()
        }} />
      <span className="tabular-nums">{clock(position)} / {clock(duration)}</span>
      {error && <span role="alert" className="text-accent-red">{error}</span>}
    </div>
  )
}
