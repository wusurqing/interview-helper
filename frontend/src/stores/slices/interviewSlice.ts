import type { StateCreator } from 'zustand'
import type { RootState } from './rootState'
import type { QAPair, QAStatus, SingleMicState } from './types'
import { mmpgAnswerLabel } from '@/lib/mmpg/settings'

const CHUNK_THROTTLE_MS = 50
const MAX_TRANSCRIPTIONS = 200
const MAX_CANDIDATE_TRANSCRIPTIONS = 200
export const MAX_QA_PAIRS = 80

type AnswerMeta = {
  source?: string
  modelName?: string
  updatesQaId?: string
  answerFormat?: QAPair['answerFormat']
  attempt?: number
}

function compareAttempt(qa: QAPair | undefined, attempt: number | undefined) {
  return qa && attempt !== undefined ? attempt - (qa.attempt ?? 1) : 0
}

function appendBufferedText(qa: QAPair, buf: { answer: string; think: string }): QAPair {
  return {
    ...qa,
    thinkContent: qa.thinkContent + buf.think,
    answer: qa.answer + buf.answer,
    isThinking: buf.answer ? false : buf.think ? true : qa.isThinking,
    retrying: buf.answer ? false : qa.retrying,
  }
}

function takeTail<T>(items: T[], maxItems: number): T[] {
  return items.length > maxItems ? items.slice(-maxItems) : items
}

function normalizeVisionVerify(raw: unknown): QAPair['visionVerify'] | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const data = raw as { verdict?: unknown; reason?: unknown }
  const verdict = String(data.verdict || '').toUpperCase()
  if (verdict !== 'PASS' && verdict !== 'FAIL' && verdict !== 'UNKNOWN') return undefined
  return {
    verdict,
    reason: String(data.reason ?? ''),
  }
}

export interface InterviewSliceState {
  isRecording: boolean
  singleMic: SingleMicState | null
  interviewerStreaming: boolean
  candidateCaptureProvider: string | null
  mmpgAnswerLabel: string | null
  isPaused: boolean
  audioLevel: number
  isTranscribing: boolean
  transcriptions: string[]
  candidateTranscriptions: string[]
  qaPairs: QAPair[]
  streamingIds: string[]
  currentStreamingId: string | null
}

export interface InterviewSliceActions {
  setRecording: (v: boolean, interviewerStreaming?: boolean, candidateCaptureProvider?: string) => void
  setSingleMic: (value: unknown) => void
  setPaused: (v: boolean) => void
  setMmpgAnswerPhase: (phase: unknown) => void
  setAudioLevel: (v: number) => void
  setTranscribing: (v: boolean) => void
  addTranscription: (text: string) => void
  addCandidateTranscription: (text: string, meta?: { segmentId?: string; isFinal?: boolean }) => void
  startAnswer: (
    id: string,
    question: string,
    meta?: AnswerMeta,
  ) => void
  appendThinkChunk: (id: string, chunk: string, attempt?: number) => void
  appendAnswerChunk: (id: string, chunk: string, attempt?: number) => void
  finalizeAnswer: (
    id: string,
    question: string,
    answer: string,
    thinkContent?: string,
    modelName?: string,
    firstTokenMs?: number,
    totalMs?: number,
    updatesQaId?: string,
    meta?: Pick<AnswerMeta, 'source' | 'answerFormat' | 'attempt'>,
  ) => void
  cancelAnswer: (id: string, reason?: string, attempt?: number) => void
  errorAnswer: (id: string, message: string, attempt?: number) => void
  setVisionVerify: (id: string, verdict: 'PASS' | 'FAIL' | 'UNKNOWN', reason: string) => void
  setInitData: (data: any) => void
  clearSession: () => void
}

export type InterviewSlice = InterviewSliceState & InterviewSliceActions

function readSingleMic(value: unknown): SingleMicState | null {
  if (!value || typeof value !== 'object') return null
  const { id, device_id, role, switching, revision } = value as Partial<SingleMicState>
  if (typeof id !== 'string' || typeof device_id !== 'number' || !Number.isInteger(device_id) || device_id < 0) return null
  if ((role !== 'interviewer' && role !== 'candidate') || typeof switching !== 'boolean') return null
  return {
    id, device_id, role, switching,
    ...(typeof revision === 'number' && Number.isSafeInteger(revision) && revision >= 0 ? { revision } : {}),
  }
}

export const createInterviewSlice: StateCreator<RootState, [], [], InterviewSlice> = (set, get) => {
  const chunkBuffer: Map<string, { answer: string; think: string }> = new Map()
  let chunkFlushTimer: ReturnType<typeof setTimeout> | null = null
  let candidateSegmentIds: Array<string | null> = []

  const clearFlushTimer = () => {
    if (chunkFlushTimer !== null) {
      clearTimeout(chunkFlushTimer)
      chunkFlushTimer = null
    }
  }

  const clearTransientBuffers = () => {
    chunkBuffer.clear()
    candidateSegmentIds = []
    clearFlushTimer()
  }

  const deleteChunkBufferEntry = (id: string) => {
    chunkBuffer.delete(id)
    if (chunkBuffer.size === 0) {
      clearFlushTimer()
    }
  }

  const scheduleChunkFlush = () => {
    if (chunkFlushTimer !== null) return
    chunkFlushTimer = setTimeout(() => {
      chunkFlushTimer = null
      const pending = new Map(chunkBuffer)
      chunkBuffer.clear()
      if (pending.size === 0) return
      set((s) => {
        const activeStreamingIds = new Set(s.streamingIds)
        return {
          qaPairs: s.qaPairs.map((qa) => {
            const buf = pending.get(qa.id)
            if (!buf || qa.status !== 'streaming' || !activeStreamingIds.has(qa.id)) return qa
            return appendBufferedText(qa, buf)
          }),
        }
      })
    }, CHUNK_THROTTLE_MS)
  }

  const trimQaWindow = (qaPairs: QAPair[], streamingIds: string[]) => {
    if (qaPairs.length <= MAX_QA_PAIRS) {
      return {
        qaPairs,
        streamingIds,
        currentStreamingId: streamingIds.length ? streamingIds[streamingIds.length - 1] : null,
      }
    }
    const overflow = qaPairs.length - MAX_QA_PAIRS
    const evicted = qaPairs.slice(0, overflow)
    for (const item of evicted) {
      deleteChunkBufferEntry(item.id)
    }
    const kept = qaPairs.slice(overflow)
    const keptIds = new Set(kept.map((item) => item.id))
    const nextStreamingIds = streamingIds.filter((id) => keptIds.has(id))
    return {
      qaPairs: kept,
      streamingIds: nextStreamingIds,
      currentStreamingId: nextStreamingIds.length ? nextStreamingIds[nextStreamingIds.length - 1] : null,
    }
  }

  return {
    isRecording: false,
    singleMic: null,
    interviewerStreaming: false,
    candidateCaptureProvider: null,
    mmpgAnswerLabel: null,
    isPaused: false,
    audioLevel: 0,
    isTranscribing: false,
    transcriptions: [],
    candidateTranscriptions: [],
    qaPairs: [],
    streamingIds: [],
    currentStreamingId: null,

    setRecording: (v, interviewerStreaming, candidateCaptureProvider) => set((state) => ({
      isRecording: v, interviewerStreaming: v && interviewerStreaming === true,
      candidateCaptureProvider: v ? candidateCaptureProvider ?? state.candidateCaptureProvider : null,
      ...(!v ? { mmpgAnswerLabel: null, singleMic: null } : {}),
    })),
    setSingleMic: (value) => {
      const next = readSingleMic(value)
      if (!next && value !== null) return
      set((state) => {
        const current = state.singleMic
        if (current && next && current.id === next.id && current.revision !== undefined
          && (next.revision === undefined || next.revision < current.revision)) return state
        return { singleMic: next }
      })
    },
    setMmpgAnswerPhase: (phase) => set({ mmpgAnswerLabel: mmpgAnswerLabel(phase) }),
    setPaused: (v) => set({ isPaused: v }),
    setAudioLevel: (v) => set({ audioLevel: v }),
    setTranscribing: (v) => set({ isTranscribing: v }),
    addTranscription: (text) => set((s) => ({
      transcriptions: takeTail([...s.transcriptions, text], MAX_TRANSCRIPTIONS),
    })),
    addCandidateTranscription: (text, meta) => set((s) => {
      if (s.candidateTranscriptions.length === 0 && candidateSegmentIds.length > 0) {
        candidateSegmentIds = []
      }
      const segmentId = meta?.segmentId || null
      if (segmentId) {
        const existingIndex = candidateSegmentIds.lastIndexOf(segmentId)
        if (existingIndex >= 0 && existingIndex < s.candidateTranscriptions.length) {
          const next = [...s.candidateTranscriptions]
          next[existingIndex] = text
          return { candidateTranscriptions: next }
        }
      }
      if (s.candidateTranscriptions[s.candidateTranscriptions.length - 1] === text) {
        return { candidateTranscriptions: s.candidateTranscriptions }
      }
      const nextTranscriptions = [...s.candidateTranscriptions, text]
      const nextSegmentIds = [...candidateSegmentIds, segmentId]
      const overflow = Math.max(0, nextTranscriptions.length - MAX_CANDIDATE_TRANSCRIPTIONS)
      candidateSegmentIds = overflow > 0 ? nextSegmentIds.slice(overflow) : nextSegmentIds
      return {
        candidateTranscriptions: overflow > 0
          ? nextTranscriptions.slice(overflow)
          : nextTranscriptions,
      }
    }),

    startAnswer: (id, question, meta) =>
      set((s) => {
        const existing = s.qaPairs.find((qa) => qa.id === id)
        if (existing) {
          const attempt = meta?.attempt ?? existing.attempt ?? 1
          const previousAttempt = existing.attempt ?? 1
          if (existing.status !== 'streaming' || attempt < previousAttempt) return s
          const restarted = attempt > previousAttempt
          if (restarted) deleteChunkBufferEntry(id)
          const streamingIds = s.streamingIds.includes(id) ? s.streamingIds : [...s.streamingIds, id]
          return {
            currentStreamingId: id,
            streamingIds,
            qaPairs: s.qaPairs.map((qa) =>
              qa.id === id
                ? {
                    ...qa,
                    question,
                    questionSource: meta?.source ?? qa.questionSource,
                    modelLabel: meta?.modelName ?? qa.modelLabel,
                    updatesQaId: meta?.updatesQaId ?? qa.updatesQaId,
                    answerFormat: meta?.answerFormat ?? qa.answerFormat,
                    attempt,
                    ...(restarted ? {
                      answer: '', thinkContent: '', isThinking: false, retrying: true,
                      firstTokenMs: undefined, totalMs: undefined,
                    } : {}),
                  }
                : qa,
            ),
          }
        }
        const nextQaPairs: QAPair[] = [
          ...s.qaPairs,
          {
            id,
            question,
            answer: '',
            thinkContent: '',
            isThinking: false,
            timestamp: Date.now() / 1000,
            questionSource: meta?.source,
            modelLabel: meta?.modelName,
            updatesQaId: meta?.updatesQaId,
            answerFormat: meta?.answerFormat,
            attempt: meta?.attempt,
            retrying: (meta?.attempt ?? 1) > 1,
            status: 'streaming' as QAStatus,
          },
        ]
        const nextStreamingIds = [...s.streamingIds, id]
        return trimQaWindow(nextQaPairs, nextStreamingIds)
      }),

    appendThinkChunk: (id, chunk, attempt) => {
      const qa = get().qaPairs.find((item) => item.id === id)
      if (compareAttempt(qa, attempt) !== 0 || (qa && qa.status !== 'streaming')) return
      const buf = chunkBuffer.get(id) ?? { answer: '', think: '' }
      buf.think += chunk
      chunkBuffer.set(id, buf)
      scheduleChunkFlush()
    },

    appendAnswerChunk: (id, chunk, attempt) => {
      const qa = get().qaPairs.find((item) => item.id === id)
      if (compareAttempt(qa, attempt) !== 0 || (qa && qa.status !== 'streaming')) return
      const buf = chunkBuffer.get(id) ?? { answer: '', think: '' }
      buf.answer += chunk
      if (qa?.answerFormat === 'plain_text' && qa.status === 'streaming' && !qa.answer.trim()) {
        deleteChunkBufferEntry(id)
        set((s) => ({ qaPairs: s.qaPairs.map((item) => item.id === id ? appendBufferedText(item, buf) : item) }))
        return
      }
      chunkBuffer.set(id, buf)
      scheduleChunkFlush()
    },

    finalizeAnswer: (id, question, answer, thinkContent, modelName, firstTokenMs, totalMs, updatesQaId, meta) => {
      const comparison = compareAttempt(get().qaPairs.find((qa) => qa.id === id), meta?.attempt)
      if (comparison < 0) return
      const newer = comparison > 0
      const buffered = newer ? undefined : chunkBuffer.get(id)
      deleteChunkBufferEntry(id)
      set((s) => {
        const next = s.streamingIds.filter((x) => x !== id)
        const existing = s.qaPairs.find((qa) => qa.id === id)
        const answerFormat = meta?.answerFormat ?? existing?.answerFormat
        if (existing?.status === 'cancelled' && existing.cancelReason === 'continuation') {
          return s
        }
        if (answerFormat === 'plain_text' && existing && existing.status !== 'streaming') return s
        const qaPairs = existing
          ? s.qaPairs.map((qa) => {
              if (qa.id !== id) return qa
              const terminal = {
                isThinking: false,
                retrying: false,
                attempt: meta?.attempt ?? qa.attempt,
                firstTokenMs: firstTokenMs ?? qa.firstTokenMs,
                totalMs: totalMs ?? qa.totalMs,
                status: 'done' as const,
              }
              if (answerFormat === 'plain_text') {
                const streamed = buffered ? appendBufferedText(qa, buffered) : qa
                // The full answer is a recovery copy for a missed WebSocket fragment.
                const recovered = newer || (typeof answer === 'string' && answer !== streamed.answer)
                return {
                  ...streamed,
                  ...terminal,
                  answerFormat,
                  ...(recovered ? { answer, thinkContent: thinkContent ?? (newer ? '' : streamed.thinkContent) } : {}),
                  ...(meta?.attempt !== undefined && modelName ? { modelLabel: modelName } : {}),
                }
              }
              return {
                ...qa, ...terminal, question, answer,
                thinkContent: thinkContent ?? qa.thinkContent,
                modelLabel: modelName ?? qa.modelLabel,
                updatesQaId: updatesQaId ?? qa.updatesQaId,
              }
            })
          : [
              ...s.qaPairs,
              {
                id,
                question,
                answer,
                thinkContent: thinkContent ?? '',
                isThinking: false,
                timestamp: Date.now() / 1000,
                modelLabel: modelName,
                questionSource: meta?.source,
                answerFormat,
                firstTokenMs,
                totalMs,
                updatesQaId,
                attempt: meta?.attempt,
                status: 'done' as QAStatus,
              },
            ]
        return trimQaWindow(qaPairs, next)
      })
    },

    cancelAnswer: (id, reason, attempt) => {
      const comparison = compareAttempt(get().qaPairs.find((qa) => qa.id === id), attempt)
      if (comparison < 0) return
      const buffered = comparison > 0 ? undefined : chunkBuffer.get(id)
      deleteChunkBufferEntry(id)
      set((s) => {
        const next = s.streamingIds.filter((x) => x !== id)
        return {
          currentStreamingId: next.length ? next[next.length - 1] : null,
          streamingIds: next,
          qaPairs: s.qaPairs.map((qa) =>
            qa.id === id && qa.status !== 'done'
              ? {
                  ...qa,
                  ...(comparison > 0 ? { answer: '', thinkContent: '' } : {
                    answer: reason === 'continuation' || qa.answerFormat === 'plain_text'
                      ? qa.answer + (buffered?.answer ?? '') : qa.answer,
                  }),
                  attempt: attempt ?? qa.attempt,
                  isThinking: false,
                  retrying: false,
                  status: 'cancelled' as QAStatus,
                  cancelReason: reason,
                }
              : qa,
          ),
        }
      })
    },

    errorAnswer: (id, message, attempt) => {
      const previous = get().qaPairs.find((qa) => qa.id === id)
      const comparison = compareAttempt(previous, attempt)
      if (comparison < 0 || previous?.status === 'done') return
      const buffered = comparison > 0 ? undefined : chunkBuffer.get(id)
      deleteChunkBufferEntry(id)
      set((s) => {
        const next = s.streamingIds.filter((x) => x !== id)
        return {
          currentStreamingId: next.length ? next[next.length - 1] : null,
          streamingIds: next,
          qaPairs: s.qaPairs.map((qa) =>
            qa.id === id
              ? {
                  ...(qa.answerFormat === 'plain_text' && buffered ? appendBufferedText(qa, buffered) : qa),
                  ...(comparison > 0 ? { answer: '', thinkContent: '' } : {}),
                  attempt: attempt ?? qa.attempt,
                  isThinking: false, retrying: false, status: 'error' as QAStatus, errorMessage: message,
                }
              : qa,
          ),
        }
      })
    },

    setVisionVerify: (id, verdict, reason) =>
      set((s) => ({
        qaPairs: s.qaPairs.map((qa) =>
          qa.id === id ? { ...qa, visionVerify: { verdict, reason } } : qa,
        ),
      })),

    setInitData: (data) => {
      clearTransientBuffers()
      const candidateSegments = Array.isArray(data.candidate_answer_segments)
        ? data.candidate_answer_segments
        : []
      const restoredCandidatePairs: Array<{ text: string; segmentId: string | null }> = candidateSegments.length > 0
        ? candidateSegments
            .map((segment: any) => ({
              text: String(segment?.text ?? '').trim(),
              segmentId: segment?.segment_id ? String(segment.segment_id) : null,
            }))
            .filter((segment: { text: string; segmentId: string | null }) => Boolean(segment.text))
        : (data.candidate_transcriptions ?? [])
            .map((text: unknown) => ({ text: String(text ?? '').trim(), segmentId: null }))
            .filter((segment: { text: string; segmentId: string | null }) => Boolean(segment.text))
      const restoredCandidatePairsTail = takeTail(restoredCandidatePairs, MAX_CANDIDATE_TRANSCRIPTIONS)
      const restoredCandidateTranscriptions = restoredCandidatePairsTail.map((segment) => segment.text)
      candidateSegmentIds = restoredCandidatePairsTail.map((segment) => segment.segmentId)
      const restoredQaPairs: QAPair[] = takeTail(
        (data.qa_pairs ?? []).map(
          (qa: Partial<QAPair> & { id: string; question: string; answer: string; answer_format?: unknown;
            first_token_ms?: number; total_ms?: number }) => {
            const visionVerify = normalizeVisionVerify((qa as any).visionVerify ?? (qa as any).vision_verify)
            return {
              ...qa,
              thinkContent: qa.thinkContent ?? '',
              isThinking: false,
              timestamp: qa.timestamp ?? Date.now() / 1000,
              questionSource: (qa as any).source ?? qa.questionSource,
              modelLabel: (qa as any).model_name ?? qa.modelLabel,
              answerFormat: qa.answer_format === 'plain_text' ? 'plain_text' : qa.answerFormat,
              firstTokenMs: qa.first_token_ms ?? qa.firstTokenMs,
              totalMs: qa.total_ms ?? qa.totalMs,
              updatesQaId: (qa as any).updates_qa_id ?? qa.updatesQaId,
              status: qa.status ?? 'done' as QAStatus,
              ...(visionVerify ? { visionVerify } : {}),
            }
          },
        ),
        MAX_QA_PAIRS,
      )
      set({
        transcriptions: takeTail(data.transcriptions ?? [], MAX_TRANSCRIPTIONS),
        candidateTranscriptions: restoredCandidateTranscriptions,
        qaPairs: restoredQaPairs,
        currentStreamingId: null,
        streamingIds: [],
        isRecording: data.is_recording ?? false,
        singleMic: readSingleMic(data.single_mic),
        interviewerStreaming: data.is_recording === true && data.interviewer_streaming === true,
        candidateCaptureProvider: data.is_recording && typeof data.candidate_capture_provider === 'string' ? data.candidate_capture_provider : null,
        mmpgAnswerLabel: mmpgAnswerLabel(data.mmpg_answer_phase),
        isPaused: data.is_paused ?? false,
        sttLoaded: data.stt_loaded ?? false,
        candidateSttLoaded: false,
        candidateSttLoading: false,
        candidateSttProvider: '',
      })
    },

    clearSession: () => {
      clearTransientBuffers()
      set({
        mmpgAnswerLabel: null,
        transcriptions: [],
        candidateTranscriptions: [],
        qaPairs: [],
        currentStreamingId: null,
        streamingIds: [],
        isRecording: false,
        singleMic: null,
        interviewerStreaming: false,
        candidateCaptureProvider: null,
        isPaused: false,
        isTranscribing: false,
        audioLevel: 0,
        candidateSttLoaded: false,
        candidateSttLoading: false,
        candidateSttProvider: '',
      })
    },
  }
}
