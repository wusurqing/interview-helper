import type { StateCreator } from 'zustand'
import type { RootState } from './rootState'
import type { QAPair, QAStatus } from './types'
import { mmpgAnswerLabel } from '@/lib/mmpg/settings'

const CHUNK_THROTTLE_MS = 50
const MAX_TRANSCRIPTIONS = 200
const MAX_CANDIDATE_TRANSCRIPTIONS = 200

const _chunkBuffer: Map<string, { answer: string; think: string }> = new Map()
let _chunkFlushTimer: ReturnType<typeof setTimeout> | null = null
let _candidateSegmentIds: Array<string | null> = []

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

function _scheduleChunkFlush(set: (fn: (s: RootState) => Partial<RootState>) => void) {
  if (_chunkFlushTimer !== null) return
  _chunkFlushTimer = setTimeout(() => {
    _chunkFlushTimer = null
    const pending = new Map(_chunkBuffer)
    _chunkBuffer.clear()
    if (pending.size === 0) return
    set((s) => {
      const activeStreamingIds = new Set(s.streamingIds)
      return {
        qaPairs: s.qaPairs.map((qa) => {
          const buf = pending.get(qa.id)
          if (!buf || qa.status !== 'streaming' || !activeStreamingIds.has(qa.id)) return qa
          return {
            ...qa,
            thinkContent: buf.think ? qa.thinkContent + buf.think : qa.thinkContent,
            answer: buf.answer ? qa.answer + buf.answer : qa.answer,
            isThinking: buf.answer ? false : buf.think ? true : qa.isThinking,
          }
        }),
      }
    })
  }, CHUNK_THROTTLE_MS)
}

export interface InterviewSliceState {
  isRecording: boolean
  interviewerStreaming: boolean
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
  setRecording: (v: boolean, interviewerStreaming?: boolean) => void
  setPaused: (v: boolean) => void
  setMmpgAnswerPhase: (phase: unknown) => void
  setAudioLevel: (v: number) => void
  setTranscribing: (v: boolean) => void
  addTranscription: (text: string) => void
  addCandidateTranscription: (text: string, meta?: { segmentId?: string; isFinal?: boolean }) => void
  startAnswer: (
    id: string,
    question: string,
    meta?: { source?: string; modelName?: string; updatesQaId?: string },
  ) => void
  appendThinkChunk: (id: string, chunk: string) => void
  appendAnswerChunk: (id: string, chunk: string) => void
  finalizeAnswer: (
    id: string,
    question: string,
    answer: string,
    thinkContent?: string,
    modelName?: string,
    firstTokenMs?: number,
    totalMs?: number,
    updatesQaId?: string,
  ) => void
  cancelAnswer: (id: string, reason?: string) => void
  errorAnswer: (id: string, message: string) => void
  setVisionVerify: (id: string, verdict: 'PASS' | 'FAIL' | 'UNKNOWN', reason: string) => void
  setInitData: (data: any) => void
  clearSession: () => void
}

export type InterviewSlice = InterviewSliceState & InterviewSliceActions

export const createInterviewSlice: StateCreator<RootState, [], [], InterviewSlice> = (set) => ({
  isRecording: false,
  interviewerStreaming: false,
  mmpgAnswerLabel: null,
  isPaused: false,
  audioLevel: 0,
  isTranscribing: false,
  transcriptions: [],
  candidateTranscriptions: [],
  qaPairs: [],
  streamingIds: [],
  currentStreamingId: null,

  setRecording: (v, interviewerStreaming) => set({
    isRecording: v, interviewerStreaming: v && interviewerStreaming === true,
    ...(!v ? { mmpgAnswerLabel: null } : {}),
  }),
  setMmpgAnswerPhase: (phase) => set({ mmpgAnswerLabel: mmpgAnswerLabel(phase) }),
  setPaused: (v) => set({ isPaused: v }),
  setAudioLevel: (v) => set({ audioLevel: v }),
  setTranscribing: (v) => set({ isTranscribing: v }),
  addTranscription: (text) => set((s) => ({
    transcriptions: takeTail([...s.transcriptions, text], MAX_TRANSCRIPTIONS),
  })),
  addCandidateTranscription: (text, meta) => set((s) => {
    const segmentId = meta?.segmentId || null
    if (segmentId) {
      const existingIndex = _candidateSegmentIds.lastIndexOf(segmentId)
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
    const nextSegmentIds = [..._candidateSegmentIds, segmentId]
    const overflow = Math.max(0, nextTranscriptions.length - MAX_CANDIDATE_TRANSCRIPTIONS)
    _candidateSegmentIds = overflow > 0 ? nextSegmentIds.slice(overflow) : nextSegmentIds
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
        if (existing.status !== 'streaming') {
          return {
            currentStreamingId: s.currentStreamingId,
            streamingIds: s.streamingIds,
            qaPairs: s.qaPairs,
          }
        }
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
                }
              : qa,
          ),
        }
      }
      return {
        currentStreamingId: id,
        streamingIds: [...s.streamingIds, id],
        qaPairs: [
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
            status: 'streaming' as QAStatus,
          },
        ],
      }
    }),

  appendThinkChunk: (id, chunk) => {
    const buf = _chunkBuffer.get(id) ?? { answer: '', think: '' }
    buf.think += chunk
    _chunkBuffer.set(id, buf)
    _scheduleChunkFlush(set)
  },

  appendAnswerChunk: (id, chunk) => {
    const buf = _chunkBuffer.get(id) ?? { answer: '', think: '' }
    buf.answer += chunk
    _chunkBuffer.set(id, buf)
    _scheduleChunkFlush(set)
  },

  finalizeAnswer: (id, question, answer, thinkContent, modelName, firstTokenMs, totalMs, updatesQaId) => {
    _chunkBuffer.delete(id)
    set((s) => {
      const next = s.streamingIds.filter((x) => x !== id)
      const existing = s.qaPairs.some((qa) => qa.id === id)
      if (s.qaPairs.some((qa) => qa.id === id && qa.status === 'cancelled' && qa.cancelReason === 'continuation')) {
        return s
      }
      const qaPairs = existing
        ? s.qaPairs.map((qa) =>
            qa.id === id
              ? {
                  ...qa,
                  question,
                  answer,
                  thinkContent: thinkContent ?? qa.thinkContent,
                  isThinking: false,
                  modelLabel: modelName ?? qa.modelLabel,
                  firstTokenMs: firstTokenMs ?? qa.firstTokenMs,
                  totalMs: totalMs ?? qa.totalMs,
                  updatesQaId: updatesQaId ?? qa.updatesQaId,
                  status: 'done' as QAStatus,
                }
              : qa,
          )
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
              firstTokenMs,
              totalMs,
              updatesQaId,
              status: 'done' as QAStatus,
            },
          ]
      return {
        currentStreamingId: next.length ? next[next.length - 1] : null,
        streamingIds: next,
        qaPairs,
      }
    })
  },

  cancelAnswer: (id, reason) => {
    const buffered = _chunkBuffer.get(id)
    _chunkBuffer.delete(id)
    set((s) => {
      const next = s.streamingIds.filter((x) => x !== id)
      return {
        currentStreamingId: next.length ? next[next.length - 1] : null,
        streamingIds: next,
        qaPairs: s.qaPairs.map((qa) =>
          qa.id === id && qa.status !== 'done'
            ? {
                ...qa,
                answer: reason === 'continuation' ? qa.answer + (buffered?.answer ?? '') : qa.answer,
                isThinking: false,
                status: 'cancelled' as QAStatus,
                cancelReason: reason,
              }
            : qa,
        ),
      }
    })
  },

  errorAnswer: (id, message) => {
    _chunkBuffer.delete(id)
    set((s) => {
      const next = s.streamingIds.filter((x) => x !== id)
      return {
        currentStreamingId: next.length ? next[next.length - 1] : null,
        streamingIds: next,
        qaPairs: s.qaPairs.map((qa) =>
          qa.id === id
            ? { ...qa, isThinking: false, status: 'error' as QAStatus, errorMessage: message }
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
    _chunkBuffer.clear()
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
    _candidateSegmentIds = restoredCandidatePairsTail.map((segment) => segment.segmentId)
    if (_chunkFlushTimer !== null) {
      clearTimeout(_chunkFlushTimer)
      _chunkFlushTimer = null
    }
    set({
      transcriptions: takeTail(data.transcriptions ?? [], MAX_TRANSCRIPTIONS),
      candidateTranscriptions: restoredCandidateTranscriptions,
      qaPairs: (data.qa_pairs ?? []).map(
        (qa: Partial<QAPair> & { id: string; question: string; answer: string }) => {
          const visionVerify = normalizeVisionVerify((qa as any).visionVerify ?? (qa as any).vision_verify)
          return {
            ...qa,
            thinkContent: qa.thinkContent ?? '',
            isThinking: false,
            timestamp: qa.timestamp ?? Date.now() / 1000,
            questionSource: (qa as any).source ?? qa.questionSource,
            modelLabel: (qa as any).model_name ?? qa.modelLabel,
            updatesQaId: (qa as any).updates_qa_id ?? qa.updatesQaId,
            status: qa.status ?? 'done' as QAStatus,
            ...(visionVerify ? { visionVerify } : {}),
          }
        },
      ),
      currentStreamingId: null,
      streamingIds: [],
      isRecording: data.is_recording ?? false,
      interviewerStreaming: data.is_recording === true && data.interviewer_streaming === true,
      mmpgAnswerLabel: mmpgAnswerLabel(data.mmpg_answer_phase),
      isPaused: data.is_paused ?? false,
      sttLoaded: data.stt_loaded ?? false,
      candidateSttLoaded: false,
      candidateSttLoading: false,
      candidateSttProvider: '',
    })
  },

  clearSession: () => {
    _chunkBuffer.clear()
    _candidateSegmentIds = []
    if (_chunkFlushTimer !== null) {
      clearTimeout(_chunkFlushTimer)
      _chunkFlushTimer = null
    }
    set({
      mmpgAnswerLabel: null,
      transcriptions: [],
      candidateTranscriptions: [],
      qaPairs: [],
      currentStreamingId: null,
      streamingIds: [],
      isPaused: false,
      candidateSttLoaded: false,
      candidateSttLoading: false,
      candidateSttProvider: '',
    })
  },
})
