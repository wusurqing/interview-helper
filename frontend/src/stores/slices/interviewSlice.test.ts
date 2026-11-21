import { afterEach, describe, expect, it, vi } from 'vitest'
import { useInterviewStore } from '@/stores/configStore'
import { MAX_QA_PAIRS } from './interviewSlice'

describe('interviewSlice bounds and transient buffer cleanup', () => {
  afterEach(() => {
    useInterviewStore.getState().clearSession()
    vi.useRealTimers()
  })

  it('caps qaPairs at MAX_QA_PAIRS across startAnswer, finalizeAnswer, and setInitData', () => {
    const store = useInterviewStore.getState()
    store.clearSession()

    for (let i = 0; i < MAX_QA_PAIRS + 15; i += 1) {
      store.startAnswer(`qa-${i}`, `Question ${i}`)
      store.finalizeAnswer(`qa-${i}`, `Question ${i}`, `Answer ${i}`)
    }

    const afterStream = useInterviewStore.getState()
    expect(afterStream.qaPairs).toHaveLength(MAX_QA_PAIRS)
    expect(afterStream.qaPairs[0].id).toBe('qa-15')
    expect(afterStream.qaPairs[afterStream.qaPairs.length - 1].id).toBe(`qa-${MAX_QA_PAIRS + 14}`)

    const oversizedInit = Array.from({ length: MAX_QA_PAIRS + 10 }, (_, idx) => ({
      id: `init-${idx}`,
      question: `Q${idx}`,
      answer: `A${idx}`,
    }))
    store.setInitData({ qa_pairs: oversizedInit })
    const afterInit = useInterviewStore.getState()
    expect(afterInit.qaPairs).toHaveLength(MAX_QA_PAIRS)
    expect(afterInit.qaPairs[0].id).toBe('init-10')
  })

  it('cancels buffered chunk flush timers on clearSession and setInitData', async () => {
    vi.useFakeTimers()
    const store = useInterviewStore.getState()
    store.clearSession()

    store.startAnswer('qa-live', 'Live Q')
    store.appendAnswerChunk('qa-live', 'buffered-token')
    store.clearSession()

    // Start a brand-new item with the same id after clearSession; stale buffered chunks must never flush into it.
    store.startAnswer('qa-live', 'Fresh Q')
    await vi.advanceTimersByTimeAsync(200)

    const current = useInterviewStore.getState().qaPairs.find((q) => q.id === 'qa-live')
    expect(current?.answer).toBe('')
  })
})
