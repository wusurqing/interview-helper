import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiRequestError, api, buildHeaders, getErrorMessage, request } from './api'

function jsonResponse(body: unknown, init: ResponseInit) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  })
}

describe('api error handling', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('formats FastAPI validation detail arrays into readable field messages', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse(
        {
          detail: [
            {
              loc: ['body', 'screen_capture_region'],
              msg: 'Input should be one of full,left_half',
            },
          ],
        },
        { status: 422, statusText: 'Unprocessable Entity' },
      ),
    ))

    await expect(api.getConfig()).rejects.toThrow(
      '请求失败 (422): body.screen_capture_region: Input should be one of full,left_half',
    )
  })

  it('keeps non-json server failures actionable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('upstream gateway timeout', { status: 504, statusText: 'Gateway Timeout' }),
    ))

    await expect(api.getConfig()).rejects.toThrow(
      '请求失败 (504): upstream gateway timeout',
    )
  })

  it('normalizes network failures to a product-facing backend message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

    await expect(api.getConfig()).rejects.toThrow(
      '无法连接后端服务，请确认应用服务已启动后重试',
    )
  })

  it('normalizes upload network failures to the same backend message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

    await expect(api.uploadResume(new File(['demo'], 'resume.pdf'))).rejects.toThrow(
      '无法连接后端服务，请确认应用服务已启动后重试',
    )
    await expect(api.kbUpload(new File(['demo'], 'notes.md'))).rejects.toThrow(
      '无法连接后端服务，请确认应用服务已启动后重试',
    )
  })

  it('extracts safe messages from unknown caught values', () => {
    expect(getErrorMessage(new Error('boom'))).toBe('boom')
    expect(getErrorMessage({ detail: 'backend says no' })).toBe('backend says no')
    expect(getErrorMessage(null, '兜底失败')).toBe('兜底失败')
  })

  it('merges Headers instances, tuple arrays, and plain records in buildHeaders', () => {
    const fromHeaders = buildHeaders(new Headers({ 'X-Custom': '1', 'content-type': 'text/plain' }))
    expect(fromHeaders.get('x-custom')).toBe('1')
    expect(fromHeaders.get('content-type')).toBe('text/plain')

    const fromTuples = buildHeaders([
      ['X-Tuple', 'yes'],
      ['Accept', 'application/json'],
    ])
    expect(fromTuples.get('x-tuple')).toBe('yes')
    expect(fromTuples.get('accept')).toBe('application/json')
    expect(fromTuples.get('content-type')).toBe('application/json')

    const fromRecord = buildHeaders({ 'X-Obj': 'ok' })
    expect(fromRecord.get('x-obj')).toBe('ok')
  })

  it('aborts hung requests when timeoutMs elapses', async () => {
    vi.useFakeTimers()
    try {
      vi.stubGlobal('fetch', vi.fn((_input: string, init?: RequestInit) => (
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'))
          })
        })
      )))

      const pending = request('/api/slow', { timeoutMs: 250 })
      const expectation = expect(pending).rejects.toThrow('请求超时，请检查后端服务或网络后重试')
      await vi.advanceTimersByTimeAsync(300)
      await expectation
    } finally {
      vi.useRealTimers()
    }
  })

  it('preserves caller AbortSignal cancellation without masking it as timeout', async () => {
    const caller = new AbortController()
    vi.stubGlobal('fetch', vi.fn((_input: string, init?: RequestInit) => (
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          reject(new DOMException('Caller aborted', 'AbortError'))
        })
      })
    )))

    const pending = request('/api/cancel-me', { signal: caller.signal, timeoutMs: 30_000 })
    caller.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })

  it.each(['timeout', 'caller'])('honors %s cancellation while reading a response body', async (source) => {
    vi.useFakeTimers()
    try {
      const caller = new AbortController()
      vi.stubGlobal('fetch', async (_input: string, init: RequestInit) => new Response(
        new ReadableStream({
          start(controller) {
            const delivery = setTimeout(() => {
              controller.enqueue(new TextEncoder().encode('"late response"'))
              controller.close()
            }, 1000)
            init.signal!.addEventListener('abort', () => {
              clearTimeout(delivery)
              controller.error(init.signal!.reason)
            }, { once: true })
          },
        }),
      ))

      const pending = request('/api/slow-body', { timeoutMs: 250, signal: caller.signal })
      const outcome = pending.catch((error: unknown) => error)
      await vi.advanceTimersByTimeAsync(10)
      if (source === 'caller') caller.abort()
      await vi.advanceTimersByTimeAsync(1000)
      expect(await outcome).toMatchObject(source === 'caller'
        ? { name: 'AbortError' }
        : { message: '请求超时，请检查后端服务或网络后重试' })
    } finally {
      vi.useRealTimers()
    }
  })

  it('fetches review audio buffers and surfaces formatted ApiRequestError on non-ok responses', async () => {
    const payload = new Uint8Array([79, 103, 103, 83]).buffer
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(payload, { status: 200, headers: { 'Content-Type': 'audio/ogg' } }),
    ).mockResolvedValueOnce(
      jsonResponse({ detail: '录音片段不存在' }, { status: 404, statusText: 'Not Found' }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const bytes = await api.reviewAudio(7, 9, 'abc123')
    expect(new Uint8Array(bytes)).toEqual(new Uint8Array([79, 103, 103, 83]))
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/api/review/sessions/7/turns/9/audio/abc123'),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )

    const error = await api.reviewAudio(7, 9, 'missing').catch((cause: unknown) => cause)
    expect(error).toBeInstanceOf(ApiRequestError)
    expect(error).toMatchObject({ status: 404, message: '请求失败 (404): 录音片段不存在' })
  })

  it.each(['timeout', 'caller'])('honors %s cancellation while reading reviewAudio arrayBuffer', async (source) => {
    vi.useFakeTimers()
    try {
      const caller = new AbortController()
      vi.stubGlobal('fetch', vi.fn(async (_input: string, init?: RequestInit) => new Response(
        new ReadableStream({
          start(controller) {
            const delivery = setTimeout(() => {
              controller.enqueue(new Uint8Array([1, 2, 3, 4]))
              controller.close()
            }, 60_000)
            init?.signal?.addEventListener('abort', () => {
              clearTimeout(delivery)
              controller.error(init.signal!.reason)
            }, { once: true })
          },
        }),
      )))

      const pending = api.reviewAudio(1, 2, 'slow-asset', caller.signal)
      const outcome = pending.catch((error: unknown) => error)
      await vi.advanceTimersByTimeAsync(10)
      if (source === 'caller') {
        caller.abort()
      } else {
        await vi.advanceTimersByTimeAsync(35_000)
      }
      await vi.advanceTimersByTimeAsync(10)
      expect(await outcome).toMatchObject(source === 'caller'
        ? { name: 'AbortError' }
        : { message: '请求超时，请检查后端服务或网络后重试' })
    } finally {
      vi.useRealTimers()
    }
  })
})
