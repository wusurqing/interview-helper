import { describe, expect, it, vi } from 'vitest'

import { createClientId } from './id'

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

describe('createClientId', () => {
  it('uses crypto.randomUUID when available', () => {
    const randomUUID = vi.fn(() => '11111111-1111-4111-8111-111111111111')
    expect(createClientId({ randomUUID })).toBe('11111111-1111-4111-8111-111111111111')
    expect(randomUUID).toHaveBeenCalledTimes(1)
  })

  it('falls back to getRandomValues in insecure contexts without randomUUID', () => {
    const getRandomValues = vi.fn((bytes: Uint8Array) => {
      for (let i = 0; i < bytes.length; i += 1) bytes[i] = 0xff
      return bytes
    })
    const id = createClientId({ getRandomValues })
    expect(getRandomValues).toHaveBeenCalledTimes(1)
    expect(id).toMatch(UUID_V4)
  })

  it('falls back to getRandomValues when randomUUID throws', () => {
    const id = createClientId({
      randomUUID: () => {
        throw new Error('not allowed')
      },
      getRandomValues: (array: Uint8Array) => array,
    })
    expect(id).toMatch(UUID_V4)
  })

  it('still produces unique v4 ids without any crypto implementation', () => {
    const ids = new Set(Array.from({ length: 200 }, () => createClientId(null)))
    expect(ids.size).toBe(200)
    for (const id of ids) expect(id).toMatch(UUID_V4)
  })
})
