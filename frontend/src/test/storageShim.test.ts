import { describe, expect, it } from 'vitest'

describe('test storage shim', () => {
  it('provides working Storage semantics on both globalThis and window', () => {
    localStorage.clear()
    expect(localStorage).toBe(window.localStorage)
    expect(localStorage.length).toBe(0)
    expect(localStorage.getItem('missing')).toBeNull()

    localStorage.setItem('alpha', '1')
    localStorage.setItem('beta', String(2))
    expect(localStorage.length).toBe(2)
    expect(localStorage.getItem('alpha')).toBe('1')
    expect(localStorage.key(0)).toBe('alpha')
    expect(localStorage.key(1)).toBe('beta')
    expect(localStorage.key(2)).toBeNull()
    expect(localStorage.key(-1)).toBeNull()

    localStorage.removeItem('alpha')
    expect(localStorage.getItem('alpha')).toBeNull()
    expect(localStorage.length).toBe(1)

    localStorage.clear()
    expect(localStorage.length).toBe(0)
  })
})
