import '@testing-library/jest-dom/vitest'

function createMemoryStorage(): Storage {
  const store = new Map<string, string>()
  return {
    get length() {
      return store.size
    },
    clear() {
      store.clear()
    },
    getItem(key: string) {
      const normalized = String(key)
      return store.has(normalized) ? store.get(normalized)! : null
    },
    key(index: number) {
      if (!Number.isInteger(index) || index < 0 || index >= store.size) return null
      return Array.from(store.keys())[index] ?? null
    },
    removeItem(key: string) {
      store.delete(String(key))
    },
    setItem(key: string, value: string) {
      store.set(String(key), String(value))
    },
  }
}

function isWorkingStorage(candidate: unknown): candidate is Storage {
  if (!candidate || typeof candidate !== 'object') return false
  const storage = candidate as Partial<Storage>
  if (
    typeof storage.getItem !== 'function'
    || typeof storage.setItem !== 'function'
    || typeof storage.removeItem !== 'function'
    || typeof storage.clear !== 'function'
    || typeof storage.key !== 'function'
  ) {
    return false
  }
  const probeKey = '__vitest_storage_probe__'
  try {
    storage.setItem(probeKey, '1')
    const ok = storage.getItem(probeKey) === '1'
    storage.removeItem(probeKey)
    return ok
  } catch {
    return false
  }
}

function ensurePerWorkerStorage(name: 'localStorage' | 'sessionStorage') {
  let existing: unknown
  try {
    existing = (globalThis as Record<string, unknown>)[name]
  } catch {
    existing = undefined
  }
  if (isWorkingStorage(existing)) {
    if (typeof window !== 'undefined') {
      try {
        if (isWorkingStorage(window[name])) return
      } catch {
        // Fall through to install shim on window as well.
      }
      Object.defineProperty(window, name, {
        value: existing,
        configurable: true,
        writable: true,
        enumerable: true,
      })
    }
    return
  }

  const shim = createMemoryStorage()
  Object.defineProperty(globalThis, name, {
    value: shim,
    configurable: true,
    writable: true,
    enumerable: true,
  })
  if (typeof window !== 'undefined' && window !== (globalThis as unknown)) {
    Object.defineProperty(window, name, {
      value: shim,
      configurable: true,
      writable: true,
      enumerable: true,
    })
  }
}

// Node 25+/26 未传 --localstorage-file 时会暴露缺失或失效的全局 localStorage，
// 覆盖 jsdom 提供的实现。为每个 Vitest worker 注入独立的内存 Storage shim。
ensurePerWorkerStorage('localStorage')
ensurePerWorkerStorage('sessionStorage')

// jsdom 不实现 ResizeObserver — 部分组件 (QuickPromptsRow 等) 依赖它做布局监测,
// 在测试环境下 polyfill 一个空实现,避免 ReferenceError.
if (typeof globalThis.ResizeObserver === 'undefined') {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver
}
