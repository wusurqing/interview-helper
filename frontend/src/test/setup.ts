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

function installPerWorkerStorage(name: 'localStorage' | 'sessionStorage') {
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

// Node 25+/26 未传 --localstorage-file 时会暴露失效的全局 localStorage，传入时又会在
// 多 worker 间争抢同一 backing file 且与 jsdom window.localStorage 分裂。
// 统一为每个 Vitest worker 注入独立的内存 Storage shim，保证 globalThis 与 window 一致且跨 worker 隔离。
installPerWorkerStorage('localStorage')
installPerWorkerStorage('sessionStorage')

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
