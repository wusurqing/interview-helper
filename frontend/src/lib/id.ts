type CryptoLike = {
  randomUUID?: () => string
  getRandomValues?: (array: Uint8Array) => Uint8Array
}

function formatUuidV4(bytes: Uint8Array): string {
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

/**
 * 生成客户端 UUID v4。
 *
 * `crypto.randomUUID` 仅在 Secure Context（HTTPS / localhost）暴露；局域网 `http://<LAN-IP>`
 * 访问时为 undefined。这里依次退化为 `crypto.getRandomValues`（非安全上下文同样可用）
 * 与 `Math.random` + 时间戳，保证任何环境都不会抛错。
 */
export function createClientId(
  cryptoImpl: CryptoLike | null | undefined = globalThis.crypto as CryptoLike | undefined,
): string {
  if (cryptoImpl && typeof cryptoImpl.randomUUID === 'function') {
    try {
      return cryptoImpl.randomUUID()
    } catch {
      // 继续退化
    }
  }
  const bytes = new Uint8Array(16)
  if (cryptoImpl && typeof cryptoImpl.getRandomValues === 'function') {
    try {
      cryptoImpl.getRandomValues(bytes)
      return formatUuidV4(bytes)
    } catch {
      // 继续退化
    }
  }
  let seed = Date.now()
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = (Math.floor(Math.random() * 256) ^ (seed & 0xff)) & 0xff
    seed = Math.floor(seed / 256) || Date.now()
  }
  return formatUuidV4(bytes)
}
