import type { ReactNode } from 'react'
import type { Components } from 'react-markdown'

/**
 * 仅允许显式 http/https 绝对 URL，阻断 javascript:/data:/file:/相对路径导航。
 */
export function resolveSafeMarkdownHref(href?: string): string | null {
  if (typeof href !== 'string') return null
  const trimmed = href.trim()
  if (!trimmed) return null
  try {
    const parsed = new URL(trimmed)
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return null
    if (!parsed.hostname) return null
    return parsed.href
  } catch {
    return null
  }
}

export function MarkdownLink({
  href,
  title,
  children,
}: {
  href?: string
  title?: string
  children?: ReactNode
}) {
  const safeHref = resolveSafeMarkdownHref(href)
  if (!safeHref) {
    return <span>{children}</span>
  }
  return (
    <a
      href={safeHref}
      title={title}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent-blue underline underline-offset-2 hover:brightness-110 break-all"
    >
      {children}
    </a>
  )
}

export const SAFE_MARKDOWN_COMPONENTS: Components = {
  a: MarkdownLink,
}
