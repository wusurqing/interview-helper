import { render, screen } from '@testing-library/react'
import ReactMarkdown from 'react-markdown'
import { describe, expect, it } from 'vitest'

import AnswerMarkdownContent from './AnswerMarkdownContent'
import { MarkdownLink, SAFE_MARKDOWN_COMPONENTS, resolveSafeMarkdownHref } from './MarkdownLink'

describe('resolveSafeMarkdownHref', () => {
  it('accepts explicit http and https URLs', () => {
    expect(resolveSafeMarkdownHref('https://docs.python.org/3/')).toBe('https://docs.python.org/3/')
    expect(resolveSafeMarkdownHref('  http://example.com/a?x=1  ')).toBe('http://example.com/a?x=1')
  })

  it('rejects dangerous protocols, relative paths, and empty values', () => {
    for (const bad of [
      undefined,
      '',
      '   ',
      'javascript:alert(1)',
      '  JaVaScRiPt:alert(1)',
      'data:text/html,<script>alert(1)</script>',
      'file:///C:/Windows/win.ini',
      'vbscript:msgbox(1)',
      '/api/assist/stop',
      '//evil.example/phish',
      '#local',
    ]) {
      expect(resolveSafeMarkdownHref(bad)).toBeNull()
    }
  })
})

describe('MarkdownLink & SAFE_MARKDOWN_COMPONENTS', () => {
  it('renders safe external links with target="_blank" and rel="noopener noreferrer"', () => {
    render(
      <ReactMarkdown components={SAFE_MARKDOWN_COMPONENTS}>
        {'参考 [Python 文档](https://docs.python.org/3/ "官方文档")'}
      </ReactMarkdown>,
    )
    const link = screen.getByRole('link', { name: 'Python 文档' })
    expect(link).toHaveAttribute('href', 'https://docs.python.org/3/')
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    expect(link).toHaveAttribute('title', '官方文档')
  })

  it('downgrades unsafe or relative links to inert spans', () => {
    const { container } = render(
      <div>
        <MarkdownLink href="javascript:alert(1)">恶意脚本</MarkdownLink>
        <MarkdownLink href="file:///etc/passwd">本地文件</MarkdownLink>
        <MarkdownLink href="/api/assist/stop">站内路径</MarkdownLink>
      </div>,
    )
    expect(container.querySelectorAll('a')).toHaveLength(0)
    expect(screen.getByText('恶意脚本')).toBeInTheDocument()
    expect(screen.getByText('本地文件')).toBeInTheDocument()
    expect(screen.getByText('站内路径')).toBeInTheDocument()
  })

  it('protects links inside AnswerMarkdownContent', () => {
    const { container } = render(
      <AnswerMarkdownContent
        answer={'看 [安全文档](https://example.com/guide) 和 [本地](/local/path) 和 [文件](file:///a.txt)'}
        colorScheme="vscode-dark-plus"
        stream={false}
      />,
    )
    const links = container.querySelectorAll('a')
    expect(links).toHaveLength(1)
    expect(links[0]).toHaveAttribute('href', 'https://example.com/guide')
    expect(links[0]).toHaveAttribute('target', '_blank')
    expect(links[0]).toHaveAttribute('rel', 'noopener noreferrer')
    expect(screen.getByText('本地')).toBeInTheDocument()
    expect(screen.getByText('文件')).toBeInTheDocument()
  })
})
