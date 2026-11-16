import { expect, test } from '@playwright/test'
import { resolveApiPayload } from './fixtures/sample-data.mjs'
import { installMocks, COMMON_WS_BOOTSTRAP } from './fixtures/setup.mjs'

const longToken = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.repeat(10)
const longMarkdown = [
  `面试回答的重点是分清条件并逐项说明。${longToken}`,
  `- 示例\n  - <https://example.com/${longToken}>`,
  '> > > > 引用中的重点',
  `\`\`\`text\n${longToken}\n\`\`\``,
].join('\n\n')

async function openPrompt(page, sizeMode, viewport, cap, autoFollow = false, mode = 'prompt') {
  await page.setViewportSize(viewport)
  await page.route('**/api/**', (route) => {
    const request = route.request()
    return route.fulfill({ json: resolveApiPayload(new URL(request.url()).pathname, request.method()) ?? {} })
  })
  await page.route('https://fonts.googleapis.com/**', (route) => route.abort())
  let socket
  await page.routeWebSocket('**/ws', (connection) => { socket = connection })
  await page.addInitScript(({ sizeMode, cap, autoFollow, mode }) => {
    localStorage.setItem('ia_overlay_enabled', '1')
    localStorage.setItem('ia_overlay_mode', mode)
    localStorage.setItem('ia_overlay_appearance', JSON.stringify({
      [mode]: { fontSize: 48, fontFamily: 'mono', textAlign: 'right', autoFollow, fontColor: '#e3d7ff', backgroundColor: '#111111' },
    }))
    window.overlayMeasurements = []
    window.promptScrollable = []
    window.electronAPI = {
      reportPromptScrollable: async (value) => { window.promptScrollable.push(value); return { ok: true } },
      onPromptWheel: (callback) => { window.promptWheel = callback; return () => { window.promptWheel = undefined } },
      getOverlayState: async () => ({ enabled: true, visible: true, mode }),
      getOverlayLayout: async () => ({ promptViewport: { ...cap, sizeMode } }),
      resizeOverlayWindow: async (dimensions) => {
        window.overlayMeasurements.push(dimensions)
        return { ok: true, ...dimensions }
      },
    }
  }, { sizeMode, cap, autoFollow, mode })
  await page.goto('/?overlay=1')
  await expect.poll(() => Boolean(socket)).toBe(true)
  return (message) => socket.send(JSON.stringify(message))
}

for (const mode of ['glass', 'prompt', 'focus']) {
  test(`${mode} Markdown inherits the configured answer color`, async ({ page }) => {
    const send = await openPrompt(page, 'fixed', { width: 800, height: 600 }, { width: 800, height: 600 }, false, mode)
    send({ type: 'answer_start', id: mode, question: '请回答' })
    send({ type: 'answer_done', id: mode, question: '请回答', answer: '# 标题\n\n正文 **重点**' })
    const markdown = page.locator('.ov-markdown')
    await expect(markdown.locator('strong')).toHaveCSS('color', 'rgb(227, 215, 255)')
    await expect(markdown.locator('h1')).toHaveCSS('color', 'rgb(227, 215, 255)')
    if (mode === 'focus') await expect(page.locator('.ov-focus-section > h2')).toHaveCSS('color', 'rgb(227, 215, 255)')
  })
}

test('settings previews use the configured color and main answer keeps its theme', async ({ context, page }) => {
  const appearance = Object.fromEntries(['glass', 'prompt', 'focus'].map((mode) => [mode, { fontColor: '#e3d7ff' }]))
  await installMocks(context, {
    localStorage: { ia_overlay_appearance: JSON.stringify(appearance), 'ia-color-scheme': 'vscode-light-plus' },
    messages: [...COMMON_WS_BOOTSTRAP, { type: 'init', delay: 50, transcriptions: [], qa_pairs: [{ id: 'theme', question: '主题测试', answer: '正文 **重点**', timestamp: Date.now() / 1000 }], is_recording: false, stt_loaded: true }],
  })
  await page.goto('/')
  const strong = page.locator('.markdown-body strong').first()
  await expect(strong).toHaveText('重点')
  const themeColor = await strong.evaluate((element) => {
    const probe = document.createElement('span')
    probe.style.color = 'rgb(var(--c-md-strong))'
    element.appendChild(probe)
    const color = getComputedStyle(probe).color
    probe.remove()
    return color
  })
  await expect(strong).toHaveCSS('color', themeColor)
  await page.getByRole('button', { name: '打开设置', exact: true }).click()
  await page.getByRole('navigation', { name: '设置分区' }).getByRole('button', { name: /悬浮窗设置/ }).click()
  for (const label of ['磨砂玻璃', '提词模式', '专注面板']) {
    await page.getByRole('group', { name: '显示模式' }).getByRole('button', { name: label }).click()
    await expect(page.locator('.ov-settings-preview .ov-answer, .ov-settings-preview .ov-focus-content')).toHaveCSS('color', 'rgb(227, 215, 255)')
  }
})

test('detects delayed image overflow after standby content is replaced', async ({ page }) => {
  const releases = []
  await page.route('**/delayed-image.svg*', async (route) => {
    await new Promise((resolve) => { releases.push(resolve) })
    await route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="900"></svg>' })
  })
  const send = await openPrompt(page, 'fixed', { width: 220, height: 180 }, { width: 220, height: 180 })
  send({ type: 'answer_start', id: 'image', question: '请回答' })
  send({ type: 'answer_done', id: 'image', question: '请回答', answer: '简答\n\n![图片](/delayed-image.svg)' })
  try {
    await expect(page.locator('.ov-markdown img')).toHaveCount(1)
    await expect.poll(() => releases.length).toBe(1)
    await expect.poll(() => page.evaluate(() => window.promptScrollable.at(-1))).toBe(false)
  } finally {
    releases.shift()?.()
  }
  await expect.poll(() => page.evaluate(() => window.promptScrollable.at(-1))).toBe(true)
  const answer = page.locator('.ov-answer')
  await page.evaluate(() => window.promptWheel(-120))
  await expect.poll(() => answer.evaluate((element) => element.scrollTop)).toBeGreaterThan(0)
  send({ type: 'session_cleared' })
  await expect.poll(() => page.evaluate(() => window.promptScrollable.at(-1))).toBe(false)
  send({ type: 'answer_start', id: 'image-again', question: '请回答' })
  send({ type: 'answer_done', id: 'image-again', question: '请回答', answer: '简答\n\n![图片](/delayed-image.svg?again)' })
  try {
    await expect.poll(() => releases.length).toBe(1)
    await expect.poll(() => page.evaluate(() => window.promptScrollable.at(-1))).toBe(false)
  } finally {
    releases.shift()?.()
  }
  await expect.poll(() => page.evaluate(() => window.promptScrollable.at(-1))).toBe(true)
})

async function expectTextFits(page, selector, text = longToken) {
  const body = page.locator(selector).first()
  await expect(body).toContainText(text)
  const bounds = await body.evaluate((element) => {
    const answer = element.closest('.ov-answer')
    const rect = answer.getBoundingClientRect()
    const left = rect.left + answer.clientLeft
    const right = left + answer.clientWidth
    const viewport = document.documentElement.clientWidth
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
    const range = document.createRange()
    let overflow = Math.max(rect.right - viewport, answer.scrollWidth - answer.clientWidth)
    while (walker.nextNode()) {
      const node = walker.currentNode
      for (const match of node.textContent.matchAll(/\S/gu)) {
        range.setStart(node, match.index)
        range.setEnd(node, match.index + match[0].length)
        for (const glyph of range.getClientRects()) {
          overflow = Math.max(overflow, left - glyph.left, glyph.right - right)
        }
      }
    }
    return overflow
  })
  expect(bounds).toBeLessThanOrEqual(1)
}

test('fits right-aligned plain text in the smallest fixed prompt', async ({ page }) => {
  const send = await openPrompt(page, 'fixed', { width: 180, height: 180 }, { width: 180, height: 180 })
  send({ type: 'answer_start', id: 'plain', question: '请回答', answer_format: 'plain_text' })
  send({ type: 'answer_chunk', id: 'plain', chunk: longToken })
  await expectTextFits(page, '[data-answer-format="plain_text"]')
  const reason = '第二张截图里的样例不通过'
  send({ type: 'vision_verify', id: 'plain', verdict: 'FAIL', reason })
  await expectTextFits(page, '.ov-verify', reason)
})

test('fits right-aligned Markdown and code at the auto width limit', async ({ page }) => {
  const send = await openPrompt(page, 'auto', { width: 220, height: 190 }, { width: 220, height: 190 })
  send({ type: 'answer_start', id: 'markdown', question: '请回答' })
  send({ type: 'answer_chunk', id: 'markdown', chunk: longMarkdown })
  await expectTextFits(page, '.ov-markdown')
})

test('fits a short Markdown block without scrolling before the height limit', async ({ page }) => {
  const cap = { width: 820, height: 800 }
  const send = await openPrompt(page, 'auto', { width: 180, height: 72 }, cap)
  send({ type: 'answer_start', id: 'short', question: '请回答' })
  send({ type: 'answer_done', id: 'short', question: '请回答', answer: '> 简短回答' })
  await expect(page.locator('.ov-markdown blockquote')).toHaveText('简短回答')
  const size = await page.evaluate(() => window.overlayMeasurements.at(-1))
  expect(size.height).toBeLessThan(cap.height)
  await page.setViewportSize(size)
  const overflow = await page.locator('.ov-answer').evaluate((element) => element.scrollHeight - element.clientHeight)
  expect(overflow).toBeLessThanOrEqual(1)
})

test('fits growing content before resize and preserves manual scrolling at the height limit', async ({ page }) => {
  const cap = { width: 820, height: 270 }
  const send = await openPrompt(page, 'auto', { width: 180, height: 104 }, cap, true)
  send({ type: 'answer_start', id: 'stream', question: '请回答' })
  send({ type: 'answer_chunk', id: 'stream', chunk: '简答' })
  await expect(page.locator('.ov-markdown')).toHaveText('简答')
  await page.evaluate(() => { window.overlayMeasurements.length = 0 })
  send({ type: 'answer_chunk', id: 'stream', chunk: `\n\n${longMarkdown}` })
  await expectTextFits(page, '.ov-markdown')
  await expect.poll(() => page.evaluate(() => window.overlayMeasurements.at(-1))).toEqual(cap)
  await page.setViewportSize(cap)
  await expectTextFits(page, '.ov-markdown')

  const answer = page.locator('.ov-answer')
  await answer.evaluate((element) => { element.scrollTop = 120; element.dispatchEvent(new Event('scroll')) })
  send({ type: 'answer_chunk', id: 'stream', chunk: '\n\n最后一行' })
  await expect(page.locator('.ov-markdown')).toContainText('最后一行')
  expect(await answer.evaluate((element) => element.scrollTop)).toBe(120)
  await answer.evaluate((element) => {
    element.scrollTop = element.scrollHeight
    element.dispatchEvent(new Event('scroll'))
  })
  send({ type: 'answer_chunk', id: 'stream', chunk: '\n\n继续跟随' })
  await expect(page.locator('.ov-markdown')).toContainText('继续跟随')
  await expect.poll(() => answer.evaluate((element) => element.scrollHeight - element.clientHeight - element.scrollTop)).toBeLessThanOrEqual(1)
  const lastLine = page.getByText('继续跟随', { exact: true })
  expect(await lastLine.evaluate((element) => {
    const answer = element.closest('.ov-answer')
    return element.getBoundingClientRect().bottom <= answer.getBoundingClientRect().top + answer.clientHeight + 1
  })).toBe(true)
  await page.setViewportSize({ width: cap.width, height: 72 })
  await expect.poll(() => answer.evaluate((element) => element.scrollHeight - element.clientHeight - element.scrollTop)).toBeLessThanOrEqual(1)
  expect(await lastLine.evaluate((element) => {
    const answer = element.closest('.ov-answer').getBoundingClientRect()
    const range = document.createRange()
    range.selectNodeContents(element)
    return Array.from(range.getClientRects()).some((line) => line.bottom > answer.top && line.top < answer.bottom)
  })).toBe(true)
})
