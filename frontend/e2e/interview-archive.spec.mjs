import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'

import { installMocks } from './fixtures/setup.mjs'

test('复盘按需读取原声与生成历史，并可跨片段拖动', async ({ context, page }) => {
  const sid = 126
  const tid = 207
  const seed = { id: sid, title: '归档回放验收', status: 'recorded', source: 'assist',
    archive_version: 1, archive_status: 'partial', started_at: 1795000000, turn_count: 2 }
  const ids = ['a'.repeat(64), 'b'.repeat(64), 'c'.repeat(64)]
  const media = { interviewer: [{ asset_id: ids[0], offset: 0, duration: 1 }, { asset_id: ids[1], offset: 0, duration: 1 }],
    candidate: [{ asset_id: ids[2], offset: 0, duration: 1 }] }
  const turn = { id: tid, session_id: sid, seq: 1, archive_version: 1, scorecard: { clarity: 8.4 },
    question_text: '你如何设计队列？', candidate_answer_text: '我先考虑消费速度。',
    reference_answer_text: '比较不同的背压策略。', candidate_status: 'recorded', ai_status: 'completed', media,
    audio_status: { interviewer: { state: 'ready', duration_ms: 2000 },
      candidate: { state: 'approximate', duration_ms: 1000 } },
    model_summary: { name: '测试模型', first_token_ms: 23, total_ms: 320 } }
  const secondTurn = { ...turn, id: tid + 1, seq: 2, is_partial: true,
    question_text: `怎么定位这个问题？${'VeryLongQuestionWithoutSpaces'.repeat(6)}`,
    candidate_answer_text: '第二题的转写仍然保留', media: { interviewer: [], candidate: [] },
    audio_status: { candidate: { state: 'missing', duration_ms: 0, reason: 'encode_failed' } } }
  const urls = []
  let detailsRequests = 0
  await installMocks(context, {
    localStorage: { ia_app_mode: 'review' },
    apiOverrides(path) {
      if (path === '/api/review/sessions') return { items: [seed], total: 1, page: 1, page_size: 20 }
      if (path === `/api/review/sessions/${sid}`) return { ...seed, turns: [turn, secondTurn] }
      if (path.endsWith('/archive')) {
        detailsRequests++
        return { round_id: 'round', utterances: [], transcription_timeouts: ['candidate'], generations: [{ id: 'run', status: 'completed',
          model: { name: '测试模型' }, answer: '', thinking: '模型返回的推理文本', attempts: [], started_at: seed.started_at }] }
      }
    },
  })
  await context.route('**/audio/*', async (route) => {
    urls.push(route.request().url())
    await route.fulfill({ path: fileURLToPath(new URL('./fixtures/archive-tone.ogg', import.meta.url)), contentType: 'audio/ogg' })
  })
  await page.goto('/')
  await page.locator('article').filter({ hasText: seed.title }).click()
  await expect(page.getByText(turn.question_text)).toBeVisible()
  await expect(page.getByText(turn.candidate_answer_text)).toBeVisible()
  await expect(page.getByText(turn.reference_answer_text)).toBeVisible()
  await expect(page.getByText('本轮原声 · 近似定位')).toBeVisible()
  expect(urls).toHaveLength(0)
  expect(detailsRequests).toBe(0)

  const firstHeader = page.getByRole('button', { name: /^第 1 题/ })
  const secondHeader = page.getByRole('button', { name: /^第 2 题/ })
  await expect(firstHeader).toHaveAttribute('aria-expanded', 'true')
  await expect(secondHeader).toHaveAttribute('aria-expanded', 'false')
  await expect(secondHeader).toContainText('候选人录音缺失')
  const firstCard = page.locator('article').filter({ has: firstHeader })
  const secondCard = page.locator('article').filter({ has: secondHeader })
  const firstBox = await firstCard.boundingBox()
  const secondBox = await secondCard.boundingBox()
  expect(secondBox.y - (firstBox.y + firstBox.height)).toBeGreaterThanOrEqual(10)
  await expect(firstCard).toHaveCSS('border-top-width', '1px')
  await expect(firstCard).toHaveCSS('border-top-left-radius', '12px')
  const badge = await firstHeader.getByText('第 1 题').boundingBox()
  const text = await firstHeader.getByText(turn.question_text).boundingBox()
  expect(text.x).toBeGreaterThan(badge.x + badge.width)
  expect(Math.abs(text.y - badge.y)).toBeLessThan(6)
  expect((await firstHeader.getByText('8.4', { exact: true }).boundingBox()).x).toBeGreaterThan(text.x + text.width)

  await page.setViewportSize({ width: 390, height: 844 })
  await secondHeader.focus()
  await page.keyboard.press('Enter')
  await expect(secondHeader).toHaveAttribute('aria-expanded', 'true')
  await expect(firstHeader).toHaveAttribute('aria-expanded', 'true')
  await expect(page.getByText(secondTurn.candidate_answer_text)).toBeVisible()
  expect(await secondHeader.evaluate((element) => element.scrollWidth)).toBeLessThanOrEqual(
    await secondHeader.evaluate((element) => element.clientWidth),
  )
  await page.keyboard.press('Enter')
  await expect(secondHeader).toHaveAttribute('aria-expanded', 'false')
  await page.locator('html').evaluate((element) => element.setAttribute('data-theme', 'vscode-dark-plus'))
  await expect(firstHeader.getByText('第 1 题')).toHaveCSS('background-color', 'rgb(42, 42, 68)')
  await expect(firstCard).toHaveCSS('background-color', 'rgb(26, 26, 36)')

  // Seek while paused to the start of the second segment, then decode/play it.
  const slider = page.getByRole('slider', { name: '面试官播放进度' })
  await slider.fill('1.02')
  await page.getByRole('button', { name: '播放面试官录音' }).click()
  await expect.poll(() => urls.some((url) => url.endsWith(ids[1]))).toBe(true)
  await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(1.1)
  expect(urls.some((url) => url.endsWith(ids[0]))).toBe(false)
  await expect(slider).toHaveValue('2')

  await page.getByRole('button', { name: '查看对话和 AI 调用详情' }).click()
  await page.getByText('测试模型 · 已生成').click()
  await expect(page.getByText('候选人末尾转写超时')).toBeVisible()
  await expect(page.getByText('模型返回的推理文本')).toBeVisible()
  expect(detailsRequests).toBe(1)
})
