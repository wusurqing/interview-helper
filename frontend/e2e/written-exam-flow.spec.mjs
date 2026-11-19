import { expect, test } from '@playwright/test'

import { COMMON_WS_BOOTSTRAP, installMocks } from './fixtures/setup.mjs'
import { SAMPLE_WRITTEN_EXAM_CONFIG } from './fixtures/sample-data.mjs'

const WRITTEN_EXAM_FLOW_MESSAGES = [
  ...COMMON_WS_BOOTSTRAP,
  {
    type: 'recording',
    value: true,
    delay: 4000,
  },
  {
    type: 'answer_start',
    id: 'exam-code',
    question: '代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。',
    source: 'manual_text',
    model_name: 'GPT-4.1 Mini',
    delay: 4300,
  },
  {
    type: 'answer_chunk',
    id: 'exam-code',
    chunk: '```python\n# 哈希表一次遍历 O(n)\ndef two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        j = seen.get(target - x)\n        if j is not None:\n            return [j, i]\n        seen[x] = i\n    return []\n```',
    delay: 4600,
  },
  {
    type: 'answer_done',
    id: 'exam-code',
    question: '代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。',
    answer: '```python\n# 哈希表一次遍历 O(n)\ndef two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        j = seen.get(target - x)\n        if j is not None:\n            return [j, i]\n        seen[x] = i\n    return []\n```',
    think: '',
    model_name: 'GPT-4.1 Mini',
    first_token_ms: 218,
    total_ms: 734,
    delay: 4800,
  },
]

test.describe('written exam flow', () => {
  test('opens the written exam workspace and answers without audio capture', async ({ context, page }) => {
    const requests = []
    let config = {
      ...SAMPLE_WRITTEN_EXAM_CONFIG,
      written_exam_mode: false,
      candidate_asr_enabled: true,
      candidate_stt_provider: 'doubao_mmpg',
    }
    const recordRequest = (pathname, method, request) => {
      const rawBody = request.postData()
      let body = rawBody
      try {
        body = rawBody ? JSON.parse(rawBody) : null
      } catch {
        /* keep raw body */
      }
      requests.push({ pathname, method, body })
    }
    await installMocks(context, {
      messages: WRITTEN_EXAM_FLOW_MESSAGES,
      localStorage: {
        'ia-color-scheme': 'vscode-light-plus',
      },
      apiOverrides: (pathname, method, request) => {
        if (pathname === '/api/config') {
          if (method === 'POST') {
            recordRequest(pathname, method, request)
            config = { ...config, ...request.postDataJSON() }
          }
          return config
        }
        if (pathname === '/api/start') {
          recordRequest(pathname, method, request)
          return { ok: true }
        }
        if (pathname === '/api/ask' || pathname === '/api/ask-from-server-screen' || pathname.startsWith('/api/mmpg/')) {
          recordRequest(pathname, method, request)
          return { ok: true }
        }
        if (pathname === '/api/exam-preflight/run') {
          recordRequest(pathname, method, request)
          return { ok: true }
        }
        if (pathname === '/api/exam-preflight/status') {
          return {
            running: false,
            question: '代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。',
            steps: {},
          }
        }
        return undefined
      },
    })

    await page.goto('/')

    await expect(page.getByRole('tab', { name: '面试辅助' })).toHaveAttribute('aria-selected', 'true')
    const examTab = page.getByRole('tab', { name: '笔试辅助' })
    await examTab.click()
    await expect(examTab).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByLabel('转写记录')).toHaveCount(0)
    await expect(page.getByLabel('选择会议音频设备')).toHaveCount(0)
    await expect(page.getByText('AI 笔试助手')).toBeVisible({ timeout: 5000 })
    await expect(page.getByText('点击「开始笔试」进入答题模式，可通过截图或手动输入提问')).toBeVisible()
    await expect(page.getByText('固定截图代码题').first()).toBeVisible()
    await page.getByRole('button', { name: '开始检测' }).dispatchEvent('click')

    await page.getByRole('button', { name: /开始笔试/ }).dispatchEvent('click')
    await expect(page.getByText('EXAM')).toBeVisible({ timeout: 5000 })
    await page.getByRole('button', { name: '截图答题', exact: true }).click()

    const input = page.getByPlaceholder('输入问题，Enter 发送…')
    await input.fill('代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。')
    await page.getByRole('button', { name: '发送问题' }).click()

    await expect(page.getByText('代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。')).toBeVisible({
      timeout: 5000,
    })
    await expect(page.getByText('哈希表一次遍历 O(n)')).toBeVisible()
    await expect(page.getByText('def two_sum(nums, target):')).toBeVisible()

    expect(requests).toEqual([
      { pathname: '/api/config', method: 'POST', body: { written_exam_mode: true } },
      { pathname: '/api/exam-preflight/run', method: 'POST', body: {} },
      { pathname: '/api/start', method: 'POST', body: {} },
      { pathname: '/api/ask-from-server-screen', method: 'POST', body: {} },
      {
        pathname: '/api/ask',
        method: 'POST',
        body: {
          text: '代码题：给定整数数组 nums 和目标值 target，返回两数之和的下标。',
        },
      },
    ])
  })
})
