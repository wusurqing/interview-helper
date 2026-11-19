import { expect, test } from '@playwright/test'
import { COMMON_WS_BOOTSTRAP, installMocks } from './fixtures/setup.mjs'
import { SAMPLE_CONFIG } from './fixtures/sample-data.mjs'

test('selects and drags answer models while keeping the screenshot model separate', async ({ context, page }) => {
  const models = SAMPLE_CONFIG.models.map((model, index) => ({
    ...model,
    id: ['vision', 'backup', 'primary'][index],
    has_api_key: true,
  }))
  let config = {
    ...SAMPLE_CONFIG,
    models,
    assist_model_ids: ['primary'],
    assist_screenshot_model_id: null,
    stt_provider: 'doubao',
    candidate_asr_enabled: false,
  }
  const updates = []
  await installMocks(context, {
    messages: COMMON_WS_BOOTSTRAP,
    apiOverrides: (path, method, request) => {
      if (path !== '/api/config') return undefined
      if (method === 'POST') {
        const patch = request.postDataJSON()
        updates.push(patch)
        config = { ...config, ...patch }
      }
      return config
    },
  })
  await page.goto('/')
  await page.getByRole('button', { name: '打开设置', exact: true }).click()
  const settings = page.getByRole('dialog', { name: /设置/ })
  await expect(settings.getByRole('button', { name: /^实时辅助 / })).toHaveAttribute('aria-current', 'page')
  await settings.getByRole('checkbox', { name: '用于答题 DeepSeek V3' }).check()

  const queue = settings.getByRole('list', { name: '答题可用模型' })
  await expect(queue.getByRole('listitem')).toHaveCount(2)
  await expect(queue.getByRole('listitem').first()).toHaveText('Qwen 2.5 72B')
  const backupHandle = queue.getByRole('button', { name: '排序 DeepSeek V3' })
  const primaryHandle = queue.getByRole('button', { name: '排序 Qwen 2.5 72B' })
  await backupHandle.scrollIntoViewIfNeeded()
  const backup = await backupHandle.boundingBox()
  const primary = await primaryHandle.boundingBox()
  if (!backup || !primary) throw new Error('The model drag handles are not visible')
  const startX = backup.x + backup.width / 2
  const startY = backup.y + backup.height / 2
  await page.mouse.move(startX, startY)
  await page.mouse.down()
  await page.mouse.move(startX, primary.y + primary.height / 2, { steps: 15 })
  await page.mouse.up()
  await expect(queue.getByRole('listitem').first()).toHaveText('DeepSeek V3')

  await settings.getByRole('combobox', { name: '截图答题模型' }).selectOption('vision')
  await expect(settings.getByRole('checkbox', { name: '用于答题 GPT-4.1 Mini' })).not.toBeChecked()
  await settings.getByRole('button', { name: '保存答题模型设置' }).click()
  await expect.poll(() => updates).toEqual([{
    assist_model_ids: ['backup', 'primary'],
    assist_screenshot_model_id: 'vision',
  }])
  await expect(settings.getByRole('button', { name: '保存答题模型设置' })).toBeDisabled()
})
