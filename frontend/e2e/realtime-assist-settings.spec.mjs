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

  const unchecked = settings.getByRole('list', { name: '未勾选模型' })
  const visionCheckbox = page.getByRole('checkbox', { name: '用于答题 GPT-4.1 Mini' })
  await expect(unchecked.getByRole('listitem').filter({ has: visionCheckbox }).getByText('识图', { exact: true })).toHaveClass(/text-accent-blue/)
  await visionCheckbox.check()
  await expect(queue.getByRole('listitem').filter({ has: visionCheckbox }).getByText('识图', { exact: true })).toHaveClass(/text-accent-blue/)
  await visionCheckbox.uncheck()
  await expect(unchecked.getByRole('checkbox', { name: '用于答题 GPT-4.1 Mini' })).not.toBeChecked()

  await settings.getByRole('button', { name: '截图答题模型' }).click()
  const screenshotChoices = settings.getByRole('group', { name: '截图答题模型选项' })
  await expect(screenshotChoices.getByText('识图', { exact: true })).toHaveClass(/text-accent-blue/)
  await screenshotChoices.getByRole('radio', { name: 'GPT-4.1 Mini' }).check()
  await expect(settings.getByRole('checkbox', { name: '用于答题 GPT-4.1 Mini' })).not.toBeChecked()
  await settings.getByRole('button', { name: '保存答题模型设置' }).click()
  await expect.poll(() => updates).toEqual([{
    assist_model_ids: ['backup', 'primary'],
    assist_screenshot_model_id: 'vision',
  }])
  await expect(settings.getByRole('button', { name: '保存答题模型设置' })).toBeDisabled()
})

test('orders the exam model queue and saves its separate generation settings', async ({ context, page }) => {
  let config = {
    ...SAMPLE_CONFIG,
    models: [
      { id: 'interview-vision', name: '面试截图模型', supports_vision: true, has_api_key: true },
      { id: 'text', name: '纯文本模型', supports_vision: false, has_api_key: true },
      { id: 'exam-vision', name: '笔试视觉模型', supports_vision: true, has_api_key: true },
    ],
    assist_model_ids: ['text'],
    assist_screenshot_model_id: 'interview-vision',
    written_exam_model_ids: [],
    system_prompts: { assist_exam: '按笔试要求作答。' },
    assist_generation: {
      assist_manual: { temperature: 0.4, max_output_tokens: 1400 },
      assist_exam: { temperature: 0.3, max_output_tokens: 2048 },
    },
  }
  const scenes = [
    {
      id: 'assist_manual',
      group: '实时辅助',
      label: '手动提问',
      entry: '面试手动输入问题时使用。',
      default_prompt: '回答面试问题。',
      output_contract: '',
      max_output_tokens: 4096,
    },
    {
      id: 'assist_exam',
      group: '笔试配置',
      label: '笔试答题',
      entry: '笔试文字、粘图和截图答题时使用。',
      default_prompt: '给出可提交的答案。',
      output_contract: '',
      max_output_tokens: 4096,
    },
  ]
  const updates = []
  await installMocks(context, {
    messages: COMMON_WS_BOOTSTRAP,
    apiOverrides: (path, method, request) => {
      if (path === '/api/prompts') return { scenes }
      if (path !== '/api/config') return undefined
      if (method === 'POST') {
        const patch = request.postDataJSON()
        updates.push(patch)
        config = {
          ...config,
          ...patch,
          system_prompts: { ...config.system_prompts, ...patch.system_prompts },
          assist_generation: { ...config.assist_generation, ...patch.assist_generation },
        }
      }
      return config
    },
  })
  await page.goto('/')
  const openSettings = page.getByRole('button', { name: '打开设置', exact: true })
  await openSettings.click()
  const settings = page.getByRole('dialog', { name: /设置/ })
  const nav = settings.getByRole('navigation', { name: '设置分区' })
  const scenario = settings.getByRole('combobox', { name: '使用场景' })
  await expect(scenario.getByRole('option', { name: '手动提问' })).toHaveCount(1)
  await expect(scenario.getByRole('option', { name: '笔试答题' })).toHaveCount(0)
  await expect(settings.getByRole('heading', { name: '笔试答题模型' })).toHaveCount(0)

  const examTab = nav.getByRole('button', { name: /^笔试配置/ })
  await examTab.click()
  const unchecked = settings.getByRole('list', { name: '未勾选模型' })
  await expect(unchecked.getByRole('checkbox', { name: '用于答题 纯文本模型' })).toHaveCount(0)
  const examModelCheckbox = page.getByRole('checkbox', { name: '用于答题 笔试视觉模型' })
  const examModelOption = unchecked.getByRole('listitem').filter({ has: examModelCheckbox })
  await expect(examModelOption.getByText('识图', { exact: true })).toHaveClass(/text-accent-blue/)
  await unchecked.getByRole('checkbox', { name: '用于答题 面试截图模型' }).click()
  await unchecked.getByRole('checkbox', { name: '用于答题 笔试视觉模型' }).click()

  const examQueue = settings.getByRole('list', { name: '笔试答题模型' })
  await expect(examQueue.getByRole('listitem').filter({ has: examModelCheckbox }).getByText('识图', { exact: true })).toHaveClass(/text-accent-blue/)
  await examModelCheckbox.uncheck()
  await expect(examModelOption.getByText('识图', { exact: true })).toHaveClass(/text-accent-blue/)
  await unchecked.getByRole('checkbox', { name: '用于答题 笔试视觉模型' }).click()
  const examModelHandle = examQueue.getByRole('button', { name: '排序 笔试视觉模型' })
  await examModelHandle.press('ArrowUp')
  await expect(examQueue.getByRole('listitem').first()).toContainText('笔试视觉模型')
  await expect(examModelHandle).toBeFocused()
  await page.keyboard.press('ArrowDown')
  await expect(examQueue.getByRole('listitem').last()).toContainText('笔试视觉模型')
  await expect(examModelHandle).toBeFocused()
  await page.keyboard.press('ArrowUp')
  await expect(examQueue.getByRole('listitem').first()).toContainText('笔试视觉模型')
  await expect(nav.getByText('未保存：笔试配置')).toBeVisible()
  expect(updates).toHaveLength(0)
  await settings.getByRole('button', { name: '保存笔试模型' }).click()
  await expect(settings.getByRole('button', { name: '保存笔试模型' })).toBeDisabled()
  await expect.poll(() => updates).toEqual([{ written_exam_model_ids: ['exam-vision', 'interview-vision'] }])

  const editor = settings.getByRole('textbox', { name: '系统提示词' })
  const temperature = settings.getByRole('spinbutton', { name: '温度' })
  const tokenLimit = settings.getByRole('spinbutton', { name: '最大输出 Token' })
  await expect(editor).toHaveValue('按笔试要求作答。')
  await expect(temperature).toHaveValue('0.3')
  await expect(tokenLimit).toHaveValue('2048')
  await editor.fill('只写答案。')
  await temperature.fill('0.2')
  await tokenLimit.fill('1024')
  await settings.getByRole('button', { name: '保存当前场景' }).click()
  await expect(settings.getByText('已保存当前场景')).toBeVisible()
  expect(updates[1]).toEqual({
    system_prompts: { assist_exam: '只写答案。' },
    assist_generation: { assist_exam: { temperature: 0.2, max_output_tokens: 1024 } },
  })
  await settings.getByRole('button', { name: '关闭', exact: true }).click()
  await expect(settings).toHaveCount(0)

  await openSettings.click()
  await expect(settings.getByRole('button', { name: '截图答题模型' })).toContainText('面试截图模型')
  await expect(settings.getByRole('spinbutton', { name: '温度' })).toHaveValue('0.4')
  await examTab.click()
  await expect(examQueue.getByRole('listitem').first()).toContainText('笔试视觉模型')
  await expect(examQueue.getByRole('listitem').last()).toContainText('面试截图模型')
  await expect(editor).toHaveValue('只写答案。')
  await expect(temperature).toHaveValue('0.2')
  await expect(tokenLimit).toHaveValue('1024')
})
