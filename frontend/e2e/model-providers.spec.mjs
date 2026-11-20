import { expect, test } from '@playwright/test'
import { COMMON_WS_BOOTSTRAP, installMocks } from './fixtures/setup.mjs'
import { SAMPLE_CONFIG } from './fixtures/sample-data.mjs'

test('verifies supplier models and confirms removing an existing assignment', async ({ context, page }) => {
  const supplier = {
    id: 'ark', name: '火山方舟', api_type: 'openai',
    api_base_url: 'https://ark.example/api/v3', has_api_key: true,
  }
  let supplierSettings = {
    revision: 'rev-1',
    providers: [supplier],
    models: [{
      id: 'old-vision', provider_id: supplier.id, name: '已分配模型', model: 'old',
      enabled: true, verified: true, supports_vision: true, supports_think: false,
      context_window_tokens: 131072, think_enabled_params: {}, think_disabled_params: {},
    }],
    active_model_id: null,
    usages: { 'old-vision': ['实时辅助', '截图答题', '笔试答题'] },
  }
  let config = {
    ...SAMPLE_CONFIG,
    models: supplierSettings.models.map((model) => ({ ...model, provider_name: supplier.name, has_api_key: true })),
    assist_model_ids: ['old-vision'],
    assist_screenshot_model_id: 'old-vision',
    written_exam_model_ids: ['old-vision'],
  }
  const verifications = []
  const saves = []
  await installMocks(context, {
    messages: COMMON_WS_BOOTSTRAP,
    apiOverrides: (path, method, request) => {
      if (path === '/api/config') return config
      if (path === '/api/model-providers/catalog') return { models: [{ id: 'listed-ok', name: '目录模型' }] }
      if (path === '/api/model-providers/verify') {
        const body = request.postDataJSON()
        verifications.push(body)
        return { ok: true, latency_ms: 24 }
      }
      if (path !== '/api/model-providers') return undefined
      if (method === 'PUT') {
        const update = request.postDataJSON()
        saves.push(update)
        supplierSettings = {
          ...supplierSettings,
          revision: 'rev-2',
          models: update.models,
          usages: {},
        }
        config = {
          ...config,
          models: supplierSettings.models.map((model) => ({ ...model, provider_name: supplier.name, has_api_key: true })),
          assist_model_ids: [],
          assist_screenshot_model_id: null,
          written_exam_model_ids: [],
        }
      }
      return supplierSettings
    },
  })

  await context.route('https://fonts.googleapis.com/**', (route) => route.abort())
  await page.goto('/')
  await page.getByRole('button', { name: '打开设置', exact: true }).click()
  const settings = page.getByRole('dialog', { name: /设置/ })
  const navigation = settings.getByRole('navigation', { name: '设置分区' })
  await navigation.getByRole('button', { name: /模型供应商配置/ }).click()
  await expect(settings.getByText('上次验证可用')).toBeVisible()
  expect(verifications).toHaveLength(0)
  await settings.getByRole('button', { name: '获取模型' }).click()
  await expect(settings.getByRole('listitem', { name: '目录模型' }).getByText('可用')).toBeVisible()
  expect(verifications.map((body) => body.model)).toEqual(['listed-ok'])

  await settings.getByRole('textbox', { name: '手填 Model ID' }).fill('ep-private')
  await settings.getByRole('button', { name: '添加 Model ID' }).click()
  const newModel = settings.getByRole('checkbox', { name: '启用 ep-private' })
  await expect(newModel).toBeDisabled()
  expect(verifications).toHaveLength(1)
  await settings.getByRole('listitem', { name: 'ep-private' }).getByRole('button', { name: '验证模型' }).click()
  await expect(newModel).toBeEnabled()
  await newModel.check()

  const oldModel = settings.getByRole('checkbox', { name: '启用 已分配模型' })
  const firstConfirm = page.waitForEvent('dialog').then(async (dialog) => {
    expect(dialog.message()).toContain('实时辅助、截图答题、笔试答题')
    await dialog.dismiss()
  })
  await Promise.all([oldModel.click(), firstConfirm])
  await expect(oldModel).toBeChecked()

  const secondConfirm = page.waitForEvent('dialog').then((dialog) => dialog.accept())
  await Promise.all([oldModel.click(), secondConfirm])
  await expect(oldModel).not.toBeChecked()
  await settings.getByRole('button', { name: '保存供应商配置' }).click()
  await expect(settings.getByText('已保存', { exact: true })).toBeVisible()
  expect(saves).toMatchObject([{ revision: 'rev-1', confirmed_remove_ids: ['old-vision'] }])
  expect(saves[0].providers[0]).not.toHaveProperty('api_key')
  expect(saves[0].models.map(({ model, enabled }) => [model, enabled])).toEqual([['old', false], ['ep-private', true]])

  await navigation.getByRole('button', { name: /^实时辅助 / }).click()
  await expect(settings.getByRole('checkbox', { name: /ep-private/ })).toBeVisible()
  await expect(settings.getByRole('checkbox', { name: /已分配模型/ })).toHaveCount(0)
})
