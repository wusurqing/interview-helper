import { test, expect } from '@playwright/test'
import { installMocks, COMMON_WS_BOOTSTRAP } from './fixtures/setup.mjs'

if (process.env.PLAYWRIGHT_CHROME_EXECUTABLE) {
  test.use({ launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROME_EXECUTABLE }, video: 'off' })
}

test('opacity slider and LLM selector stay usable at the minimum desktop width', async ({ context, page }) => {
  await page.setViewportSize({ width: 800, height: 600 })
  await installMocks(context, { messages: COMMON_WS_BOOTSTRAP })
  await context.addInitScript(() => {
    window.electronAPI = {
      getWindowState: () => Promise.resolve({ opacity: 1, opacitySupported: true, alwaysOnTop: false, contentProtection: true, visible: true }),
      setWindowOpacity: (opacity) => Promise.resolve(opacity),
    }
  })

  await page.goto('/')
  const slider = page.getByRole('slider', { name: '窗口透明度' })
  const model = page.getByRole('button', { name: /优先答题模型/ })
  await expect(slider).toBeVisible()
  await expect(model).toBeVisible()
  const sliderBox = await slider.boundingBox()
  const modelBox = await model.boundingBox()
  expect(sliderBox.x + sliderBox.width).toBeLessThanOrEqual(modelBox.x)
  await page.getByRole('group', { name: '窗口透明度调节' }).hover()
  await page.mouse.wheel(0, -100)
  await expect(slider).toHaveValue('2')
})
