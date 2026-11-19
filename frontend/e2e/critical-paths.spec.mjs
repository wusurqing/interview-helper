/**
 * Critical-path smoke tests with mocked backend and WebSocket.
 *
 * Goal: catch regressions that break the app's main navigation, settings
 * panel, and module switches without needing a live backend or any audio
 * device. Each test runs in <5s.
 */
import { test, expect } from '@playwright/test'
import { installMocks, COMMON_WS_BOOTSTRAP } from './fixtures/setup.mjs'
import { SAMPLE_CONFIG } from './fixtures/sample-data.mjs'

test.describe('app shell', () => {
  test.beforeEach(async ({ context }) => {
    await installMocks(context, {
      messages: COMMON_WS_BOOTSTRAP,
      localStorage: { 'ia-color-scheme': 'vscode-light-plus' },
    })
  })

  test('boots and shows the brand header and configured ASR status', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: '学习助手' })).toBeVisible()
    await expect(page.getByText('ASR 已配置', { exact: true })).toBeVisible()
  })

  test('module tabs switch between assist / review / knowledge / resume / job-tracker', async ({ page }) => {
    await page.goto('/')

    const assistTab = page.getByRole('tab', { name: '实时辅助' })
    const reviewTab = page.getByRole('tab', { name: '面试复盘' })
    const knowledgeTab = page.getByRole('tab', { name: '能力分析' })
    const resumeTab = page.getByRole('tab', { name: '简历优化' })
    const jobTab = page.getByRole('tab', { name: /求职看板/ })

    await expect(assistTab).toHaveAttribute('aria-selected', 'true')

    await reviewTab.click()
    await expect(reviewTab).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByRole('heading', { name: '已完成', exact: true })).toBeVisible({ timeout: 8000 })

    await knowledgeTab.click()
    await expect(knowledgeTab).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByText('薄弱点排名')).toBeVisible({ timeout: 8000 })
    await expect(page.getByRole('button', { name: '分析设置' })).toBeEnabled()
    await expect(page.getByText('待分析 0 条问答')).toBeVisible()

    await resumeTab.click()
    await expect(resumeTab).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByPlaceholder('将招聘 JD 粘贴到这里...')).toBeVisible({ timeout: 8000 })

    await jobTab.click()
    await expect(jobTab).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByRole('heading', { name: '求职进度', exact: true })).toBeVisible({ timeout: 8000 })
  })

  test('does not log uncaught errors during initial render', async ({ page }) => {
    await page.route('https://fonts.googleapis.com/**', route => route.fulfill({
      contentType: 'text/css',
      body: '',
    }))
    /** @type {string[]} */
    const errors = []
    page.on('pageerror', (err) => errors.push(String(err)))
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text())
    })

    await page.goto('/')
    await expect(page.getByRole('heading', { name: '学习助手' })).toBeVisible()
    await page.waitForTimeout(500)

    const significant = errors.filter(
      (e) =>
        !e.includes('favicon') &&
        !e.includes('downloadable font') &&
        !e.toLowerCase().includes('manifest'),
    )
    expect(significant, `Console errors:\n${significant.join('\n')}`).toEqual([])
  })
})

test.describe('assist mode with WebSocket-driven Q/A', () => {
  test('single-mic test switches to the confirmed role and ends the session', async ({ context, page }) => {
    await installMocks(context, {
      messages: COMMON_WS_BOOTSTRAP,
      apiOverrides: {
        '/api/config': { ...SAMPLE_CONFIG, candidate_asr_enabled: true },
        'POST /api/start': { ok: true },
        'POST /api/single-mic/role': { ok: true },
        'POST /api/stop': { ok: true },
      },
    })
    const emitAssist = (message) => page.evaluate((payload) => window.__IA_MOCK_WS_MESSAGE__(payload), message)
    await page.goto('/')
    await page.getByRole('button', { name: '单人测试', exact: true }).click()
    await expect(page.getByRole('button', { name: '选择测试麦克风' })).toHaveText(/MacBook Pro 麦克风/)
    await expect(page.getByRole('button', { name: '选择会议音频设备' })).toHaveCount(0)

    const startRequest = page.waitForRequest('**/api/start')
    await page.getByRole('button', { name: '开始单人测试' }).click()
    const startBody = (await startRequest).postDataJSON()
    expect(startBody).toEqual({ device_id: 1002, candidate_mic_device_id: 1002, single_mic_id: expect.stringMatching(/^[0-9a-f]{32}$/) })
    const state = { id: startBody.single_mic_id, device_id: 1002, role: 'interviewer', switching: false }
    await emitAssist({ type: 'recording', value: true })
    await emitAssist({ type: 'single_mic', value: state })

    const interviewer = page.getByRole('button', { name: '面试官', exact: true })
    const candidate = page.getByRole('button', { name: '候选人', exact: true })
    await expect(interviewer).toHaveAttribute('aria-pressed', 'true')
    const roleRequest = page.waitForRequest('**/api/single-mic/role')
    await candidate.click()
    expect((await roleRequest).postDataJSON()).toEqual({ session_id: state.id, role: 'candidate' })
    await emitAssist({ type: 'single_mic', value: { ...state, switching: true } })
    await expect(page.getByText('切换中，等待上一句转写完成')).toBeVisible()
    await expect(candidate).toBeDisabled()
    await expect(page.getByRole('button', { name: '结束单人测试' })).toBeEnabled()

    await emitAssist({ type: 'single_mic', value: { ...state, role: 'candidate' } })
    await expect(candidate).toHaveAttribute('aria-pressed', 'true')
    await expect(interviewer).toHaveAttribute('aria-pressed', 'false')
    await expect(interviewer).toBeEnabled()

    page.once('dialog', (dialog) => dialog.accept())
    const stopRequest = page.waitForRequest('**/api/stop')
    await page.getByRole('button', { name: '结束单人测试' }).click()
    await stopRequest
    await emitAssist({ type: 'recording', value: false })
    await expect(candidate).toHaveCount(0)
    await expect(page.getByRole('button', { name: '开始单人测试' })).toBeEnabled()
  })

  test('renders streamed transcripts and answer payload from mock WS', async ({ context, page }) => {
    const now = Math.floor(Date.now() / 1000)

    await installMocks(context, {
      messages: [
        ...COMMON_WS_BOOTSTRAP,
        {
          type: 'init',
          delay: 50,
          transcriptions: ['请介绍一下你做过最有挑战的一个项目。'],
          qa_pairs: [
            {
              id: 'qa-mock-1',
              question: '请介绍一下你做过最有挑战的一个项目。',
              answer: '核心要点是先讲背景与目标、再讲关键决策、最后讲量化结果。',
              thinkContent: '',
              timestamp: now - 5,
              source: 'manual_text',
              model_name: 'GPT-4.1 Mini',
            },
          ],
          is_recording: false,
          is_paused: false,
          stt_loaded: true,
        },
      ],
    })

    await page.goto('/')
    await expect(page.getByRole('heading', { name: '学习助手' })).toBeVisible()
    await expect(page.getByText('请介绍一下你做过最有挑战的一个项目。').first()).toBeVisible({
      timeout: 5000,
    })
    await expect(
      page.getByText('核心要点是先讲背景与目标、再讲关键决策、最后讲量化结果。'),
    ).toBeVisible({ timeout: 5000 })
  })
})

test.describe('settings drawer', () => {
  for (const layout of [
    { name: 'desktop', viewport: { width: 1440, height: 900 }, desktopBridge: true },
    { name: 'narrow browser', viewport: { width: 390, height: 844 }, desktopBridge: false },
  ]) {
    test(`separates appearance and quick actions on ${layout.name}`, async ({ context, page }) => {
      await page.setViewportSize(layout.viewport)
      await installMocks(context, {
        messages: COMMON_WS_BOOTSTRAP,
        apiOverrides: { '/api/network-info': { url: 'http://192.168.1.9:8000' } },
      })
      if (layout.desktopBridge) {
        await context.addInitScript(() => {
          window.shortcutsSuspended = false
          window.electronAPI = {
            getShortcuts: async () => ({
              hideOrShowWindow: {
                action: 'hideOrShowWindow', key: 'CommandOrControl+B',
                defaultKey: 'CommandOrControl+B', label: '隐藏/显示窗口',
                category: '窗口', status: 'registered',
              },
            }),
            setShortcutsSuspended: async (suspended) => { window.shortcutsSuspended = suspended },
          }
        })
      }
      await page.goto('/')
      await page.getByRole('button', { name: '打开设置', exact: true }).click()
      const dialog = page.getByRole('dialog')
      const appearance = dialog.getByRole('button', { name: /^外观/ })
      const shortcuts = dialog.getByRole('button', { name: /^快捷操作/ })
      const search = dialog.getByRole('searchbox', { name: '搜索设置项' })

      await appearance.click()
      await expect(appearance).toHaveAttribute('aria-current', 'page')
      await expect(dialog.getByText('配色方案', { exact: true })).toBeVisible()
      await expect(dialog.getByText('流式跟滚阈值（像素）', { exact: true })).toBeVisible()
      await expect(dialog.getByText('手机扫码访问', { exact: true })).toHaveCount(0)
      await expect(dialog.getByRole('heading', { name: '快捷提示词', exact: true })).toHaveCount(0)
      await expect(dialog.getByRole('heading', { name: '全局快捷键', exact: true })).toHaveCount(0)

      await shortcuts.click()
      await expect(shortcuts).toHaveAttribute('aria-current', 'page')
      await expect(appearance).not.toHaveAttribute('aria-current', 'page')
      await expect(dialog.getByText('配色方案', { exact: true })).toHaveCount(0)
      await expect(dialog.getByText('流式跟滚阈值（像素）', { exact: true })).toHaveCount(0)
      await expect(dialog.getByText('手机扫码访问', { exact: true })).toBeVisible()
      await expect(dialog.getByRole('img', { name: 'QR Code' })).toBeVisible()
      await expect(dialog.getByRole('heading', { name: '快捷提示词', exact: true })).toBeVisible()
      const globalShortcuts = dialog.getByRole('heading', { name: '全局快捷键', exact: true })
      if (layout.desktopBridge) await expect(globalShortcuts).toBeVisible()
      else await expect(globalShortcuts).toHaveCount(0)

      await search.fill('快捷提示词')
      await expect(dialog.getByRole('heading', { name: '快捷提示词', exact: true })).toBeVisible()
      await expect(dialog.getByText('手机扫码访问', { exact: true })).toHaveCount(0)
      await expect(globalShortcuts).toHaveCount(0)
      await search.fill('二维码')
      await expect(dialog.getByRole('img', { name: 'QR Code' })).toBeVisible()
      await expect(dialog.getByRole('heading', { name: '快捷提示词', exact: true })).toHaveCount(0)
      await expect(globalShortcuts).toHaveCount(0)
      if (layout.desktopBridge) {
        await search.fill('全局快捷键')
        await expect(globalShortcuts).toBeVisible()
        await expect(dialog.getByText('手机扫码访问', { exact: true })).toHaveCount(0)
        await expect(dialog.getByRole('heading', { name: '快捷提示词', exact: true })).toHaveCount(0)
      }
      await search.fill('不存在的设置项')
      await expect(dialog.getByText(/没有找到匹配/)).toBeVisible()
      await search.fill('')

      if (layout.desktopBridge) {
        await dialog.getByRole('button', { name: /^(?:Ctrl\+B|⌘B)$/ }).click()
        await expect.poll(() => page.evaluate(() => window.shortcutsSuspended)).toBe(true)
      }
      await appearance.click()
      if (layout.desktopBridge) {
        await expect.poll(() => page.evaluate(() => window.shortcutsSuspended)).toBe(false)
      }
      await search.fill('主题')
      await expect(dialog.getByText('配色方案', { exact: true })).toBeVisible()
      await search.fill('流式跟滚阈值')
      await expect(dialog.getByText('流式跟滚阈值（像素）', { exact: true })).toBeVisible()
      await search.fill('快捷提示词')
      await expect(dialog.getByText(/没有找到匹配/)).toBeVisible()
    })
  }

  test('keeps quick prompts usable when network discovery fails', async ({ context, page }) => {
    await installMocks(context, { messages: COMMON_WS_BOOTSTRAP })
    await context.route('**/api/network-info', (route) => route.abort())
    await page.goto('/')
    await page.getByRole('button', { name: '打开设置', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('button', { name: /^快捷操作/ }).click()
    await dialog.getByPlaceholder('输入新的快捷词...').fill('解释实现取舍')
    await dialog.getByRole('button', { name: '添加', exact: true }).click()
    await expect(dialog.getByText('解释实现取舍', { exact: true })).toBeVisible()
    await dialog.getByRole('button', { name: /^外观/ }).click()
    await dialog.getByRole('button', { name: /^快捷操作/ }).click()
    await expect(dialog.getByText('解释实现取舍', { exact: true })).toBeVisible()
    await expect(dialog.getByText('手机扫码访问', { exact: true })).toHaveCount(0)
  })

  test('opens and shows model list / position config', async ({ context, page }) => {
    await installMocks(context, { messages: COMMON_WS_BOOTSTRAP })
    await page.goto('/')
    await expect(page.getByRole('heading', { name: '学习助手' })).toBeVisible()

    await page.getByRole('button', { name: /设置|Settings/i }).first().click()

    await expect(page.getByText(/GPT-4\.1 Mini/).first()).toBeVisible({ timeout: 5000 })
  })
})
