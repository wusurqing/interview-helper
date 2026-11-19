import { expect, test } from '@playwright/test'
import { resolveApiPayload, SAMPLE_CONFIG, SAMPLE_DEVICES, SAMPLE_OPTIONS } from './fixtures/sample-data.mjs'

if (process.env.PLAYWRIGHT_CHROME_EXECUTABLE) {
  test.use({ launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROME_EXECUTABLE }, video: 'off' })
}

test('MMPG settings test publishes PCM through the native AudioWorklet and stops from the main page', async ({ context, page }) => {
  const commands = []
  const apiPaths = []
  let ownerClosures = 0
  const config = { ...SAMPLE_CONFIG, stt_provider: 'doubao_mmpg' }
  await context.route('**/api/**', (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    apiPaths.push(path)
    const payloads = {
      '/api/config': config,
      '/api/options': SAMPLE_OPTIONS,
      '/api/devices': { ...SAMPLE_DEVICES, platform: { platform: 'Windows' } },
    }
    return route.fulfill({ json: payloads[path] ?? resolveApiPayload(path, request.method()) ?? {} })
  })
  await context.route('**/fonts.googleapis.com/**', (route) => route.abort())
  await context.route('**/assets/index.esm.min-*.js', (route) => route.fulfill({
    contentType: 'text/javascript',
    body: `
      export const AudioSourceType = { AUDIO_SOURCE_TYPE_EXTERNAL: 1 };
      export const StreamIndex = { STREAM_INDEX_MAIN: 0 };
      export const MediaType = { AUDIO: 1 };
      export const ConnectionState = { CONNECTION_STATE_DISCONNECTED: 1, CONNECTION_STATE_CONNECTED: 3, CONNECTION_STATE_RECONNECTING: 4, CONNECTION_STATE_RECONNECTED: 5, CONNECTION_STATE_LOST: 6 };
      export default {
        events: { onError: 'onError', onTokenWillExpire: 'onTokenWillExpire', onConnectionStateChanged: 'onConnectionStateChanged' },
        ErrorCode: { TOKEN_EXPIRED: 'TOKEN_EXPIRED', INVALID_TOKEN: 'INVALID_TOKEN' },
        setLogConfig() {},
        createEngine() {
          return {
            on(event, callback) { (window.rtcEvents ||= {})[event] = callback; },
            async setAudioSourceType(stream, source) { window.rtcSource = source; },
            async setExternalAudioTrack(stream, track) {
              window.mmpgTrack = track;
              const context = window.mmpgAnalysis = new AudioContext();
              const source = context.createMediaStreamSource(new MediaStream([track]));
              const analyser = window.mmpgAnalyser = context.createAnalyser();
              source.connect(analyser);
              analyser.connect(context.createMediaStreamDestination());
            },
            async joinRoom() {},
            async publishStream() {},
            async updateToken(token) { window.updatedTestToken = token; },
            async unpublishStream() {},
            async leaveRoom() {},
          };
        },
        destroyEngine() { void window.mmpgAnalysis.close(); },
      };
    `,
  }))

  let frames
  await context.routeWebSocket('**/ws/stt/mmpg*', (socket) => {
    const state = (status, generation, extras = {}) => socket.send(JSON.stringify({ type: 'state', state: status, generation, ...extras }))
    socket.send(JSON.stringify({ type: 'state', initial: true, state: 'idle', reason: 'paused', generation: 1 }))
    socket.onMessage((incoming) => {
      const command = JSON.parse(incoming)
      commands.push(command)
      if (command.type === 'start') {
        state('connecting', 3)
      } else if (command.type === 'test') {
        state('connecting', 1)
        socket.send(JSON.stringify({ type: 'rtc_join', generation: 2, appId: 'test', roomId: 'test-room', userId: 'test-user', token: 'test-token', expires_at: null }))
      } else if (command.type === 'rtc_ready') {
        state('listening', 2)
        socket.send(JSON.stringify({ type: 'diagnostic', generation: 2, event: 'speech_paused' }))
        socket.send(JSON.stringify({ type: 'transcript', generation: 2, transcript: '本地模拟识别文本' }))
        frames = setInterval(() => socket.send(Buffer.from(new Float32Array(320).fill(0.5).buffer)), 20)
      } else if (command.type === 'renew') {
        socket.send(JSON.stringify({ type: 'rtc_token', generation: 2, token: 'renewed-test-token', expires_at: null }))
      } else if (command.type === 'rtc_reconnecting') {
        clearInterval(frames)
        state('reconnecting', 2)
      } else if (command.type === 'rtc_recovered') {
        state('listening', 2)
      } else if (command.type === 'stop') {
        clearInterval(frames)
        socket.send(JSON.stringify({ type: 'rtc_leave', generation: 2, reason: 'stopped' }))
        state('idle', 2, { reason: 'stopped' })
      }
    })
    socket.onClose(() => {
      ownerClosures++
      clearInterval(frames)
    })
  })
  await context.routeWebSocket('**/ws', (socket) => {
    socket.send(JSON.stringify({ type: 'init', is_recording: false, stt_loaded: false }))
  })

  await page.goto('/')
  await page.getByRole('button', { name: '打开豆包 MMPG 测试设置' }).click()
  await page.getByText('豆包 MMPG 高级设置', { exact: true }).click()
  await expect(page.getByLabel('判停静音时间', { exact: true })).toHaveValue('800')
  await expect(page.getByLabel('答题前等待时长（毫秒）', { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('回答启动后的续说窗口（毫秒）', { exact: true })).toHaveCount(0)
  await page.getByLabel('MMPG 测试音频设备').selectOption('1001')
  await page.getByRole('button', { name: '保存并测试 MMPG' }).click()
  await expect(page.getByText('检测到停顿，等待本轮结束')).toBeVisible()
  await expect(page.getByText('判停完成')).toBeVisible()
  await expect(page.getByText('本地模拟识别文本')).toBeVisible()
  await expect.poll(() => page.evaluate(async () => {
    await window.mmpgAnalysis.resume()
    const waveform = new Float32Array(window.mmpgAnalyser.fftSize)
    window.mmpgAnalyser.getFloatTimeDomainData(waveform)
    return Math.max(...waveform)
  })).toBeGreaterThan(0.25)
  expect(commands[0]).toEqual({ type: 'test', device_ids: [1001], candidate_mic_id: null })
  expect(commands.find(({ type }) => type === 'rtc_ready')).toEqual({ type: 'rtc_ready', generation: 2 })
  expect(commands.filter(({ type }) => type === 'rtc_stage').map(({ stage }) => stage)).toContain('room_joined')
  expect(await page.evaluate(() => window.rtcSource)).toBe(1)
  await page.evaluate(() => window.rtcEvents.onTokenWillExpire())
  await expect.poll(() => page.evaluate(() => window.updatedTestToken)).toBe('renewed-test-token')
  await page.evaluate(() => window.rtcEvents.onConnectionStateChanged({ state: 4 }))
  await expect(page.getByText('正在重连', { exact: true }).first()).toBeVisible()
  await page.evaluate(() => window.rtcEvents.onConnectionStateChanged({ state: 5 }))
  await expect(page.getByText('正在识别', { exact: true }).first()).toBeVisible()
  await page.getByRole('button', { name: '结束 MMPG 测试' }).click()
  await expect(page.getByText('测试已结束')).toBeVisible()
  await expect.poll(() => page.evaluate(() => window.mmpgTrack.readyState)).toBe('ended')
  await expect.poll(() => ownerClosures).toBe(1)
  await page.getByRole('button', { name: '关闭', exact: true }).click()
  await page.getByRole('button', { name: '开始面试', exact: true }).click()
  await expect.poll(() => commands.filter(({ type }) => type === 'start').length).toBe(1)
  await page.getByRole('button', { name: '取消连接' }).click()
  await expect.poll(() => ownerClosures).toBe(2)
  await expect(page.getByRole('button', { name: '开始面试', exact: true })).toBeEnabled()
  expect(apiPaths).not.toContain('/api/ask')
  expect(apiPaths).not.toContain('/api/start')
})

test('MMPG continuation stops a partial answer and adds an update alongside the completed answer', async ({ context, page }) => {
  const config = { ...SAMPLE_CONFIG, stt_provider: 'doubao_mmpg' }
  await context.route('**/api/**', (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    return route.fulfill({ json: path === '/api/config' ? config : resolveApiPayload(path, request.method()) ?? {} })
  })
  const sockets = new Set()
  const send = (data) => {
    for (const socket of sockets) socket.send(JSON.stringify(data))
  }
  await context.routeWebSocket('**/ws', (socket) => {
    sockets.add(socket)
    socket.onClose(() => sockets.delete(socket))
    socket.send(JSON.stringify({ type: 'init', is_recording: true, interviewer_streaming: true, stt_loaded: true,
      mmpg_answer_phase: 'waiting', qa_pairs: [{ id: 'first', question: '熟悉 C++ 吗？', answer: '已有完整回答。' }] }))
    socket.send(JSON.stringify({ type: 'stt_status', provider: 'doubao_mmpg', state: 'listening', loaded: true }))
  })
  await page.goto('/')
  await expect(page.getByRole('status').filter({ hasText: '等待续说' })).toBeVisible()
  send({ type: 'answer_start', id: 'partial', question: '讲讲编译。' })
  await expect(page.getByText('讲讲编译。', { exact: true })).toBeVisible()
  send({ type: 'answer_chunk', id: 'partial', chunk: '已经生成的部分。' })
  send({ type: 'answer_cancelled', id: 'partial', reason: 'continuation' })
  send({ type: 'answer_done', id: 'partial', question: '讲讲编译。', answer: '迟到的旧内容。' })
  await expect(page.getByText('已因续说停止', { exact: true })).toBeVisible()
  await expect(page.getByText('已经生成的部分。', { exact: true })).toBeVisible()
  await expect(page.getByText('迟到的旧内容。', { exact: true })).toHaveCount(0)
  send({ type: 'answer_start', id: 'updated', question: '熟悉 C++ 吗？\n讲讲编译。', updates_qa_id: 'first' })
  send({ type: 'answer_done', id: 'updated', question: '熟悉 C++ 吗？\n讲讲编译。', answer: '新回答。', updates_qa_id: 'first' })
  await expect(page.getByText('补充后的回答', { exact: true })).toBeVisible()
  await expect(page.getByText('已有完整回答。', { exact: true })).toBeVisible()
  await expect(page.getByText('新回答。', { exact: true })).toBeVisible()
})
