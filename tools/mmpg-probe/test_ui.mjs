// Offline smoke test. The interaction stage replaces both RTC and WebSocket.
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const require = createRequire(new URL('../../frontend/package.json', import.meta.url));
const { chromium } = require('playwright');
const origin = 'http://localhost:18765';
const browser = await chromium.launch({ headless: true, channel: 'msedge' });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.route('**/*', route => route.request().url().startsWith(origin) ? route.continue() : route.abort());
  await page.goto(origin);
  await page.locator('#status').filter({ hasText: /等待配置|准备就绪/ }).waitFor();
  assert.equal(await page.locator('#silence-ms').getAttribute('min'), '500');
  for (const path of ['/config.local.json', '/server.py', '/logs/test.jsonl']) {
    const response = await page.request.get(origin + path);
    assert.equal(response.status(), 404, path + ' must not be served');
  }
  const rejected = await page.request.get(origin + '/api/config', { headers: { Origin: 'https://example.com' } });
  assert.equal(rejected.status(), 403);
  assert.deepEqual(errors, []);
  await page.screenshot({ path: fileURLToPath(new URL('ui-local-check.png', import.meta.url)), fullPage: true });

  const app = (await readFile(new URL('app.js', import.meta.url), 'utf8')).replace(/^import .*?;\r?\n/, '');
  const mock = `
    window.probeTest = { commands: [], stoppedTracks: 0, left: 0 };
    const track = { stop() { window.probeTest.stoppedTracks++; } };
    const engine = { on() {}, async startAudioCapture() {}, getLocalStreamTrack() { return track; },
      async joinRoom() {}, async publishStream() {}, async unpublishStream() {},
      async stopAudioCapture() {}, async leaveRoom() { window.probeTest.left++; } };
    const VERTC = { createEngine() { return engine; }, destroyEngine() {}, getSdkVersion() { return 'OFFLINE_MOCK'; } };
    const AudioSourceType = { AUDIO_SOURCE_TYPE_EXTERNAL: 1 }, MediaType = { AUDIO: 1 }, StreamIndex = { STREAM_INDEX_MAIN: 0 };
    class MockSocket extends EventTarget {
      static OPEN = 1; constructor() { super(); this.readyState = 0; setTimeout(() => { this.readyState = 1; this.dispatchEvent(new Event('open')); }, 0); }
      deliver(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })); }
      send(value) {
        const valueObject = JSON.parse(value); window.probeTest.commands.push(valueObject);
        if (valueObject.type === 'start') setTimeout(() => this.deliver({ type: 'status', status: 'session_created' }), 10);
        if (valueObject.type === 'mark_end') setTimeout(() => {
          this.deliver({ type: 'mark_end_ack', elapsed_ms: 1000 });
          this.deliver({ type: 'done', conv_id: 'mock-turn', transcript: '请解释 Redis 的持久化。', since_mark_ms: 350 });
        }, 10);
      }
      close() { this.readyState = 3; this.dispatchEvent(new Event('close')); }
    }
    window.WebSocket = MockSocket;
  `;
  await page.route(origin + '/api/config', route => route.fulfill({ json: {
    configured: true, missing: [], appId: 'OFFLINE', roomId: 'mock-room', userId: 'mock-user', botId: 'mock-bot', rtcToken: 'OFFLINE',
    aivad: true, twopass: true, silenceMs: 800, maxSessionSeconds: 1800,
  } }));
  await page.route(origin + '/dist/app.js', route => route.fulfill({ contentType: 'text/javascript', body: mock + app }));
  await page.reload();
  await page.locator('#status').filter({ hasText: '准备就绪' }).waitFor();
  await page.locator('#aivad').uncheck();
  await page.locator('#start').click();
  await page.locator('#status').filter({ hasText: '可以讲话' }).waitFor();
  await page.locator('#mark-end').click();
  await page.locator('.final-item').filter({ hasText: '请解释 Redis' }).waitFor();
  assert.match(await page.locator('.latency').innerText(), /350 ms/);
  await page.locator('#stop').click();
  await page.locator('#status').filter({ hasText: '已停止' }).waitFor();
  const test = await page.evaluate(() => window.probeTest);
  assert.equal(test.commands[0].type, 'start');
  assert.equal(test.commands[0].options.aivad, false);
  assert.ok(test.commands.some(command => command.type === 'stop'));
  assert.ok(test.stoppedTracks > 0);
  assert.equal(test.left, 1);
  assert.deepEqual(errors, []);
  console.log('PASS: page load, private-file denial, cross-origin denial, and mocked start/mark/done/stop; no cloud calls.');
} finally {
  await browser.close();
}
