const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { runInNewContext } = require('node:vm');

const mainFile = path.join(__dirname, 'main.js');
const mainSource = readFileSync(mainFile, 'utf8');

const primary = { id: 1, label: 'Primary', workArea: { x: 0, y: 0, width: 1280, height: 800 } };
const secondary = { id: 2, label: 'Secondary', workArea: { x: -1000, y: -100, width: 1000, height: 900 } };

function startDesktop(t, { http, savedLayout, connected = true, main = false, backend } = {}) {
  t.mock.timers.enable({ apis: ['Date', 'setInterval', 'setTimeout'], now: 1000 });
  const handlers = new Map();
  const appEvents = new Map();
  let quitCalls = 0;
  const shortcuts = new Map();
  const userData = mkdtempSync(path.join(tmpdir(), 'interview-overlay-test-'));
  t.after(() => rmSync(userData, { recursive: true, force: true }));
  const layoutFile = path.join(userData, 'overlay-window.json');
  if (savedLayout) writeFileSync(layoutFile, JSON.stringify(savedLayout));
  let displays = connected ? [primary, secondary] : [primary];
  const screen = Object.assign(new EventEmitter(), {
    getPrimaryDisplay: () => primary,
    getAllDisplays: () => displays,
    getCursorScreenPoint: () => ({ x: 400, y: 300 }),
    getDisplayNearestPoint: () => primary,
  });
  let overlay;
  const helpers = [];

  class OverlayWindow extends EventEmitter {
    constructor({ x, y, width, height }) {
      super();
      this.webContents = new EventEmitter();
      this.messages = [];
      this.webContents.send = (channel, payload) => { this.messages.push({ channel, payload }); };
      this.visible = false;
      this.destroyed = false;
      this.moveTopCalls = 0;
      this.bounds = { x, y, width, height };
      overlay = this;
    }

    isVisible() { return this.visible; }
    isDestroyed() { return this.destroyed; }
    setAlwaysOnTop() {}
    setContentProtection() {}
    setBackgroundColor() {}
    setVisibleOnAllWorkspaces() {}
    setFocusable() {}
    setIgnoreMouseEvents(value) { this.ignoreMouse = value; }
    getNativeWindowHandle() { const b = Buffer.alloc(8); b.writeBigUInt64LE(123n); return b; }
    hookWindowMessage(message, callback) { this.hookedMessage = message; this.messageCallback = callback; }
    unhookWindowMessage() { this.messageCallback = null; }
    setMinimumSize() {}
    setBounds(bounds) { this.bounds = { ...bounds }; }
    getBounds() { return this.bounds; }
    setPosition(x, y) { this.bounds = { ...this.bounds, x, y }; }
    loadURL() {}

    showInactive() {
      this.visible = true;
      this.emit('show');
    }

    hide() {
      if (!this.visible) return;
      this.visible = false;
      this.emit('hide');
    }

    moveTop() {
      this.moveTopCalls += 1;
      // Electron's Windows implementation passes SWP_SHOWWINDOW to SetWindowPos.
      this.visible = true;
    }

    destroy() {
      this.destroyed = true;
      this.visible = false;
      this.emit('closed');
    }
  }

  const electron = {
    app: {
      commandLine: { appendSwitch() {} },
      getPath: () => userData,
      whenReady: () => new Promise(() => {}),
      on(event, handler) { appEvents.set(event, handler); },
      quit() { quitCalls++; },
    },
    BrowserWindow: OverlayWindow,
    screen,
    globalShortcut: {
      register(key, handler) { shortcuts.set(key, handler); return true; },
      unregister(key) { shortcuts.delete(key); },
    },
    ipcMain: {
      handle(channel, handler) { handlers.set(channel, handler); },
      on() {},
    },
  };

  const context = {
    __dirname,
    require: (name) => {
      if (name === 'electron') return electron;
      if (name === 'http' && http) return http;
      if (name === 'child_process') return { spawn() {
        const child = new EventEmitter();
        child.stdout = new EventEmitter();
        child.stderr = new EventEmitter();
        child.messages = [];
        child.stdin = { write(data) { child.messages.push(JSON.parse(data)); return true; }, end() {} };
        child.kill = () => {};
        helpers.push(child);
        return child;
      } };
      return require(name);
    },
    process: { platform: 'win32', env: {} },
    testBackend: backend,
    Date,
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
    Buffer,
  };
  runInNewContext(mainSource, context, { filename: mainFile });
  if (backend) runInNewContext('pythonProcess = testBackend', context);
  if (main) {
    runInNewContext(`mainWindow = {
      isDestroyed: () => false, isVisible: () => false,
      webContents: { send() {} },
    }`, context);
  }
  context.registerOverlayDisplayListeners?.();

  const invoke = (channel, ...args) => handlers.get(channel)?.({ sender: overlay?.webContents }, ...args);
  invoke('set-shortcuts-suspended', false);
  return {
    requestAssistStop: context.requestAssistStop,
    beforeQuit: (event) => appEvents.get('before-quit')(event),
    get quitCalls() { return quitCalls; },
    invoke,
    toggle: () => shortcuts.get('CommandOrControl+O')(),
    moveToMouse: () => shortcuts.get('CommandOrControl+M')(),
    readSaved: () => JSON.parse(readFileSync(layoutFile, 'utf8')),
    disconnect() { displays = [primary]; screen.emit('display-removed', {}, secondary); },
    reconnect() { displays = [primary, secondary]; screen.emit('display-added', {}, secondary); },
    get overlay() { return overlay; },
    get helper() { return helpers[helpers.length - 1]; },
    tick: (ms) => t.mock.timers.tick(ms),
    finishLoad() {
      overlay.webContents.emit('did-finish-load');
      t.mock.timers.tick(120);
    },
  };
}

test('prompt window passes mouse through and only enables wheel routing when ready and scrollable', (t) => {
  const desktop = startDesktop(t);
  desktop.invoke('sync-overlay-window', { mode: 'prompt', promptWheelModifier: 'ctrl' });
  desktop.toggle();
  desktop.finishLoad();
  assert.equal(desktop.overlay.ignoreMouse, true);
  assert.ok(desktop.helper);
  desktop.helper.stdout.emit('data', Buffer.from('{"type":"ready"}\n'));
  desktop.invoke('report-prompt-scrollable', true);
  const state = desktop.helper.messages.at(-1);
  assert.equal(state.active, true);
  assert.equal(state.scrollable, true);
  assert.equal(state.modifier, 'ctrl');
  desktop.invoke('sync-overlay-window', { mode: 'glass' });
  assert.equal(desktop.overlay.ignoreMouse, false);
  assert.equal(desktop.helper.messages.at(-1).active, false);
});

test('wheel routing ignores stale messages and stops when the helper exits or overlay hides', (t) => {
  const desktop = startDesktop(t);
  desktop.invoke('sync-overlay-window', { mode: 'prompt' });
  desktop.toggle();
  desktop.finishLoad();
  const helper = desktop.helper;
  helper.stdout.emit('data', Buffer.from('{"type":"ready"}\n'));
  desktop.invoke('report-prompt-scrollable', true);
  const generation = helper.messages.at(-1).generation;
  const wParam = Buffer.alloc(8);
  const lParam = Buffer.alloc(8);
  wParam.writeBigUInt64LE(BigInt(generation));
  lParam.writeInt32LE(-120);
  desktop.overlay.messageCallback(wParam, lParam);
  assert.equal(desktop.overlay.messages.at(-1).payload, -120);
  helper.emit('exit', 1);
  assert.equal(desktop.invoke('get-prompt-wheel-status'), 'unavailable');
  const count = desktop.overlay.messages.length;
  desktop.overlay.messageCallback(wParam, lParam);
  assert.equal(desktop.overlay.messages.length, count);
  desktop.tick(400);
  desktop.toggle();
  assert.equal(desktop.invoke('get-prompt-wheel-status'), 'inactive');
  desktop.tick(400);
  desktop.toggle();
  assert.notEqual(desktop.helper, helper);
});

test('a broken helper pipe during shutdown does not recursively fail', (t) => {
  const desktop = startDesktop(t);
  desktop.invoke('sync-overlay-window', { mode: 'prompt' });
  desktop.toggle();
  desktop.finishLoad();
  desktop.helper.stdout.emit('data', Buffer.from('{"type":"ready"}\n'));
  desktop.helper.stdin.write = () => { throw new Error('broken pipe'); };
  desktop.tick(400);
  assert.doesNotThrow(() => desktop.toggle());
  assert.equal(desktop.invoke('get-prompt-wheel-status'), 'inactive');
});

test('saved geometry is independent by mode and Ctrl+M only moves the visible instance', (t) => {
  const desktop = startDesktop(t);
  const update = (mode, patch) => desktop.invoke('update-overlay-layout', mode, patch);
  assert.equal(update('glass', { displayId: 2, anchor: 'top-left', x: 60, y: 40, width: 610, height: 300 })?.ok, true);
  assert.equal(update('prompt', { sizeMode: 'fixed', width: 300, height: 180 })?.ok, true);
  assert.equal(desktop.readSaved().layouts.glass.width, 610);
  desktop.toggle();
  desktop.finishLoad();
  assert.deepEqual(desktop.overlay.getBounds(), { x: -940, y: -60, width: 610, height: 300 });
  desktop.moveToMouse();
  assert.equal(desktop.overlay.getBounds().x, 380);
  desktop.tick(400);
  desktop.toggle();
  desktop.tick(400);
  desktop.toggle();
  assert.equal(desktop.overlay.getBounds().x, -940);
  desktop.invoke('sync-overlay-window', { mode: 'prompt' });
  assert.equal(desktop.overlay.getBounds().width, 300);
  assert.equal(desktop.invoke('resize-overlay-window', { width: 250, height: 90 }).skipped, true);
  assert.equal(desktop.overlay.getBounds().height, 180);
  desktop.invoke('sync-overlay-window', { mode: 'glass' });
  assert.equal(desktop.overlay.getBounds().width, 610);
});

test('disconnect temporarily falls back to primary without losing preferred display', (t) => {
  const desktop = startDesktop(t, {
    savedLayout: { position: { x: -900, y: 12, w: 510, h: 220 } },
  });
  const layout = desktop.invoke('get-overlay-layout');
  assert.equal(layout?.layouts.glass.displayId, secondary.id);
  assert.equal(layout.layouts.glass.width, 510);
  assert.equal(layout.layouts.prompt.maxWidth, 820);
  desktop.toggle();
  desktop.finishLoad();
  desktop.disconnect();
  assert.ok(desktop.overlay.getBounds().x >= 0);
  assert.equal(desktop.invoke('get-overlay-layout').layouts.glass.displayId, secondary.id);
  desktop.reconnect();
  assert.ok(desktop.overlay.getBounds().x >= 0);
  desktop.tick(400);
  desktop.toggle();
  desktop.tick(400);
  desktop.toggle();
  assert.equal(desktop.overlay.getBounds().x, -900);
});

test('auto prompt preserves its anchor and never writes its measured dimensions', (t) => {
  const desktop = startDesktop(t);
  desktop.invoke('sync-overlay-window', { mode: 'prompt', focusWidthPct: 76, promptMaxWidth: 700 });
  assert.equal(desktop.invoke('get-overlay-layout')?.layouts.focus.widthPct, 76);
  assert.equal(desktop.invoke('get-overlay-layout')?.layouts.prompt.maxWidth, 700);
  assert.equal(desktop.invoke('update-overlay-layout', 'prompt', { anchor: 'bottom-right', x: -20, y: -30 })?.ok, true);
  desktop.toggle();
  desktop.finishLoad();
  assert.equal(desktop.overlay.getBounds().y, 698);
  desktop.invoke('resize-overlay-window', { width: 240, height: 110 });
  assert.deepEqual(desktop.overlay.getBounds(), { x: 1020, y: 698, width: 240, height: 102 });
  assert.equal(desktop.readSaved().layouts.prompt.maxWidth, 700);
  assert.equal(desktop.invoke('reset-overlay-layout', 'prompt')?.ok, true);
  assert.equal(desktop.invoke('get-overlay-layout').layouts.prompt.maxWidth, 820);
});

test('an auto prompt offset may use space beyond its maximum-size rectangle', (t) => {
  const desktop = startDesktop(t);
  desktop.invoke('sync-overlay-window', { mode: 'prompt' });
  desktop.invoke('update-overlay-layout', 'prompt', { anchor: 'top-left', x: 800, y: 680 });
  desktop.toggle();
  desktop.finishLoad();
  desktop.invoke('resize-overlay-window', { width: 240, height: 110 });
  assert.deepEqual(desktop.overlay.getBounds(), { x: 800, y: 680, width: 240, height: 110 });
});

test('an auto prompt grows down to the work-area edge, then shrinks for a short answer', (t) => {
  const desktop = startDesktop(t, { savedLayout: { layouts: { prompt: { maxHeight: 420 } } } });
  desktop.invoke('sync-overlay-window', { mode: 'prompt' });
  desktop.invoke('update-overlay-layout', 'prompt', { anchor: 'top-left', x: 10, y: 120 });
  const measure = (height) => desktop.invoke('resize-overlay-window', { width: 240, height });
  desktop.toggle();
  desktop.finishLoad();
  assert.equal(desktop.overlay.getBounds().height, 72);
  assert.equal(desktop.invoke('get-overlay-layout').promptViewport.height, 680);
  measure(500);
  assert.deepEqual(desktop.overlay.getBounds(), { x: 10, y: 120, width: 240, height: 500 });
  measure(900);
  assert.deepEqual(desktop.overlay.getBounds(), { x: 10, y: 120, width: 240, height: 680 });
  measure(100);
  assert.deepEqual(desktop.overlay.getBounds(), { x: 10, y: 120, width: 240, height: 100 });
});

test('auto prompt preserves a custom legacy height and recalculates room after Ctrl+M', (t) => {
  const desktop = startDesktop(t, {
    savedLayout: { layouts: { prompt: { displayId: 2, anchor: 'bottom-left', y: -10, maxHeight: 360 } } },
  });
  desktop.invoke('sync-overlay-window', { mode: 'prompt' });
  desktop.toggle();
  desktop.finishLoad();
  const layout = () => desktop.invoke('get-overlay-layout');
  const measure = () => desktop.invoke('resize-overlay-window', { width: 240, height: 750 });
  assert.equal(layout().layouts.prompt.heightLimit, 'custom');
  measure();
  assert.deepEqual(desktop.overlay.getBounds(), { x: -1000, y: 718, width: 240, height: 82 });
  desktop.moveToMouse();
  assert.equal(layout().promptViewport.height, 360);
  measure();
  assert.deepEqual(desktop.overlay.getBounds(), { x: 380, y: 280, width: 240, height: 360 });
  desktop.invoke('update-overlay-layout', 'prompt', { heightLimit: 'screen' });
  assert.equal(layout().layouts.prompt.maxHeight, 360);
});

test('legacy coordinates can be migrated after the preferred display comes online', (t) => {
  const desktop = startDesktop(t, {
    connected: false,
    savedLayout: { position: { x: -900, y: 12, w: 510, h: 220 } },
  });
  desktop.invoke('sync-overlay-window', { mode: 'glass', focusWidthPct: 76, promptMaxWidth: 700 });
  assert.equal(desktop.invoke('get-overlay-layout').layouts.glass.displayId, null);
  desktop.reconnect();
  assert.equal(desktop.readSaved().layouts.glass.displayId, secondary.id);
  assert.equal(desktop.readSaved().layouts.prompt.maxWidth, 700);
  desktop.toggle();
  desktop.finishLoad();
  assert.equal(desktop.overlay.getBounds().x, -900);
});

test('a startup shortcut opens using the renderer’s stored mode after preference sync', (t) => {
  const desktop = startDesktop(t, { main: true });
  desktop.toggle();
  assert.equal(desktop.overlay, undefined);
  desktop.invoke('sync-overlay-window', { mode: 'focus', enabled: false, focusWidthPct: 75, focusHeightPct: 80 });
  desktop.finishLoad();
  assert.equal(desktop.overlay.isVisible(), true);
  assert.equal(desktop.invoke('get-overlay-state').mode, 'focus');
  assert.equal(desktop.invoke('get-overlay-state').enabled, true);
  assert.equal(desktop.overlay.getBounds().width, 939);
});

test('focus fixed dimensions leave room for its toolbar and answer', (t) => {
  const desktop = startDesktop(t);
  const saved = desktop.invoke('update-overlay-layout', 'focus', { sizeMode: 'fixed', width: 300, height: 100 });
  assert.equal(saved.layouts.focus.width, 480);
  assert.equal(saved.layouts.focus.height, 240);
});

test('desktop quit posts the session stop and waits for its response', async (t) => {
  let requested;
  let sendResponse;
  const request = new EventEmitter();
  request.end = () => {};
  const http = {
    request(url, options, onResponse) {
      requested = { url, options };
      sendResponse = onResponse;
      return request;
    },
  };
  const desktop = startDesktop(t, { http });
  let settled = false;
  const stopping = desktop.requestAssistStop().then((acknowledged) => { settled = true; return acknowledged; });
  assert.equal(new URL(requested.url).pathname, '/api/stop');
  assert.equal(requested.options.method, 'POST');
  assert.equal(requested.options.timeout, undefined);

  const response = new EventEmitter();
  response.resume = () => {};
  response.statusCode = 200;
  sendResponse(response);
  await Promise.resolve();
  assert.equal(settled, false);
  response.emit('end');
  assert.equal(await stopping, true);
  assert.equal(settled, true);
});

test('desktop keeps the backend alive while archive acknowledgement takes over 15 seconds', async (t) => {
  let sendResponse;
  let requests = 0;
  const signals = [];
  const backend = new EventEmitter();
  backend.kill = (signal) => { signals.push(signal); backend.emit('exit'); };
  const desktop = startDesktop(t, {
    main: true, backend, http: { request(_url, _options, callback) {
      requests++;
      sendResponse = callback;
      return Object.assign(new EventEmitter(), { end() {} });
    } },
  });

  let prevented = 0;
  const event = { preventDefault() { prevented++; } };
  desktop.beforeQuit(event);
  desktop.beforeQuit(event);
  desktop.tick(20000);
  assert.equal(prevented, 2);
  assert.equal(requests, 1);
  assert.deepEqual(signals, []);

  const response = Object.assign(new EventEmitter(), { resume() {}, statusCode: 200 });
  sendResponse(response);
  response.emit('end');
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(signals, ['SIGTERM']);
  assert.equal(desktop.quitCalls, 1);
});

for (const [entry, hide] of [
  ['Ctrl+O', (desktop) => desktop.toggle()],
  ['overlay IPC', (desktop) => desktop.invoke('sync-overlay-window', { visible: false })],
  ['native hide', (desktop) => desktop.overlay.hide()],
]) {
  test(`overlay stays hidden after ${entry} while the topmost timer was active`, (t) => {
    const desktop = startDesktop(t);
    desktop.toggle();
    desktop.finishLoad();
    assert.equal(desktop.overlay.isVisible(), true);

    desktop.tick(500);
    const movesBeforeHide = desktop.overlay.moveTopCalls;
    hide(desktop);
    assert.equal(desktop.overlay.isVisible(), false);

    desktop.tick(500);
    assert.equal(desktop.overlay.isVisible(), false, 'a topmost tick resurrected the overlay');
    assert.equal(desktop.overlay.moveTopCalls, movesBeforeHide);
  });
}

test('reopening starts one topmost timer and destroying the overlay stops it', (t) => {
  const desktop = startDesktop(t);
  desktop.toggle();
  desktop.finishLoad();
  desktop.tick(400);
  desktop.toggle();
  desktop.tick(400);
  desktop.toggle();
  const reopened = desktop.overlay;
  assert.equal(reopened.isVisible(), true);

  const movesOnReopen = reopened.moveTopCalls;
  desktop.tick(500);
  assert.equal(reopened.moveTopCalls, movesOnReopen + 1);

  desktop.invoke('destroy-overlay');
  desktop.tick(1000);
  assert.equal(reopened.isDestroyed(), true);
  assert.equal(reopened.isVisible(), false);
  assert.equal(reopened.moveTopCalls, movesOnReopen + 1);
});

test('hiding an overlay before it finishes loading cancels the pending show', (t) => {
  const desktop = startDesktop(t);
  desktop.toggle();
  desktop.tick(400);
  desktop.toggle();
  desktop.finishLoad();
  desktop.tick(500);
  assert.equal(desktop.overlay.isVisible(), false);
  assert.equal(desktop.overlay.moveTopCalls, 0);
});
