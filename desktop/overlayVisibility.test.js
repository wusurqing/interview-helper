const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { readFileSync } = require('node:fs');
const { randomUUID } = require('node:crypto');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { runInNewContext } = require('node:vm');

const mainFile = path.join(__dirname, 'main.js');
const mainSource = readFileSync(mainFile, 'utf8');

function startDesktop(t, { http } = {}) {
  t.mock.timers.enable({ apis: ['Date', 'setInterval', 'setTimeout'], now: 1000 });
  const handlers = new Map();
  const shortcuts = new Map();
  const userData = path.join(tmpdir(), `interview-overlay-test-${randomUUID()}`);
  let overlay;

  class OverlayWindow extends EventEmitter {
    constructor() {
      super();
      this.webContents = new EventEmitter();
      this.webContents.send = () => {};
      this.visible = false;
      this.destroyed = false;
      this.moveTopCalls = 0;
      overlay = this;
    }

    isVisible() { return this.visible; }
    isDestroyed() { return this.destroyed; }
    setAlwaysOnTop() {}
    setContentProtection() {}
    setBackgroundColor() {}
    setVisibleOnAllWorkspaces() {}
    setFocusable() {}
    setMinimumSize() {}
    setBounds() {}
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
      on() {},
    },
    BrowserWindow: OverlayWindow,
    screen: {
      getPrimaryDisplay: () => ({ workArea: { x: 0, y: 0, width: 1280, height: 800 } }),
    },
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
    require: (name) => name === 'electron' ? electron : name === 'http' && http ? http : require(name),
    process: { platform: 'win32', env: {} },
    Date,
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
  };
  runInNewContext(mainSource, context, { filename: mainFile });

  const invoke = (channel, ...args) => handlers.get(channel)({}, ...args);
  invoke('set-shortcuts-suspended', false);
  return {
    requestAssistStop: context.requestAssistStop,
    invoke,
    toggle: () => shortcuts.get('CommandOrControl+O')(),
    get overlay() { return overlay; },
    tick: (ms) => t.mock.timers.tick(ms),
    finishLoad() {
      overlay.webContents.emit('did-finish-load');
      t.mock.timers.tick(120);
    },
  };
}

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
  const stopping = desktop.requestAssistStop(15000).then(() => { settled = true; });
  assert.equal(new URL(requested.url).pathname, '/api/stop');
  assert.equal(requested.options.method, 'POST');
  assert.equal(requested.options.timeout, 15000);

  const response = new EventEmitter();
  response.resume = () => {};
  sendResponse(response);
  await Promise.resolve();
  assert.equal(settled, false);
  response.emit('end');
  await stopping;
  assert.equal(settled, true);
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
