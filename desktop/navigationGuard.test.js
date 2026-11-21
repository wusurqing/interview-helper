const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { mkdtempSync, readFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { runInNewContext } = require('node:vm');

const {
  createSafeWebPreferences,
  installNavigationGuards,
  isAllowedAppOrigin,
  isSafeExternalUrl,
} = require('./windowOptions');

test('createSafeWebPreferences enforces sandbox, contextIsolation and disables nodeIntegration', () => {
  const prefs = createSafeWebPreferences('/app/preload.js');
  assert.equal(prefs.preload, '/app/preload.js');
  assert.equal(prefs.sandbox, true);
  assert.equal(prefs.contextIsolation, true);
  assert.equal(prefs.nodeIntegration, false);
  assert.equal(prefs.backgroundThrottling, false);
});

test('isAllowedAppOrigin allows same-origin HTTP(S) only', () => {
  const allowed = 'http://127.0.0.1:18080';
  assert.equal(isAllowedAppOrigin('http://127.0.0.1:18080/', allowed), true);
  assert.equal(isAllowedAppOrigin('http://127.0.0.1:18080/?overlay=1', allowed), true);
  assert.equal(isAllowedAppOrigin('http://127.0.0.1:18081/', allowed), false);
  assert.equal(isAllowedAppOrigin('http://localhost:18080/', allowed), false);
  assert.equal(isAllowedAppOrigin('https://evil.example/', allowed), false);
  assert.equal(isAllowedAppOrigin('file:///C:/Windows/win.ini', allowed), false);
  assert.equal(isAllowedAppOrigin('javascript:alert(1)', allowed), false);
  assert.equal(isAllowedAppOrigin('not-a-url', allowed), false);
});

test('isSafeExternalUrl allows only http/https URLs with a host', () => {
  assert.equal(isSafeExternalUrl('https://docs.python.org/3/'), true);
  assert.equal(isSafeExternalUrl('http://example.com/path'), true);
  assert.equal(isSafeExternalUrl('javascript:alert(1)'), false);
  assert.equal(isSafeExternalUrl('file:///etc/passwd'), false);
  assert.equal(isSafeExternalUrl('data:text/html,<script>alert(1)</script>'), false);
  assert.equal(isSafeExternalUrl('ms-msdt:/id'), false);
  assert.equal(isSafeExternalUrl(''), false);
});

test('installNavigationGuards denies popups, opens safe external URLs only when allowed, and blocks foreign navigation', () => {
  const opened = [];
  let windowOpenHandler = null;
  const webContents = Object.assign(new EventEmitter(), {
    setWindowOpenHandler(fn) {
      windowOpenHandler = fn;
    },
  });

  installNavigationGuards(webContents, {
    allowedOrigin: 'http://127.0.0.1:18080',
    allowOpenExternal: true,
    openExternal: (url) => opened.push(url),
  });

  assert.deepEqual(windowOpenHandler({ url: 'https://example.com/docs' }), { action: 'deny' });
  assert.deepEqual(windowOpenHandler({ url: 'javascript:alert(1)' }), { action: 'deny' });
  assert.deepEqual(windowOpenHandler({ url: 'file:///C:/secret.txt' }), { action: 'deny' });
  assert.deepEqual(windowOpenHandler({ url: 'http://127.0.0.1:18080/settings' }), { action: 'deny' });
  assert.deepEqual(opened, ['https://example.com/docs']);

  const sameOriginEvent = { prevented: false, preventDefault() { this.prevented = true; } };
  webContents.emit('will-navigate', sameOriginEvent, 'http://127.0.0.1:18080/?tab=jobs');
  assert.equal(sameOriginEvent.prevented, false);

  const foreignEvent = { prevented: false, preventDefault() { this.prevented = true; } };
  webContents.emit('will-navigate', foreignEvent, 'https://evil.example/phish');
  assert.equal(foreignEvent.prevented, true);

  const redirectEvent = { prevented: false, preventDefault() { this.prevented = true; } };
  webContents.emit('will-redirect', redirectEvent, 'file:///etc/passwd');
  assert.equal(redirectEvent.prevented, true);
});

test('main.js configures sandbox and navigation guards on both mainWindow and overlayWindow', (t) => {
  const mainFile = path.join(__dirname, 'main.js');
  const mainSource = readFileSync(mainFile, 'utf8');
  const userData = mkdtempSync(path.join(tmpdir(), 'interview-nav-guard-test-'));
  t.after(() => rmSync(userData, { recursive: true, force: true }));

  const createdWindows = [];
  const externalOpens = [];
  const primary = { id: 1, label: 'Primary', workArea: { x: 0, y: 0, width: 1280, height: 800 } };

  class FakeBrowserWindow extends EventEmitter {
    constructor(options) {
      super();
      this.options = options;
      this.windowOpenHandler = null;
      this.webContents = Object.assign(new EventEmitter(), {
        session: { clearCache: () => Promise.resolve() },
        send() {},
        executeJavaScript: () => Promise.resolve(),
        setWindowOpenHandler: (fn) => { this.windowOpenHandler = fn; },
      });
      createdWindows.push(this);
    }

    isDestroyed() { return false; }
    isVisible() { return false; }
    setContentProtection() {}
    setBackgroundColor() {}
    setAlwaysOnTop() {}
    setVisibleOnAllWorkspaces() {}
    setHiddenInMissionControl() {}
    setIgnoreMouseEvents() {}
    hookWindowMessage() {}
    setFocusable() {}
    setTitle() {}
    loadURL() {}
    show() {}
  }

  const electron = {
    app: {
      commandLine: { appendSwitch() {} },
      getPath: () => userData,
      whenReady: () => new Promise(() => {}),
      on() {},
    },
    BrowserWindow: FakeBrowserWindow,
    globalShortcut: { register: () => true, unregister() {} },
    Tray: function FakeTray() {},
    Menu: { buildFromTemplate: () => ({}) },
    nativeImage: { createEmpty: () => ({}) },
    ipcMain: { handle() {}, on() {} },
    screen: Object.assign(new EventEmitter(), {
      getPrimaryDisplay: () => primary,
      getAllDisplays: () => [primary],
      getCursorScreenPoint: () => ({ x: 400, y: 300 }),
      getDisplayNearestPoint: () => primary,
    }),
    shell: {
      openExternal: (url) => {
        externalOpens.push(url);
        return Promise.resolve();
      },
    },
  };

  const context = {
    __dirname,
    require: (name) => (name === 'electron' ? electron : require(name)),
    process: { platform: 'win32', env: { PORT: '18080' } },
    Date,
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
    Buffer,
    console,
  };
  runInNewContext(mainSource, context, { filename: mainFile });
  runInNewContext('createWindow(); createOverlayWindow();', context);

  assert.equal(createdWindows.length, 2);
  const [mainWin, overlayWin] = createdWindows;

  for (const win of [mainWin, overlayWin]) {
    assert.equal(win.options.webPreferences.sandbox, true);
    assert.equal(win.options.webPreferences.contextIsolation, true);
    assert.equal(win.options.webPreferences.nodeIntegration, false);
    assert.equal(typeof win.windowOpenHandler, 'function');

    const navBlocked = { prevented: false, preventDefault() { this.prevented = true; } };
    win.webContents.emit('will-navigate', navBlocked, 'https://attacker.example/');
    assert.equal(navBlocked.prevented, true);

    const navAllowed = { prevented: false, preventDefault() { this.prevented = true; } };
    win.webContents.emit('will-navigate', navAllowed, 'http://127.0.0.1:18080/');
    assert.equal(navAllowed.prevented, false);
  }

  assert.deepEqual(mainWin.windowOpenHandler({ url: 'https://example.com/help' }), { action: 'deny' });
  assert.deepEqual(mainWin.windowOpenHandler({ url: 'file:///C:/Windows/win.ini' }), { action: 'deny' });
  assert.deepEqual(overlayWin.windowOpenHandler({ url: 'https://example.com/from-overlay' }), { action: 'deny' });
  assert.deepEqual(externalOpens, ['https://example.com/help']);
});
