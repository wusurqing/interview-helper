const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { mkdtempSync, readFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { runInNewContext } = require('node:vm');

const mainFile = path.join(__dirname, 'main.js');
const mainSource = readFileSync(mainFile, 'utf8');

function createHarness(t, {
  platform = 'win32',
  env = {},
  existsSyncImpl,
  httpMock,
  clearCacheImpl,
  spawnSyncResult,
  useMockTimers = true,
} = {}) {
  if (useMockTimers) {
    t.mock.timers.enable({ apis: ['Date', 'setInterval', 'setTimeout'], now: 1000 });
  }
  const userData = mkdtempSync(path.join(tmpdir(), 'interview-main-lifecycle-'));
  t.after(() => rmSync(userData, { recursive: true, force: true }));

  const handlers = new Map();
  const appEvents = new EventEmitter();
  const spawnCalls = [];
  const spawnSyncCalls = [];
  const createdWindows = [];
  let trayMenuTemplate = null;

  class MockWindow extends EventEmitter {
    constructor(opts = {}) {
      super();
      this.opts = opts;
      this.bounds = { x: opts.x ?? 0, y: opts.y ?? 0, width: opts.width ?? 800, height: opts.height ?? 600 };
      this.destroyed = false;
      this.visible = false;
      this.loadedUrls = [];
      this.contentProtectionCalls = [];
      this.webContents = Object.assign(new EventEmitter(), {
        send() {},
        setWindowOpenHandler() {},
        executeJavaScript: () => Promise.resolve(),
        session: {
          clearCache: () => (clearCacheImpl ? clearCacheImpl(this) : Promise.resolve()),
        },
      });
      createdWindows.push(this);
    }

    isDestroyed() { return this.destroyed; }
    isVisible() { return this.visible; }
    isAlwaysOnTop() { return false; }
    getOpacity() { return 1; }
    setOpacity() {}
    setTitle() {}
    setAlwaysOnTop() {}
    setContentProtection(val) { this.contentProtectionCalls.push(val); }
    setBackgroundColor() {}
    setVisibleOnAllWorkspaces() {}
    setFocusable() {}
    setIgnoreMouseEvents() {}
    setMinimumSize() {}
    getBounds() { return this.bounds; }
    setBounds(b) { this.bounds = { ...b }; }
    getPosition() { return [this.bounds.x, this.bounds.y]; }
    setPosition(x, y) { this.bounds = { ...this.bounds, x, y }; }
    getNativeWindowHandle() { const b = Buffer.alloc(8); b.writeBigUInt64LE(123n); return b; }
    hookWindowMessage() {}
    loadURL(url) { this.loadedUrls.push(url); }
    show() { this.visible = true; this.emit('show'); }
    showInactive() { this.visible = true; this.emit('show'); }
    hide() { this.visible = false; this.emit('hide'); }
    focus() {}
    blur() {}
    moveTop() {}
    destroy() { this.destroyed = true; this.visible = false; this.emit('closed'); }
  }

  const primary = { id: 1, label: 'Primary', workArea: { x: 0, y: 0, width: 1280, height: 800 } };
  const electron = {
    app: {
      commandLine: { appendSwitch() {} },
      disableHardwareAcceleration() {},
      getPath: () => userData,
      whenReady: () => new Promise(() => {}),
      on(ev, fn) { appEvents.on(ev, fn); },
      quit() {},
    },
    BrowserWindow: MockWindow,
    screen: Object.assign(new EventEmitter(), {
      getPrimaryDisplay: () => primary,
      getAllDisplays: () => [primary],
      getCursorScreenPoint: () => ({ x: 100, y: 100 }),
      getDisplayNearestPoint: () => primary,
    }),
    globalShortcut: {
      register: () => true,
      unregister() {},
      unregisterAll() {},
    },
    ipcMain: {
      handle(channel, handler) { handlers.set(channel, handler); },
      on() {},
    },
    Tray: class {
      setToolTip() {}
      setContextMenu() {}
      on() {}
    },
    Menu: {
      buildFromTemplate(template) {
        trayMenuTemplate = template;
        const menu = Object.assign([...template], {
          getMenuItemById: (id) => template.find((item) => item.id === id) || null,
        });
        return menu;
      },
    },
    nativeImage: {
      createFromBuffer: () => ({}),
      createFromPath: () => ({ resize: () => ({}) }),
    },
    shell: { openExternal: () => Promise.resolve() },
  };

  const fsReal = require('fs');
  const fsMock = {
    ...fsReal,
    existsSync: existsSyncImpl || fsReal.existsSync,
  };

  let nextPid = 4000;
  const childProcessMock = {
    spawn(cmd, args, options) {
      const child = new EventEmitter();
      child.pid = ++nextPid;
      child.exitCode = null;
      child.signalCode = null;
      child.stdout = new EventEmitter();
      child.stderr = new EventEmitter();
      child.stdin = { write: () => true, end: () => {} };
      child.killCalls = [];
      child.kill = (sig) => { child.killCalls.push(sig); };
      spawnCalls.push({ cmd, args, options, child });
      return child;
    },
    spawnSync(cmd, args, options) {
      spawnSyncCalls.push({ cmd, args, options });
      return spawnSyncResult || { status: 0 };
    },
  };

  const context = {
    __dirname,
    console: { log() {}, warn() {}, error() {} },
    require: (name) => {
      if (name === 'electron') return electron;
      if (name === 'child_process') return childProcessMock;
      if (name === 'fs') return fsMock;
      if (name === 'http' && httpMock) return httpMock;
      return require(name);
    },
    process: {
      pid: 1234,
      platform,
      env: { ...env },
      stdout: { write() {} },
      stderr: { write() {} },
    },
    Date,
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
    Buffer,
    Promise,
  };

  runInNewContext(mainSource, context, { filename: mainFile });

  return {
    context,
    handlers,
    appEvents,
    spawnCalls,
    spawnSyncCalls,
    createdWindows,
    getTrayMenu: () => trayMenuTemplate,
    tick: (ms) => t.mock.timers.tick(ms),
  };
}

test('startPythonBackend resolves IA_DESKTOP_PYTHON -> .venv -> python/python3', (t) => {
  const root = path.resolve(__dirname, '..');
  const winVenv = path.join(root, '.venv', 'Scripts', 'python.exe');
  const posixVenv = path.join(root, '.venv', 'bin', 'python');

  const hEnv = createHarness(t, {
    platform: 'win32',
    env: { IA_DESKTOP_PYTHON: 'D:\\custom\\python.exe' },
    existsSyncImpl: () => true,
    useMockTimers: false,
  });
  hEnv.context.startPythonBackend();
  assert.equal(hEnv.spawnCalls[0].cmd, 'D:\\custom\\python.exe');

  const hWinVenv = createHarness(t, {
    platform: 'win32',
    env: {},
    existsSyncImpl: (p) => p === winVenv,
    useMockTimers: false,
  });
  hWinVenv.context.startPythonBackend();
  assert.equal(hWinVenv.spawnCalls[0].cmd, winVenv);

  const hWinFallback = createHarness(t, {
    platform: 'win32',
    env: {},
    existsSyncImpl: () => false,
    useMockTimers: false,
  });
  hWinFallback.context.startPythonBackend();
  assert.equal(hWinFallback.spawnCalls[0].cmd, 'python');

  const hPosixVenv = createHarness(t, {
    platform: 'darwin',
    env: {},
    existsSyncImpl: (p) => p === posixVenv,
    useMockTimers: false,
  });
  hPosixVenv.context.startPythonBackend();
  assert.equal(hPosixVenv.spawnCalls[0].cmd, posixVenv);

  const hPosixFallback = createHarness(t, {
    platform: 'linux',
    env: {},
    existsSyncImpl: () => false,
    useMockTimers: false,
  });
  hPosixFallback.context.startPythonBackend();
  assert.equal(hPosixFallback.spawnCalls[0].cmd, 'python3');
});

test('waitForServer prevents timer doubling on timeout+error and always calls res.resume()', async (t) => {
  const requests = [];
  const resumedStatuses = [];
  const httpMock = {
    get(_url, _opts, cb) {
      const req = new EventEmitter();
      req.destroy = () => {
        // Real Node http.ClientRequest emits 'error' when destroyed after timeout.
        req.emit('error', new Error('socket hang up'));
      };
      requests.push({ req, cb });
      return req;
    },
  };

  const h = createHarness(t, { httpMock });
  const waitPromise = h.context.waitForServer(10000);

  // Attempt 1: timeout triggers destroy() + error -> must schedule ONLY 1 retry
  assert.equal(requests.length, 1);
  requests[0].req.emit('timeout');
  h.tick(300);
  assert.equal(requests.length, 2, 'timeout + error must not double retry timers');

  // Attempt 2: non-200 response -> must call res.resume() and retry once
  const res503 = Object.assign(new EventEmitter(), {
    statusCode: 503,
    resume() { resumedStatuses.push(503); },
  });
  requests[1].cb(res503);
  h.tick(300);
  assert.equal(requests.length, 3);
  assert.deepEqual(resumedStatuses, [503]);

  // Attempt 3: 200 response -> must call res.resume() and resolve
  const res200 = Object.assign(new EventEmitter(), {
    statusCode: 200,
    resume() { resumedStatuses.push(200); },
  });
  requests[2].cb(res200);
  await waitPromise;
  assert.deepEqual(resumedStatuses, [503, 200]);

  // No lingering retries after resolve
  h.tick(1000);
  assert.equal(requests.length, 3);
});

test('postBackend and getBackend enforce request timeouts and settle once', async (t) => {
  const recorded = [];
  const httpMock = {
    request(url, options, _cb) {
      const req = new EventEmitter();
      req.destroyed = false;
      req.write = () => {};
      req.end = () => {};
      req.destroy = () => {
        req.destroyed = true;
        req.emit('error', new Error('socket hang up'));
      };
      recorded.push({ url, options, req });
      return req;
    },
  };

  const h = createHarness(t, { httpMock });

  const postPromise = h.context.postBackend('/api/clear', '{}', 1234);
  assert.equal(recorded[0].options.timeout, 1234);
  recorded[0].req.emit('timeout');
  await assert.rejects(postPromise, /超时/);
  assert.equal(recorded[0].req.destroyed, true);

  const getPromise = h.context.getBackend('/api/config', 2345);
  assert.equal(recorded[1].options.timeout, 2345);
  recorded[1].req.emit('timeout');
  await assert.rejects(getPromise, /超时/);
  assert.equal(recorded[1].req.destroyed, true);
});

test('Windows shutdown uses taskkill /T /F for backend process tree and force-kills hung prompt wheel helper', async (t) => {
  const httpMock = {
    request(_url, _opts, cb) {
      const req = new EventEmitter();
      req.end = () => {
        const res = new EventEmitter();
        res.resume = () => {};
        cb(res);
        res.emit('end');
      };
      req.destroy = () => {};
      return req;
    },
  };

  const h = createHarness(t, { platform: 'win32', httpMock });
  h.context.startPythonBackend();
  const backendChild = h.spawnCalls[0].child;

  const stopPromise = h.context.gracefulStopPython(6000);
  await Promise.resolve();
  await Promise.resolve();

  const killCall = h.spawnCalls.find((call) => call.cmd === 'taskkill');
  assert.ok(killCall, 'expected taskkill to be spawned on Windows');
  assert.deepEqual(Array.from(killCall.args), ['/PID', String(backendChild.pid), '/T', '/F']);

  backendChild.exitCode = 0;
  backendChild.emit('exit', 0);
  await stopPromise;

  // Test hung prompt wheel helper gets force-killed via taskkill after timeout
  const hungHelper = Object.assign(new EventEmitter(), {
    pid: 9876,
    exitCode: null,
    signalCode: null,
    ready: true,
    stdin: { write: () => true, end: () => {} },
    kill: () => {},
  });
  runInNewContext('promptWheelHelper = __hungHelper;', Object.assign(h.context, { __hungHelper: hungHelper }));
  h.context.stopPromptWheelHelper('inactive');
  h.tick(1500);

  const helperKill = h.spawnCalls.find((call) => call.cmd === 'taskkill' && call.args.includes('9876'));
  assert.ok(helperKill, 'expected taskkill fallback for hung prompt wheel helper');
  assert.deepEqual(Array.from(helperKill.args), ['/PID', '9876', '/T', '/F']);
});

test('move-overlay-window validates sender and finite coordinates; sync-overlay-window tolerates null payload', (t) => {
  const h = createHarness(t, { useMockTimers: false });

  // sync-overlay-window must not throw on null / primitive / array
  assert.doesNotThrow(() => h.handlers.get('sync-overlay-window')({}, null));
  assert.doesNotThrow(() => h.handlers.get('sync-overlay-window')({}, 'invalid'));
  assert.doesNotThrow(() => h.handlers.get('sync-overlay-window')({}, [1, 2]));

  // Open overlay window
  h.handlers.get('sync-overlay-window')({}, { visible: true, mode: 'glass' });
  const overlayWin = h.createdWindows.at(-1);
  overlayWin.setPosition(100, 200);

  const move = h.handlers.get('move-overlay-window');
  // Reject non-overlay sender
  assert.equal(move({ sender: {} }, 10, 20).ok, false);
  assert.deepEqual(overlayWin.getPosition(), [100, 200]);

  // Reject non-finite / non-number deltas
  assert.equal(move({ sender: overlayWin.webContents }, NaN, 10).ok, false);
  assert.equal(move({ sender: overlayWin.webContents }, Infinity, 10).ok, false);
  assert.equal(move({ sender: overlayWin.webContents }, '10', 20).ok, false);
  assert.deepEqual(overlayWin.getPosition(), [100, 200]);

  // Accept valid sender and finite deltas
  assert.equal(move({ sender: overlayWin.webContents }, 15.4, -20.6).ok, true);
  assert.deepEqual(overlayWin.getPosition(), [115, 179]);
});

test('tray content protection toggle syncs mainWindow._contentProtection and get-window-state', (t) => {
  const h = createHarness(t, { useMockTimers: false });
  h.context.createWindow();
  h.context.createTray();

  const mainWin = h.createdWindows[0];
  assert.equal(h.handlers.get('get-window-state')().contentProtection, true);

  const stealthItem = h.getTrayMenu().find((item) => item.label === '屏幕共享隐身');
  assert.ok(stealthItem);
  stealthItem.click({ checked: false });

  assert.equal(mainWin._contentProtection, false);
  assert.equal(mainWin.contentProtectionCalls.at(-1), false);
  assert.equal(h.handlers.get('get-window-state')().contentProtection, false);

  stealthItem.click({ checked: true });
  assert.equal(mainWin._contentProtection, true);
  assert.equal(h.handlers.get('get-window-state')().contentProtection, true);
});

test('createWindow clearCache guards against destroyed window and cache rejection', async (t) => {
  let resolveCache;
  const hDestroyed = createHarness(t, {
    useMockTimers: false,
    clearCacheImpl: () => new Promise((res) => { resolveCache = res; }),
  });
  hDestroyed.context.createWindow();
  const win1 = hDestroyed.createdWindows[0];
  win1.destroy();
  resolveCache();
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(win1.loadedUrls, [], 'must not call loadURL on destroyed window');

  const hReject = createHarness(t, {
    useMockTimers: false,
    clearCacheImpl: () => Promise.reject(new Error('disk full')),
  });
  hReject.context.createWindow();
  const win2 = hReject.createdWindows[0];
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(win2.loadedUrls.length, 1, 'should still load URL after clearCache error');
});

test('will-quit force-kills exiting hung promptWheelHelper and falls back when taskkill spawnSync errors', (t) => {
  const h = createHarness(t, {
    platform: 'win32',
    spawnSyncResult: { error: new Error('ENOENT') },
    useMockTimers: false,
  });

  const killCalls = [];
  const hungHelper = Object.assign(new EventEmitter(), {
    pid: 7788,
    exitCode: null,
    signalCode: null,
    ready: true,
    stdin: { write: () => true, end: () => {} },
    kill: (sig) => { killCalls.push(sig); },
  });
  runInNewContext('promptWheelHelper = __hungHelper;', Object.assign(h.context, { __hungHelper: hungHelper }));

  // Stop helper (schedules 1500ms timer) then immediately fire will-quit before 1500ms elapses
  h.context.stopPromptWheelHelper('inactive');
  h.appEvents.emit('will-quit');

  assert.equal(h.spawnSyncCalls.length, 1);
  assert.equal(h.spawnSyncCalls[0].cmd, 'taskkill');
  assert.deepEqual(Array.from(h.spawnSyncCalls[0].args), ['/PID', '7788', '/T', '/F']);
  assert.deepEqual(killCalls, ['SIGKILL'], 'must fall back to proc.kill when spawnSync reports error');
});

test('toggle-content-protection syncs tray checkbox and handles already-exited backend in gracefulStopPython', async (t) => {
  const h = createHarness(t, { useMockTimers: false });
  h.context.createWindow();
  h.context.createTray();

  const stealthItem = h.getTrayMenu().find((item) => item.id === 'content-protection');
  assert.ok(stealthItem);
  assert.equal(stealthItem.checked, true);

  const next = h.handlers.get('toggle-content-protection')();
  assert.equal(next, false);
  assert.equal(stealthItem.checked, false);

  // Already-exited pythonProcess must resolve gracefulStopPython immediately without waiting
  const exitedProc = Object.assign(new EventEmitter(), { pid: 5566, exitCode: 0, signalCode: null, kill: () => {} });
  runInNewContext('pythonProcess = __exited;', Object.assign(h.context, { __exited: exitedProc }));
  await h.context.gracefulStopPython(5000);
});

test('postBackend and getBackend reject when server closes connection before response completes', async (t) => {
  const httpMock = {
    request(_url, _opts, cb) {
      const req = new EventEmitter();
      req.write = () => {};
      req.end = () => {
        const res = new EventEmitter();
        res.statusCode = 200;
        res.complete = false;
        res.setEncoding = () => {};
        cb(res);
        res.emit('close');
      };
      req.destroy = () => {};
      return req;
    },
  };
  const h = createHarness(t, { httpMock, useMockTimers: false });
  await assert.rejects(() => h.context.postBackend('/api/test', '{}'), /连接已中断/);
  await assert.rejects(() => h.context.getBackend('/api/config'), /连接已中断/);
});
