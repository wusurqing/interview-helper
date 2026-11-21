const { app, BrowserWindow, globalShortcut, Tray, Menu, nativeImage, ipcMain, screen, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

// Windows: 透明 BrowserWindow 需要 DWM 硬件加速。
// 历史遗留 disableHardwareAcceleration() 已移除 ——
// 它会让 setContentProtection 退化成「窗口被捕获时显示黑色」(WDA_MONITOR)，
// 而非 Win10 2004+ 才支持的「直接从屏幕捕获中排除」(WDA_EXCLUDEFROMCAPTURE)，
// 后者才是真正的屏幕共享隐身。
// 如个别老 Win7/Win8 设备透明窗口出现渲染异常，可设环境变量 ELECTRON_DISABLE_HW_ACCEL=1 回退。
if (process.platform === 'win32') {
  app.commandLine.appendSwitch('enable-transparent-visuals');
  if (process.env.ELECTRON_DISABLE_HW_ACCEL === '1') {
    app.disableHardwareAcceleration();
  }
}

// 跨平台：阻止系统 occlusion / window list 计算把 overlay 暴露给屏幕共享 / 录屏 API。
// 与 BrowserWindow 的 setContentProtection(true) (+ macOS type:'panel') 形成多重防御。
//
// macOS 真正生效的方案 (Electron PR #34362, refs issue #19880):
// 1. enable ScreenCaptureKitMac —— 切换到 macOS 12.3+ 的 ScreenCaptureKit 实现，
//    PR #34362 在这个 capturer 路径里 *硬编码* 跳过 setContentProtection(true) 的窗口，
//    是目前唯一让 overlay 在「always-on-top + transparent」配置下真正隐身的方案。
// 2. disable IOSurfaceCapturer / DesktopCaptureMacV2 —— 关掉旧 capturer 防回退,
//    它们绕过 NSWindowSharingNone 让 setContentProtection 失效 (这正是 overlay 截图可见的根因)。
// 3. disable CalculateNativeWinOcclusion —— 防 OS 把 overlay 列入 window list,
//    被 CGWindowListCreateImage 类老 API 抓到。
//
// 三件齐备后, ScreenCaptureKit / CGWindowListCreateImage / Lark/钉钉/Zoom 屏幕共享、
// macOS 系统截图 (Cmd+Shift+5) 都不再录到 overlay。
app.commandLine.appendSwitch('enable-features', 'ScreenCaptureKitMac');
app.commandLine.appendSwitch(
  'disable-features',
  'CalculateNativeWinOcclusion,IOSurfaceCapturer,DesktopCaptureMacV2',
);
const {
  ShortcutStatus,
  createShortcutState,
  loadShortcutConfig,
  saveShortcutConfig,
  validateShortcutMap,
} = require('./shortcuts');
const {
  PROMPT_OVERLAY_DEFAULT_WIDTH,
  PROMPT_OVERLAY_MAX_WIDTH,
  consumePendingOverlayShow,
  createOverlayChromeOptions,
  createSafeWebPreferences,
  installNavigationGuards,
  relayChildOutput,
} = require('./windowOptions');
const {
  MODES, MIN_SIZE, defaultLayout, normalizeLayout, readOverlayLayout,
  writeOverlayLayout, canMigrateLayout, readModeLayout, getPromptGrowthBounds, resolveOverlayBounds,
} = require('./overlayLayout');
const { createMultiScreenBatch } = require('./multiScreenBatch');
const { normalizeOpacity, readMainWindowOpacity, writeMainWindowOpacity } = require('./mainWindowOpacity');

const pkg = require('./package.json');

/** 应用显示名：环境变量 ELECTRON_APP_DISPLAY_NAME > desktop/app-title.json > package.json appDisplayName > 默认 */
function loadAppDisplayName() {
  const env = process.env.ELECTRON_APP_DISPLAY_NAME;
  if (env && String(env).trim()) return String(env).trim();
  const titlePath = path.join(__dirname, 'app-title.json');
  try {
    const raw = fs.readFileSync(titlePath, 'utf8');
    const j = JSON.parse(raw);
    if (j && typeof j.appDisplayName === 'string' && j.appDisplayName.trim()) {
      return j.appDisplayName.trim();
    }
  } catch {
    /* 无文件或解析失败 */
  }
  if (pkg.appDisplayName && String(pkg.appDisplayName).trim()) {
    return String(pkg.appDisplayName).trim();
  }
  return '学习助手';
}

const APP_DISPLAY_NAME = loadAppDisplayName();

const ROOT = path.resolve(__dirname, '..');
const BACKEND_DIR = path.join(ROOT, 'backend');
const PORT = parseInt(process.env.PORT || '18080', 10);
const SERVER_URL = `http://127.0.0.1:${PORT}`;
const DEV_FRONTEND_PORT = process.env.IA_DEV_FRONTEND_PORT;
const FRONTEND_URL = DEV_FRONTEND_PORT ? `http://127.0.0.1:${DEV_FRONTEND_PORT}` : SERVER_URL;

let mainWindow = null;
let mainWindowOpacity = 1;
let mainWindowOpacitySaveTimer = null;
let overlayWindow = null;
let tray = null;
let pythonProcess = null;
let isQuitting = false;
let shortcuts = {};
let _overlayDragging = false;
let _blurTimer = null;
let overlayLayoutFileState;
const legacyLayoutDisplays = new Map();
let overlayDisplayId;
let temporaryOverlayPosition;
let measuredOverlaySize;
let overlayToggleBlockedUntil = 0;
let queuedOverlayToggle = false;
const PROMPT_WHEEL_MESSAGE = 0x8000 + 0x179;
let promptWheelHelper = null;
let promptWheelGeneration = 0;
let promptWheelScrollable = false;
let promptWheelStatus = 'inactive';
let lastOverlayState = {
  initialized: false,
  enabled: false,
  visible: false,
  opacity: 0.88,
  fontSize: 14,
  fontColor: '#e2e8f0',
  showBg: true,
  mode: 'glass',
  focusWidthPct: 96,
  focusHeightPct: 90,
  promptMaxWidth: PROMPT_OVERLAY_DEFAULT_WIDTH,
  promptAutoFollow: false,
  promptWheelModifier: 'alt',
  maxLines: 0,
};

function sendPromptWheelStatus(status) {
  if (promptWheelStatus === status) return;
  promptWheelStatus = status;
  for (const win of [mainWindow, overlayWindow]) {
    if (win && !win.isDestroyed()) win.webContents.send('prompt-wheel-status', status);
  }
}

function promptWheelActive() {
  return process.platform === 'win32' && !isQuitting && overlayWindow
    && !overlayWindow.isDestroyed() && overlayWindow._overlayReady
    && overlayWindow.isVisible() && lastOverlayState.visible && getOverlayMode() === 'prompt';
}

function updatePromptWheelState(active = Boolean(promptWheelActive())) {
  if (!promptWheelHelper || !promptWheelHelper.ready) return;
  const win = overlayWindow;
  const handle = win && !win.isDestroyed() ? win.getNativeWindowHandle() : null;
  const hwnd = handle ? (handle.length === 8 ? handle.readBigUInt64LE() : BigInt(handle.readUInt32LE())).toString() : '0';
  const state = {
    type: 'state', active, scrollable: active && promptWheelScrollable,
    modifier: lastOverlayState.promptWheelModifier, hwnd,
    message: PROMPT_WHEEL_MESSAGE, generation: promptWheelGeneration,
  };
  try {
    promptWheelHelper.stdin.write(`${JSON.stringify(state)}\n`);
  } catch {
    if (!promptWheelHelper.stopping) stopPromptWheelHelper('unavailable');
  }
}

function resolvePythonExecutable() {
  const envPython = typeof process.env.IA_DESKTOP_PYTHON === 'string'
    ? process.env.IA_DESKTOP_PYTHON.trim()
    : '';
  if (envPython) return envPython;
  const venvPython = process.platform === 'win32'
    ? path.join(ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  return process.platform === 'win32' ? 'python' : 'python3';
}

function terminateChildProcess(proc, { sync = false } = {}) {
  if (!proc) return;
  if (proc.exitCode !== null && proc.exitCode !== undefined) return;
  if (proc.signalCode !== null && proc.signalCode !== undefined) return;
  if (process.platform === 'win32' && proc.pid) {
    try {
      if (sync && typeof spawnSync === 'function') {
        spawnSync('taskkill', ['/PID', String(proc.pid), '/T', '/F'], {
          windowsHide: true,
          stdio: 'ignore',
          timeout: 3000,
        });
        return;
      }
      const killer = spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], {
        windowsHide: true,
        stdio: 'ignore',
      });
      killer?.on?.('error', () => {
        try { proc.kill(); } catch { /* ignore */ }
      });
      return;
    } catch {
      /* fallback to proc.kill */
    }
  }
  try { proc.kill('SIGKILL'); } catch { /* ignore */ }
}

function stopPromptWheelHelper(status = 'inactive') {
  const helper = promptWheelHelper;
  if (helper) {
    helper.stopping = true;
    updatePromptWheelState(false);
    promptWheelHelper = null;
    promptWheelGeneration += 1;
    clearTimeout(helper.startupTimer);
    clearTimeout(helper.killTimer);
    try { helper.stdin.end(); } catch { /* process may have exited */ }
    if (helper.exitCode == null && helper.signalCode == null) {
      helper.killTimer = setTimeout(() => {
        helper.killTimer = null;
        terminateChildProcess(helper);
      }, 1500);
      helper.once?.('exit', () => {
        clearTimeout(helper.killTimer);
        helper.killTimer = null;
      });
    }
  }
  sendPromptWheelStatus(status);
}

function startPromptWheelHelper() {
  if (!promptWheelActive() || promptWheelHelper) return;
  const python = resolvePythonExecutable();
  let helper;
  try {
    helper = spawn(python, ['-u', path.join(__dirname, 'prompt_mouse_hook.py'), '--parent-pid', String(process.pid)], {
      cwd: ROOT, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch {
    sendPromptWheelStatus('unavailable');
    return;
  }
  promptWheelGeneration += 1;
  promptWheelHelper = helper;
  helper.ready = false;
  helper.startupTimer = setTimeout(() => {
    if (promptWheelHelper === helper && !helper.ready) stopPromptWheelHelper('unavailable');
  }, 5000);
  let output = '';
  helper.stdout.on('data', (chunk) => {
    output += chunk.toString();
    const lines = output.split('\n');
    output = lines.pop();
    for (const line of lines) {
      try {
        const message = JSON.parse(line);
        if (promptWheelHelper !== helper) return;
        if (message.type === 'ready') {
          helper.ready = true;
          clearTimeout(helper.startupTimer);
          sendPromptWheelStatus('ready');
          updatePromptWheelState();
        } else if (message.type === 'error') {
          stopPromptWheelHelper('unavailable');
        }
      } catch { /* ignore malformed helper output */ }
    }
  });
  helper.stderr.on('data', (chunk) => console.warn('prompt mouse helper:', String(chunk)));
  helper.stdin.on?.('error', () => {
    if (promptWheelHelper === helper) stopPromptWheelHelper('unavailable');
  });
  const failed = () => {
    if (promptWheelHelper === helper) stopPromptWheelHelper('unavailable');
  };
  helper.on('error', failed);
  helper.on('exit', failed);
}


let _frontReassertTimer = null;
const FRONT_REASSERT_LEVEL = 1;
const FOCUS_OVERLAY_SHORTCUT_ACTIONS = new Set(['focusPrevTab', 'focusNextTab']);
const VISIBLE_OVERLAY_SHORTCUT_ACTIONS = new Set(['cancelAnswer', 'overlayPrevQuestion', 'overlayNextQuestion']);
const FRONT_REASSERT_DURATION = 5000;
const FRONT_REASSERT_INTERVAL = 500;

function stopFrontReassert() {
  if (_frontReassertTimer === null) return;
  clearInterval(_frontReassertTimer);
  _frontReassertTimer = null;
}

function canReassertOverlayTopMost(win) {
  return win && win === overlayWindow && !win.isDestroyed()
    && lastOverlayState.visible && win.isVisible();
}

function applyTopMost(win) {
  // On Windows, Electron's moveTop() also shows a hidden window.
  if (!canReassertOverlayTopMost(win)) return;
  win.setAlwaysOnTop(true, 'screen-saver', FRONT_REASSERT_LEVEL);
  win.setContentProtection(true);
  win.moveTop();
}

function keepWindowInFront(win) {
  stopFrontReassert();
  if (!canReassertOverlayTopMost(win)) return;
  const start = Date.now();
  applyTopMost(win);
  _frontReassertTimer = setInterval(() => {
    if (!canReassertOverlayTopMost(win) || Date.now() - start > FRONT_REASSERT_DURATION) {
      stopFrontReassert();
      return;
    }
    applyTopMost(win);
  }, FRONT_REASSERT_INTERVAL);
}

function getOverlayStateFilePath() {
  return path.join(app.getPath('userData'), 'overlay-window.json');
}

function getOverlayMode() {
  return lastOverlayState.mode;
}

function getOverlayLayouts() {
  overlayLayoutFileState ??= readOverlayLayout(getOverlayStateFilePath());
  for (const display of screen.getAllDisplays()) legacyLayoutDisplays.set(display.id, display);
  const displays = Array.from(legacyLayoutDisplays.values());
  return Object.fromEntries(MODES.map((mode) => [
    mode, readModeLayout(overlayLayoutFileState, mode, displays, lastOverlayState),
  ]));
}

function getOverlayPlacement(mode, retainDisplay = false) {
  let layout = getOverlayLayouts()[mode];
  if (temporaryOverlayPosition && mode === getOverlayMode()) {
    layout = { ...layout, ...temporaryOverlayPosition };
  }
  const preferredId = (retainDisplay ? overlayDisplayId : undefined) ?? layout.displayId;
  const display = screen.getAllDisplays().find(({ id }) => id === preferredId) ?? screen.getPrimaryDisplay();
  return { layout, display };
}

function getOverlayLayoutSnapshot() {
  const primaryId = screen.getPrimaryDisplay().id;
  const displays = screen.getAllDisplays().map(({ id, label, workArea }) => ({
    id, label, workArea, primary: id === primaryId,
  }));
  const { layout, display } = getOverlayPlacement('prompt', getOverlayMode() === 'prompt' && lastOverlayState.visible);
  const area = display.workArea;
  const bounds = resolveOverlayBounds('prompt', layout, area);
  const width = bounds.width;
  const height = layout.sizeMode === 'auto' ? getPromptGrowthBounds(layout, area).maxHeight : bounds.height;
  return { layouts: getOverlayLayouts(), displays, promptViewport: { sizeMode: layout.sizeMode, width, height } };
}

function sendOverlayLayout() {
  const snapshot = getOverlayLayoutSnapshot();
  for (const win of [mainWindow, overlayWindow]) {
    if (win && !win.isDestroyed()) win.webContents.send('overlay-layout', snapshot);
  }
  return snapshot;
}

function applyOverlayModeBounds({ retainDisplay = false } = {}) {
  if (!overlayWindow || overlayWindow.isDestroyed()) return;
  const mode = getOverlayMode();
  const { layout, display } = getOverlayPlacement(mode, retainDisplay);
  const area = display.workArea;
  const min = MIN_SIZE[mode];
  overlayDisplayId = display.id;
  overlayWindow.setMinimumSize(Math.min(min.width, area.width), Math.min(min.height, area.height));
  const bounds = resolveOverlayBounds(mode, layout, area, layout.sizeMode === 'auto' ? measuredOverlaySize : undefined);
  const previous = overlayWindow.getBounds();
  if (Object.keys(bounds).some((key) => bounds[key] !== previous[key])) {
    overlayWindow.setBounds(bounds, false);
  }
  return bounds;
}

function persistOverlayLayouts(updates = {}) {
  const layouts = getOverlayLayouts();
  const stored = { ...overlayLayoutFileState.layouts, ...updates };
  const displays = Array.from(legacyLayoutDisplays.values());
  for (const mode of MODES) {
    if (!stored[mode] && canMigrateLayout(overlayLayoutFileState, mode, displays)) stored[mode] = layouts[mode];
  }
  if (Object.keys(updates).length === 0 && MODES.every((mode) => stored[mode] === overlayLayoutFileState.layouts?.[mode])) {
    return true;
  }
  // Retain the legacy record for older builds using the same user data directory.
  const next = { ...overlayLayoutFileState, layouts: stored };
  try {
    writeOverlayLayout(getOverlayStateFilePath(), next);
    overlayLayoutFileState = next;
    return true;
  } catch (error) {
    console.warn('saveOverlayLayout failed:', error);
    return false;
  }
}

function setOverlayLayout(mode, patch) {
  if (!MODES.includes(mode) || (patch !== null && (!patch || typeof patch !== 'object' || Array.isArray(patch)))) {
    return { ok: false };
  }
  const next = patch === null ? defaultLayout(mode) : normalizeLayout(mode, patch, getOverlayLayouts()[mode]);
  if (!persistOverlayLayouts({ [mode]: next })) return { ok: false };
  if (mode === getOverlayMode() && lastOverlayState.visible) {
    temporaryOverlayPosition = undefined;
    applyOverlayModeBounds();
  }
  return { ok: true, ...sendOverlayLayout() };
}

function onOverlayDisplayChanged(event, display, metrics) {
  if (lastOverlayState.visible && (!metrics || metrics.some((name) =>
    ['bounds', 'workArea', 'scaleFactor', 'rotation'].includes(name)))) {
    applyOverlayModeBounds({ retainDisplay: true });
  }
  sendOverlayLayout();
}

function registerOverlayDisplayListeners() {
  screen.on('display-removed', onOverlayDisplayChanged);
  screen.on('display-metrics-changed', onOverlayDisplayChanged);
  screen.on('display-added', () => {
    if (lastOverlayState.initialized) persistOverlayLayouts();
    sendOverlayLayout();
  });
}

function createTrayIcon() {
  const size = 16;
  const canvas = nativeImage.createFromBuffer(
    Buffer.alloc(size * size * 4, 0),
    { width: size, height: size }
  );
  return canvas;
}

function waitForServer(timeout = 40000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    let done = false;
    let retryTimer = null;
    const finish = (err) => {
      if (done) return;
      done = true;
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      if (err) reject(err);
      else resolve();
    };
    const scheduleRetry = () => {
      if (done || retryTimer) return;
      if (Date.now() - start > timeout) {
        finish(new Error('Server start timeout'));
        return;
      }
      retryTimer = setTimeout(() => {
        retryTimer = null;
        check();
      }, 300);
    };
    const check = () => {
      if (done) return;
      let attemptSettled = false;
      const onAttemptFailed = () => {
        if (attemptSettled || done) return;
        attemptSettled = true;
        scheduleRetry();
      };
      const req = http.get(`${SERVER_URL}/api/options`, { timeout: 1000 }, (res) => {
        res.resume?.();
        if (attemptSettled || done) return;
        attemptSettled = true;
        if (res.statusCode === 200) {
          finish();
          return;
        }
        scheduleRetry();
      });
      req.on('error', onAttemptFailed);
      req.on('timeout', () => {
        try { req.destroy(); } catch { /* ignore */ }
        onAttemptFailed();
      });
    };
    check();
  });
}

function startPythonBackend() {
  const python = resolvePythonExecutable();
  pythonProcess = spawn(python, [
    path.join(ROOT, 'start.py'),
    '--mode', 'network',
    '--no-build',
    '--port', String(PORT),
  ], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env },
    windowsHide: true,
  });

  relayChildOutput(pythonProcess.stdout, process.stdout, '[py] ');
  relayChildOutput(pythonProcess.stderr, process.stderr, '[py] ');
  pythonProcess.on('close', (code) => {
    console.log(`[py] exited with code ${code}`);
    pythonProcess = null;
    if (!isQuitting) {
      if (mainWindow) {
        const { dialog } = require('electron');
        dialog.showErrorBox('后端已退出', `Python 后端进程异常退出 (code ${code})。\n可能原因：端口 ${PORT} 被占用。\n请关闭占用该端口的进程后重试。`);
      }
      app.quit();
    }
  });
}

function createWindow() {
  const isWindows = process.platform === 'win32';
  const opacitySupported = isWindows || process.platform === 'darwin';
  mainWindowOpacity = opacitySupported
    ? readMainWindowOpacity(path.join(app.getPath('userData'), 'main-window.json'))
    : 1;
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 500,
    title: APP_DISPLAY_NAME,
    frame: false,
    show: false,
    ...(opacitySupported ? { opacity: mainWindowOpacity } : {}),
    // Windows 下不在任务栏显示，只在托盘
    skipTaskbar: isWindows,
    webPreferences: createSafeWebPreferences(path.join(__dirname, 'preload.js')),
  });

  installNavigationGuards(mainWindow.webContents, {
    allowedOrigin: FRONTEND_URL,
    allowOpenExternal: true,
    openExternal: (url) => shell?.openExternal?.(url),
  });

  mainWindow._contentProtection = true;
  mainWindow.setContentProtection(true);

  // 每次启动清除缓存，确保加载到最新的前端构建（避免设置里识别引擎等不更新）
  const win = mainWindow;
  const loadMainWindow = () => {
    if (!win || win.isDestroyed()) return;
    win.loadURL(FRONTEND_URL);
  };
  Promise.resolve(win.webContents.session.clearCache())
    .catch((err) => {
      console.warn('clearCache failed:', err?.message || err);
    })
    .then(loadMainWindow);

  mainWindow.webContents.on('did-finish-load', () => {
    const t = JSON.stringify(APP_DISPLAY_NAME);
    mainWindow.webContents.executeJavaScript(`document.title = ${t}`).catch(() => {});
    mainWindow.setTitle(APP_DISPLAY_NAME);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Windows 下最小化 = 隐藏到托盘
  mainWindow.on('minimize', (e) => {
    if (process.platform === 'win32') {
      e.preventDefault();
      mainWindow.hide();
    }
  });

  // 关闭按钮 = 真正退出
  mainWindow.on('close', () => {
    isQuitting = true;
  });
}

function createOverlayWindow() {
  if (overlayWindow && !overlayWindow.isDestroyed()) {
    return overlayWindow;
  }

  const initialMode = getOverlayMode();
  const { layout, display } = getOverlayPlacement(initialMode);
  const initialBounds = resolveOverlayBounds(initialMode, layout, display.workArea);
  const min = MIN_SIZE[initialMode];

  // 透明浮窗: 视觉上能看到桌面, 需要 transparent: true + alpha=0 背景.
  // 注意: setContentProtection 在 macOS 的透明窗口上只是 best effort,
  // 对部分截图路径 (尤其是 ScreenCaptureKit) 可能无效; 这是 OS 级限制.
  overlayWindow = new BrowserWindow({
    width: initialBounds.width,
    height: initialBounds.height,
    x: initialBounds.x,
    y: initialBounds.y,
    minWidth: Math.min(min.width, display.workArea.width),
    minHeight: Math.min(min.height, display.workArea.height),
    ...createOverlayChromeOptions(process.platform, true),
    maximizable: false,
    minimizable: false,
    fullscreenable: false,
    focusable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    hiddenInMissionControl: true,
    show: false,
    autoHideMenuBar: true,
    roundedCorners: true,
    title: `${APP_DISPLAY_NAME} Overlay`,
    webPreferences: createSafeWebPreferences(path.join(__dirname, 'preload.js')),
  });

  installNavigationGuards(overlayWindow.webContents, {
    allowedOrigin: FRONTEND_URL,
    allowOpenExternal: false,
  });

  // Content protection 必须尽早调用 —— 等到 ready-to-show 时,
  // 窗口可能已经被 window server 登记过一次, 导致 NSWindowSharingNone 漏掉初始帧
  overlayWindow.setContentProtection(true);
  overlayWindow.setBackgroundColor('#00000000');
  overlayWindow.setAlwaysOnTop(true, 'screen-saver', FRONT_REASSERT_LEVEL);
  overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true, skipTransformProcessType: true });
  if (process.platform === 'darwin') {
    // macOS: NSWindowCollectionBehaviorCanJoinAllSpaces + Transient
    // 让窗口不进入 Cmd+Tab, Mission Control, Exposé, 以及 CGWindowList
    try {
      overlayWindow.setHiddenInMissionControl(true);
    } catch { /* older electron */ }
  }

  overlayWindow._overlayReady = false;
  if (process.platform === 'win32') {
    overlayWindow.setIgnoreMouseEvents(initialMode === 'prompt');
    overlayWindow.hookWindowMessage(PROMPT_WHEEL_MESSAGE, (wParam, lParam) => {
      const generation = wParam.length === 8 ? Number(wParam.readBigUInt64LE()) : wParam.readUInt32LE();
      const delta = lParam.readInt32LE();
      if (generation !== promptWheelGeneration || !promptWheelActive() || !promptWheelScrollable) return;
      overlayWindow.webContents.send('prompt-wheel', delta);
    });
  }
  overlayWindow.webContents.on('did-start-loading', () => {
    if (!overlayWindow || overlayWindow.isDestroyed()) return;
    overlayWindow._overlayReady = false;
    promptWheelScrollable = false;
    stopPromptWheelHelper();
  });
  overlayWindow.webContents.on('render-process-gone', () => {
    if (overlayWindow && !overlayWindow.isDestroyed()) overlayWindow._overlayReady = false;
    promptWheelScrollable = false;
    stopPromptWheelHelper('unavailable');
  });

  overlayWindow.once('ready-to-show', () => {
    if (!overlayWindow || overlayWindow.isDestroyed()) return;
    overlayWindow.setContentProtection(true);
    overlayWindow.setBackgroundColor('#00000000');
    overlayWindow.setFocusable(false);
    overlayWindow.setAlwaysOnTop(true, 'screen-saver', FRONT_REASSERT_LEVEL);
  });
  overlayWindow.loadURL(`${FRONTEND_URL}?overlay=1`);
  overlayWindow.webContents.on('did-finish-load', () => {
    if (!overlayWindow || overlayWindow.isDestroyed()) return;
    overlayWindow.setContentProtection(true);
    overlayWindow.setBackgroundColor('#00000000');
    overlayWindow.setFocusable(false);
    if (lastOverlayState) {
      overlayWindow.webContents.send('overlay-state', lastOverlayState);
    }
    overlayWindow.webContents.send('overlay-layout', getOverlayLayoutSnapshot());
    setTimeout(() => {
      if (!overlayWindow || overlayWindow.isDestroyed()) return;
      overlayWindow._overlayReady = true;
      if (consumePendingOverlayShow(overlayWindow, lastOverlayState?.visible)) {
        showOverlayWindow();
      } else if (lastOverlayState?.visible && overlayWindow.isVisible()) {
        startPromptWheelHelper();
      }
    }, process.platform === 'win32' ? 120 : 0);
  });
  overlayWindow.on('show', () => {
    if (!overlayWindow || overlayWindow.isDestroyed()) return;
    overlayWindow.setContentProtection(true);
    overlayWindow.setBackgroundColor('#00000000');
    overlayWindow.setFocusable(false);
  });
  overlayWindow.on('hide', () => {
    stopFrontReassert();
    promptWheelScrollable = false;
    stopPromptWheelHelper();
  });

  overlayWindow.on('closed', () => {
    stopPromptWheelHelper();
    overlayWindow = null;
    measuredOverlaySize = undefined;
    temporaryOverlayPosition = undefined;
    overlayDisplayId = undefined;
    _overlayDragging = false;
    if (_blurTimer) { clearTimeout(_blurTimer); _blurTimer = null; }
    stopFrontReassert();
  });
  overlayWindow.on('focus', () => {
    if (_blurTimer) clearTimeout(_blurTimer);
    _blurTimer = setTimeout(() => {
      _blurTimer = null;
      if (!_overlayDragging && overlayWindow && !overlayWindow.isDestroyed()) {
        overlayWindow.blur();
      }
    }, 150);
  });

  return overlayWindow;
}

function sendOverlayState(payload, trigger) {
  lastOverlayState = payload;
  [mainWindow, overlayWindow].forEach((win) => {
    if (!win || win.isDestroyed()) return;
    win.webContents.send('overlay-state', trigger ? { ...payload, trigger } : payload);
  });
}

function sendShortcutsState() {
  [mainWindow, overlayWindow].forEach((win) => {
    if (!win || win.isDestroyed()) return;
    win.webContents.send('shortcuts-state', shortcuts);
  });
}

function hideOverlayWindow() {
  stopFrontReassert();
  if (!overlayWindow || overlayWindow.isDestroyed()) return;
  overlayWindow._pendingShow = false;
  overlayWindow.hide();
}

function showOverlayWindow() {
  if (!overlayWindow || overlayWindow.isDestroyed()) return;
  if (!lastOverlayState?.visible) {
    hideOverlayWindow();
    return;
  }
  if (!overlayWindow._overlayReady) {
    overlayWindow._pendingShow = true;
    return;
  }
  if (process.platform === 'win32') overlayWindow.setIgnoreMouseEvents(getOverlayMode() === 'prompt');
  temporaryOverlayPosition = undefined;
  applyOverlayModeBounds();
  sendOverlayLayout();
  overlayWindow.setFocusable(false);
  if (process.platform === 'darwin' || process.platform === 'win32') {
    overlayWindow.showInactive();
  } else {
    overlayWindow.show();
  }
  overlayWindow.setContentProtection(true);
  startPromptWheelHelper();
  keepWindowInFront(overlayWindow);
}

function toggleWindow() {
  if (!mainWindow) return;
  if (mainWindow.isVisible()) {
    mainWindow.hide();
  } else {
    mainWindow.show();
    mainWindow.focus();
  }
}

function notifyMainWindow(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function toggleOverlayFromShortcut() {
  const visible = !Boolean(lastOverlayState.visible);
  if (visible) measuredOverlaySize = undefined;
  sendOverlayState({ ...lastOverlayState, enabled: visible || lastOverlayState.enabled, visible }, 'shortcut');
  syncFocusOverlayShortcuts();

  if (visible) {
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible()) {
      mainWindow._hiddenByOverlay = true;
      mainWindow.hide();
    }
    createOverlayWindow();
    showOverlayWindow();
  } else {
    // Ctrl+O only hides the overlay. The end-session IPC may also restore the main window.
    hideOverlayWindow();
  }
}

function notifyDesktopToast(message, level = 'info') {
  if (!message) return;
  notifyMainWindow('desktop-toast', { message: String(message), level });
}

function notifyConfigUpdated(config) {
  if (config && typeof config === 'object') {
    notifyMainWindow('config-updated', config);
  }
}

const BACKEND_GET_TIMEOUT_MS = 15000;
const BACKEND_POST_TIMEOUT_MS = 30000;

function postBackend(pathname, body = '{}', timeoutMs = BACKEND_POST_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };
    const req = http.request(
      `${SERVER_URL}${pathname}`,
      {
        method: 'POST',
        timeout: timeoutMs,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (res) => {
        let raw = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => { raw += chunk; });
        res.on('error', (err) => settle(reject, err));
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            if (!raw) { settle(resolve, { ok: true }); return; }
            try { settle(resolve, JSON.parse(raw)); } catch { settle(resolve, { ok: true }); }
            return;
          }
          let detail;
          try { detail = JSON.parse(raw).detail; } catch { /* Some errors are plain text. */ }
          settle(reject, new Error((typeof detail === 'string' && detail) || raw || res.statusMessage || `HTTP ${res.statusCode}`));
        });
      }
    );
    req.on('timeout', () => {
      settle(reject, new Error(`请求超时 (${pathname})`));
      try { req.destroy(); } catch { /* ignore */ }
    });
    req.on('error', (err) => settle(reject, err));
    req.write(body);
    req.end();
  });
}

function getBackend(pathname, timeoutMs = BACKEND_GET_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };
    const req = http.request(`${SERVER_URL}${pathname}`, { method: 'GET', timeout: timeoutMs }, (res) => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { raw += chunk; });
      res.on('error', (err) => settle(reject, err));
      res.on('end', () => {
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
          if (!raw) { settle(resolve, {}); return; }
          try { settle(resolve, JSON.parse(raw)); } catch { settle(resolve, {}); }
          return;
        }
        settle(reject, new Error(raw || res.statusMessage || `HTTP ${res.statusCode}`));
      });
    });
    req.on('timeout', () => {
      settle(reject, new Error(`请求超时 (${pathname})`));
      try { req.destroy(); } catch { /* ignore */ }
    });
    req.on('error', (err) => settle(reject, err));
    req.end();
  });
}

async function getMultiScreenIdleMs() {
  try {
    const cfg = await getBackend('/api/config');
    const sec = Number(cfg?.multi_screen_capture_idle_sec ?? 10);
    return Math.max(1, Math.min(60, Number.isFinite(sec) ? sec : 10)) * 1000;
  } catch {
    return 10000;
  }
}

const multiServerScreenBatch = createMultiScreenBatch({
  submitImages: (images, writtenExamMode) => (
    postBackend('/api/ask-from-server-screens', JSON.stringify({ images, expected_written_exam_mode: writtenExamMode }))
  ),
  onError: (error) => {
    console.error('flushMultiServerScreenBatch failed:', error);
    notifyDesktopToast(error.message || '连续截图提交失败', 'error');
  },
});

async function addMultiServerScreenShot() {
  try {
    const res = await postBackend('/api/capture-server-screen');
    if (!res?.image) throw new Error('capture response missing image');
    if (typeof res.written_exam_mode !== 'boolean') throw new Error('无法确认截图所属答题模式，请检查后端版本后重新截图。');
    const idleMs = await getMultiScreenIdleMs();
    multiServerScreenBatch.add(res.image, idleMs, res.written_exam_mode);
  } catch (error) {
    console.error('addMultiServerScreenShot failed:', error);
    notifyDesktopToast(error.message || '连续截图失败', 'error');
  }
}

function createTray() {
  const iconPath = path.join(__dirname, 'icon.png');
  let icon;
  try {
    icon = nativeImage.createFromPath(iconPath).resize({ width: 16, height: 16 });
  } catch {
    icon = createTrayIcon();
  }

  tray = new Tray(icon);
  tray.setToolTip(APP_DISPLAY_NAME);

  const contextMenu = Menu.buildFromTemplate([
    { label: '显示窗口', click: () => { mainWindow?.show(); mainWindow?.focus(); } },
    { label: '隐藏到托盘', click: () => toggleWindow() },
    { type: 'separator' },
    {
      label: '窗口置顶',
      type: 'checkbox',
      checked: false,
      click: (menuItem) => {
        mainWindow?.setAlwaysOnTop(menuItem.checked, 'floating');
      },
    },
    {
      label: '屏幕共享隐身',
      type: 'checkbox',
      checked: true,
      click: (menuItem) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow._contentProtection = Boolean(menuItem.checked);
          mainWindow.setContentProtection(mainWindow._contentProtection);
        }
      },
    },
    { type: 'separator' },
    { label: '退出', click: () => { isQuitting = true; app.quit(); } },
  ]);

  tray.setContextMenu(contextMenu);
  tray.on('click', () => { mainWindow?.show(); mainWindow?.focus(); });
  // Windows 双击托盘图标也能显示
  tray.on('double-click', () => { mainWindow?.show(); mainWindow?.focus(); });
}

function isFocusOverlayActive() {
  const mode = lastOverlayState?.mode || (lastOverlayState?.showBg === false ? 'prompt' : 'glass');
  return Boolean(lastOverlayState?.visible) && mode === 'focus';
}

function isOverlayShortcutActive(action) {
  if (FOCUS_OVERLAY_SHORTCUT_ACTIONS.has(action)) return isFocusOverlayActive();
  if (VISIBLE_OVERLAY_SHORTCUT_ACTIONS.has(action)) return Boolean(lastOverlayState?.visible);
  // Non-overlay shortcuts are intentionally global and should stay registered.
  return true;
}

function registerManagedShortcut(shortcut) {
  const callback = shortcutCallbacks[shortcut.action];
  if (!callback) {
    shortcut.status = ShortcutStatus.Available;
    return false;
  }
  if (globalShortcut.register(shortcut.key, callback)) {
    shortcut.status = ShortcutStatus.Registered;
    return true;
  }
  shortcut.status = ShortcutStatus.Failed;
  return false;
}

function registerShortcuts() {
  shortcuts = loadShortcutConfig(app);
  Object.values(shortcuts).forEach((shortcut) => {
    if (!isOverlayShortcutActive(shortcut.action)) {
      shortcut.status = ShortcutStatus.Available;
      return;
    }
    registerManagedShortcut(shortcut);
  });
  syncFocusOverlayShortcuts();
}

function unregisterShortcut(action) {
  const shortcut = shortcuts[action];
  if (!shortcut) return;
  globalShortcut.unregister(shortcut.key);
  shortcut.status = ShortcutStatus.Available;
}

function unregisterAllManagedShortcuts() {
  Object.keys(shortcuts).forEach((action) => unregisterShortcut(action));
}

function syncFocusOverlayShortcuts() {
  for (const action of FOCUS_OVERLAY_SHORTCUT_ACTIONS) {
    const shortcut = shortcuts[action];
    if (!shortcut) continue;
    if (isOverlayShortcutActive(action)) {
      if (shortcut.status !== ShortcutStatus.Registered) registerManagedShortcut(shortcut);
    } else if (shortcut.status === ShortcutStatus.Registered || shortcut.status === ShortcutStatus.Failed) {
      unregisterShortcut(action);
    }
  }
  for (const action of VISIBLE_OVERLAY_SHORTCUT_ACTIONS) {
    const shortcut = shortcuts[action];
    if (!shortcut) continue;
    if (isOverlayShortcutActive(action)) {
      if (shortcut.status !== ShortcutStatus.Registered) registerManagedShortcut(shortcut);
    } else if (shortcut.status === ShortcutStatus.Registered || shortcut.status === ShortcutStatus.Failed) {
      unregisterShortcut(action);
    }
  }
  sendShortcutsState();
}

function registerShortcutSet(nextShortcuts) {
  const validation = validateShortcutMap(nextShortcuts);
  if (!validation.ok) {
    return { ok: false, error: validation.error, shortcuts };
  }

  const prevShortcuts = shortcuts;
  unregisterAllManagedShortcuts();

  const nextState = JSON.parse(JSON.stringify(nextShortcuts));
  let failedKey = null;
  for (const shortcut of Object.values(nextState)) {
    if (!isOverlayShortcutActive(shortcut.action)) {
      shortcut.status = ShortcutStatus.Available;
      continue;
    }
    const callback = shortcutCallbacks[shortcut.action];
    if (!callback) continue;
    if (globalShortcut.register(shortcut.key, callback)) {
      shortcut.status = ShortcutStatus.Registered;
    } else {
      shortcut.status = ShortcutStatus.Failed;
      failedKey = shortcut.key;
      break;
    }
  }

  if (failedKey) {
    Object.values(nextState).forEach((shortcut) => globalShortcut.unregister(shortcut.key));
    shortcuts = prevShortcuts;
    Object.values(shortcuts).forEach((shortcut) => {
      if (!isOverlayShortcutActive(shortcut.action)) {
        shortcut.status = ShortcutStatus.Available;
        return;
      }
      const callback = shortcutCallbacks[shortcut.action];
      if (!callback) return;
      globalShortcut.register(shortcut.key, callback);
      shortcut.status = ShortcutStatus.Registered;
    });
    return { ok: false, error: `快捷键注册失败：${failedKey}`, shortcuts };
  }

  shortcuts = nextState;
  syncFocusOverlayShortcuts();
  saveShortcutConfig(app, shortcuts);
  return { ok: true, shortcuts };
}

const shortcutCallbacks = {
  hideOrShowWindow: () => toggleWindow(),
  hardClearSession: async () => {
    try {
      await postBackend('/api/clear');
    } catch (error) {
      console.error('hardClearSession failed:', error);
    }
  },
  askFromServerScreen: async () => {
    try {
      await postBackend('/api/ask-from-server-screen');
    } catch (error) {
      console.error('askFromServerScreen failed:', error);
      notifyDesktopToast(error.message || '截图答题失败', 'error');
    }
  },
  askScreenForceThink: async () => {
    try {
      await postBackend('/api/ask-from-server-screen', JSON.stringify({ override_think_mode: true }));
      notifyDesktopToast('已提交强制思考截图审题', 'success');
    } catch (error) {
      console.error('askScreenForceThink failed:', error);
      notifyDesktopToast(error?.message || '强制思考截图失败', 'error');
    }
  },
  askScreenForceNoThink: async () => {
    try {
      await postBackend('/api/ask-from-server-screen', JSON.stringify({ override_think_mode: false }));
      notifyDesktopToast('已提交强制非思考截图审题', 'success');
    } catch (error) {
      console.error('askScreenForceNoThink failed:', error);
      notifyDesktopToast(error?.message || '强制非思考截图失败', 'error');
    }
  },
  toggleThinkMode: async () => {
    try {
      const cfg = await getBackend('/api/config');
      let nextPayload;
      let toastMsg;
      if (cfg?.written_exam_mode) {
        const next = !Boolean(cfg.written_exam_think);
        nextPayload = await postBackend('/api/config', JSON.stringify({
          written_exam_think: next,
        }));
        toastMsg = next ? '笔试思考模式已开启' : '笔试思考模式已关闭';
      } else {
        const next = !Boolean(cfg?.think_mode);
        nextPayload = await postBackend('/api/config', JSON.stringify({
          think_mode: next,
          think_effort: next ? 'xhigh' : 'off',
        }));
        toastMsg = next ? '全局思考模式已开启' : '全局思考模式已关闭';
      }
      notifyConfigUpdated(nextPayload);
      notifyDesktopToast(toastMsg, 'success');
    } catch (error) {
      console.error('toggleThinkMode failed:', error);
      notifyDesktopToast(error?.message || '切换思考模式失败', 'error');
    }
  },
  cancelAnswer: async () => {
    try {
      await postBackend('/api/ask/cancel');
    } catch (error) {
      console.error('cancelAnswer failed:', error);
    }
  },
  addMultiServerScreenShot,
  toggleInterviewOverlay: () => {
    // Electron may repeat a global shortcut while the key is held. Debounce
    // the toggle so one physical press cannot hide and immediately re-show it.
    if (Date.now() < overlayToggleBlockedUntil) return;
    overlayToggleBlockedUntil = Date.now() + 350;
    if (mainWindow && !mainWindow.isDestroyed() && !lastOverlayState.initialized) {
      // Main renderer owns legacy preferences and the active mode on startup.
      queuedOverlayToggle = !queuedOverlayToggle;
      return;
    }
    toggleOverlayFromShortcut();
  },
  moveOverlayToMouse: () => {
    if (!overlayWindow || overlayWindow.isDestroyed() || !overlayWindow.isVisible()) return;
    const cursor = screen.getCursorScreenPoint();
    const display = screen.getDisplayNearestPoint(cursor);
    temporaryOverlayPosition = {
      displayId: display.id,
      anchor: 'top-left',
      x: cursor.x - display.workArea.x - 20,
      y: cursor.y - display.workArea.y - 20,
    };
    applyOverlayModeBounds();
    sendOverlayLayout();
  },
  focusPrevTab: () => sendFocusTabCommand('prev'),
  focusNextTab: () => sendFocusTabCommand('next'),
  overlayPrevQuestion: () => sendOverlayQuestionCommand('prev'),
  overlayNextQuestion: () => sendOverlayQuestionCommand('next'),
};

function sendFocusTabCommand(direction) {
  if (!isFocusOverlayActive()) return;
  if (!overlayWindow || overlayWindow.isDestroyed()) return;
  overlayWindow.webContents.send('focus-tab-command', direction);
}

function sendOverlayQuestionCommand(direction) {
  if (!lastOverlayState?.visible) return;
  if (!overlayWindow || overlayWindow.isDestroyed()) return;
  overlayWindow.webContents.send('overlay-question-command', direction);
}

ipcMain.handle('hide-window', () => mainWindow?.hide());
ipcMain.handle('minimize-window', () => mainWindow?.minimize());
ipcMain.handle('quit-app', () => { isQuitting = true; app.quit(); });
ipcMain.handle('show-window', () => { mainWindow?.show(); mainWindow?.focus(); });
ipcMain.handle('get-shortcuts', () => shortcuts);
ipcMain.handle('update-shortcuts', (_event, updates) => {
  const next = JSON.parse(JSON.stringify(shortcuts));
  for (const update of updates || []) {
    if (!update || typeof update.action !== 'string' || typeof update.key !== 'string') continue;
    if (!next[update.action]) continue;
    next[update.action].key = update.key;
  }
  const result = registerShortcutSet(next);
  if (result.ok) sendShortcutsState();
  return result;
});
ipcMain.handle('reset-shortcuts', () => {
  try { fs.unlinkSync(path.join(app.getPath('userData'), 'shortcuts.json')); } catch {}
  const defaults = createShortcutState();
  const result = registerShortcutSet(defaults);
  if (result.ok) sendShortcutsState();
  return result;
});
ipcMain.handle('toggle-always-on-top', () => {
  if (!mainWindow) return false;
  const next = !mainWindow.isAlwaysOnTop();
  mainWindow.setAlwaysOnTop(next, 'floating');
  return next;
});
ipcMain.handle('toggle-content-protection', () => {
  if (!mainWindow) return true;
  const webContents = mainWindow.webContents;
  const isProtected = mainWindow._contentProtection !== false;
  mainWindow._contentProtection = !isProtected;
  mainWindow.setContentProtection(!isProtected);
  return !isProtected;
});
ipcMain.handle('get-window-state', () => ({
  alwaysOnTop: mainWindow?.isAlwaysOnTop() ?? false,
  contentProtection: mainWindow?._contentProtection !== false,
  visible: mainWindow?.isVisible() ?? false,
  opacity: mainWindow && !mainWindow.isDestroyed() ? mainWindow.getOpacity() : mainWindowOpacity,
  opacitySupported: process.platform === 'win32' || process.platform === 'darwin',
}));
ipcMain.handle('set-window-opacity', (_event, requestedOpacity) => {
  if (!mainWindow || mainWindow.isDestroyed()) throw new Error('Main window is unavailable');
  if (process.platform !== 'win32' && process.platform !== 'darwin') {
    throw new Error('Window opacity is not supported on this platform');
  }
  const opacity = normalizeOpacity(requestedOpacity);
  mainWindow.setOpacity(opacity);
  mainWindowOpacity = opacity;
  if (mainWindowOpacitySaveTimer) clearTimeout(mainWindowOpacitySaveTimer);
  mainWindowOpacitySaveTimer = setTimeout(() => {
    mainWindowOpacitySaveTimer = null;
    try {
      writeMainWindowOpacity(path.join(app.getPath('userData'), 'main-window.json'), mainWindowOpacity);
    } catch (error) {
      console.warn('saveMainWindowOpacity failed:', error);
    }
  }, 250);
  return mainWindow.getOpacity();
});
ipcMain.handle('sync-overlay-window', (_event, rawPayload) => {
  const payload = rawPayload && typeof rawPayload === 'object' && !Array.isArray(rawPayload)
    ? rawPayload
    : {};
  const style = {};
  if ('opacity' in payload) {
    const opacity = Number(payload.opacity);
    if (Number.isFinite(opacity)) style.opacity = Math.max(0, Math.min(1, opacity));
  }
  if ('fontSize' in payload) {
    const fontSize = Number(payload.fontSize);
    if (Number.isFinite(fontSize)) style.fontSize = Math.max(10, Math.min(48, Math.round(fontSize)));
  }
  if ('fontColor' in payload && typeof payload.fontColor === 'string' && /^#[0-9a-fA-F]{6}$/.test(payload.fontColor)) {
    style.fontColor = payload.fontColor;
  }
  if ('showBg' in payload && typeof payload.showBg === 'boolean') {
    style.showBg = payload.showBg;
  }
  if ('mode' in payload && ['glass', 'prompt', 'focus'].includes(payload.mode)) {
    style.mode = payload.mode;
    style.showBg = payload.mode !== 'prompt';
  } else if ('showBg' in style && !('mode' in payload)) {
    style.mode = style.showBg ? 'glass' : 'prompt';
  }
  if ('focusWidthPct' in payload) {
    const focusWidthPct = Number(payload.focusWidthPct);
    if (Number.isFinite(focusWidthPct)) style.focusWidthPct = Math.max(50, Math.min(100, Math.round(focusWidthPct)));
  }
  if ('focusHeightPct' in payload) {
    const focusHeightPct = Number(payload.focusHeightPct);
    if (Number.isFinite(focusHeightPct)) style.focusHeightPct = Math.max(35, Math.min(100, Math.round(focusHeightPct)));
  }
  if ('promptMaxWidth' in payload) {
    const promptMaxWidth = Number(payload.promptMaxWidth);
    if (Number.isFinite(promptMaxWidth)) style.promptMaxWidth = Math.max(200, Math.min(PROMPT_OVERLAY_MAX_WIDTH, Math.round(promptMaxWidth)));
  }
  if ('promptAutoFollow' in payload && typeof payload.promptAutoFollow === 'boolean') {
    style.promptAutoFollow = payload.promptAutoFollow;
  }
  if (['alt', 'ctrl', 'shift'].includes(payload.promptWheelModifier)) {
    style.promptWheelModifier = payload.promptWheelModifier;
  }
  if ('maxLines' in payload) {
    const maxLines = Number(payload.maxLines);
    if (Number.isFinite(maxLines)) style.maxLines = Math.max(0, Math.min(50, Math.round(maxLines)));
  }

  const nextEnabled = typeof payload.enabled === 'boolean' ? payload.enabled : Boolean(lastOverlayState.enabled);
  const nextVisible = typeof payload.visible === 'boolean' ? payload.visible : Boolean(lastOverlayState.visible);

  const state = { ...lastOverlayState, ...style, enabled: nextEnabled, visible: nextVisible };
  state.initialized = true;
  const visibleChanged = Boolean(lastOverlayState.visible) !== state.visible;
  // 结构性变化才需要重设窗口 bounds：纯样式（透明度/颜色/自动跟读等）
  // 的 sync 若也走 showOverlayWindow，会把 prompt 自适应宽度重置回满宽。
  const modeChanged = Boolean(lastOverlayState.mode) !== Boolean(state.mode)
    || (lastOverlayState.mode || '') !== (state.mode || '');
  if (visibleChanged || modeChanged) measuredOverlaySize = undefined;
  if (modeChanged) {
    promptWheelScrollable = false;
    stopPromptWheelHelper();
  }

  lastOverlayState = state;
  updatePromptWheelState();
  if ('focusWidthPct' in style || 'promptMaxWidth' in style) persistOverlayLayouts();
  sendOverlayState(state);
  syncFocusOverlayShortcuts();

  if (queuedOverlayToggle) {
    queuedOverlayToggle = false;
    if (typeof payload.visible !== 'boolean') {
      toggleOverlayFromShortcut();
      return { ok: true, visible: lastOverlayState.visible };
    }
  }

  if (visibleChanged) {
    if (state.visible) {
      if (mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible()) {
        mainWindow._hiddenByOverlay = true;
        mainWindow.hide();
      }
    } else {
      if (mainWindow && !mainWindow.isDestroyed() && mainWindow._hiddenByOverlay) {
        mainWindow._hiddenByOverlay = false;
        mainWindow.show();
        mainWindow.focus();
      }
    }
  }

  if (!state.visible) {
    hideOverlayWindow();
    return { ok: true, visible: false };
  }

  if (!overlayWindow || overlayWindow.isDestroyed()) {
    createOverlayWindow();
  }

  if (visibleChanged || modeChanged) {
    showOverlayWindow();
  } else {
    // 纯样式 sync：窗口保持当前 bounds，避免把内容自适应的 prompt 宽度打回满宽
    if (lastOverlayState.visible && !overlayWindow.isVisible() && overlayWindow._overlayReady) {
      overlayWindow.showInactive();
    }
  }
  return { ok: true, visible: true };
});
ipcMain.handle('get-overlay-state', () => lastOverlayState);
ipcMain.handle('get-prompt-wheel-status', () => promptWheelStatus);
ipcMain.handle('report-prompt-scrollable', (event, scrollable) => {
  if (!overlayWindow || event.sender !== overlayWindow.webContents || typeof scrollable !== 'boolean') return { ok: false };
  promptWheelScrollable = scrollable;
  updatePromptWheelState();
  return { ok: true };
});
ipcMain.handle('set-shortcuts-suspended', (_event, on) => {
  // 设置页正在录制新快捷键时暂停全部全局热键：否则按下的组合键会
  // 触发旧绑定（Ctrl+B 藏主窗口 / Ctrl+. 清空会话），录制 UI 直接被带走。
  const suspend = on === true;
  try {
    if (suspend) {
      unregisterAllManagedShortcuts();
    } else {
      registerShortcuts();
    }
  } catch (err) {
    // 恢复失败不阻塞：下次 registerShortcuts 时会重试
  }
  return { ok: true, suspended: suspend };
});
ipcMain.handle('get-overlay-layout', () => getOverlayLayoutSnapshot());
ipcMain.handle('update-overlay-layout', (_event, mode, patch) => setOverlayLayout(mode, patch));
ipcMain.handle('reset-overlay-layout', (_event, mode) => setOverlayLayout(mode, null));
ipcMain.handle('resize-overlay-window', (event, payload = {}) => {
  if (!overlayWindow || overlayWindow.isDestroyed() || event.sender !== overlayWindow.webContents) {
    return { ok: false };
  }
  const { layout } = getOverlayPlacement('prompt', true);
  if (!lastOverlayState.visible || getOverlayMode() !== 'prompt' || layout.sizeMode !== 'auto') {
    return { ok: true, skipped: true };
  }
  if (!Number.isFinite(payload?.width) || !Number.isFinite(payload?.height)) return { ok: false };
  measuredOverlaySize = { width: payload.width, height: payload.height };
  const { width, height } = applyOverlayModeBounds({ retainDisplay: true });
  return { ok: true, width, height };
});
// M3: 添加 destroyOverlay 接口，支持显式销毁悬浮窗
ipcMain.handle('destroy-overlay', () => {
  if (overlayWindow && !overlayWindow.isDestroyed()) {
    overlayWindow.destroy();
    overlayWindow = null;
  }
  return { ok: true };
});
ipcMain.handle('move-overlay-window', (event, dx, dy) => {
  if (!overlayWindow || overlayWindow.isDestroyed() || event?.sender !== overlayWindow.webContents) {
    return { ok: false };
  }
  if (typeof dx !== 'number' || typeof dy !== 'number' || !Number.isFinite(dx) || !Number.isFinite(dy)) {
    return { ok: false };
  }
  const [x, y] = overlayWindow.getPosition();
  overlayWindow.setPosition(x + Math.round(dx), y + Math.round(dy));
  return { ok: true };
});
ipcMain.on('overlay-drag-start', (event) => {
  _overlayDragging = true;
  if (_blurTimer) { clearTimeout(_blurTimer); _blurTimer = null; }
  event.returnValue = true;
});
ipcMain.on('overlay-drag-end', () => {
  _overlayDragging = false;
  if (_blurTimer) { clearTimeout(_blurTimer); _blurTimer = null; }
  if (overlayWindow && !overlayWindow.isDestroyed()) overlayWindow.blur();
});

function createAppMenu() {
  if (process.platform !== 'darwin') return;
  const template = [
    {
      label: app.name,
      submenu: [
        { role: 'about', label: `关于 ${app.name}` },
        { type: 'separator' },
        { label: '隐藏/显示窗口', click: () => toggleWindow() },
        { type: 'separator' },
        { role: 'hide', label: '隐藏应用' },
        { role: 'unhide', label: '显示应用' },
        { type: 'separator' },
        { label: '退出', accelerator: 'CommandOrControl+Q', click: () => { isQuitting = true; app.quit(); } },
      ],
    },
    {
      label: '编辑',
      submenu: [
        { role: 'undo' }, { role: 'redo' }, { type: 'separator' },
        { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function requestAssistStop(timeoutMs = 12000) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    const req = http.request(`${SERVER_URL}/api/stop`, {
      method: 'POST',
      timeout: timeoutMs,
    }, (res) => {
      res.resume();
      res.on('end', finish);
      res.on('close', finish);
    });
    req.on('error', finish);
    req.on('timeout', () => {
      try { req.destroy(); } catch (_) { /* ignore */ }
      finish();
    });
    req.end();
  });
}

// 先等 /api/stop 释放 MMPG 音频和云端会话，并完成复盘归档，再停止后端进程。
let pythonStopPromise = null;
function gracefulStopPython(timeoutMs = 20000) {
  if (pythonStopPromise) return pythonStopPromise;
  const proc = pythonProcess;
  if (!proc) return Promise.resolve();
  pythonStopPromise = new Promise((resolve) => {
    let settled = false;
    const finish = () => { if (settled) return; settled = true; resolve(); };
    proc.once('exit', finish);
    const stopTimeoutMs = Math.max(1000, timeoutMs - 5000);
    requestAssistStop(stopTimeoutMs).then(() => {
      if (settled) return;
      if (process.platform === 'win32') {
        terminateChildProcess(proc);
      } else {
        try { proc.kill('SIGTERM'); } catch (err) { console.warn('[py] SIGTERM failed:', err.message); }
      }
      const escalateTimer = setTimeout(() => {
        if (settled) return;
        try {
          if (proc.exitCode == null && proc.signalCode == null) {
            console.warn('[py] graceful timeout, escalating to SIGKILL');
            terminateChildProcess(proc);
          }
        } catch (err) {
          console.warn('[py] SIGKILL failed:', err.message);
        }
        finish();
      }, Math.max(1000, timeoutMs - stopTimeoutMs));
      proc.once?.('exit', () => clearTimeout(escalateTimer));
    });
  });
  return pythonStopPromise;
}

app.on('before-quit', (event) => {
  isQuitting = true;
  stopPromptWheelHelper();
  if (!pythonProcess) return;
  event.preventDefault();
  if (pythonStopPromise) return;
  gracefulStopPython().then(() => {
    pythonProcess = null;
    app.quit();
  });
});

app.whenReady().then(async () => {
  registerOverlayDisplayListeners();
  try {
    app.setName(APP_DISPLAY_NAME);
  } catch {
    /* 个别平台/版本可能不支持 */
  }
  createAppMenu();
  if (!DEV_FRONTEND_PORT) {
    console.log('Starting Python backend...');
    startPythonBackend();
  }

  try {
    await waitForServer();
    console.log('Backend ready, creating window...');
  } catch (err) {
    console.error('Failed to start backend:', err.message);
    app.quit();
    return;
  }

  createWindow();
  createTray();
  registerShortcuts();

  setImmediate(() => {
    if (overlayWindow && !overlayWindow.isDestroyed()) return;
    try {
      createOverlayWindow();
    } catch (error) {
      console.warn('overlay preheat failed:', error?.message || error);
    }
  });
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  mainWindow?.show();
  mainWindow?.focus();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  stopFrontReassert();
  if (mainWindowOpacitySaveTimer) {
    clearTimeout(mainWindowOpacitySaveTimer);
    mainWindowOpacitySaveTimer = null;
    try {
      writeMainWindowOpacity(path.join(app.getPath('userData'), 'main-window.json'), mainWindowOpacity);
    } catch (error) {
      console.warn('saveMainWindowOpacity failed:', error);
    }
  }
  if (overlayWindow && !overlayWindow.isDestroyed()) {
    overlayWindow.destroy();
    overlayWindow = null;
  }
  // 兜底:before-quit 通常已经 graceful 停过 pythonProcess,
  // 这里 fallback 防止异常路径泄漏子进程。
  if (pythonProcess) {
    terminateChildProcess(pythonProcess, { sync: true });
    pythonProcess = null;
  }
});
