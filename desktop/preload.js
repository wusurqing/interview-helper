const { contextBridge, ipcRenderer } = require('electron');

const overlayStateWrappers = new Map();

contextBridge.exposeInMainWorld('electronAPI', {
  hideWindow: () => ipcRenderer.invoke('hide-window'),
  minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
  quitApp: () => ipcRenderer.invoke('quit-app'),
  showWindow: () => ipcRenderer.invoke('show-window'),
  getShortcuts: () => ipcRenderer.invoke('get-shortcuts'),
  updateShortcuts: (shortcuts) => ipcRenderer.invoke('update-shortcuts', shortcuts),
  resetShortcuts: () => ipcRenderer.invoke('reset-shortcuts'),
  toggleAlwaysOnTop: () => ipcRenderer.invoke('toggle-always-on-top'),
  toggleContentProtection: () => ipcRenderer.invoke('toggle-content-protection'),
  getWindowState: () => ipcRenderer.invoke('get-window-state'),
  setWindowOpacity: (opacity) => ipcRenderer.invoke('set-window-opacity', opacity),
  syncOverlayWindow: (payload) => ipcRenderer.invoke('sync-overlay-window', payload),
  resizeOverlayWindow: (payload) => ipcRenderer.invoke('resize-overlay-window', payload),
  isWindows: process.platform === 'win32',
  reportPromptScrollable: (scrollable) => ipcRenderer.invoke('report-prompt-scrollable', scrollable),
  getPromptWheelStatus: () => ipcRenderer.invoke('get-prompt-wheel-status'),
  onPromptWheel: (listener) => {
    const wrapped = (_event, delta) => listener(delta);
    ipcRenderer.on('prompt-wheel', wrapped);
    return () => ipcRenderer.removeListener('prompt-wheel', wrapped);
  },
  onPromptWheelStatus: (listener) => {
    const wrapped = (_event, status) => listener(status);
    ipcRenderer.on('prompt-wheel-status', wrapped);
    return () => ipcRenderer.removeListener('prompt-wheel-status', wrapped);
  },
  getOverlayLayout: () => ipcRenderer.invoke('get-overlay-layout'),
  updateOverlayLayout: (mode, patch) => ipcRenderer.invoke('update-overlay-layout', mode, patch),
  resetOverlayLayout: (mode) => ipcRenderer.invoke('reset-overlay-layout', mode),
  onOverlayLayout: (listener) => {
    const wrapped = (_event, payload) => listener(payload);
    ipcRenderer.on('overlay-layout', wrapped);
    return () => ipcRenderer.removeListener('overlay-layout', wrapped);
  },
  setShortcutsSuspended: (on) => ipcRenderer.invoke('set-shortcuts-suspended', on),
  destroyOverlay: () => ipcRenderer.invoke('destroy-overlay'),
  moveOverlayWindow: (dx, dy) => ipcRenderer.invoke('move-overlay-window', dx, dy),
  overlayDragStart: () => ipcRenderer.sendSync('overlay-drag-start'),
  overlayDragEnd: () => ipcRenderer.send('overlay-drag-end'),
  getOverlayState: () => ipcRenderer.invoke('get-overlay-state'),
  onOverlayState: (listener) => {
    if (typeof listener !== 'function') return () => {};
    const wrapped = (_event, payload) => listener(payload);
    let wrappers = overlayStateWrappers.get(listener);
    if (!wrappers) {
      wrappers = new Set();
      overlayStateWrappers.set(listener, wrappers);
    }
    wrappers.add(wrapped);
    ipcRenderer.on('overlay-state', wrapped);
    return () => {
      ipcRenderer.removeListener('overlay-state', wrapped);
      wrappers.delete(wrapped);
      if (wrappers.size === 0) overlayStateWrappers.delete(listener);
    };
  },
  onShortcuts: (listener) => {
    const wrapped = (_event, payload) => listener(payload);
    ipcRenderer.on('shortcuts-state', wrapped);
    return () => ipcRenderer.removeListener('shortcuts-state', wrapped);
  },
  onFocusTabCommand: (listener) => {
    const wrapped = (_event, payload) => listener(payload);
    ipcRenderer.on('focus-tab-command', wrapped);
    return () => ipcRenderer.removeListener('focus-tab-command', wrapped);
  },
  onOverlayQuestionCommand: (listener) => {
    const wrapped = (_event, payload) => listener(payload);
    ipcRenderer.on('overlay-question-command', wrapped);
    return () => ipcRenderer.removeListener('overlay-question-command', wrapped);
  },
  onDesktopToast: (listener) => {
    const wrapped = (_event, payload) => listener(payload);
    ipcRenderer.on('desktop-toast', wrapped);
    return () => ipcRenderer.removeListener('desktop-toast', wrapped);
  },
  onConfigUpdated: (listener) => {
    const wrapped = (_event, payload) => listener(payload);
    ipcRenderer.on('config-updated', wrapped);
    return () => ipcRenderer.removeListener('config-updated', wrapped);
  },
  removeOverlayStateListener: (listener) => {
    if (!listener) return;
    const wrappers = overlayStateWrappers.get(listener);
    if (wrappers) {
      for (const wrapped of wrappers) {
        ipcRenderer.removeListener('overlay-state', wrapped);
      }
      overlayStateWrappers.delete(listener);
      return;
    }
    ipcRenderer.removeListener('overlay-state', listener);
  },
});
