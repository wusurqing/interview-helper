const test = require('node:test');
const assert = require('node:assert/strict');

const invoked = [];
const listeners = new Map();
const fakeIpcRenderer = {
  invoke(channel, ...args) {
    invoked.push({ channel, args });
    return Promise.resolve();
  },
  send() {},
  sendSync() { return undefined; },
  on(channel, fn) {
    if (!listeners.has(channel)) listeners.set(channel, new Set());
    listeners.get(channel).add(fn);
  },
  removeListener(channel, fn) {
    listeners.get(channel)?.delete(fn);
  },
  emit(channel, ...args) {
    for (const fn of [...(listeners.get(channel) || [])]) {
      fn({}, ...args);
    }
  },
};

const fakeContextBridge = {
  exposed: null,
  exposeInMainWorld(name, api) {
    fakeContextBridge.exposed = { name, api };
  },
};

require.cache[require.resolve('electron')] = {
  id: 'electron',
  filename: 'electron',
  loaded: true,
  exports: {
    contextBridge: fakeContextBridge,
    ipcRenderer: fakeIpcRenderer,
  },
};

delete require.cache[require.resolve('./preload')];
require('./preload');

const api = fakeContextBridge.exposed.api;

test('preload exposes minimizeWindow that invokes minimize-window channel', async () => {
  invoked.length = 0;
  await api.minimizeWindow();
  assert.equal(invoked[0].channel, 'minimize-window');
});

test('preload exposes quitApp that invokes quit-app channel', async () => {
  invoked.length = 0;
  await api.quitApp();
  assert.equal(invoked[0].channel, 'quit-app');
});

test('preload sends main window opacity to the dedicated IPC channel', async () => {
  invoked.length = 0;
  await api.setWindowOpacity(0.42);
  assert.deepEqual(invoked[0], { channel: 'set-window-opacity', args: [0.42] });
});

test('preload exposes all required window control methods', () => {
  assert.equal(typeof api.hideWindow, 'function');
  assert.equal(typeof api.minimizeWindow, 'function');
  assert.equal(typeof api.quitApp, 'function');
  assert.equal(typeof api.showWindow, 'function');
});

test('preload exposes desktop toast and config update listeners', () => {
  assert.equal(typeof api.onDesktopToast, 'function');
  assert.equal(typeof api.onConfigUpdated, 'function');
});

test('removeOverlayStateListener removes listeners registered via onOverlayState wrapper', () => {
  const received = [];
  const listener = (payload) => received.push(payload);

  api.onOverlayState(listener);
  fakeIpcRenderer.emit('overlay-state', { visible: true });
  assert.deepEqual(received, [{ visible: true }]);

  api.removeOverlayStateListener(listener);
  fakeIpcRenderer.emit('overlay-state', { visible: false });
  assert.deepEqual(received, [{ visible: true }]);
  assert.equal(listeners.get('overlay-state')?.size ?? 0, 0);
});

test('onOverlayState returned disposer removes only its own wrapper', () => {
  const receivedA = [];
  const receivedB = [];
  const disposeA = api.onOverlayState((payload) => receivedA.push(payload));
  const listenerB = (payload) => receivedB.push(payload);
  api.onOverlayState(listenerB);

  disposeA();
  fakeIpcRenderer.emit('overlay-state', { mode: 'prompt' });
  assert.deepEqual(receivedA, []);
  assert.deepEqual(receivedB, [{ mode: 'prompt' }]);

  api.removeOverlayStateListener(listenerB);
  fakeIpcRenderer.emit('overlay-state', { mode: 'glass' });
  assert.deepEqual(receivedB, [{ mode: 'prompt' }]);
});
