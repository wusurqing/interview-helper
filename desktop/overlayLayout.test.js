const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { canMigrateLayout, readModeLayout, readOverlayLayout, writeOverlayLayout, resolveOverlayBounds } = require('./overlayLayout');

const primary = { id: 1, workArea: { x: 0, y: 0, width: 1280, height: 800 } };
const secondary = { id: 2, workArea: { x: -1000, y: 0, width: 1000, height: 800 } };

test('migrates legacy positions just beyond the top and left edges', () => {
  for (const position of [{ x: -15, y: 100 }, { x: -40, y: 100 }, { x: 100, y: -40 }]) {
    const state = { position: { ...position, w: 510, h: 220 } };
    for (const mode of ['glass', 'prompt']) {
      assert.equal(canMigrateLayout(state, mode, [primary]), true);
      const layout = readModeLayout(state, mode, [primary], {});
      assert.equal(layout.displayId, 1);
      assert.equal(layout.x, position.x);
      assert.equal(layout.y, position.y);
      const bounds = resolveOverlayBounds(mode, layout, primary.workArea);
      assert.ok(bounds.x >= 0 && bounds.y >= 0);
    }
  }
});

test('chooses the lowest display id when legacy candidates are equally near', () => {
  const duplicate = { id: 5, workArea: primary.workArea };
  const state = { position: { x: -10, y: -10 } };
  assert.equal(readModeLayout(state, 'glass', [duplicate, primary], {}).displayId, 1);
});

test('migrated edge positions survive saving and loading', (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), 'overlay-layout-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, 'overlay-window.json');
  const state = { position: { x: -15, y: 100, w: 510, h: 220 } };
  const layouts = Object.fromEntries(['glass', 'prompt'].map((mode) =>
    [mode, readModeLayout(state, mode, [primary], {})]));
  writeOverlayLayout(file, { ...state, layouts });
  const saved = readOverlayLayout(file);
  for (const mode of ['glass', 'prompt']) {
    const restored = readModeLayout(saved, mode, [primary], {});
    assert.equal(restored.displayId, 1);
    assert.equal(restored.x, -15);
    assert.equal(resolveOverlayBounds(mode, restored, primary.workArea).x, 0);
  }
});

test('prefers an exact display match and defers unmatched old positions', () => {
  const state = { position: { x: -15, y: 100 } };
  assert.equal(readModeLayout(state, 'glass', [primary, secondary], {}).displayId, 2);
  assert.equal(canMigrateLayout({ position: { x: -41, y: 100 } }, 'glass', [primary]), false);
  assert.equal(readModeLayout({ ...state, layouts: { glass: { displayId: 42 } } }, 'glass', [primary], {}).displayId, 42);
});
