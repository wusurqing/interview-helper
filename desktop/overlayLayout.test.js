const { test } = require('node:test');
const assert = require('node:assert/strict');
const { canMigrateLayout, readModeLayout, resolveOverlayBounds } = require('./overlayLayout');

const primary = { id: 1, workArea: { x: 0, y: 0, width: 1280, height: 800 } };
const secondary = { id: 2, workArea: { x: -1000, y: 0, width: 1000, height: 800 } };

test('migrates legacy positions just beyond the top and left edges', () => {
  for (const position of [{ x: -15, y: 100 }, { x: 100, y: -40 }]) {
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

test('prefers an exact display match and defers unmatched old positions', () => {
  const state = { position: { x: -15, y: 100 } };
  assert.equal(readModeLayout(state, 'glass', [primary, secondary], {}).displayId, 2);
  assert.equal(canMigrateLayout({ position: { x: -41, y: 100 } }, 'glass', [primary]), false);
  assert.equal(readModeLayout({ ...state, layouts: { glass: { displayId: 42 } } }, 'glass', [primary], {}).displayId, 42);
});
