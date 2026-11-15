const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { normalizeOpacity, readMainWindowOpacity, writeMainWindowOpacity } = require('./mainWindowOpacity');

test('opacity accepts finite numbers and keeps the main window visible', () => {
  assert.equal(normalizeOpacity(0.42), 0.42);
  assert.equal(normalizeOpacity(0), 0.1);
  assert.equal(normalizeOpacity(2), 1);
  for (const value of [NaN, Infinity, -Infinity, '0.5', null]) {
    assert.throws(() => normalizeOpacity(value), TypeError);
  }
});

test('main window opacity survives restart and invalid storage restores full opacity', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ia-main-opacity-'));
  try {
    const file = path.join(dir, 'main-window.json');
    assert.equal(readMainWindowOpacity(file), 1);
    writeMainWindowOpacity(file, 0.35);
    assert.equal(readMainWindowOpacity(file), 0.35);
    fs.writeFileSync(file, '{broken', 'utf8');
    assert.equal(readMainWindowOpacity(file), 1);
    fs.writeFileSync(file, JSON.stringify({ opacity: '0' }), 'utf8');
    assert.equal(readMainWindowOpacity(file), 1);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
