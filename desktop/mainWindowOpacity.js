const fs = require('node:fs');

function normalizeOpacity(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('Window opacity must be a finite number');
  }
  return Math.max(0.1, Math.min(1, value));
}

function readMainWindowOpacity(filePath) {
  try {
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    return normalizeOpacity(data?.opacity);
  } catch {
    return 1;
  }
}

function writeMainWindowOpacity(filePath, opacity) {
  fs.writeFileSync(filePath, JSON.stringify({ opacity: normalizeOpacity(opacity) }), 'utf8');
}

module.exports = { normalizeOpacity, readMainWindowOpacity, writeMainWindowOpacity };
