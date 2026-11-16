const fs = require('node:fs');

const MODES = ['glass', 'prompt', 'focus'];
const ROWS = ['top', 'middle', 'bottom'];
const COLUMNS = ['left', 'center', 'right'];
const MIN_SIZE = {
  glass: { width: 300, height: 100 },
  prompt: { width: 180, height: 72 },
  focus: { width: 480, height: 240 },
};

function defaultLayout(mode) {
  return {
    displayId: null,
    anchor: mode === 'focus' ? 'middle-center' : 'top-center',
    x: 0,
    y: mode === 'focus' ? 0 : 24,
    sizeMode: mode === 'focus' ? 'percent' : mode === 'prompt' ? 'auto' : 'fixed',
    width: mode === 'prompt' ? 820 : mode === 'focus' ? 960 : 480,
    height: mode === 'focus' ? 600 : 320,
    ...(mode === 'focus' && { widthPct: 96, heightPct: 90 }),
    ...(mode === 'prompt' && { maxWidth: 820, maxHeight: 420 }),
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function integer(value, fallback, min, max) {
  return Number.isFinite(value) ? clamp(Math.round(value), min, max) : fallback;
}

function normalizeLayout(mode, input, fallback = defaultLayout(mode)) {
  const { displayId, anchor, x, y, sizeMode, width, height, widthPct, heightPct, maxWidth, maxHeight } = input ?? {};
  const [row, column, extra] = typeof anchor === 'string' ? anchor.split('-') : [];
  const sizes = mode === 'prompt' ? ['fixed', 'auto'] : mode === 'focus' ? ['fixed', 'percent'] : ['fixed'];
  const min = MIN_SIZE[mode];
  return {
    displayId: displayId === null || Number.isSafeInteger(displayId) ? displayId : fallback.displayId,
    anchor: ROWS.includes(row) && COLUMNS.includes(column) && extra === undefined ? anchor : fallback.anchor,
    x: integer(x, fallback.x, -32768, 32768),
    y: integer(y, fallback.y, -32768, 32768),
    sizeMode: sizes.includes(sizeMode) ? sizeMode : fallback.sizeMode,
    width: integer(width, fallback.width, min.width, 32768),
    height: integer(height, fallback.height, min.height, 32768),
    ...(mode === 'focus' && {
      widthPct: integer(widthPct, fallback.widthPct, 50, 100),
      heightPct: integer(heightPct, fallback.heightPct, 35, 100),
    }),
    ...(mode === 'prompt' && {
      maxWidth: integer(maxWidth, fallback.maxWidth, min.width, 32768),
      maxHeight: integer(maxHeight, fallback.maxHeight, min.height, 32768),
    }),
  };
}

function readOverlayLayout(file) {
  try {
    const state = JSON.parse(fs.readFileSync(file, 'utf8'));
    return state && typeof state === 'object' && !Array.isArray(state) ? state : {};
  } catch {
    return {};
  }
}

function writeOverlayLayout(file, state) {
  const tempFile = `${file}.tmp`;
  try {
    fs.writeFileSync(tempFile, JSON.stringify(state, null, 2), 'utf8');
    fs.renameSync(tempFile, file);
  } catch (error) {
    try { fs.unlinkSync(tempFile); } catch { /* already gone */ }
    throw error;
  }
}

function displayAt(displays, x, y) {
  return displays.find(({ workArea: area }) =>
    x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height);
}

function canMigrateLayout(state, mode, displays) {
  const { x, y } = state.position ?? {};
  return mode === 'focus' || !Number.isFinite(x) || !Number.isFinite(y) || Boolean(displayAt(displays, x, y));
}

function readModeLayout(state, mode, displays, legacyPrefs) {
  if (state.layouts?.[mode]) return normalizeLayout(mode, state.layouts[mode]);

  const previous = state.position;
  const display = previous && mode !== 'focus' ? displayAt(displays, previous.x, previous.y) : null;
  return normalizeLayout(mode, {
    ...(display && {
      displayId: display.id,
      anchor: 'top-left',
      x: previous.x - display.workArea.x,
      y: previous.y - display.workArea.y,
    }),
    ...(mode === 'glass' && {
      width: previous?.w > 0 ? previous.w : undefined,
      height: previous?.h > 0 ? previous.h : undefined,
    }),
    ...(mode === 'focus' && { widthPct: legacyPrefs.focusWidthPct, heightPct: legacyPrefs.focusHeightPct }),
    ...(mode === 'prompt' && { maxWidth: legacyPrefs.promptMaxWidth }),
  });
}

function resolveOverlayBounds(mode, layout, area, measured) {
  const margin = layout.sizeMode === 'percent' ? 14 : 0;
  const usableWidth = Math.max(1, area.width - margin * 2);
  const usableHeight = Math.max(1, area.height - margin * 2);
  const minWidth = Math.min(MIN_SIZE[mode].width, usableWidth);
  const minHeight = Math.min(MIN_SIZE[mode].height, usableHeight);
  let envelopeWidth = layout.width;
  let envelopeHeight = layout.height;
  if (layout.sizeMode === 'auto') {
    envelopeWidth = layout.maxWidth;
    envelopeHeight = layout.maxHeight;
  } else if (layout.sizeMode === 'percent') {
    envelopeWidth = Math.round(usableWidth * layout.widthPct / 100);
    envelopeHeight = Math.round(usableHeight * layout.heightPct / 100);
  }
  envelopeWidth = clamp(envelopeWidth, minWidth, usableWidth);
  envelopeHeight = clamp(envelopeHeight, minHeight, usableHeight);

  const width = integer(measured?.width, envelopeWidth, minWidth, envelopeWidth);
  const height = integer(measured?.height, envelopeHeight, minHeight, envelopeHeight);
  const [row, column] = layout.anchor.split('-');
  const horizontal = COLUMNS.indexOf(column) / 2;
  const vertical = ROWS.indexOf(row) / 2;
  const left = clamp(Math.round((usableWidth - width) * horizontal) + layout.x, 0, usableWidth - width);
  const top = clamp(Math.round((usableHeight - height) * vertical) + layout.y, 0, usableHeight - height);
  return {
    x: area.x + margin + left,
    y: area.y + margin + top,
    width,
    height,
  };
}

module.exports = {
  MODES, MIN_SIZE, defaultLayout, normalizeLayout, readOverlayLayout, writeOverlayLayout,
  canMigrateLayout, readModeLayout, resolveOverlayBounds,
};
