import VERTC, { AudioSourceType, MediaType, StreamIndex } from '@volcengine/rtc';

const el = Object.fromEntries([
  'status', 'config-summary', 'error', 'controls', 'audio-source', 'silence-ms',
  'silence-range', 'aivad', 'twopass', 'source-help', 'start', 'stop', 'mark-end',
  'duration', 'partial-text', 'partial-state', 'final-count', 'final-empty',
  'final-results', 'timeline',
].map((id) => [id, document.getElementById(id)]));

let config = null;
let currentRun = null;
let finalCount = 0;

function status(text, kind = '') {
  el.status.textContent = text;
  el.status.className = `badge ${kind}`;
}

function errorMessage(text) {
  el.error.textContent = text;
  el.error.hidden = !text;
}

function timeline(name, detail = '', message = null) {
  const row = document.createElement('li');
  const time = document.createElement('time');
  time.textContent = Number.isFinite(message?.elapsed_ms)
    ? `后端 +${(message.elapsed_ms / 1000).toFixed(3)}s`
    : `本地 ${new Date().toLocaleTimeString('zh-CN', { hour12: false })}`;
  if (message?.received_at) time.title = String(message.received_at);
  const content = document.createElement('div');
  const title = document.createElement('span');
  title.className = 'event-name';
  title.textContent = name;
  content.append(title);
  if (detail) content.append(document.createTextNode(` · ${String(detail).slice(0, 500)}`));
  row.append(time, content);
  el.timeline.prepend(row);
  while (el.timeline.children.length > 250) el.timeline.lastElementChild.remove();
}

function updateControls() {
  const busy = Boolean(currentRun);
  el.controls.disabled = busy;
  el.start.disabled = busy || !config?.configured;
  el.stop.disabled = !busy || Boolean(currentRun?.cancelled);
  el['mark-end'].disabled = !currentRun?.ready || Boolean(currentRun?.cancelled);
}

function publicError(error) {
  if (error?.probeMessage) return error.probeMessage;
  const labels = {
    NotAllowedError: '未获得音频权限，或已取消分享。请允许麦克风或选择带音频的分享来源。',
    NotFoundError: '没有找到可用的音频设备。',
    NotReadableError: '音频设备无法读取，可能正被其他程序独占。',
    InvalidStateError: '浏览器无法开始采集，请保持页面前台并重新点击开始。',
    AbortError: '浏览器中断了音频采集，请重试。',
  };
  if (labels[error?.name]) return labels[error.name];
  // Never display an arbitrary SDK error object: it may contain request credentials.
  const code = String(error?.errorCode ?? error?.code ?? '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 70);
  return `启动失败${code ? `（${code}）` : ''}。请检查 RTC Token、房间与用户是否匹配，以及浏览器网络。`;
}

function normalizeSilenceMs(rawValue) {
  const text = String(rawValue ?? '').trim();
  const parsed = text === '' ? 800 : Number(text);
  const clamped = Math.min(2999, Math.max(500, Number.isFinite(parsed) ? parsed : 800));
  return Math.round(clamped);
}

function fail(message) {
  return Object.assign(new Error(message), { probeMessage: message });
}

function ensureActive(run) {
  if (run.cancelled || currentRun !== run) throw fail('本次启动已取消。');
}

function send(run, message) {
  if (run.socket?.readyState !== WebSocket.OPEN) throw fail('本机诊断连接已断开，请停止后重新开始。');
  run.socket.send(JSON.stringify(message));
}

function connectBackend(run) {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/ws/probe`);
    run.socket = socket;
    const timer = setTimeout(() => {
      reject(fail('连接本机诊断服务超时。'));
      socket.close();
    }, 10000);
    socket.addEventListener('open', () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
    socket.addEventListener('error', () => {
      clearTimeout(timer);
      reject(fail('无法连接本机诊断服务。'));
    }, { once: true });
    socket.addEventListener('message', ({ data }) => {
      if (run.cancelled || currentRun !== run) return;
      try { handleMessage(run, JSON.parse(data)); }
      catch { timeline('诊断消息无法解析'); }
    });
    socket.addEventListener('close', () => {
      clearTimeout(timer);
      reject(fail('本机诊断连接已关闭。'));
      if (!run.cancelled && currentRun === run) {
        errorMessage('本机诊断连接已关闭，已停止音频上行。');
        void stopRun(run, '诊断连接关闭');
      }
    });
  });
}

function handleMessage(run, message) {
  if (message.type === 'status') {
    timeline(`服务端 ${message.status}`, '', message);
    if (message.status === 'session_created') {
      run.ready = true;
      clearTimeout(run.startTimeout);
      run.readyAt = performance.now();
      status('可以讲话 · MMPG 已就绪', 'ready');
      updateControls();
      run.durationTimer = setInterval(() => {
        const seconds = Math.floor((performance.now() - run.readyAt) / 1000);
        el.duration.textContent = `已运行 ${seconds} 秒`;
        if (seconds >= (config.maxSessionSeconds || 1800)) void stopRun(run, '达到单次时长上限');
      }, 1000);
    } else if (message.status === 'stopped') {
      void stopRun(run, '服务端已停止');
    } else if (message.status === 'cloud_connected') {
      status('云端已连接 · 等待会话就绪', 'busy');
    }
    return;
  }
  if (message.type === 'error') {
    errorMessage(String(message.message || '服务端报告错误，请查看本机日志。'));
    timeline('服务端错误', message.code || '', message);
    void stopRun(run, '错误后停止');
    return;
  }
  if (message.type === 'mark_end_ack') {
    timeline('我说完了 · 后端已记录', message.associated === false
      ? '当前没有未结束的云端话轮；仅记录时间，不关联下一轮'
      : '已关联当前话轮；这是人工时间标记，不会触发判停', message);
    return;
  }
  if (message.type === 'done') {
    if (message.duplicate) return;
    const transcript = String(message.transcript || '').trim();
    if (!transcript) return;
    finalCount += 1;
    el['final-count'].textContent = `${finalCount} 轮`;
    el['final-empty'].hidden = true;
    const card = document.createElement('article');
    card.className = 'final-item';
    const meta = document.createElement('div');
    meta.className = 'final-meta';
    meta.textContent = `第 ${finalCount} 轮${message.conv_id ? ` · ${message.conv_id}` : ''}`;
    const text = document.createElement('p');
    text.className = 'transcript';
    text.textContent = transcript;
    const metric = document.createElement('p');
    metric.className = 'final-meta latency';
    metric.textContent = Number.isFinite(message.since_mark_ms)
      ? `点击标记 → 收到 done：${message.since_mark_ms.toFixed(0)} ms`
      : '本轮未关联人工结束标记';
    card.append(meta, text, metric);
    el['final-results'].prepend(card);
    el['partial-state'].textContent = '本轮已收到 done；继续讲话可测试下一轮';
    timeline('最终文本已收到', transcript, message);
    return;
  }
  if (message.type === 'event') {
    const event = message.event || {};
    const name = String(event.type || '未知事件');
    const text = typeof event.transcript === 'string' ? event.transcript
      : typeof event.text === 'string' ? event.text : '';
    timeline(name, text, message);
    if (name === 'user.conversation.created') {
      el['partial-text'].textContent = '检测到讲话，等待识别文字……';
      el['partial-text'].classList.add('empty');
      el['partial-state'].textContent = '检测到新一轮讲话';
    } else if (name === 'user.audio.transcription.started') {
      el['partial-state'].textContent = 'ASR 已识别到有效文字';
    } else if (name === 'user.audio.transcription.segment') {
      if (text) {
        el['partial-text'].textContent = text;
        el['partial-text'].classList.remove('empty');
      }
      el['partial-state'].textContent = '中间结果仍可修正；等待整轮 done';
    }
  }
}

async function startRun() {
  if (currentRun || !config?.configured) return;
  const run = { cancelled: false, ready: false, internalCapture: false, engine: null, socket: null, tracks: [] };
  currentRun = run;
  errorMessage('');
  finalCount = 0;
  el['final-count'].textContent = '0 轮';
  el['final-empty'].hidden = false;
  el['final-results'].replaceChildren();
  el.timeline.replaceChildren();
  el['partial-text'].textContent = '会话准备中，请等“可以讲话”。';
  el['partial-text'].classList.add('empty');
  el['partial-state'].textContent = '尚无识别事件';
  el.duration.textContent = '准备中';
  updateControls();
  status('准备音频与 RTC', 'busy');
  try {
    const source = el['audio-source'].value;
    let sharedAudioTrack = null;
    if (source === 'system') {
      if (!navigator.mediaDevices?.getDisplayMedia) throw fail('此浏览器不支持分享系统音频，请使用桌面版 Chrome 或 Edge。');
      // Called directly from the button gesture, before network awaits.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true, audio: true, systemAudio: 'include',
      });
      if (run.cancelled || currentRun !== run) {
        stream.getTracks().forEach((track) => track.stop());
        ensureActive(run);
      }
      run.tracks = stream.getTracks();
      sharedAudioTrack = stream.getAudioTracks()[0];
      if (!sharedAudioTrack) throw fail('分享来源没有音频。请重新开始，选择带声音的标签页，或在共享整个屏幕时勾选“共享系统音频”。');
      run.tracks.forEach((track) => track.addEventListener('ended', () => {
        if (!run.cancelled) void stopRun(run, '浏览器分享已结束');
      }, { once: true }));
      timeline('分享音频已授权', '只发布音轨；视频轨仅用于维持浏览器分享授权');
    }
    await connectBackend(run);
    ensureActive(run);
    const engine = VERTC.createEngine(config.appId);
    run.engine = engine;
    engine.on('onError', (event) => {
      if (run.cancelled || currentRun !== run) return;
      errorMessage(publicError(event));
      timeline('RTC 错误', String(event?.errorCode ?? event?.code ?? '未知代码'));
      void stopRun(run, 'RTC 错误后停止');
    });
    engine.on('onUserJoined', (event) => {
      if (!run.cancelled && event?.userInfo?.userId === config.botId) timeline('MMPG Bot 已进入 RTC 房间');
    });
    if (sharedAudioTrack) {
      await engine.setAudioSourceType(StreamIndex.STREAM_INDEX_MAIN, AudioSourceType.AUDIO_SOURCE_TYPE_EXTERNAL);
      ensureActive(run);
      await engine.setExternalAudioTrack(StreamIndex.STREAM_INDEX_MAIN, sharedAudioTrack);
    } else {
      run.internalCapture = true;
      await engine.startAudioCapture();
      const track = engine.getLocalStreamTrack(StreamIndex.STREAM_INDEX_MAIN, 'audio');
      if (track) run.tracks.push(track);
      if (run.cancelled) run.tracks.forEach((item) => item.stop());
    }
    ensureActive(run);
    await engine.joinRoom(config.rtcToken, config.roomId, { userId: config.userId }, {
      isAutoPublish: false, isAutoSubscribeAudio: false, isAutoSubscribeVideo: false,
    });
    ensureActive(run);
    timeline('RTC 已进房');
    await engine.publishStream(MediaType.AUDIO);
    ensureActive(run);
    timeline('RTC 音频已发布', '开始创建 MMPG 会话');
    const silenceMs = normalizeSilenceMs(el['silence-ms'].value);
    el['silence-ms'].value = el['silence-range'].value = String(silenceMs);
    send(run, { type: 'start', options: {
      aivad: el.aivad.checked,
      silence_ms: silenceMs,
      twopass: el.twopass.checked,
    } });
    status('RTC 已发布 · 等待 MMPG', 'busy');
    run.startTimeout = setTimeout(() => {
      if (!run.ready && !run.cancelled) {
        errorMessage('MMPG 会话 30 秒内未就绪，已停止音频上行。请查看事件与后端日志。');
        void stopRun(run, '会话创建超时');
      }
    }, 30000);
  } catch (error) {
    if (!run.cancelled && currentRun === run) {
      errorMessage(publicError(error));
      timeline('启动未完成', publicError(error));
    }
    await stopRun(run, '启动清理');
  }
}

async function boundedCleanup(action) {
  let timer;
  try {
    await Promise.race([
      Promise.resolve().then(action),
      new Promise((resolve) => { timer = setTimeout(resolve, 1800); }),
    ]);
  } catch { /* Continue cleanup even when RTC already disconnected. */ }
  finally { clearTimeout(timer); }
}

function stopRun(run = currentRun, reason = '手动停止') {
  if (!run) return Promise.resolve();
  if (run.cleanup) return run.cleanup;
  run.cancelled = true;
  run.ready = false;
  clearTimeout(run.startTimeout);
  clearInterval(run.durationTimer);
  if (currentRun === run) {
    status('正在停止', 'busy');
    updateControls();
  }
  // Stop tracks and notify the backend immediately, before waiting for RTC ACKs.
  run.tracks.forEach((track) => track.stop());
  if (run.socket?.readyState === WebSocket.OPEN) run.socket.send(JSON.stringify({ type: 'stop' }));
  run.cleanup = (async () => {
    if (run.engine) {
      await boundedCleanup(() => run.engine.unpublishStream(MediaType.AUDIO));
      if (run.internalCapture) await boundedCleanup(() => run.engine.stopAudioCapture());
      await boundedCleanup(() => run.engine.leaveRoom());
      try { VERTC.destroyEngine(run.engine); } catch { /* Already destroyed. */ }
    }
    run.socket?.close();
    if (currentRun === run) {
      currentRun = null;
      timeline('会话停止', reason);
      status(el.error.hidden ? '已停止' : '已停止 · 请检查提示', el.error.hidden ? '' : 'failed');
      el.duration.textContent = '已停止音频上行';
      updateControls();
    }
  })();
  return run.cleanup;
}

function updateSourceHelp() {
  el['source-help'].textContent = el['audio-source'].value === 'system'
    ? '建议使用桌面版 Chrome / Edge，选择带声音的标签页，或勾选共享整个屏幕的系统音频。浏览器要求同时授权画面，但本工具只上传音轨；没有音轨会拒绝启动。'
    : '麦克风模式只采集麦克风。开始后等状态显示“可以讲话”再朗读。';
}

el.start.addEventListener('click', () => void startRun());
el.stop.addEventListener('click', () => void stopRun());
el['mark-end'].addEventListener('click', () => {
  if (!currentRun?.ready || currentRun.cancelled) return;
  try {
    send(currentRun, { type: 'mark_end' });
    timeline('我说完了 · 已点击', '等待后端记录');
  } catch (error) { errorMessage(publicError(error)); }
});
el['audio-source'].addEventListener('change', updateSourceHelp);
for (const id of ['silence-ms', 'silence-range']) {
  el[id].addEventListener('input', () => {
    const value = Math.min(2900, Math.max(500, Number(el[id].value) || 800));
    el[id === 'silence-ms' ? 'silence-range' : 'silence-ms'].value = String(value);
  });
  el[id].addEventListener('change', () => {
    const value = Math.round(Math.min(2900, Math.max(500, Number(el[id].value) || 800)) / 100) * 100;
    el['silence-ms'].value = el['silence-range'].value = String(value);
  });
}
window.addEventListener('pagehide', () => {
  if (currentRun) void stopRun(currentRun, '页面关闭');
});

async function initialize() {
  try {
    const response = await fetch('/api/config', { cache: 'no-store' });
    if (!response.ok) throw fail('无法读取本机配置，请确认 Python 服务已启动。');
    config = await response.json();
    el.aivad.checked = config.aivad !== false;
    el.twopass.checked = config.twopass !== false;
    el['silence-ms'].value = el['silence-range'].value = String(config.silenceMs || 800);
    if (!config.configured) {
      const missing = Array.isArray(config.missing) ? config.missing.join('、') : 'RTC / MMPG 凭证';
      errorMessage(`本机配置尚未完成：${missing}。填写配置后刷新页面。`);
      el['config-summary'].textContent = '配置未就绪；当前不会创建云端会话。';
      status('等待配置');
    } else {
      el['config-summary'].textContent = `RTC 房间 ${config.roomId} · 用户 ${config.userId} · 单次上限 ${Math.round((config.maxSessionSeconds || 1800) / 60)} 分钟`;
      status('准备就绪');
    }
    timeline('页面就绪', `RTC Web SDK ${VERTC.getSdkVersion()}`);
    updateControls();
  } catch (error) {
    errorMessage(publicError(error));
    status('无法读取配置', 'failed');
    el['config-summary'].textContent = '请通过本机 Python 服务打开本页面。';
  }
}

void initialize();
