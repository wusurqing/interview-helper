import type { IRTCEngine } from '@volcengine/rtc'
import pcmWorkletUrl from './pcm.worklet?worker&url'
import { buildWsUrl } from '../backendUrl'

export type MmpgState = 'idle' | 'connecting' | 'listening' | 'muted' | 'reconnecting' | 'error'
type Command = 'start' | 'resume' | 'test'
type RtcSdk = typeof import('@volcengine/rtc')
type Snapshot = {
  state: MmpgState
  mode: Command | null
  error: string | null
  notice: string | null
  tokenExpiresAt: string | null
  audioDropped: number
  underruns: number
  overrunSamples: number
  timeline: { offsetMs: number; label: string; text?: string }[]
}

const initialSnapshot: Snapshot = {
  state: 'idle', mode: null, error: null, notice: null, tokenExpiresAt: null,
  audioDropped: 0, underruns: 0, overrunSamples: 0, timeline: [],
}
const statuses: readonly MmpgState[] = ['idle', 'connecting', 'listening', 'muted', 'reconnecting', 'error']
export function isMmpgState(state: unknown): state is MmpgState {
  return typeof state === 'string' && statuses.some((status) => status === state)
}
export const mmpgStateLabels: Record<MmpgState, string> = {
  idle: '等待启动', connecting: '正在连接', listening: '正在识别',
  muted: '已静音', reconnecting: '正在重连', error: '连接失败',
}
const diagnosticLabels = new Map([
  ['speech_recognized', '识别到语音'],
  ['speech_paused', '检测到停顿，等待本轮结束'],
  ['turn_cancelled', '本轮语音已取消'],
])
const errorMessages: Record<string, string> = {
  mmpg_busy: '另一个窗口已占用豆包 MMPG，请先结束其识别。',
  invalid_command: '豆包 MMPG 无法启动，请检查主链路和音频设备设置。',
  rtc_error: 'RTC 连接出错，请检查网络后重新开始。',
  rtc_join_failed: 'RTC 进房或发布音频失败，请检查房间、Token 和网络。',
  rtc_token_update_failed: 'RTC Token 更新失败，请检查 RTC 凭据。',
  rtc_token_expired: 'RTC Token 已过期，请更新凭据后重新开始。',
  rtc_token_invalid: 'RTC Token 无效，请检查 Token 与 App ID、房间和用户是否一致。',
  rtc_audio_unavailable: '无法初始化浏览器音频桥，请使用桌面应用或支持 AudioWorklet 的本机浏览器。',
  ws_disconnected: '豆包 MMPG 音频连接已断开，请从主窗口重新启动。',
  connect_timeout: '豆包 MMPG 等待连接超时，请检查配置和网络。',
  config_missing: '请在语音设置补全豆包 MMPG 的 RTC 与火山引擎凭据。',
  config_invalid: '豆包 MMPG 配置无效，请检查房间、用户、模型和 Token 设置。',
  dependencies_missing: '后端缺少豆包 MMPG 所需依赖，请重新安装项目依赖。',
  owner_unavailable: '豆包 MMPG 主窗口已断开。',
  rtc_join_timeout: 'RTC 进房超时，请检查本机网络。',
  rtc_recovery_timeout: 'RTC 重连超时。',
  audio_overrun: '音频连接积压，请检查本机性能和网络后重试。',
  audio_underrun: '音频缓冲持续欠载，请检查本机性能和音频设备。',
  audio_stalled: '后端音频传输中断，请检查音频设备。',
  cloud_auth_failed: 'MMPG 鉴权失败，请检查火山 Access Key、Secret Key 和豆包 ASR API Key。',
  cloud_connection_failed: '无法连接豆包 MMPG 服务，请检查网络。',
  cloud_connection_lost: '豆包 MMPG 服务连接中断。',
  cloud_error: '豆包 MMPG 服务返回错误，请检查服务开通状态和模型资源。',
  cloud_protocol_error: '豆包 MMPG 服务返回了无法处理的响应。',
  session_create_timeout: '豆包 MMPG 会话创建超时。',
  capture_failed: '后端音频采集失败，请检查和重新选择设备。',
  capture_busy: '音频设备正在被其他操作使用，请结束该操作后重试。',
  device_required: '请选择一个后端音频设备。',
  device_unavailable: '所选音频设备不可用，请刷新设备后重新选择。',
  cannot_resume: '当前没有已暂停的 MMPG 面试可继续。',
  interview_active: '请结束当前面试后再运行 MMPG 测试。',
  provider_not_selected: '请将主链路语音引擎选为豆包 MMPG 并保存。',
}

export function mmpgErrorMessage(code: string) {
  return errorMessages[code] ?? '豆包 MMPG 无法继续识别，请检查配置、音频设备和网络。'
}

export function mmpgUnavailable(platform: string | undefined): string | null {
  const host = window.location.hostname
  if (new URLSearchParams(window.location.search).get('overlay') === '1') return '请在主窗口启动豆包 MMPG。'
  if (platform && !/^(windows|win32)$/i.test(platform)) return '豆包 MMPG 需要 Windows 后端。'
  if (!window.isSecureContext || !['localhost', '127.0.0.1', '[::1]'].includes(host)) {
    return '请使用桌面应用或通过 localhost 打开安全的浏览器页面。'
  }
  if (!window.AudioContext || !window.AudioWorkletNode || !window.WebSocket) {
    return mmpgErrorMessage('rtc_audio_unavailable')
  }
  const backendHost = new URL(buildWsUrl('/ws/stt/mmpg')).hostname
  if (!['localhost', '127.0.0.1', '[::1]'].includes(backendHost)) return '豆包 MMPG 需要本机后端。'
  return null
}

type RtcSession = {
  generation: number
  engine: IRTCEngine
  sdk: RtcSdk
  node: AudioWorkletNode
  track: MediaStreamTrack
  destination: MediaStreamAudioDestinationNode
  pendingSamples: number
  bufferedSamples: number
  published: boolean
  reconnecting: boolean
  renewing: boolean
  expiresAt: number
  renewTimer?: number
  expiryTimer?: number
}

type Owner = {
  mode: Command
  context: AudioContext
  socket?: WebSocket
  generation: number
  rtc?: RtcSession
  joinGeneration?: number
  cleanup: Promise<void>
  connectTimer?: number
  heartbeat?: number
  lastMessageAt: number
  startedAt: number
  stopping: boolean
  resolve: () => void
  reject: (error: Error) => void
}

class MmpgController {
  private snapshot = initialSnapshot
  private readonly subscribers = new Set<() => void>()
  private owner?: Owner
  private cleanup = Promise.resolve()

  getSnapshot = () => this.snapshot

  subscribe = (listener: () => void) => {
    this.subscribers.add(listener)
    return () => this.subscribers.delete(listener)
  }

  private update(patch: Partial<Snapshot>) {
    this.snapshot = { ...this.snapshot, ...patch }
    this.subscribers.forEach((listener) => listener())
  }

  async start(mode: Command, deviceId: number, candidateMicId: number | null, platform?: string, prepare?: () => Promise<boolean>) {
    if (this.owner) throw new Error(mmpgErrorMessage('mmpg_busy'))
    const unavailable = mmpgUnavailable(platform)
    if (unavailable) throw new Error(unavailable)

    // Resume synchronously from the initiating click, including when settings must save first.
    let context: AudioContext
    try {
      context = new AudioContext()
    } catch {
      throw new Error(mmpgErrorMessage('rtc_audio_unavailable'))
    }
    const ready = context.resume()
    return new Promise<void>((resolve, reject) => {
      const owner: Owner = {
        mode, context, generation: -1, cleanup: this.cleanup, stopping: false,
        lastMessageAt: Date.now(), startedAt: performance.now(), resolve, reject,
      }
      this.owner = owner
      this.update({ ...initialSnapshot, mode, state: 'connecting' })
      this.testEvent(owner, '初始化本机音频')
      window.addEventListener('pagehide', this.onPageHide)
      owner.connectTimer = window.setTimeout(() => this.finish(owner, 'connect_timeout'), 15_000)
      void this.connect(owner, ready, deviceId, candidateMicId, prepare)
    })
  }

  cancelConnection = () => {
    const owner = this.owner
    if (owner && owner.mode !== 'test' && ['connecting', 'reconnecting'].includes(this.snapshot.state)) {
      this.finish(owner)
    }
  }

  stopTest = () => {
    const owner = this.owner
    if (owner?.mode !== 'test' || owner.stopping) return
    owner.stopping = true
    this.releaseRtc(owner)
    if (!this.send(owner, { type: 'stop' })) this.finish(owner)
    else {
      window.clearTimeout(owner.connectTimer)
      owner.connectTimer = window.setTimeout(() => this.finish(owner), 2500)
    }
  }

  private onPageHide = () => {
    if (this.owner) this.finish(this.owner)
  }

  private async connect(owner: Owner, ready: Promise<void>, deviceId: number, candidateMicId: number | null, prepare?: () => Promise<boolean>) {
    try {
      const [prepared] = await Promise.all([
        prepare?.() ?? true,
        ready,
        owner.context.audioWorklet.addModule(pcmWorkletUrl),
        owner.cleanup,
      ])
      if (this.owner !== owner) return
      if (!prepared) {
        this.finish(owner, 'invalid_command')
        return
      }
      if (owner.context.state !== 'running') throw new Error()
    } catch {
      this.finish(owner, 'rtc_audio_unavailable')
      return
    }
    try {
      const socket = new WebSocket(buildWsUrl('/ws/stt/mmpg'))
      owner.socket = socket
      socket.binaryType = 'arraybuffer'
      socket.onopen = () => {
        window.clearTimeout(owner.connectTimer)
        this.testEvent(owner, '已连接本机后端')
        this.send(owner, { type: owner.mode, device_ids: [deviceId], candidate_mic_id: candidateMicId })
        owner.heartbeat = window.setInterval(() => {
          if (Date.now() - owner.lastMessageAt > 45_000) this.finish(owner, 'ws_disconnected')
          else this.send(owner, { type: 'ping' })
        }, 15_000)
      }
      socket.onmessage = ({ data }) => this.receive(owner, data)
      socket.onclose = (event) => this.finish(owner, owner.stopping ? undefined : event.code === 1008 ? 'mmpg_busy' : 'ws_disconnected')
      // WebSocket errors are followed by close, which provides the final status.
    } catch {
      this.finish(owner, 'ws_disconnected')
    }
  }

  private receive(owner: Owner, data: unknown) {
    if (this.owner !== owner) return
    owner.lastMessageAt = Date.now()
    if (data instanceof ArrayBuffer) {
      const rtc = owner.rtc
      if (!rtc || !rtc.published || rtc.reconnecting || this.snapshot.state !== 'listening') return
      const samples = data.byteLength / Float32Array.BYTES_PER_ELEMENT
      if (!Number.isInteger(samples) || !samples) return
      if (rtc.pendingSamples + rtc.bufferedSamples + samples > 16_000) {
        this.recordAudioHealth(owner, rtc, 0, samples)
        return
      }
      rtc.pendingSamples += samples
      rtc.node.port.postMessage(data, [data])
      return
    }
    if (typeof data !== 'string') return
    let message: Record<string, unknown>
    try {
      message = JSON.parse(data)
    } catch {
      return
    }
    if (!message || typeof message !== 'object' || message.initial === true) return
    if (message.type === 'error') {
      this.finish(owner, typeof message.code === 'string' ? message.code : 'invalid_command')
      return
    }
    const generation = message.generation
    if (typeof generation !== 'number' || generation < owner.generation) return
    switch (message.type) {
      case 'state': {
        const state = message.state
        if (!isMmpgState(state) || (state === 'idle' && !message.reason && owner.generation === -1)) return
        owner.generation = generation
        const previousState = this.snapshot.state
        const code = typeof message.error === 'string' && message.error ? message.error : null
        this.update({
          state, error: code ? mmpgErrorMessage(code) : null,
          tokenExpiresAt: typeof message.token_expires_at === 'string' ? message.token_expires_at : null,
          audioDropped: typeof message.audio_dropped === 'number' ? message.audio_dropped : 0,
        })
        if (owner.rtc?.generation === generation && state !== previousState) {
          owner.rtc.node.port.postMessage(state === 'listening' ? 'listening' : 'clear')
        }
        if (state !== previousState) {
          this.testEvent(owner, state === 'idle' ? (message.reason === 'test_timeout' ? '达到 60 秒测试上限' : '测试结束') : mmpgStateLabels[state])
        }
        if (state === 'listening' && !owner.stopping) {
          window.clearTimeout(owner.connectTimer)
          owner.resolve()
        }
        if (state === 'idle' || state === 'error') this.finish(owner, state === 'error' ? code ?? 'rtc_error' : undefined)
        break
      }
      case 'rtc_join':
        if (owner.stopping) return
        owner.generation = generation
        owner.joinGeneration = generation
        this.update({ notice: null, tokenExpiresAt: typeof message.expires_at === 'string' ? message.expires_at : null })
        this.testEvent(owner, 'RTC 进房')
        void this.join(owner, generation, message)
        break
      case 'notice':
        if (generation === owner.generation && message.code === 'manual_token_renewal_required') {
          this.update({ notice: '手动 RTC Token 即将过期。请在语音设置更新 Token，再暂停并继续面试。' })
        }
        break
      case 'rtc_leave':
        if (owner.joinGeneration === generation) owner.joinGeneration = undefined
        if (owner.rtc?.generation === generation) this.releaseRtc(owner)
        break
      case 'rtc_token':
        if (owner.rtc?.generation === generation && typeof message.token === 'string') {
          void this.updateToken(owner, owner.rtc, message.token, message.expires_at)
        }
        break
      case 'diagnostic': {
        if (owner.mode !== 'test' || owner.generation !== generation || typeof message.event !== 'string') break
        const label = diagnosticLabels.get(message.event)
        if (label) this.testEvent(owner, label)
        break
      }
      case 'transcript':
        if (owner.mode === 'test' && owner.generation === generation && typeof message.transcript === 'string') {
          this.testEvent(owner, '判停完成', message.transcript)
        }
        break
    }
  }

  private async join(owner: Owner, generation: number, message: Record<string, unknown>) {
    this.releaseRtc(owner)
    await owner.cleanup
    const current = () => this.owner === owner && owner.generation === generation && owner.joinGeneration === generation && !owner.stopping
    if (!current()) return
    const { appId, roomId, userId, token } = message
    if (typeof appId !== 'string' || typeof roomId !== 'string' || typeof userId !== 'string' || typeof token !== 'string') {
      this.send(owner, { type: 'rtc_fault', generation, code: 'rtc_join_failed' })
      return
    }
    let fault = 'rtc_audio_unavailable'
    let node: AudioWorkletNode | undefined
    let destination: MediaStreamAudioDestinationNode | undefined
    let track: MediaStreamTrack | undefined
    let sdk: RtcSdk | undefined
    try {
      sdk = await import('@volcengine/rtc')
      if (!current()) return
      node = new AudioWorkletNode(owner.context, 'mmpg-pcm-source', {
        numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1],
      })
      destination = new MediaStreamAudioDestinationNode(owner.context, { channelCount: 1 })
      node.connect(destination)
      track = destination.stream.getAudioTracks()[0]
      if (!track || track.readyState !== 'live') throw new Error()
      fault = 'rtc_join_failed'
      sdk.default.setLogConfig({ logLevel: 'none' })
      const engine = sdk.default.createEngine(appId)
      const rtc: RtcSession = {
        generation, engine, sdk, node, destination, track,
        pendingSamples: 0, bufferedSamples: 0, published: false, reconnecting: false, renewing: false,
        expiresAt: typeof message.expires_at === 'string' ? Date.parse(message.expires_at) : NaN,
      }
      owner.rtc = rtc
      node.port.onmessage = ({ data }) => {
        if (!this.isCurrentRtc(owner, rtc)) return
        rtc.pendingSamples = Math.max(0, rtc.pendingSamples - (data.samples_received ?? 0))
        if (typeof data.buffered_samples === 'number') rtc.bufferedSamples = data.buffered_samples
        this.recordAudioHealth(owner, rtc, data.underruns ?? 0, data.overrun_samples ?? 0)
      }
      const events = sdk.default.events
      engine.on(events.onError, ({ errorCode }) => {
        if (this.isCurrentRtc(owner, rtc)) this.rtcFault(owner, rtc, errorCode === sdk.default.ErrorCode.TOKEN_EXPIRED ? 'rtc_token_expired' : 'rtc_error')
      })
      engine.on(events.onTokenWillExpire, () => this.requestRenew(owner, rtc))
      engine.on(events.onConnectionStateChanged, ({ state }) => {
        if (!this.isCurrentRtc(owner, rtc) || !rtc.published) return
        if ([sdk.ConnectionState.CONNECTION_STATE_RECONNECTING, sdk.ConnectionState.CONNECTION_STATE_DISCONNECTED, sdk.ConnectionState.CONNECTION_STATE_LOST].includes(state)) {
          if (rtc.reconnecting) return
          rtc.reconnecting = true
          node.port.postMessage('clear')
          this.send(owner, { type: 'rtc_reconnecting', generation })
        } else if (rtc.reconnecting && (state === sdk.ConnectionState.CONNECTION_STATE_RECONNECTED || state === sdk.ConnectionState.CONNECTION_STATE_CONNECTED)) {
          rtc.reconnecting = false
          this.send(owner, { type: 'rtc_recovered', generation })
        }
      })
      track.onended = () => this.rtcFault(owner, rtc, 'rtc_audio_unavailable')
      const mainStream = sdk.StreamIndex.STREAM_INDEX_MAIN
      await engine.setAudioSourceType(mainStream, sdk.AudioSourceType.AUDIO_SOURCE_TYPE_EXTERNAL)
      if (!this.isCurrentRtc(owner, rtc)) return
      await engine.setExternalAudioTrack(mainStream, track)
      if (!this.isCurrentRtc(owner, rtc)) return
      await engine.joinRoom(token, roomId, { userId }, {
        isAutoPublish: false, isAutoSubscribeAudio: false, isAutoSubscribeVideo: false,
      })
      if (!this.isCurrentRtc(owner, rtc)) return
      await engine.publishStream(sdk.MediaType.AUDIO)
      if (!this.isCurrentRtc(owner, rtc)) return
      rtc.published = true
      this.testEvent(owner, 'RTC 音轨已发布')
      this.send(owner, { type: 'rtc_ready', generation })
      this.scheduleRenewal(owner, rtc, message.expires_at)
    } catch (error) {
      if (fault === 'rtc_join_failed' && sdk) {
        const code = error?.errorCode ?? error?.code ?? error
        if (code === sdk.default.ErrorCode.INVALID_TOKEN) {
          fault = owner.rtc?.expiresAt <= Date.now() ? 'rtc_token_expired' : 'rtc_token_invalid'
        }
      }
      if (current() && owner.rtc) this.rtcFault(owner, owner.rtc, fault)
      else if (current()) {
        track?.stop()
        node?.disconnect()
        node?.port.close()
        destination?.disconnect()
        this.send(owner, { type: 'rtc_fault', generation, code: fault })
      }
    }
  }

  private isCurrentRtc(owner: Owner, rtc: RtcSession) {
    return this.owner === owner && owner.rtc === rtc && !owner.stopping
  }

  private requestRenew(owner: Owner, rtc: RtcSession) {
    if (!this.isCurrentRtc(owner, rtc) || rtc.renewing) return
    rtc.renewing = this.send(owner, { type: 'renew', generation: rtc.generation })
  }

  private scheduleRenewal(owner: Owner, rtc: RtcSession, expiry: unknown) {
    window.clearTimeout(rtc.renewTimer)
    window.clearTimeout(rtc.expiryTimer)
    const remaining = typeof expiry === 'string' ? Date.parse(expiry) - Date.now() : NaN
    rtc.expiresAt = remaining + Date.now()
    if (!Number.isFinite(remaining)) return
    // Bound JS timers for long-lived manual tokens and reconsider on early wakeups.
    if (remaining > 2_000_000_000) {
      rtc.renewTimer = window.setTimeout(() => this.scheduleRenewal(owner, rtc, expiry), 1_000_000_000)
      return
    }
    rtc.renewTimer = window.setTimeout(() => this.requestRenew(owner, rtc), Math.max(0, remaining - 120_000))
    rtc.expiryTimer = window.setTimeout(() => this.rtcFault(owner, rtc, 'rtc_token_expired'), Math.max(0, remaining))
  }

  private async updateToken(owner: Owner, rtc: RtcSession, token: string, expiry: unknown) {
    try {
      await rtc.engine.updateToken(token)
      if (!this.isCurrentRtc(owner, rtc)) return
      rtc.renewing = false
      this.update({ tokenExpiresAt: typeof expiry === 'string' ? expiry : null, notice: null })
      this.testEvent(owner, 'RTC Token 已更新')
      this.scheduleRenewal(owner, rtc, expiry)
    } catch {
      this.rtcFault(owner, rtc, 'rtc_token_update_failed')
    }
  }

  private recordAudioHealth(owner: Owner, rtc: RtcSession, underruns: number, overrunSamples: number) {
    if (!underruns && !overrunSamples) return
    this.update({ underruns: this.snapshot.underruns + underruns, overrunSamples: this.snapshot.overrunSamples + overrunSamples })
    this.send(owner, { type: 'audio_status', generation: rtc.generation, underruns, overrun_samples: overrunSamples })
  }

  private testEvent(owner: Owner, label: string, text?: string) {
    if (owner.mode !== 'test' || this.owner !== owner) return
    const event = { offsetMs: performance.now() - owner.startedAt, label, text }
    this.update({ timeline: [...this.snapshot.timeline, event].slice(-50) })
  }

  private rtcFault(owner: Owner, rtc: RtcSession, code: string) {
    if (!this.isCurrentRtc(owner, rtc)) return
    this.releaseRtc(owner)
    this.send(owner, { type: 'rtc_fault', generation: rtc.generation, code })
  }

  private send(owner: Owner, message: Record<string, unknown>): boolean {
    if (this.owner !== owner || owner.socket?.readyState !== WebSocket.OPEN) return false
    try {
      owner.socket.send(JSON.stringify(message))
      return true
    } catch {
      this.finish(owner, 'ws_disconnected')
      return false
    }
  }

  private releaseRtc(owner: Owner) {
    const rtc = owner.rtc
    if (!rtc) return
    owner.rtc = undefined
    window.clearTimeout(rtc.renewTimer)
    window.clearTimeout(rtc.expiryTimer)
    rtc.track.onended = null
    rtc.track.stop()
    rtc.node.disconnect()
    rtc.node.port.close()
    rtc.destination.disconnect()
    owner.cleanup = this.cleanup = (async () => {
      if (rtc.published) await quietRtcCleanup(() => rtc.engine.unpublishStream(rtc.sdk.MediaType.AUDIO))
      await quietRtcCleanup(() => rtc.engine.leaveRoom())
      try { rtc.sdk.default.destroyEngine(rtc.engine) } catch { /* SDK already destroyed. */ }
    })()
  }

  private finish(owner: Owner, code?: string) {
    if (this.owner !== owner) return
    this.owner = undefined
    window.removeEventListener('pagehide', this.onPageHide)
    this.releaseRtc(owner)
    window.clearTimeout(owner.connectTimer)
    window.clearInterval(owner.heartbeat)
    owner.socket?.close()
    if (owner.context.state !== 'closed') void owner.context.close().catch(() => undefined)
    const error = code ? mmpgErrorMessage(code) : null
    this.update({ state: code ? 'error' : 'idle', error, notice: null, tokenExpiresAt: null })
    owner.reject(error ? new Error(error) : new DOMException('已停止豆包 MMPG。', 'AbortError'))
  }
}

async function quietRtcCleanup(action: () => Promise<unknown>) {
  let timer: number | undefined
  try {
    await Promise.race([action(), new Promise((resolve) => { timer = window.setTimeout(resolve, 1000) })])
  } catch { /* Continue cleanup when RTC has disconnected. */ }
  finally { window.clearTimeout(timer) }
}

export const mmpgController = new MmpgController()
