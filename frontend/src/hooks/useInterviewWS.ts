import { useEffect, useRef } from 'react'
import { useInterviewStore } from '@/stores/configStore'
import { useKbStore } from '@/stores/kbStore'
import { receiveArchiveStatus } from '@/stores/archiveStatusStore'
import { buildWsUrl } from '@/lib/backendUrl'
import { emitExamPreflight, emitSoundPreflight } from '@/lib/preflightWsBus'
import { subscribeLeader } from '@/lib/wsLeader'
import { isMmpgState, mmpgErrorMessage } from '@/lib/mmpg/rtcController'

// 指数退避重连步长（ms）。最后一档是稳态，不再翻倍。
const RECONNECT_BACKOFF_MS = [1000, 2000, 4000, 8000, 16000, 30000]
// 客户端 ping 间隔（与服务端 25s 心跳错开，避免对撞）。
const CLIENT_PING_INTERVAL_MS = 25000
// pong 超时阈值：若超过该时长没收到任何服务端帧（pong/ping/普通广播），视为僵尸。
const PONG_TIMEOUT_MS = 60000

interface WsMsg {
  type: string
  scope?: string
  [k: string]: unknown
}

export function useInterviewWS(active = true) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<number | null>(null)
  const errorClearTimer = useRef<number | null>(null)
  const reconnectAttempts = useRef(0)
  const shouldReconnect = useRef(true)
  const lastServerPongAt = useRef(0)
  const pendingWorkspace = useRef<boolean | undefined>(undefined)
  const workspaceChannel = useRef<BroadcastChannel | null>(null)
  const preflightChannel = useRef<BroadcastChannel | null>(null)

  const applyWorkspace = (exam: boolean) => {
    const { config, setConfig } = useInterviewStore.getState()
    if (!config) {
      pendingWorkspace.current = exam
    } else if (config.written_exam_mode !== exam) {
      setConfig({ ...config, written_exam_mode: exam })
    }
  }

  const scheduleReconnect = () => {
    if (!shouldReconnect.current) return
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    const idx = Math.min(reconnectAttempts.current, RECONNECT_BACKOFF_MS.length - 1)
    const delay = RECONNECT_BACKOFF_MS[idx]
    reconnectAttempts.current += 1
    reconnectTimer.current = window.setTimeout(connect, delay)
  }

  const connect = () => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN
      || wsRef.current?.readyState === WebSocket.CONNECTING
    ) return
    let ws: WebSocket
    try {
      ws = new WebSocket(buildWsUrl('/ws'))
    } catch (err) {
      console.warn('[ws] failed to construct WebSocket', err)
      scheduleReconnect()
      return
    }
    wsRef.current = ws
    lastServerPongAt.current = Date.now()

    ws.onopen = () => {
      if (wsRef.current !== ws) return
      reconnectAttempts.current = 0
      useInterviewStore.getState().setWsConnected(true)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
    }
    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return
      lastServerPongAt.current = Date.now()
      let data: WsMsg
      try {
        data = JSON.parse(event.data)
      } catch (err) {
        console.warn('[ws] JSON parse failed', err, typeof event.data === 'string' ? event.data.slice(0, 120) : event.data)
        return
      }
      if (!data || typeof data !== 'object') return
      // 服务端 ping → 主动回 pong 维持心跳
      if (data.type === 'ping') {
        try {
          ws.send(JSON.stringify({ type: 'pong' }))
        } catch (err) {
          console.warn('[ws] send pong failed', err)
        }
        return
      }
      if (data.type === 'pong') return
      handleMessage(data)
    }
    ws.onclose = (ev) => {
      if (wsRef.current !== ws) return
      wsRef.current = null
      useInterviewStore.getState().setWsConnected(false)
      if (!shouldReconnect.current) return
      if (ev && ev.code === 1008) {
        // policy violation (鉴权失败)，重试也无效，停止并向用户显式提示
        console.warn('[ws] closed by server with policy violation; stop reconnecting')
        shouldReconnect.current = false
        if (errorClearTimer.current !== null) {
          clearTimeout(errorClearTimer.current)
          errorClearTimer.current = null
        }
        const authMsg = '实时连接鉴权失败（访问令牌无效或已过期），请重新扫码或刷新授权后重试'
        const store = useInterviewStore.getState()
        store.setLastWSError(authMsg)
        store.pushToast(authMsg, 'error', 10000)
        return
      }
      scheduleReconnect()
    }
    ws.onerror = (ev) => {
      if (wsRef.current !== ws) return
      console.warn('[ws] error event', ev)
      try { ws.close() } catch (err) { console.warn('[ws] close after error failed', err) }
    }
  }

  const handleMessage = (msg: WsMsg) => {
    const s = useInterviewStore.getState()
    if ((msg.type === 'init' || msg.type === 'workspace_mode') && typeof msg.written_exam_mode === 'boolean') {
      applyWorkspace(msg.written_exam_mode)
      workspaceChannel.current?.postMessage(msg.written_exam_mode)
    }
    if (msg.type === 'preflight_step') {
      emitSoundPreflight(msg)
      preflightChannel.current?.postMessage({ target: 'sound', msg })
      return
    }
    if (msg.type === 'exam_preflight_step' || msg.exam_preflight_id) {
      emitExamPreflight(msg)
      preflightChannel.current?.postMessage({ target: 'exam', msg })
      return
    }
    switch (msg.type) {
      case 'init':
        s.setInitData(msg as Parameters<typeof s.setInitData>[0])
        receiveArchiveStatus(msg.archive, true)
        break
      case 'review_archive_status':
        receiveArchiveStatus(msg)
        break
      case 'recording':
        s.setRecording(
          msg.value as boolean,
          msg.interviewer_streaming === true,
          typeof msg.candidate_capture_provider === 'string' ? msg.candidate_capture_provider : undefined,
        )
        break
      case 'single_mic':
        s.setSingleMic(msg.value)
        break
      case 'paused':
        s.setPaused(msg.value as boolean)
        break
      case 'audio_level':
        s.setAudioLevel(msg.value as number)
        break
      case 'transcribing':
        s.setTranscribing(msg.value as boolean)
        break
      case 'transcription':
        s.addTranscription(msg.text as string)
        break
      case 'candidate_transcription':
        s.addCandidateTranscription(msg.text as string, {
          segmentId: msg.segment_id as string | undefined,
          isFinal: msg.is_final as boolean | undefined,
        })
        break
      case 'session_cleared':
        s.clearSession()
        break
      case 'mmpg_answer_phase':
        s.setMmpgAnswerPhase(msg.phase)
        break
      case 'answer_start':
      case 'answer_retry':
        if (msg.exam_preflight_id) return
        s.startAnswer(msg.id as string, msg.question as string, {
          source: msg.source as string,
          modelName: msg.model_name as string,
          ...(msg.answer_format === 'plain_text' ? { answerFormat: 'plain_text' as const } : {}),
          updatesQaId: msg.updates_qa_id as string | undefined,
          attempt: typeof msg.attempt === 'number' ? msg.attempt : undefined,
        })
        break
      case 'answer_think_chunk':
        if (msg.exam_preflight_id) return
        s.appendThinkChunk(msg.id as string, msg.chunk as string, msg.attempt as number | undefined)
        break
      case 'answer_chunk':
        if (msg.exam_preflight_id) return
        s.appendAnswerChunk(msg.id as string, msg.chunk as string, msg.attempt as number | undefined)
        break
      case 'answer_done':
        if (msg.exam_preflight_id) return
        s.finalizeAnswer(
          msg.id as string,
          msg.question as string,
          msg.answer as string,
          msg.think as string,
          msg.model_name as string,
          msg.first_token_ms as number | undefined,
          msg.total_ms as number | undefined,
          msg.updates_qa_id as string | undefined,
          {
            ...(msg.answer_format === 'plain_text' ? { answerFormat: 'plain_text' as const } : {}),
            source: msg.source as string | undefined,
            attempt: msg.attempt as number | undefined,
          },
        )
        break
      case 'answer_cancelled':
        if (msg.exam_preflight_id) return
        s.cancelAnswer(msg.id as string, msg.reason as string | undefined, msg.attempt as number | undefined)
        break
      case 'answer_error':
        if (msg.exam_preflight_id) return
        {
          const qa = s.qaPairs.find(item => item.id === msg.id)
          if (qa && (qa.status === 'done' || (typeof msg.attempt === 'number' && msg.attempt < (qa.attempt ?? 1)))) return
          const message = (msg.message as string) || '答案处理失败'
          const prefix = msg.stage === 'persistence' ? '答案保存失败' : '答案生成失败'
          s.errorAnswer(msg.id as string, message, msg.attempt as number | undefined)
          s.pushToast(message.startsWith(prefix) ? message : `${prefix}: ${message}`, 'error')
        }
        break
      case 'vision_verify':
        {
          const verdict = msg.verdict as Parameters<typeof s.setVisionVerify>[1]
          const reason = String(msg.reason || '').trim()
          const reasonHint = reason ? `：${reason.slice(0, 80)}` : ''
          if (verdict === 'FAIL') {
            s.pushToast(`截图自检不一致，请人工复核${reasonHint}`, 'warn', 6000)
          }
          s.setVisionVerify(
            msg.id as string,
            verdict,
            reason,
          )
        }
        break
      case 'stt_status':
        s.setSttStatus((msg.loaded as boolean) ?? false, (msg.loading as boolean) ?? false, msg.provider as string | undefined)
        if (msg.provider === 'doubao_mmpg' && isMmpgState(msg.state)) {
          s.setMmpgStatus(msg.state, typeof msg.error === 'string' ? msg.error : null)
        }
        break
      case 'candidate_asr_status':
        s.setCandidateSttStatus(
          (msg.loaded as boolean) ?? false,
          (msg.loading as boolean) ?? false,
          msg.provider as string | undefined,
        )
        if (msg.error && msg.provider === 'doubao_mmpg') {
          if (msg.state === 'error') s.pushToast(`我的麦克风 MMPG · ${mmpgErrorMessage(String(msg.error))}`, 'warn')
        } else if (msg.error) {
          const prefix = s.singleMic
            ? '测试麦克风识别失败'
            : '我的回答识别失败'
          s.pushToast(`${prefix}: ${msg.error as string}`, 'warn')
        }
        break
      case 'model_health':
        s.setModelHealth(
          msg.index as number,
          msg.status as Parameters<typeof s.setModelHealth>[1],
          msg.detail as string | undefined,
          msg.latency_ms as number | undefined,
          msg.model_fingerprint as string | undefined,
        )
        break
      case 'token_update':
        s.setTokenUsage({
          prompt: msg.prompt as number,
          completion: msg.completion as number,
          total: msg.total as number,
          byModel: (msg.by_model as Record<string, { prompt: number; completion: number }>) ?? {},
        })
        break
      case 'model_fallback':
        s.setFallbackToast({ from: msg.from as string, to: msg.to as string, reason: msg.reason as string })
        break
      case 'stt_error':
        if (typeof msg.message === 'string') {
          const role = msg.role === 'candidate' ? '我的回答' : '面试官'
          s.pushToast(`${role}识别失败 · ${msg.message}`, 'warn')
        }
        break
      case 'resume_opt_start':
        s.resetResumeOpt(typeof msg.job_id === 'string' ? msg.job_id : null)
        s.setResumeOptLoading(true)
        break
      case 'resume_opt_chunk':
        s.appendResumeOptChunk(msg.chunk as string, typeof msg.job_id === 'string' ? msg.job_id : null)
        break
      case 'resume_opt_done':
        s.setResumeOptResult(msg.text as string, typeof msg.job_id === 'string' ? msg.job_id : null)
        break
      case 'error':
        s.setLastWSError((msg.message as string) || '未知错误')
        if (errorClearTimer.current !== null) {
          clearTimeout(errorClearTimer.current)
        }
        errorClearTimer.current = window.setTimeout(() => {
          errorClearTimer.current = null
          useInterviewStore.getState().setLastWSError(null)
        }, 5000)
        break
      case 'kb_hits':
        useKbStore.getState().appendHits({
          qa_id: msg.qa_id as string,
          latency_ms: (msg.latency_ms as number) ?? 0,
          degraded: !!msg.degraded,
          hit_count: (msg.hit_count as number) ?? (Array.isArray(msg.hits) ? (msg.hits as unknown[]).length : 0),
          hits: Array.isArray(msg.hits) ? (msg.hits as Parameters<ReturnType<typeof useKbStore.getState>['appendHits']>[0]['hits']) : [],
        })
        break
    }
  }

  useEffect(() => {
    let unsubscribe: (() => void) | null = null
    let pingTimer: ReturnType<typeof setInterval> | null = null

    const teardown = () => {
      shouldReconnect.current = false
      reconnectAttempts.current = 0
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      if (errorClearTimer.current !== null) {
        clearTimeout(errorClearTimer.current)
        errorClearTimer.current = null
      }
      if (wsRef.current) {
        try { wsRef.current.close() } catch (err) { console.warn('[ws] teardown close failed', err) }
        wsRef.current = null
      }
      useInterviewStore.getState().setWsConnected(false)
      useInterviewStore.getState().setWsIsLeader(false)
    }

    if (!active) {
      teardown()
      return teardown
    }

    // Follower tabs share the leader's socket. Relay workspace mode and preflight events.
    const channel = typeof window.BroadcastChannel === 'function'
      ? new window.BroadcastChannel('ia-workspace-mode') : null
    workspaceChannel.current = channel
    if (channel) channel.onmessage = (event) => {
      if (!useInterviewStore.getState().wsIsLeader && typeof event.data === 'boolean') applyWorkspace(event.data)
    }
    const pfChannel = typeof window.BroadcastChannel === 'function'
      ? new window.BroadcastChannel('ia-preflight-events') : null
    preflightChannel.current = pfChannel
    if (pfChannel) pfChannel.onmessage = (event) => {
      if (useInterviewStore.getState().wsIsLeader) return
      const payload = event.data as { target?: string; msg?: WsMsg } | null
      if (!payload || !payload.msg || typeof payload.msg.type !== 'string') return
      if (payload.target === 'sound') emitSoundPreflight(payload.msg)
      else if (payload.target === 'exam') emitExamPreflight(payload.msg)
    }
    const unsubscribeBootstrap = useInterviewStore.subscribe((state) => {
      if (!state.config || pendingWorkspace.current === undefined) return
      const exam = pendingWorkspace.current
      pendingWorkspace.current = undefined
      applyWorkspace(exam)
    })

    unsubscribe = subscribeLeader((isLeader) => {
      useInterviewStore.getState().setWsIsLeader(isLeader)
      if (isLeader) {
        shouldReconnect.current = true
        reconnectAttempts.current = 0
        connect()
      } else {
        teardown()
      }
    })

    pingTimer = setInterval(() => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      // watchdog：长时间没收到任何服务端帧，认定连接已僵尸，主动重连。
      if (Date.now() - lastServerPongAt.current > PONG_TIMEOUT_MS) {
        console.warn('[ws] pong watchdog timeout, reconnecting')
        try { ws.close() } catch (err) { console.warn('[ws] watchdog close failed', err) }
        return
      }
      try {
        ws.send(JSON.stringify({ type: 'ping' }))
      } catch (err) {
        console.warn('[ws] client ping send failed', err)
      }
    }, CLIENT_PING_INTERVAL_MS)

    return () => {
      unsubscribe?.()
      unsubscribeBootstrap()
      pendingWorkspace.current = undefined
      workspaceChannel.current = null
      channel?.close()
      preflightChannel.current = null
      pfChannel?.close()
      if (pingTimer) clearInterval(pingTimer)
      teardown()
    }
    // 除 active 外，内部依赖都通过 ref/store getState 访问，不需要因 store 变化重建连接。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])
}
