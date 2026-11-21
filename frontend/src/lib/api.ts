import { buildApiUrl } from './backendUrl'
import { getAuthToken } from './auth'
import type {
  AppConfig,
  DeviceItem,
  ModelFullInfo,
  OptionsInfo,
  PlatformInfo,
  SingleMicRole,
  SingleMicState,
} from '@/stores/slices/types'
import type { HistoryRecord } from '@/components/knowledge-map/mergeHistory'

export type KnowledgeThinkEffort = 'off' | 'low' | 'medium' | 'high' | 'xhigh'

export interface KnowledgeSettings {
  model_index: number | null
  model_id?: string | null
  system_prompt: string | null
  temperature: number
  max_tokens: number
  think_effort: KnowledgeThinkEffort
}

export interface KnowledgeSettingsPayload {
  settings: KnowledgeSettings
  default_system_prompt: string
  output_contract: string
  models: {
    id?: string
    index: number
    name: string
    provider_name?: string
    model?: string
    fingerprint: string
    available: boolean
    supports_think: boolean
    supports_vision?: boolean
  }[]
}

export interface KnowledgeAnalysisState {
  pending_count: number
  job: {
    job_id: string
    status: 'running' | 'completed'
    total: number
    succeeded: number
    failed: number
    skipped: number
  } | null
}

const BACKEND_UNREACHABLE_MESSAGE = '无法连接后端服务，请确认应用服务已启动后重试'
const BACKEND_TIMEOUT_MESSAGE = '请求超时，请检查后端服务或网络后重试'
export const DEFAULT_REQUEST_TIMEOUT_MS = 30_000
const LONG_REQUEST_TIMEOUT_MS = 120_000

export interface ApiRequestInit extends RequestInit {
  timeoutMs?: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function formatDetailValue(value: unknown): string | null {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => {
        if (!isRecord(item)) return formatDetailValue(item)
        const msg = formatDetailValue(item.msg ?? item.message ?? item.detail)
        const loc = Array.isArray(item.loc) ? item.loc.map(String).join('.') : null
        if (loc && msg) return `${loc}: ${msg}`
        return msg ?? formatDetailValue(item)
      })
      .filter(Boolean)
    return parts.length ? parts.join('; ') : null
  }
  if (isRecord(value)) {
    const direct = formatDetailValue(value.detail ?? value.message ?? value.error)
    if (direct) return direct
    try {
      return JSON.stringify(value)
    } catch {
      return null
    }
  }
  return null
}

function parseResponseBody(text: string): unknown {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

function buildResponseErrorMessage(res: Response, text: string, fallback?: string): string {
  const body = parseResponseBody(text)
  const detail = isRecord(body) ? body.detail ?? body.message ?? body.error : body
  const detailText = formatDetailValue(detail) ?? fallback ?? res.statusText ?? '请求失败'
  return `请求失败 (${res.status}): ${detailText}`
}

async function fetchWithCancellation<T>(
  input: string,
  init: ApiRequestInit | undefined,
  readBody: (res: Response) => Promise<T>,
): Promise<T> {
  const { timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS, signal: callerSignal, ...rest } = init ?? {}
  const controller = new AbortController()
  let timedOut = false
  let timer: ReturnType<typeof setTimeout> | undefined
  let onCallerAbort: (() => void) | undefined

  if (callerSignal) {
    if (callerSignal.aborted) {
      controller.abort(callerSignal.reason)
    } else {
      onCallerAbort = () => controller.abort(callerSignal.reason)
      callerSignal.addEventListener('abort', onCallerAbort, { once: true })
    }
  }

  if (!controller.signal.aborted && Number.isFinite(timeoutMs) && timeoutMs > 0) {
    timer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, timeoutMs)
  }

  try {
    const res = await fetch(input, { ...rest, signal: controller.signal })
    // Fetch resolves at the headers. Keep cancellation active until the body finishes.
    return await readBody(res)
  } catch (error) {
    if (callerSignal?.aborted) throw error
    if (timedOut) throw new Error(BACKEND_TIMEOUT_MESSAGE)
    if (error instanceof ApiRequestError) throw error
    throw new Error(BACKEND_UNREACHABLE_MESSAGE)
  } finally {
    if (timer !== undefined) clearTimeout(timer)
    if (callerSignal && onCallerAbort) {
      callerSignal.removeEventListener('abort', onCallerAbort)
    }
  }
}

async function fetchBackend(input: string, init?: ApiRequestInit) {
  return fetchWithCancellation(input, init, async (res) => ({
    res,
    text: await res.text(),
  }))
}

async function fetchBackendArrayBuffer(input: string, init?: ApiRequestInit): Promise<ArrayBuffer> {
  return fetchWithCancellation(input, init, async (res) => {
    if (!res.ok) {
      const text = await res.text()
      throw new ApiRequestError(buildResponseErrorMessage(res, text), res.status)
    }
    return await res.arrayBuffer()
  })
}

export function getErrorMessage(error: unknown, fallback = '操作失败'): string {
  if (error instanceof Error && error.message) return error.message
  if (typeof error === 'string' && error.trim()) return error
  if (isRecord(error)) return formatDetailValue(error) ?? fallback
  return fallback
}

export function buildHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers({ 'Content-Type': 'application/json' })
  const token = getAuthToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (extra) {
    const extraHeaders = new Headers(extra)
    extraHeaders.forEach((value, key) => {
      headers.set(key, value)
    })
  }
  return headers
}

export class ApiRequestError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
  }
}

export async function request<T = unknown>(url: string, opts?: ApiRequestInit): Promise<T> {
  const { res, text } = await fetchBackend(buildApiUrl(url), {
    ...opts,
    headers: buildHeaders(opts?.headers),
  })
  if (!res.ok) {
    throw new ApiRequestError(buildResponseErrorMessage(res, text), res.status)
  }
  if (res.status === 204) return undefined as T
  return parseResponseBody(text) as T
}

export interface ResumeHistoryItem {
  id: number
  original_filename: string
  file_size: number
  created_at: number
  last_used_at: number
  parsed_ok: boolean
  preview: string
  parse_error: string | null
  is_active: boolean
}

/** GET /resume/history/:id 返回在列表项基础上增加 summary */
export type ResumeHistoryDetail = ResumeHistoryItem & {
  summary: string
  /** false 表示仅展示入库时的节选，点「选用」成功后会写入全文摘要 */
  summary_is_full?: boolean
}

// --- Knowledge Base ---
export type KBOrigin = 'text' | 'ocr' | 'vision' | 'mixed'

export interface KBHit {
  path: string
  section_path: string
  page?: number | null
  origin: KBOrigin
  score: number
  excerpt: string
}

export interface KBDoc {
  id: number
  path: string
  mtime: number
  size: number
  loader: string
  title: string | null
  status: 'ok' | 'failed' | 'pending'
  error: string | null
  chunk_count: number
}

export interface KBStatus {
  enabled: boolean
  trigger_modes: string[]
  top_k: number
  deadline_ms: number
  asr_deadline_ms: number
  total_docs: number
  total_chunks: number
  last_mtime: number
  deps: { docx: boolean; pdf: boolean; ocr: boolean; vision: boolean }
}

export interface KBRecentHit {
  ts: number
  query: string
  mode: string
  hit_count: number
  latency_ms: number
  timed_out?: boolean
  error?: string | null
  top_section_paths: string[]
}

export interface ResumeUploadResult {
  ok: boolean
  history_id: number
  parsed: boolean
  length?: number | null
  preview?: string | null
  parse_error?: string | null
}

export interface SystemPromptScene {
  id: string
  group: string
  label: string
  entry: string
  default_prompt: string
  output_contract: string
  max_output_tokens: number
}

export interface KnowledgeSummaryTag {
  tag: string
  count: number
  avg_score: number | null
  trend: 'up' | 'down' | 'stable'
}

export interface PreflightStatusResponse {
  running?: boolean
  steps?: Record<string, unknown> | unknown[]
  captured_transcript?: string
  expected_phrase?: string
  [key: string]: unknown
}

export const api = {
  getConfig: () => request<AppConfig>('/api/config'),
  updateConfig: (data: Record<string, unknown>) =>
    request<AppConfig>('/api/config', { method: 'POST', body: JSON.stringify(data) }),
  getSystemPromptScenes: () => request<{ scenes: SystemPromptScene[] }>('/api/prompts'),
  importMmpgProbe: () => request<{ ok: boolean }>('/api/stt/mmpg/import-probe', { method: 'POST' }),
  modelsLayout: (data: {
    order?: number[]
    enabled?: boolean[]
    max_parallel_answers?: number
    active_model?: number
  }) => request<AppConfig>('/api/config/models-layout', { method: 'POST', body: JSON.stringify(data) }),
  getModelsFull: () => request<{ models: ModelFullInfo[] }>('/api/config/models-full'),
  getOptions: () => request<OptionsInfo>('/api/options'),
  getDevices: () => request<{ devices: DeviceItem[]; platform: PlatformInfo }>('/api/devices'),
  uploadResume: async (file: File): Promise<ResumeUploadResult> => {
    const fd = new FormData()
    fd.append('file', file)
    const token = getAuthToken()
    const headers: Record<string, string> = {}
    if (token) headers.Authorization = `Bearer ${token}`
    const { res, text } = await fetchBackend(buildApiUrl('/api/resume'), {
      method: 'POST',
      body: fd,
      headers,
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    })
    if (!res.ok) {
      throw new Error(buildResponseErrorMessage(res, text, '上传失败'))
    }
    return JSON.parse(text)
  },
  deleteResume: () => request<{ ok: boolean }>('/api/resume', { method: 'DELETE' }),
  resumeHistory: () => request<{ items: ResumeHistoryItem[]; max: number }>('/api/resume/history'),
  resumeHistoryApply: (id: number) =>
    request<{ ok: boolean; history_id: number; length: number; preview: string }>(
      `/api/resume/history/${id}/apply`,
      { method: 'POST' },
    ),
  resumeHistoryDelete: (id: number) =>
    request<{ ok: boolean }>(`/api/resume/history/${id}`, { method: 'DELETE' }),
  resumeHistoryDetail: (id: number) => request<ResumeHistoryDetail>(`/api/resume/history/${id}`),
  resumeHistoryUpdate: (id: number, summary: string) =>
    request<{ ok: boolean; length: number }>(`/api/resume/history/${id}`, {
      method: 'PUT',
      body: JSON.stringify({ summary }),
    }),
  start: (device_id?: number | null, candidate_mic_device_id?: number | null, single_mic_id?: string) =>
    request<{ ok: boolean }>('/api/start', {
      method: 'POST',
      body: JSON.stringify({
        ...(device_id != null ? { device_id } : {}),
        ...(candidate_mic_device_id != null ? { candidate_mic_device_id } : {}),
        ...(single_mic_id ? { single_mic_id } : {}),
      }),
    }),
  stop: () => request<{ ok: boolean }>('/api/stop', { method: 'POST' }),
  pause: (single_mic_id?: string) => request<{ ok: boolean }>('/api/pause', {
    method: 'POST',
    ...(single_mic_id ? { body: JSON.stringify({ single_mic_id }) } : {}),
  }),
  resume: (device_id?: number, candidate_mic_device_id?: number | null, single_mic_id?: string) => request<{ ok: boolean }>('/api/unpause', {
    method: 'POST',
    body: JSON.stringify({
      ...(device_id != null ? { device_id } : {}),
      ...(candidate_mic_device_id != null ? { candidate_mic_device_id } : {}),
      ...(single_mic_id ? { single_mic_id } : {}),
    }),
  }),
  switchSingleMicRole: (session_id: string, role: SingleMicRole) =>
    request<{ ok: true; single_mic?: SingleMicState }>('/api/single-mic/role', {
      method: 'POST',
      body: JSON.stringify({ session_id, role }),
    }),
  clear: () => request<{ ok: boolean }>('/api/clear', { method: 'POST' }),
  ask: (text: string, image?: string, purpose?: 'knowledge_practice') =>
    request<{ ok: boolean }>('/api/ask', { method: 'POST', body: JSON.stringify({ text, image, purpose }) }),
  cancelAsk: () => request<{ ok: boolean }>('/api/ask/cancel', { method: 'POST' }),
  preflightScenarios: () => request<{ scenarios: { id: string; label: string; question: string; recommended: boolean }[] }>('/api/preflight/scenarios'),
  preflightRun: (scenario_id: string, device_id?: number | null) =>
    request<Record<string, unknown>>('/api/preflight/run', {
      method: 'POST',
      body: JSON.stringify({ scenario_id, device_id: device_id ?? undefined }),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  preflightStatus: () => request<PreflightStatusResponse>('/api/preflight/status'),
  examPreflightRun: () =>
    request<{ ok: boolean; preflight_id: string }>('/api/exam-preflight/run', {
      method: 'POST',
      body: '{}',
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  examPreflightStatus: () => request<Record<string, unknown>>('/api/exam-preflight/status'),
  audioOutputTest: () =>
    request<{ ok: boolean; elapsed_sec: number }>('/api/audio-test/output', {
      method: 'POST',
      body: '{}',
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  audioInputTest: (device_id: number, duration_sec = 1.2) =>
    request<{
      ok: boolean
      device_id: number
      elapsed_sec: number
      samples?: number
      rms: number
      peak: number
      has_signal: boolean
      detail: string
    }>('/api/audio-test/input', {
      method: 'POST',
      body: JSON.stringify({ device_id, duration_sec }),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  audioInputMonitorStart: (device_id: number) =>
    request<{
      running: boolean
      device_id: number
      rms: number
      peak: number
      level_pct: number
      has_signal: boolean
      error?: string | null
    }>('/api/audio-test/input/start', { method: 'POST', body: JSON.stringify({ device_id }) }),
  audioInputMonitorStatus: () =>
    request<{
      running: boolean
      device_id: number | null
      rms: number
      peak: number
      level_pct: number
      has_signal: boolean
      error?: string | null
    }>('/api/audio-test/input/status'),
  audioInputMonitorStop: () =>
    request<{ running: boolean }>('/api/audio-test/input/stop', { method: 'POST', body: '{}' }),
  /** 服务端截取本机主屏左半幅 + VL 写码（手机端用，不经过手机截图 API） */
  askFromServerScreen: () =>
    request<Record<string, unknown>>('/api/ask-from-server-screen', {
      method: 'POST',
      body: '{}',
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  captureServerScreen: () =>
    request<{ ok: boolean; image: string; written_exam_mode: boolean }>('/api/capture-server-screen', {
      method: 'POST',
      body: '{}',
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  askFromServerScreens: (images: string[], expectedWrittenExamMode?: boolean) =>
    request<Record<string, unknown>>('/api/ask-from-server-screens', {
      method: 'POST',
      body: JSON.stringify({ images, expected_written_exam_mode: expectedWrittenExamMode }),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  getSttStatus: () => request<Record<string, unknown>>('/api/stt/status'),
  checkModelsHealth: (indices?: number[]) => request<import('@/stores/slices/types').ModelCatalogSnapshot>(
    '/api/models/health',
    {
      method: 'POST',
      body: indices ? JSON.stringify({ indices }) : undefined,
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    },
  ),
  /** 当前各模型健康状态（检测中/可用/不可用） */
  getModelsHealth: () => request<{
    health: Record<string, string>
    detail?: Record<string, string>
    latency?: Record<string, number>
    fingerprint?: Record<string, string>
  }>('/api/models/health'),
  listRemoteModels: (payload: { api_base_url: string; api_key: string; model_id?: string; model_index?: number; api_type?: string }) =>
    request<{ models: { id: string; owned_by?: string | null }[] }>('/api/models/list', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  checkSingleModelHealth: (index: number) =>
    request<Record<string, unknown>>('/api/models/health/' + index, {
      method: 'POST',
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  probeModelCapabilities: (index: number, modelId?: string) =>
    request<{
      ok: boolean
      detail?: string
      latency_ms?: number
      supports_vision: boolean
      supports_think: boolean
      think_style: string
      think_params: Record<string, unknown>
      think_disabled_params: Record<string, unknown>
      vision_detail?: string
      think_detail?: string
      think_disabled_detail?: string
    }>(`/api/models/probe/${index}${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''}`, {
      method: 'POST',
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  sttTest: (payload: { provider: import('@/lib/speechProviders').SegmentedSpeechProvider }) =>
    request<{ ok: boolean; detail?: string; text?: string }>('/api/stt/test', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),

  // Knowledge
  knowledgeSettings: () => request<KnowledgeSettingsPayload>('/api/knowledge/settings'),
  saveKnowledgeSettings: (settings: KnowledgeSettings & { model_fingerprint: string | null }) =>
    request<KnowledgeSettingsPayload>('/api/knowledge/settings', { method: 'PUT', body: JSON.stringify(settings) }),
  knowledgeAnalysisStatus: () => request<KnowledgeAnalysisState>('/api/knowledge/analysis'),
  startKnowledgeAnalysis: () => request<KnowledgeAnalysisState>('/api/knowledge/analysis', { method: 'POST' }),
  knowledgeSummary: () => request<{ tags: KnowledgeSummaryTag[] }>('/api/knowledge/summary'),
  knowledgeHistory: (page: number = 1, pageSize: number = 20) =>
    request<{ records: HistoryRecord[]; total: number }>(`/api/knowledge/history?page=${page}&page_size=${pageSize}`),
  knowledgeReset: () => request<{ ok: boolean }>('/api/knowledge/reset', { method: 'DELETE' }),

  // Review (面试复盘)
  reviewSessions: (page: number = 1, pageSize: number = 20) =>
    request<Record<string, unknown>>(`/api/review/sessions?page=${page}&page_size=${pageSize}`),
  reviewSessionDetail: (sessionId: number) =>
    request<Record<string, unknown>>(`/api/review/sessions/${sessionId}`),
  reviewTurnArchive: (sessionId: number, turnId: number, signal?: AbortSignal) =>
    request<import('@/components/review/types').ArchiveDetails>(`/api/review/sessions/${sessionId}/turns/${turnId}/archive`, { signal }),
  reviewAudio: (sessionId: number, turnId: number, assetId: string, signal?: AbortSignal) =>
    fetchBackendArrayBuffer(
      buildApiUrl(`/api/review/sessions/${sessionId}/turns/${turnId}/audio/${encodeURIComponent(assetId)}`),
      { signal, headers: buildHeaders() },
    ),
  reviewUpdateSession: (sessionId: number, data: { title?: string; company?: string; role?: string; application_id?: number | null }) =>
    request<Record<string, unknown>>(`/api/review/sessions/${sessionId}`, { method: 'PATCH', body: JSON.stringify(data) }),
  reviewTriggerAnalysis: (sessionId: number) =>
    request<{ status: string; message: string }>(`/api/review/sessions/${sessionId}/generate`, { method: 'POST', body: '{}' }),
  reviewCreateManual: (data: { transcript: string; title?: string; company?: string; role?: string; analyze?: boolean }) =>
    request<{ session_id: number; turn_count: number; status: string }>('/api/review/sessions/manual', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  reviewAsrCorrectionTest: (data?: { question?: string; answer?: string }) =>
    request<{
      ok: boolean
      model_name: string
      model: string
      original: string
      corrected: string
      changed: boolean
      detail?: string
    }>('/api/review/asr-correction-test', {
      method: 'POST',
      body: JSON.stringify(data ?? {}),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),
  reviewCurrent: () =>
    request<Record<string, unknown>>('/api/review/current'),
  reviewProfile: () =>
    request<Record<string, unknown>>('/api/review/profile'),

  // Resume optimizer
  resumeOptimize: (jd: string) =>
    request<Record<string, unknown>>('/api/resume/optimize', {
      method: 'POST',
      body: JSON.stringify({ jd }),
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    }),

  // Token
  tokenStats: () => request<Record<string, unknown>>('/api/token/stats'),

  // Job tracker (desktop / local SQLite)
  jobTrackerStages: () => request<{ stages: string[] }>('/api/job-tracker/stages'),
  jobTrackerApplications: (params?: { stage?: string; q?: string; sort_by?: string; sort_dir?: string }) => {
    const sp = new URLSearchParams()
    if (params?.stage) sp.set('stage', params.stage)
    if (params?.q) sp.set('q', params.q)
    if (params?.sort_by) sp.set('sort_by', params.sort_by)
    if (params?.sort_dir) sp.set('sort_dir', params.sort_dir)
    const qs = sp.toString()
    return request<{ items: Record<string, unknown>[] }>(`/api/job-tracker/applications${qs ? `?${qs}` : ''}`)
  },
  jobTrackerCreateApplication: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>('/api/job-tracker/applications', { method: 'POST', body: JSON.stringify(body) }),
  jobTrackerPatchApplication: (id: number, body: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/api/job-tracker/applications/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  jobTrackerDeleteApplication: (id: number) =>
    request<{ ok: boolean }>(`/api/job-tracker/applications/${id}`, { method: 'DELETE' }),
  jobTrackerApplicationReviews: (id: number) =>
    request<{ items: Record<string, unknown>[] }>(`/api/job-tracker/applications/${id}/reviews`),
  jobTrackerBatchStage: (ids: number[], stage: string) =>
    request<Record<string, unknown>>('/api/job-tracker/applications/batch-stage', {
      method: 'PATCH',
      body: JSON.stringify({ ids, stage }),
    }),
  jobTrackerReorderStage: (stage: string, orderedIds: number[]) =>
    request<{ updated: number }>('/api/job-tracker/applications/reorder-stage', {
      method: 'PATCH',
      body: JSON.stringify({ stage, ordered_ids: orderedIds }),
    }),
  jobTrackerListOffers: () => request<{ items: Record<string, unknown>[] }>('/api/job-tracker/offers'),
  jobTrackerUpsertOffer: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>('/api/job-tracker/offers', { method: 'POST', body: JSON.stringify(body) }),
  jobTrackerPatchOffer: (id: number, body: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/api/job-tracker/offers/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  jobTrackerDeleteOffer: (id: number) =>
    request<{ ok: boolean }>(`/api/job-tracker/offers/${id}`, { method: 'DELETE' }),
  jobTrackerCompare: (offer_ids: number[]) =>
    request<{ items: Record<string, unknown>[] }>('/api/job-tracker/compare', {
      method: 'POST',
      body: JSON.stringify({ offer_ids }),
    }),

  // Knowledge Base
  kbStatus: () => request<KBStatus>('/api/kb/status'),
  kbDocs: (limit?: number) =>
    request<{ items: KBDoc[] }>(`/api/kb/docs${limit ? `?limit=${limit}` : ''}`),
  kbSearch: (query: string, k = 4, min_score = 0) =>
    request<{ hits: KBHit[] }>('/api/kb/search', {
      method: 'POST',
      body: JSON.stringify({ query, k, min_score }),
    }),
  kbHitsRecent: (limit = 50) =>
    request<{ items: KBRecentHit[] }>(`/api/kb/hits/recent?limit=${limit}`),
  kbReindex: () => request<Record<string, unknown>>('/api/kb/reindex', {
    method: 'POST',
    body: '{}',
    timeoutMs: LONG_REQUEST_TIMEOUT_MS,
  }),
  kbDelete: (path: string) =>
    request<{ ok: boolean }>(`/api/kb/docs?path=${encodeURIComponent(path)}`, { method: 'DELETE' }),
  kbUpload: async (file: File, subdir = ''): Promise<{ path: string; size: number }> => {
    const fd = new FormData()
    fd.append('file', file)
    if (subdir) fd.append('subdir', subdir)
    const token = getAuthToken()
    const headers: Record<string, string> = {}
    if (token) headers.Authorization = `Bearer ${token}`
    const { res, text } = await fetchBackend(buildApiUrl('/api/kb/upload'), {
      method: 'POST',
      body: fd,
      headers,
      timeoutMs: LONG_REQUEST_TIMEOUT_MS,
    })
    if (!res.ok) {
      throw new Error(buildResponseErrorMessage(res, text, '上传失败'))
    }
    return JSON.parse(text)
  },
}
