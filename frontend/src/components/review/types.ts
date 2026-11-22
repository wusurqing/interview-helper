import { parseReviewScore } from '@/lib/reviewScore'

export const ARCHIVE_SPEAKERS = [['interviewer', '面试官'], ['candidate', '候选人']] as const
export type ArchiveSpeaker = typeof ARCHIVE_SPEAKERS[number][0]
const AUDIO_STATES = ['ready', 'approximate', 'partial', 'missing', 'none'] as const
type AudioState = typeof AUDIO_STATES[number]

interface ArchiveAudioCoverage {
  state: AudioState
  duration_ms: number
  reason?: string
}

export function archiveAudioIssue(state?: AudioState) {
  if (state === 'partial') return '部分录音缺失'
  if (state === 'missing') return '录音缺失'
  return null
}

export interface ArchiveClip {
  asset_id: string
  offset: number
  duration: number
  sample_start?: number
  sample_count?: number
  capture_id?: string
}

export interface ArchiveDetails {
  round_id: string
  policy_version: string
  transcription_timeouts?: ArchiveSpeaker[]
  utterances: Array<{
    id: string; speaker: string; text: string; started_mono: number; is_final: boolean; alignment: string
    transcription_failed?: boolean
  }>
  generations: Array<{
    id: string; status: string; answer: string; thinking: string; error?: string
    started_at: number; finished_at?: number; first_token_at: number | null; first_token_ms: number | null; total_ms: number | null
    model: { name?: string; model?: string; protocol?: string; endpoint?: string }
    attempts: Array<{ attempt_id: string; call_id: string; status: string; request_at: string; finished_at: string; duration_ms: number; first_output_latency_ms?: number; reasoning_text: string; output_text: string; error?: string }>
  }>
}

export interface ReviewSession {
  id: number
  status: 'recording' | 'recorded' | 'analyzing' | 'completed' | 'partial_capture' | 'analysis_failed'
  started_at: number
  ended_at: number | null
  source: string
  title?: string | null
  company?: string | null
  role?: string | null
  auto_sync_eligible?: boolean
  application_id?: number | null
  application?: {
    id: number
    company: string
    position: string
    city: string
    stage: string
    applied_at?: number | null
    next_followup_at?: number | null
    updated_at?: number | null
  } | null
  jd_snapshot?: string | null
  resume_snapshot?: string | null
  interviewer_capture_enabled: boolean
  candidate_capture_enabled: boolean
  turn_count: number
  avg_score: number | null
  summary_markdown?: string | null
  strong_points?: string[] | null
  weak_points?: string[] | null
  behavior_traits?: string[] | null
  domain_summary?: Record<string, unknown> | null
  created_at: number
  updated_at: number
  archive_status?: string | null
  archive_warning?: string | null
  archive_version?: number
}

export interface ReviewTurn {
  id: number
  session_id: number
  qa_id: string
  seq: number
  question_text: string
  candidate_answer_text: string
  original_candidate_answer_text?: string | null
  reference_answer_text?: string | null
  code_text?: string | null
  duration_ms: number
  is_partial: boolean
  analysis_status: 'pending' | 'analyzing' | 'completed' | 'failed'
  strengths?: string[] | null
  risks?: string[] | null
  evidence?: Record<string, unknown> | null
  scorecard?: Record<string, unknown> | null
  created_at: number
  updated_at: number
  round_id?: string | null
  archive_version?: number
  candidate_status?: string | null
  ai_status?: string | null
  media?: { interviewer: ArchiveClip[]; candidate: ArchiveClip[] } | null
  audio_status?: Record<ArchiveSpeaker, ArchiveAudioCoverage | null> | null
  model_summary?: { name?: string; model?: string; first_token_ms?: number | null; total_ms?: number | null } | null
}

export interface ReviewSessionDetail extends ReviewSession {
  turns: ReviewTurn[]
}

export interface ReviewSessionsResponse {
  total: number
  page: number
  page_size: number
  items: ReviewSession[]
}

const REVIEW_STATUSES: ReviewSession['status'][] = [
  'recording',
  'recorded',
  'analyzing',
  'completed',
  'partial_capture',
  'analysis_failed',
]

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? value as Record<string, unknown> : {}
}

function finiteNumber(value: unknown): number | null {
  if (value == null) return null
  if (typeof value === 'string' && value.trim() === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function parseNumber(value: unknown, fallback = 0): number {
  return finiteNumber(value) ?? fallback
}

function parseNullableNumber(value: unknown): number | null {
  return finiteNumber(value)
}

function parseNonNegativeInteger(value: unknown, fallback = 0): number {
  return Math.max(0, Math.floor(parseNumber(value, fallback)))
}

function parsePositiveInteger(value: unknown, fallback = 1): number {
  return Math.max(1, Math.floor(parseNumber(value, fallback)))
}

function parseOptionalBoolean(value: unknown): boolean | undefined {
  if (value == null) return undefined
  return parseBoolean(value)
}

function parseBoolean(value: unknown, fallback = false): boolean {
  if (value == null) return fallback
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (['true', '1', 'yes'].includes(normalized)) return true
    if (['false', '0', 'no'].includes(normalized)) return false
  }
  return Boolean(value)
}

function parseStringList(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null
  return value.map((item) => String(item)).filter(Boolean)
}

function parseStatus(value: unknown): ReviewSession['status'] {
  const status = String(value ?? '')
  return REVIEW_STATUSES.includes(status as ReviewSession['status'])
    ? status as ReviewSession['status']
    : 'recorded'
}

function parseReviewApplication(value: unknown): ReviewSession['application'] {
  if (!value || typeof value !== 'object') return null
  const raw = value as Record<string, unknown>
  return {
    id: parseNumber(raw.id),
    company: String(raw.company ?? ''),
    position: String(raw.position ?? ''),
    city: String(raw.city ?? ''),
    stage: String(raw.stage ?? 'applied'),
    applied_at: parseNullableNumber(raw.applied_at),
    next_followup_at: parseNullableNumber(raw.next_followup_at),
    updated_at: parseNullableNumber(raw.updated_at),
  }
}

export function parseReviewTurn(raw: Record<string, unknown>): ReviewTurn {
  return {
    round_id: raw.round_id == null ? null : String(raw.round_id),
    archive_version: parseNonNegativeInteger(raw.archive_version),
    candidate_status: raw.candidate_status == null ? null : String(raw.candidate_status),
    ai_status: raw.ai_status == null ? null : String(raw.ai_status),
    media: parseArchiveMedia(raw.media),
    audio_status: parseAudioCoverage(raw.audio_status),
    model_summary: raw.model_summary && typeof raw.model_summary === 'object' ? raw.model_summary as ReviewTurn['model_summary'] : null,
    id: parseNumber(raw.id),
    session_id: parseNumber(raw.session_id),
    qa_id: String(raw.qa_id ?? ''),
    seq: parseNumber(raw.seq),
    question_text: String(raw.question_text ?? ''),
    candidate_answer_text: String(raw.candidate_answer_text ?? ''),
    original_candidate_answer_text: raw.original_candidate_answer_text != null ? String(raw.original_candidate_answer_text) : null,
    reference_answer_text: raw.reference_answer_text != null ? String(raw.reference_answer_text) : null,
    code_text: raw.code_text != null ? String(raw.code_text) : null,
    duration_ms: parseNonNegativeInteger(raw.duration_ms),
    is_partial: parseBoolean(raw.is_partial),
    analysis_status: ['pending', 'analyzing', 'completed', 'failed'].includes(String(raw.analysis_status ?? ''))
      ? String(raw.analysis_status) as ReviewTurn['analysis_status']
      : 'pending',
    strengths: parseStringList(raw.strengths),
    risks: parseStringList(raw.risks),
    evidence: raw.evidence && typeof raw.evidence === 'object' ? raw.evidence as Record<string, unknown> : null,
    scorecard: raw.scorecard && typeof raw.scorecard === 'object' ? raw.scorecard as Record<string, unknown> : null,
    created_at: parseNumber(raw.created_at),
    updated_at: parseNumber(raw.updated_at),
  }
}

export function parseReviewSession(raw: Record<string, unknown>): ReviewSession {
  return {
    id: parseNumber(raw.id),
    status: parseStatus(raw.status),
    started_at: parseNumber(raw.started_at),
    ended_at: parseNullableNumber(raw.ended_at),
    source: String(raw.source ?? 'assist'),
    title: raw.title != null ? String(raw.title) : null,
    company: raw.company != null ? String(raw.company) : null,
    role: raw.role != null ? String(raw.role) : null,
    auto_sync_eligible: parseOptionalBoolean(raw.auto_sync_eligible),
    application_id: parseNullableNumber(raw.application_id),
    application: parseReviewApplication(raw.application),
    jd_snapshot: raw.jd_snapshot != null ? String(raw.jd_snapshot) : null,
    resume_snapshot: raw.resume_snapshot != null ? String(raw.resume_snapshot) : null,
    interviewer_capture_enabled: parseBoolean(raw.interviewer_capture_enabled),
    candidate_capture_enabled: parseBoolean(raw.candidate_capture_enabled),
    turn_count: parseNonNegativeInteger(raw.turn_count),
    avg_score: parseReviewScore(raw.avg_score),
    summary_markdown: raw.summary_markdown != null ? String(raw.summary_markdown) : null,
    strong_points: parseStringList(raw.strong_points),
    weak_points: parseStringList(raw.weak_points),
    behavior_traits: parseStringList(raw.behavior_traits),
    domain_summary: raw.domain_summary && typeof raw.domain_summary === 'object' ? raw.domain_summary as Record<string, unknown> : null,
    created_at: parseNumber(raw.created_at),
    updated_at: parseNumber(raw.updated_at),
  }
}

export function parseReviewSessionDetail(raw: Record<string, unknown>): ReviewSessionDetail {
  const turns = Array.isArray(raw.turns) ? raw.turns.map((turn) => parseReviewTurn(asRecord(turn))) : []
  return {
    ...parseReviewSession(raw),
    archive_status: raw.archive_status == null ? null : String(raw.archive_status),
    archive_warning: raw.archive_warning == null ? null : String(raw.archive_warning),
    archive_version: parseNonNegativeInteger(raw.archive_version),
    turns,
  }
}

function parseArchiveMedia(raw: unknown): ReviewTurn['media'] {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const clips = (value: unknown): ArchiveClip[] => Array.isArray(value) ? value.flatMap((item) => {
    const clip = asRecord(item)
    const duration = finiteNumber(clip.duration)
    const offset = finiteNumber(clip.offset)
    return typeof clip.asset_id === 'string' && /^[a-f0-9]{64}$/.test(clip.asset_id) && duration != null && duration > 0 && offset != null && offset >= 0
      ? [{ asset_id: clip.asset_id, offset, duration }] : []
  }) : []
  return { interviewer: clips(source.interviewer), candidate: clips(source.candidate) }
}

function parseAudioCoverage(value: unknown): ReviewTurn['audio_status'] {
  if (!value || typeof value !== 'object') return null
  const raw = asRecord(value)
  const coverage = (speaker: ArchiveSpeaker): ArchiveAudioCoverage | null => {
    const item = asRecord(raw[speaker])
    const state = item.state as AudioState
    if (!AUDIO_STATES.includes(state)) return null
    return { state, duration_ms: parseNonNegativeInteger(item.duration_ms), reason: typeof item.reason === 'string' ? item.reason : undefined }
  }
  return { interviewer: coverage('interviewer'), candidate: coverage('candidate') }
}

export function parseReviewSessionsResponse(raw: unknown): ReviewSessionsResponse {
  const record = asRecord(raw)
  const items = Array.isArray(record.items)
    ? record.items.map((item) => parseReviewSession(asRecord(item)))
    : []
  return {
    total: parseNonNegativeInteger(record.total, items.length),
    page: parsePositiveInteger(record.page, 1),
    page_size: parsePositiveInteger(record.page_size, Math.max(1, items.length)),
    items,
  }
}
