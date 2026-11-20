export const mmpgSecrets = ['volc_access_key', 'volc_secret_key', 'rtc_app_key', 'rtc_token', 'doubao_api_key'] as const
export type MmpgSecret = typeof mmpgSecrets[number]

export const mmpgAnswerLabels = {
  waiting: '等待续说',
  continuing: '继续识别',
  generating: '生成回答',
} as const

export function mmpgAnswerLabel(phase: unknown): string | null {
  if (phase === 'waiting' || phase === 'continuing' || phase === 'generating') return mmpgAnswerLabels[phase]
  return null
}

export type MmpgPublicFields = {
  token_mode: 'auto' | 'manual'
  rtc_app_id: string
  rtc_room_id: string
  rtc_user_id: string
  rtc_bot_id: string
  rtc_token_expires_at: number | null
  doubao_resource_id: string
  aivad: boolean
  silence_ms: number
  answer_delay_ms: number
  followup_window_ms: number
  twopass: boolean
  enable_prefetch: boolean
  expire_time_ms: number | null
  force_end: boolean
}

export type MmpgConfig = MmpgPublicFields & Partial<Record<`${MmpgSecret}_configured`, boolean>>
export type MmpgForm = MmpgPublicFields & Partial<Record<MmpgSecret, string | null>>
export type MmpgEngineForm = Omit<MmpgForm, 'answer_delay_ms' | 'followup_window_ms'>

export function mmpgFormFromConfig(config?: MmpgConfig): MmpgForm {
  return {
    token_mode: config?.token_mode ?? 'auto',
    rtc_app_id: config?.rtc_app_id ?? '',
    rtc_room_id: config?.rtc_room_id ?? '',
    rtc_user_id: config?.rtc_user_id ?? '',
    rtc_bot_id: config?.rtc_bot_id ?? '',
    rtc_token_expires_at: config?.rtc_token_expires_at ?? null,
    doubao_resource_id: config?.doubao_resource_id ?? 'volc.seedasr.sauc.duration',
    aivad: config?.aivad ?? true,
    silence_ms: config?.silence_ms ?? 800,
    answer_delay_ms: config?.answer_delay_ms ?? 1500,
    followup_window_ms: config?.followup_window_ms ?? 1500,
    twopass: config?.twopass ?? true,
    enable_prefetch: config?.enable_prefetch ?? true,
    expire_time_ms: config?.expire_time_ms ?? null,
    force_end: config?.force_end ?? false,
  }
}

export function mmpgEngineFormFromConfig(config?: MmpgConfig): MmpgEngineForm {
  const { answer_delay_ms: _delay, followup_window_ms: _followup, ...engine } = mmpgFormFromConfig(config)
  return engine
}
