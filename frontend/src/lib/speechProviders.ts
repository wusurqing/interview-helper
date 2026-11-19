export const speechProviders = [
  { id: 'doubao', label: '火山原生豆包' },
  { id: 'generic', label: 'OpenAI 兼容 ASR' },
  { id: 'doubao_mmpg', label: '豆包 MMPG' },
] as const

export type SpeechProvider = typeof speechProviders[number]['id']
export type SegmentedSpeechProvider = Exclude<SpeechProvider, 'doubao_mmpg'>

export function speechProvider(value: string | undefined): SpeechProvider | '' {
  return speechProviders.find(({ id }) => id === value)?.id ?? ''
}

export function isSegmentedSpeech(provider: SpeechProvider | '') {
  return provider === 'doubao' || provider === 'generic'
}
