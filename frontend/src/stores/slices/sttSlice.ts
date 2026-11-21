import type { StateCreator } from 'zustand'
import type { RootState } from './rootState'
import type { ModelCatalogSnapshot, ModelCatalogStatus, ModelHealthStatus, TokenUsage } from './types'
import type { MmpgState } from '@/lib/mmpg/rtcController'

export interface SttSliceState {
  sttLoaded: boolean
  sttLoading: boolean
  sttActiveProvider: string
  mmpgPhase: MmpgState | null
  mmpgError: string | null
  candidateSttLoaded: boolean
  candidateSttLoading: boolean
  candidateSttProvider: string
  modelHealth: Record<number, ModelHealthStatus>
  modelHealthDetail: Record<number, string>
  modelHealthLatency: Record<number, number>
  modelCatalog: Record<number, { status: ModelCatalogStatus; detail?: string }>
  tokenUsage: TokenUsage
}

export interface SttSliceActions {
  setSttStatus: (loaded: boolean, loading: boolean, provider?: string) => void
  setMmpgStatus: (phase: MmpgState | null, error: string | null) => void
  setCandidateSttStatus: (loaded: boolean, loading: boolean, provider?: string) => void
  setModelHealth: (
    index: number,
    status: ModelHealthStatus,
    detail?: string,
    latencyMs?: number,
    modelFingerprint?: string,
  ) => void
  setModelCatalog: (snapshot: ModelCatalogSnapshot) => void
  setTokenUsage: (usage: TokenUsage) => void
}

export type SttSlice = SttSliceState & SttSliceActions

export const createSttSlice: StateCreator<RootState, [], [], SttSlice> = (set) => ({
  sttLoaded: false,
  sttLoading: false,
  sttActiveProvider: '',
  mmpgPhase: null,
  mmpgError: null,
  candidateSttLoaded: false,
  candidateSttLoading: false,
  candidateSttProvider: '',
  modelHealth: {},
  modelHealthDetail: {},
  modelHealthLatency: {},
  modelCatalog: {},
  tokenUsage: { prompt: 0, completion: 0, total: 0, byModel: {} },

  setSttStatus: (loaded, loading, provider) => set({
    sttLoaded: loaded,
    sttLoading: loading,
    ...(provider != null ? { sttActiveProvider: provider } : {}),
  }),
  setCandidateSttStatus: (loaded, loading, provider) => set({
    candidateSttLoaded: loaded,
    candidateSttLoading: loading,
    ...(provider != null ? { candidateSttProvider: provider } : {}),
  }),
  setMmpgStatus: (mmpgPhase, mmpgError) => set({ mmpgPhase, mmpgError }),
  setModelHealth: (index, status, detail, latencyMs, modelFingerprint) => set((s) => {
    const currentFingerprint = s.config?.models?.[index]?.health_fingerprint
    if (modelFingerprint && currentFingerprint && modelFingerprint !== currentFingerprint) return {}
    return {
      modelHealth: { ...s.modelHealth, [index]: status },
      ...(detail != null ? { modelHealthDetail: { ...s.modelHealthDetail, [index]: detail } } : {}),
      ...(latencyMs != null ? { modelHealthLatency: { ...s.modelHealthLatency, [index]: latencyMs } } : {}),
    }
  }),
  setModelCatalog: (snapshot) => set((s) => {
    const catalog = { ...s.modelCatalog }
    for (const [key, status] of Object.entries(snapshot.catalog ?? {})) {
      const index = Number(key)
      const model = s.config?.models[index]
      const fingerprint = snapshot.fingerprint?.[index]
      if (!model || model.enabled === false || (fingerprint && fingerprint !== model.health_fingerprint)) continue
      catalog[index] = { status, detail: snapshot.detail?.[index] }
    }
    return { modelCatalog: catalog }
  }),
  setTokenUsage: (usage) =>
    set({
      tokenUsage: {
        prompt: usage.prompt,
        completion: usage.completion,
        total: usage.total,
        byModel: usage.byModel ?? {},
      },
    }),
})
