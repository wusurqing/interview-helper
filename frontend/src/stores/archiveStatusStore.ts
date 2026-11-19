import { create } from 'zustand'

export interface ArchiveStatus {
  session_id: number
  revision: number
  archive_status: string
  persisted_turn_count: number
  pending_round_count: number
  warning: string
}

export const useArchiveStatusStore = create<{ value: ArchiveStatus | null }>(() => ({ value: null }))

export function receiveArchiveStatus(raw: unknown, reset = false) {
  if (raw == null && reset) {
    useArchiveStatusStore.setState({ value: null })
    return
  }
  if (!raw || typeof raw !== 'object') return
  const value = raw as ArchiveStatus
  if (!Number.isInteger(value.session_id) || !Number.isInteger(value.revision)) return
  const current = useArchiveStatusStore.getState().value
  if (reset || !current || current.session_id < value.session_id ||
      (current.session_id === value.session_id && current.revision < value.revision)) {
    useArchiveStatusStore.setState({ value })
  }
}
