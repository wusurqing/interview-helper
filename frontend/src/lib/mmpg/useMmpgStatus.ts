import { useSyncExternalStore } from 'react'
import { mmpgController } from './rtcController'

export function useMmpgStatus() {
  return useSyncExternalStore(mmpgController.subscribe, mmpgController.getSnapshot)
}
