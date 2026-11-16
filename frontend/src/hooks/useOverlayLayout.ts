import { useEffect, useState } from 'react'
import type { OverlayLayoutSnapshot } from '@/lib/interviewOverlay'

export function useOverlayLayout() {
  const [layout, setLayout] = useState<OverlayLayoutSnapshot>()
  const [layoutError, setLayoutError] = useState(false)

  useEffect(() => {
    const api = window.electronAPI
    if (!api?.getOverlayLayout) return
    let subscribed = true
    let receivedEvent = false
    const unsubscribe = api.onOverlayLayout?.((snapshot) => {
      receivedEvent = true
      setLayout(snapshot)
      setLayoutError(false)
    })
    api.getOverlayLayout()
      .then((snapshot) => { if (subscribed && !receivedEvent) setLayout(snapshot) })
      .catch(() => { if (subscribed && !receivedEvent) setLayoutError(true) })
    return () => {
      subscribed = false
      unsubscribe?.()
    }
  }, [])

  return { layout, setLayout, layoutError }
}
