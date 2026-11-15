import { useCallback, useEffect, useRef, useState } from 'react'
import { useInterviewStore } from '@/stores/configStore'

const clampPercent = (value: number) => Math.max(0, Math.min(90, Math.round(value)))

export function WindowOpacitySlider() {
  const [supported, setSupported] = useState(false)
  const [percent, setPercent] = useState(0)
  const regionRef = useRef<HTMLDivElement>(null)
  const desiredRef = useRef(0)
  const confirmedRef = useRef(0)
  const busyRef = useRef(false)
  const mountedRef = useRef(false)

  useEffect(() => {
    mountedRef.current = true
    const api = window.electronAPI
    if (api?.getWindowState && api.setWindowOpacity) {
      void api.getWindowState().then((state) => {
        if (!mountedRef.current || !state.opacitySupported) return
        const initial = clampPercent((1 - state.opacity) * 100)
        confirmedRef.current = initial
        desiredRef.current = initial
        setPercent(initial)
        setSupported(true)
      }).catch(() => {})
    }
    return () => { mountedRef.current = false }
  }, [])

  const sendLatest = useCallback(async () => {
    if (busyRef.current) return
    busyRef.current = true
    try {
      while (mountedRef.current && desiredRef.current !== confirmedRef.current) {
        const requested = desiredRef.current
        try {
          const actual = await window.electronAPI!.setWindowOpacity(Number((1 - requested / 100).toFixed(2)))
          if (!Number.isFinite(actual)) throw new Error('Invalid opacity response')
          confirmedRef.current = clampPercent((1 - actual) * 100)
          if (mountedRef.current && desiredRef.current === requested) setPercent(confirmedRef.current)
        } catch {
          if (mountedRef.current && desiredRef.current === requested) {
            desiredRef.current = confirmedRef.current
            setPercent(confirmedRef.current)
            useInterviewStore.getState().setToastMessage('窗口透明度调节失败，请重试')
          }
        }
      }
    } finally {
      busyRef.current = false
    }
  }, [])

  const changePercent = useCallback((value: number) => {
    const next = clampPercent(value)
    desiredRef.current = next
    setPercent(next)
    void sendLatest()
  }, [sendLatest])

  useEffect(() => {
    if (!supported) return
    const region = regionRef.current
    if (!region) return
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY === 0) return
      event.preventDefault()
      changePercent(desiredRef.current + (event.deltaY < 0 ? 2 : -2))
    }
    region.addEventListener('wheel', onWheel, { passive: false })
    return () => region.removeEventListener('wheel', onWheel)
  }, [supported, changePercent])

  if (!supported) return null

  return (
    <div
      ref={regionRef}
      role="group"
      aria-label="窗口透明度调节"
      title="拖动滑块或悬停滚轮调节窗口透明度"
      className="app-no-drag inline-flex flex-shrink-0 items-center gap-1 rounded-lg border border-bg-hover/40 bg-bg-tertiary/40 px-1.5 py-1 text-text-secondary"
    >
      <span className="hidden text-[10px] font-medium xl:inline">透明度</span>
      <input
        type="range"
        min={0}
        max={90}
        step={1}
        value={percent}
        onChange={(event) => changePercent(Number(event.target.value))}
        aria-label="窗口透明度"
        aria-valuetext={`${percent}% 透明`}
        className="w-14 cursor-pointer accent-accent-blue sm:w-20 xl:w-24"
      />
      <span className="w-7 text-right text-[10px] tabular-nums">{percent}%</span>
    </div>
  )
}
