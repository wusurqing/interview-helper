import { useRef, useState } from 'react'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { useSettingsDirtyRegistration, type SaveState } from './shared'

function diff<T extends object>(value: Partial<T>, saved: T): Partial<T> {
  return Object.fromEntries(Object.entries(value).filter(([key, entry]) => !Object.is(entry, saved[key as keyof T]))) as Partial<T>
}

export function useSpeechSettingsDraft<T extends object>(key: string, saved: T) {
  const [draft, setDraft] = useState<Partial<T>>({})
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [error, setError] = useState<string | null>(null)
  const savingRef = useRef(false)
  const changes = diff(draft, saved)
  const dirty = Object.keys(changes).length > 0
  const form = { ...saved, ...changes }
  useSettingsDirtyRegistration(key, dirty)

  function edit(patch: Partial<T>) {
    setDraft((current) => diff({ ...current, ...patch }, saved))
    setSaveState('idle')
    setError(null)
  }

  function reset() {
    setDraft({})
    setSaveState('idle')
    setError(null)
  }

  async function save(toPayload: (patch: Partial<T>) => Record<string, unknown> = (patch) => ({ ...patch })) {
    if (savingRef.current) return false
    if (!dirty) return true
    savingRef.current = true
    setSaveState('saving')
    setError(null)
    try {
      const body = toPayload(changes)
      if (Object.keys(body).length) await updateConfigAndRefresh(body)
      setDraft({})
      setSaveState('saved')
      return true
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : '保存失败')
      setSaveState('error')
      return false
    } finally {
      savingRef.current = false
    }
  }

  const state = saveState === 'idle' && dirty ? 'dirty' : saveState
  return { form, edit, reset, save, state, error, dirty, saving: saveState === 'saving' }
}
