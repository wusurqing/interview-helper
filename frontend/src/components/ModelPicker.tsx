import { useEffect, useId, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import VisionBadge from './VisionBadge'

interface ModelPickerOption {
  value: string
  label: string
  supportsVision?: boolean
  disabledReason?: string
}

export default function ModelPicker({ label, description, value, onChange, options, disabled = false, emptyLabel }: {
  label: string
  description?: string
  value: string
  onChange: (value: string) => void
  options: ModelPickerOption[]
  disabled?: boolean
  emptyLabel?: string
}) {
  const id = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  const selected = options.find((option) => option.value === value)
  const choices = emptyLabel === undefined ? options : [{ value: '', label: emptyLabel }, ...options]

  useEffect(() => {
    if (!open) return
    const dismiss = (event: MouseEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('click', dismiss)
    return () => document.removeEventListener('click', dismiss)
  }, [open])

  return (
    <div className="min-w-0 space-y-1.5" ref={rootRef} onKeyDown={(event) => {
      if (event.key !== 'Escape' || !open) return
      event.stopPropagation()
      setOpen(false)
      triggerRef.current?.focus()
    }}>
      <div id={`${id}-label`} className="text-xs font-medium text-text-secondary">{label}</div>
      {description && <p className="text-[11px] text-text-muted">{description}</p>}
      <button
        type="button"
        ref={triggerRef}
        aria-labelledby={`${id}-label`}
        aria-describedby={`${id}-value`}
        aria-expanded={open}
        aria-controls={`${id}-options`}
        disabled={disabled}
        onClick={() => setOpen((previous) => !previous)}
        className="flex w-full min-w-0 items-center gap-2 rounded-lg border border-bg-hover bg-bg-secondary px-3 py-2 text-left text-xs text-text-primary hover:border-accent-blue/40 disabled:opacity-50"
      >
        <span className="min-w-0 flex-1 truncate" id={`${id}-value`} title={selected?.label}>
          {selected?.label ?? emptyLabel ?? '选择模型'}
        </span>
        {selected?.supportsVision && <VisionBadge />}
        <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-text-muted ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
        <fieldset id={`${id}-options`} aria-label={`${label}选项`} disabled={disabled}
          className="max-h-56 space-y-0.5 overflow-y-auto rounded-lg border border-bg-hover/70 bg-bg-secondary p-1">
          {choices.map((option, index) => {
            const reasonId = option.disabledReason ? `${id}-reason-${index}` : undefined
            return (
              <label key={option.value}
                className={`flex min-w-0 items-start gap-2 rounded-md px-2 py-1.5 ${option.disabledReason ? 'cursor-not-allowed text-text-muted' : 'cursor-pointer hover:bg-bg-tertiary'}`}>
                <input type="radio" name={id} aria-label={option.label} aria-describedby={reasonId}
                  checked={option.value === value} disabled={Boolean(option.disabledReason)}
                  onChange={() => onChange(option.value)} className="mt-1 shrink-0 accent-accent-blue" />
                <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5">
                  <span className="break-words text-xs">{option.label}</span>
                  {option.supportsVision && <VisionBadge />}
                  {option.disabledReason && <span id={reasonId} className="text-[10px] text-accent-amber">{option.disabledReason}</span>}
                </span>
              </label>
            )
          })}
        </fieldset>
      )}
    </div>
  )
}
