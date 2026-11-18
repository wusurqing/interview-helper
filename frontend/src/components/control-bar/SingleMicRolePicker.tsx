import type { SingleMicRole } from '@/stores/slices/types'

const roles = [
  ['interviewer', '面试官'],
  ['candidate', '候选人'],
] as const

interface SingleMicRolePickerProps {
  role: SingleMicRole
  disabled: boolean
  switching: boolean
  connecting: boolean
  onSelect: (role: SingleMicRole) => void
}

export function SingleMicRolePicker({ role, disabled, switching, connecting, onSelect }: SingleMicRolePickerProps) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex flex-wrap items-center gap-x-2 text-[10px] font-medium leading-none text-text-muted">
        <span>发言角色</span>
        <span role="status" className="text-accent-amber">
          {switching ? '切换中，等待上一句转写完成' : connecting ? '语音未就绪，请稍候' : ''}
        </span>
      </div>
      <div role="group" aria-label="单人测试发言角色" className="flex h-10 items-center gap-1 rounded-lg border border-bg-hover bg-bg-tertiary p-1">
        {roles.map(([value, label]) => (
          <button
            key={value}
            type="button"
            aria-pressed={role === value}
            disabled={disabled}
            onClick={() => onSelect(value)}
            className="h-full flex-1 rounded-md text-xs font-medium text-text-secondary transition-colors enabled:hover:bg-bg-hover aria-pressed:bg-accent-blue/15 aria-pressed:font-semibold aria-pressed:text-accent-blue disabled:cursor-not-allowed disabled:opacity-60"
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}
