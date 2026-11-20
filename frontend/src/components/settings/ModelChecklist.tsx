import { useId, useLayoutEffect, useRef, type ReactNode } from 'react'
import { GripVertical } from 'lucide-react'
import { closestCenter, DndContext, PointerSensor, useSensor, useSensors, type DragEndEvent } from '@dnd-kit/core'
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import VisionBadge from '@/components/VisionBadge'
import type { ModelInfo } from '@/stores/slices/types'

export type CatalogModel = ModelInfo & { id: string }

export function visionModelUnavailableReason(model: ModelInfo): string | undefined {
  if (!model.supports_vision) return '该模型不支持图片作为输入'
  if (model.enabled === false) return '已停用'
  if (model.has_api_key !== true) return '未配置 API Key'
}

interface ModelChoiceProps {
  model: CatalogModel
  checked: boolean
  disabled: boolean
  status?: string
  onToggle: (id: string) => void
}

function ModelChoice({ model, checked, disabled, status, onToggle }: ModelChoiceProps) {
  const statusId = useId()
  return (
    <label className={`flex min-w-0 flex-1 items-center gap-2.5 py-2.5 ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}>
      <input type="checkbox" aria-label={`用于答题 ${model.name}`} checked={checked}
        aria-describedby={status ? statusId : undefined}
        disabled={disabled} onChange={() => onToggle(model.id)} className="shrink-0 accent-accent-blue" />
      <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <span className={`truncate text-xs font-medium ${disabled || status ? 'text-text-muted' : 'text-text-primary'}`} title={model.name}>{model.name}</span>
        {model.supports_vision && <VisionBadge />}
        {status && <span id={statusId} className="text-[10px] text-accent-amber">{status}</span>}
      </span>
    </label>
  )
}

function SortableModel({ position, onStep, ...props }: Omit<ModelChoiceProps, 'checked'> & {
  position: number
  onStep: (modelId: string, offset: -1 | 1) => boolean
}) {
  const { model, disabled } = props
  const focusAfterMove = useRef<HTMLButtonElement | null>(null)
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } = useSortable({
    id: model.id,
    disabled,
  })
  useLayoutEffect(() => {
    focusAfterMove.current?.focus()
    focusAfterMove.current = null
  }, [position])

  return (
    <li ref={setNodeRef} style={{ transform: CSS.Transform.toString(transform), transition }}
      className={`relative flex items-center gap-2 rounded-lg border border-bg-hover/70 bg-bg-secondary px-2.5 ${isDragging ? 'z-10 shadow-lg' : ''}`}>
      <ModelChoice {...props} checked />
      <button {...attributes} {...listeners} ref={setActivatorNodeRef} type="button" disabled={disabled}
        aria-label={`排序 ${model.name}`}
        title="拖动或使用上、下方向键调整顺序"
        onKeyDown={(event) => {
          if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
          event.preventDefault()
          focusAfterMove.current = event.currentTarget
          if (!onStep(model.id, event.key === 'ArrowUp' ? -1 : 1)) focusAfterMove.current = null
        }}
        className="cursor-grab touch-none rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary active:cursor-grabbing disabled:opacity-50">
        <GripVertical className="h-4 w-4" aria-hidden />
      </button>
    </li>
  )
}

export default function ModelChecklist({
  models, selectedIds, onChange, title, description, action, disabled = false, requireVision = false,
}: {
  models: CatalogModel[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
  title: string
  description: string
  action?: ReactNode
  disabled?: boolean
  requireVision?: boolean
}) {
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }))
  const modelById = new Map(models.map((model) => [model.id, model]))
  const selected = selectedIds.flatMap((id) => {
    const model = modelById.get(id)
    return model ? [model] : []
  })
  const unselected = models.filter((model) => !selectedIds.includes(model.id))

  function statusFor(model: CatalogModel) {
    if (requireVision) return visionModelUnavailableReason(model)
    if (model.enabled === false) return '已停用'
    if (model.has_api_key === false) return '未配置 API Key'
  }

  function toggleModel(id: string) {
    onChange(selectedIds.includes(id) ? selectedIds.filter((item) => item !== id) : [...selectedIds, id])
  }

  function moveModel(modelId: string, targetId: string) {
    const from = selectedIds.indexOf(modelId)
    const to = selectedIds.indexOf(targetId)
    if (from < 0 || to < 0 || from === to) return false
    onChange(arrayMove(selectedIds, from, to))
    return true
  }

  function stepModel(modelId: string, offset: -1 | 1) {
    const position = selected.findIndex(({ id }) => id === modelId)
    const target = selected[position + offset]
    return Boolean(target && moveModel(modelId, target.id))
  }

  function reorderModels({ active, over }: DragEndEvent) {
    if (over) moveModel(String(active.id), String(over.id))
  }

  return (
    <>
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-xs font-semibold text-text-primary">{title}</h3>
          <p className="mt-1 text-[11px] text-text-muted">{description}</p>
        </div>
        {action}
      </div>
      {selected.length > 0 ? (
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={reorderModels}
          accessibility={{ screenReaderInstructions: { draggable: '使用上、下方向键调整模型的优先顺序。' } }}>
          <SortableContext items={selected.map(({ id }) => id)} strategy={verticalListSortingStrategy}>
            <ol aria-label={title} className="space-y-1.5">
              {selected.map((model, position) => (
                <SortableModel key={model.id} model={model} position={position} status={statusFor(model)}
                  onToggle={toggleModel} onStep={stepModel} disabled={disabled} />
              ))}
            </ol>
          </SortableContext>
        </DndContext>
      ) : <p className="rounded-lg bg-bg-tertiary/40 p-3 text-xs text-text-muted">请勾选要参与答题的模型。</p>}

      {unselected.length > 0 && (
        <div className="space-y-1">
          <h3 className="text-xs font-semibold text-text-primary">未勾选模型</h3>
          <ul aria-label="未勾选模型" className="divide-y divide-bg-hover/60 px-2.5">
            {unselected.map((model) => {
              const status = statusFor(model)
              return (
                <li key={model.id}>
                  <ModelChoice model={model} checked={false} status={status} onToggle={toggleModel}
                    disabled={disabled || (requireVision && Boolean(status))} />
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </>
  )
}
