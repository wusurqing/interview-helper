import { createClientId } from '@/lib/id'
import type { Application, TodoItem } from './types'

/**
 * 把编辑框文本映射为待办列表，尽量复用已有待办的 id/done/due。
 *
 * 渲染期（`useMemo` 里算脏状态）调用，因此**不能**生成随机 id：新增行的 id 先留空，
 * 真正提交前由 `withMaterializedTodoIds` 补齐。每条已有待办最多被复用一次，避免两行
 * 同名待办共用同一个 id。先按位置精确匹配，再按标题匹配剩余行。
 */
export function todosFromText(text: string, currentTodos: TodoItem[]): TodoItem[] {
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
  const consumed = new Set<number>()
  const matchedIndex: Array<number | undefined> = lines.map((title, index) => {
    if (currentTodos[index]?.title === title) {
      consumed.add(index)
      return index
    }
    return undefined
  })
  const indexesByTitle = new Map<string, number[]>()
  currentTodos.forEach((todo, index) => {
    const queue = indexesByTitle.get(todo.title)
    if (queue) queue.push(index)
    else indexesByTitle.set(todo.title, [index])
  })
  lines.forEach((title, lineIndex) => {
    if (matchedIndex[lineIndex] !== undefined) return
    const candidate = indexesByTitle.get(title)?.find((index) => !consumed.has(index))
    if (candidate !== undefined) {
      consumed.add(candidate)
      matchedIndex[lineIndex] = candidate
    }
  })
  return lines.map((title, lineIndex) => {
    const index = matchedIndex[lineIndex]
    const matched = index !== undefined ? currentTodos[index] : undefined
    return {
      id: matched?.id ?? '',
      title,
      done: matched?.done ?? false,
      due: matched?.due,
    }
  })
}

/** 提交前为新增待办生成 id（兼容非 Secure Context）。 */
export function withMaterializedTodoIds(
  patch: Partial<Application>,
  createId: () => string = createClientId,
): Partial<Application> {
  if (!patch.todos) return patch
  return {
    ...patch,
    todos: patch.todos.map((todo) => (todo.id ? todo : { ...todo, id: createId() })),
  }
}
