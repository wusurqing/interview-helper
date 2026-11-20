import { describe, expect, it } from 'vitest'

import { todosFromText, withMaterializedTodoIds } from './todoDraft'
import type { TodoItem } from './types'

describe('todosFromText', () => {
  it('does not generate ids during render-time diffing', () => {
    const todos = todosFromText('新待办\n另一个', [])
    expect(todos.map((todo) => todo.id)).toEqual(['', ''])
  })

  it('reuses existing todo data by position, then by title', () => {
    const current: TodoItem[] = [
      { id: 'a', title: '投简历', done: true },
      { id: 'b', title: '准备笔试', done: false, due: '2026-01-01' },
    ]
    const todos = todosFromText('准备笔试\n投简历\n新增', current)
    expect(todos).toEqual([
      { id: 'b', title: '准备笔试', done: false, due: '2026-01-01' },
      { id: 'a', title: '投简历', done: true, due: undefined },
      { id: '', title: '新增', done: false, due: undefined },
    ])
  })

  it('never reuses the same existing todo id for duplicate titles', () => {
    const current: TodoItem[] = [{ id: 'only', title: '跟进 HR', done: true }]
    const todos = todosFromText('跟进 HR\n跟进 HR', current)
    expect(todos[0]).toMatchObject({ id: 'only', done: true })
    expect(todos[1]).toMatchObject({ id: '', done: false })
  })

  it('maps duplicate existing titles one-to-one', () => {
    const current: TodoItem[] = [
      { id: 'x1', title: '同名', done: true },
      { id: 'x2', title: '同名', done: false },
    ]
    const todos = todosFromText('其他\n同名\n同名', current)
    const ids = todos.map((todo) => todo.id)
    expect(ids[0]).toBe('')
    expect(new Set(ids.slice(1))).toEqual(new Set(['x1', 'x2']))
  })
})

describe('withMaterializedTodoIds', () => {
  it('fills missing ids only and keeps the rest of the patch', () => {
    let n = 0
    const patch = withMaterializedTodoIds(
      {
        notes: 'n',
        todos: [
          { id: 'keep', title: 'a', done: false },
          { id: '', title: 'b', done: false },
          { id: '', title: 'c', done: false },
        ],
      },
      () => `gen-${++n}`,
    )
    expect(patch.notes).toBe('n')
    expect(patch.todos?.map((todo) => todo.id)).toEqual(['keep', 'gen-1', 'gen-2'])
  })

  it('returns the same patch when it has no todos', () => {
    const patch = { stage: 'offer' }
    expect(withMaterializedTodoIds(patch)).toBe(patch)
  })
})
