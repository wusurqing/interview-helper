import type { QAPair } from '@/stores/slices/types'

export default function PlainAnswerContent({ text }: { text: string }) {
  return (
    <div data-answer-format="plain_text" className="whitespace-pre-wrap [overflow-wrap:anywhere]">
      {text}
    </div>
  )
}

export function plainAnswerStatus(qa: QAPair, isStreaming: boolean) {
  if (qa.status === 'error') return qa.errorMessage || '答案保存失败'
  if (qa.status === 'cancelled') return qa.cancelReason === 'continuation' ? '已因续说停止' : '已取消'
  if (isStreaming) return qa.isThinking ? '思考中…' : '输出中…'
  return '已完成'
}
