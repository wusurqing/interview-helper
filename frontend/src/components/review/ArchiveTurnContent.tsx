import { useEffect, useState } from 'react'
import { Info } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { api, getErrorMessage } from '@/lib/api'
import ArchiveAudioPlayer from './ArchiveAudioPlayer'
import { ARCHIVE_SPEAKERS, archiveAudioIssue, type ArchiveDetails, type ArchiveSpeaker, type ReviewTurn } from './types'

const candidateStatus: Record<string, string> = {
  disabled: '未开启候选人采集', unanswered: '未回答', transcription_failed: '转写失败', unavailable: '候选人麦克风不可用',
}
const answerStatus: Record<string, string> = {
  failed: 'AI 生成失败', cancelled: 'AI 生成已取消', timed_out: 'AI 生成超时', not_requested: '未触发 AI 回答', completed: '已生成',
}
const audioFailure: Record<string, string> = {
  buffer_full: '音频缓存已达上限', queue_full: '编码队列已满', encode_failed: '音频编码失败',
  unmatched: '无法确认原声归属', samples_missing: '部分采样未收到', capture_gap: '设备采集检测到断点',
}
const milliseconds = (value?: number | null) => value == null ? '—' : `${Math.round(value)} ms`

function ArchiveRecording({ turn, speaker, label }: { turn: ReviewTurn; speaker: ArchiveSpeaker; label: string }) {
  const clips = turn.media?.[speaker]
  const coverage = turn.audio_status?.[speaker]
  const issue = archiveAudioIssue(coverage?.state)
  const reason = audioFailure[coverage?.reason ?? '']
  return (
    <>
      {coverage?.state === 'approximate' && <p className="text-xs text-text-muted">本轮原声 · 近似定位</p>}
      {issue && <p className="text-xs text-accent-amber">{label}{issue}{reason && ` · ${reason}`}</p>}
      {clips?.length ? <ArchiveAudioPlayer clips={clips} sessionId={turn.session_id} turnId={turn.id} label={label} />
        : !issue && (speaker === 'interviewer' || (turn.candidate_answer_text && !coverage))
          ? <p className="text-xs text-text-muted">{coverage ? '无录音' : '历史记录未保存此录音'}</p>
          : null}
    </>
  )
}

export function ArchiveTurnContent({ turn }: { turn: ReviewTurn }) {
  const [open, setOpen] = useState(false)
  const [details, setDetails] = useState<ArchiveDetails | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    if (!open) return
    const abort = new AbortController()
    setError('')
    api.reviewTurnArchive(turn.session_id, turn.id, abort.signal).then((value) => {
      if (!abort.signal.aborted) setDetails(value)
    }).catch((cause) => {
      if (!abort.signal.aborted) setError(getErrorMessage(cause, '无法加载归档详情'))
    })
    return () => abort.abort()
  }, [open, turn.session_id, turn.id, turn.updated_at])

  const model = turn.model_summary
  const timing = model ? `${model.name || model.model || '模型'} · 首字 ${milliseconds(model.first_token_ms)} · 完成 ${milliseconds(model.total_ms)}` : '模型和对话详情'
  return (
    <div className="space-y-3 text-sm text-text-primary">
      <section>
        <h4 className="mb-1 text-xs font-semibold text-text-muted">面试官录音</h4>
        <ArchiveRecording turn={turn} speaker="interviewer" label="面试官" />
      </section>
      <section>
        <h4 className="mb-1 text-xs font-semibold text-text-muted">候选人回答</h4>
        <p className="whitespace-pre-wrap">{turn.candidate_answer_text || candidateStatus[turn.candidate_status ?? ''] || '未获取到回答'}</p>
        <ArchiveRecording turn={turn} speaker="candidate" label="候选人" />
      </section>
      <section>
        <div className="mb-1 flex items-center gap-2">
          <h4 className="text-xs font-semibold text-text-muted">AI 参考答案</h4>
          <button type="button" title={timing} aria-label="查看对话和 AI 调用详情" aria-expanded={open}
            className="rounded p-1 text-text-muted hover:bg-bg-hover" onClick={() => setOpen(!open)}><Info className="h-4 w-4" /></button>
        </div>
        {turn.reference_answer_text ? <div className="markdown-body"><ReactMarkdown>{turn.reference_answer_text}</ReactMarkdown></div>
          : <p>{answerStatus[turn.ai_status ?? ''] || '无参考答案'}</p>}
      </section>
      {open && <section className="space-y-3 rounded border border-bg-hover p-3 text-xs">
        {error ? <p role="alert">{error}</p> : !details ? <p role="status">正在加载详情…</p> : <>
          <div>
            <h5 className="mb-2 font-semibold">对话顺序</h5>
            {ARCHIVE_SPEAKERS.map(([speaker, label]) => details.transcription_timeouts?.includes(speaker) && (
              <p key={speaker} className="mb-1 text-accent-amber">{label}末尾转写超时</p>
            ))}
            {details.utterances.map((item) => <p key={item.id} className="mb-1 whitespace-pre-wrap">
              <span className="text-text-muted">{item.speaker === 'interviewer' ? '面试官 · ' : '候选人 · '}</span>
              {item.text || (item.transcription_failed ? '转写失败' : '无转写文本')}
            </p>)}
          </div>
          <div>
            <h5 className="mb-2 font-semibold">AI 生成历史</h5>
            {details.generations.length === 0 && <p>未触发 AI 回答</p>}
            {details.generations.map((run, index) => <details key={run.id} className="border-t border-bg-hover py-2">
              <summary className="cursor-pointer">{run.model.name || run.model.model || `生成 ${index + 1}`} · {answerStatus[run.status] || run.status}</summary>
              <p className="mt-2 text-text-muted">{[run.model.model, run.model.protocol, run.model.endpoint].filter(Boolean).join(' · ')}</p>
              <p>首字 {milliseconds(run.first_token_ms)} · 完成 {milliseconds(run.total_ms)}</p>
              <p className="text-text-muted">开始 {new Date(run.started_at * 1000).toLocaleTimeString()} · 首字 {run.first_token_at == null ? '—' : new Date(run.first_token_at * 1000).toLocaleTimeString()} · 结束 {run.finished_at == null ? '—' : new Date(run.finished_at * 1000).toLocaleTimeString()}</p>
              {run.error && <p>{run.error}</p>}
              {run.answer && <p className="mt-2 whitespace-pre-wrap">{run.answer}</p>}
              {run.thinking && !run.attempts.some((attempt) => attempt.reasoning_text) && <p className="mt-2 whitespace-pre-wrap">{run.thinking}</p>}
              {run.attempts.map((attempt, attemptIndex) => <details key={attempt.attempt_id} className="mt-2 border-l border-bg-hover pl-2">
                <summary className="cursor-pointer">请求 {attemptIndex + 1} · {attempt.status} · {milliseconds(attempt.duration_ms)}</summary>
                <p className="text-text-muted">{attempt.request_at} → {attempt.finished_at}</p>
                {attempt.error && <p>{attempt.error}</p>}
                <p className="whitespace-pre-wrap">{attempt.reasoning_text || '接口未返回推理文本'}</p>
                {attempt.output_text && <p className="mt-2 whitespace-pre-wrap">{attempt.output_text}</p>}
              </details>)}
              {!run.thinking && !run.attempts.length && <p className="text-text-muted">未收到推理文本</p>}
            </details>)}
          </div>
        </>}
      </section>}
    </div>
  )
}
