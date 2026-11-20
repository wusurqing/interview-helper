import { useState, useEffect, type ReactNode } from 'react'
import {
  Palette,
  Monitor,
  ChevronDown,
  BookOpen,
} from 'lucide-react'
import { useInterviewStore } from '@/stores/configStore'
import { useUiPrefsStore } from '@/stores/uiPrefsStore'
import { updateConfigAndRefresh } from '@/lib/configSync'
import { COLOR_SCHEME_OPTIONS } from '@/lib/colorScheme'
import { Section, Field, SaveStateBadge, matchSettingsSearch, useAutoSaveSetting, useSettingsSearch } from './shared'

function Collapsible({ title, searchTitle, icon, defaultOpen = false, alwaysOpen = false, keywords, children }: {
  title: React.ReactNode
  searchTitle?: string
  icon?: React.ReactNode
  defaultOpen?: boolean
  /** 分区级单卡片：不渲染折叠按钮，内容常开 */
  alwaysOpen?: boolean
  keywords?: string
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  const query = useSettingsSearch()
  const titleText = searchTitle ?? (typeof title === 'string' ? title : '')
  if (!matchSettingsSearch(titleText, keywords, query)) return null
  const effectiveOpen = alwaysOpen || (query.trim() ? true : open)
  return (
    <section className="rounded-xl border border-bg-hover/60 bg-bg-primary/35 overflow-hidden" data-search-title={titleText}>
      {alwaysOpen ? (
        <div className="flex items-center gap-2 px-4 py-3">
          {icon && <span className="text-text-muted flex-shrink-0">{icon}</span>}
          <span className="text-sm font-semibold text-text-primary">{title}</span>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-bg-tertiary/25 transition-colors"
        >
          {icon && <span className="text-text-muted flex-shrink-0">{icon}</span>}
          <span className="text-sm font-semibold text-text-primary">{title}</span>
          <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${effectiveOpen ? 'rotate-180' : ''}`} />
        </button>
      )}
      {effectiveOpen && <div className="px-4 pb-4 space-y-3 border-t border-bg-hover/50">{children}</div>}
    </section>
  )
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 rounded bg-bg-tertiary border-bg-hover text-accent-blue focus:ring-accent-blue focus:ring-offset-0"
      />
      <span className="text-xs text-text-secondary">{label}</span>
    </label>
  )
}

export default function PreferencesTab({ variant, children }: { variant: 'mode' | 'kb' | 'appearance'; children?: ReactNode }) {
  const config = useInterviewStore((s) => s.config)
  const options = useInterviewStore((s) => s.options)
  const colorScheme = useUiPrefsStore((s) => s.colorScheme)
  const setColorScheme = useUiPrefsStore((s) => s.setColorScheme)
  const [scrollBottomPx, setScrollBottomPx] = useState(40)
  const [screenMaxLongEdge, setScreenMaxLongEdge] = useState(1600)
  const [multiScreenIdleSec, setMultiScreenIdleSec] = useState(10)
  const [kbTopK, setKbTopK] = useState(4)
  const [kbDeadlineMs, setKbDeadlineMs] = useState(150)
  const [kbAsrDeadlineMs, setKbAsrDeadlineMs] = useState(80)
  const autoSave = useAutoSaveSetting<Record<string, unknown>>(updateConfigAndRefresh)

  useEffect(() => {
    if (config?.answer_autoscroll_bottom_px != null) {
      setScrollBottomPx(config.answer_autoscroll_bottom_px)
    }
    if (config?.screen_capture_max_long_edge != null) {
      setScreenMaxLongEdge(config.screen_capture_max_long_edge)
    }
    if (config?.multi_screen_capture_idle_sec != null) {
      setMultiScreenIdleSec(config.multi_screen_capture_idle_sec)
    }
    if (config?.kb_top_k != null) {
      setKbTopK(config.kb_top_k)
    }
    if (config?.kb_deadline_ms != null) {
      setKbDeadlineMs(config.kb_deadline_ms)
    }
    if (config?.kb_asr_deadline_ms != null) {
      setKbAsrDeadlineMs(config.kb_asr_deadline_ms)
    }
  }, [
    config?.answer_autoscroll_bottom_px,
    config?.screen_capture_max_long_edge,
    config?.multi_screen_capture_idle_sec,
    config?.kb_top_k,
    config?.kb_deadline_ms,
    config?.kb_asr_deadline_ms,
  ])

  const hasScreenCapture = (options?.screen_capture_regions?.length ?? 0) > 0
  const searchQuery = useSettingsSearch()
  const inSearch = searchQuery.trim().length > 0

  return (
    <div className="p-5 space-y-5 pb-8" data-in-search={inSearch ? '1' : undefined}>
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-bg-hover/60 bg-bg-primary/35 px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-text-primary">
            {variant === 'appearance' ? '外观' : variant === 'mode' ? '笔试配置' : '本地知识库'}
          </p>
          <p className="mt-0.5 text-[11px] leading-relaxed text-text-muted">
            {variant === 'appearance'
              ? '配色即时生效，流式跟滚阈值自动保存。'
              : variant === 'mode'
                ? '从顶部「笔试辅助」进入答题。'
                : '开启后回答会引用本地知识库笔记。'}
          </p>
        </div>
        {variant !== 'appearance' && <SaveStateBadge mode="auto" state={autoSave.state} error={autoSave.error} />}
      </div>

      {variant === 'mode' && (<>
      {children}
      <Collapsible
        alwaysOpen
        title="截图审题"
        icon={<Monitor className="w-3.5 h-3.5" />}
        keywords="截图 笔试 vision exam 连续截图 多图 Think 深度思考"
      >
        {config && hasScreenCapture && (
          <Field label="截图区域" htmlFor="exam-screen-region" hint="服务端截图审题时，后端截取主显示器的范围">
            <select
              id="exam-screen-region"
              value={config.screen_capture_region ?? 'left_half'}
              onChange={async (e) => {
                await autoSave.saveNow({ screen_capture_region: e.target.value })
              }}
              className="input-field w-full max-w-[200px]"
            >
              {(options?.screen_capture_regions ?? []).map((r) => (
                <option key={r} value={r}>
                  {r === 'full' ? '全屏' : r === 'left_half' ? '左半屏' : r === 'right_half' ? '右半屏' : r === 'top_half' ? '上半屏' : '下半屏'}
                </option>
              ))}
            </select>
          </Field>
        )}

        {config && hasScreenCapture && (
          <Field
            label={`截图最长边: ${screenMaxLongEdge}px`}
            htmlFor="exam-screen-max-edge"
            hint="送入识图模型前缩放截图，降低带宽和 token；0 表示不缩放"
          >
            <input
              id="exam-screen-max-edge"
              type="number"
              min={0}
              max={4000}
              step={100}
              value={screenMaxLongEdge}
              onChange={(e) => {
                const raw = Number(e.target.value)
                const v = Number.isFinite(raw) ? Math.max(0, Math.min(4000, raw)) : 1600
                setScreenMaxLongEdge(v)
                autoSave.saveDebounced({ screen_capture_max_long_edge: v })
              }}
              onBlur={() => {
                const v = Math.max(0, Math.min(4000, screenMaxLongEdge || 0))
                setScreenMaxLongEdge(v)
                void autoSave.flush({ screen_capture_max_long_edge: v })
              }}
              className="input-field w-full max-w-[120px]"
            />
          </Field>
        )}

        {config && hasScreenCapture && (
          <Field
            label={`多图截图等待: ${multiScreenIdleSec} 秒`}
            htmlFor="exam-screen-batch-wait"
            hint="多图截图判题快捷键最后一次按下后，等待这段时间无新截图就提交整批图片 (1-60)"
          >
            <input
              id="exam-screen-batch-wait"
              type="number"
              min={1}
              max={60}
              step={1}
              value={multiScreenIdleSec}
              onChange={(e) => {
                const v = Math.max(1, Math.min(60, Number(e.target.value) || 10))
                setMultiScreenIdleSec(v)
                autoSave.saveDebounced({ multi_screen_capture_idle_sec: v })
              }}
              onBlur={() => {
                const v = Math.max(1, Math.min(60, multiScreenIdleSec || 10))
                setMultiScreenIdleSec(v)
                void autoSave.flush({ multi_screen_capture_idle_sec: v })
              }}
              className="input-field w-full max-w-[120px]"
            />
          </Field>
        )}

        {config && (
          <div className="space-y-1 border-t border-bg-hover/40 pt-3">
            <Toggle
              checked={config.written_exam_think ?? false}
              onChange={(enabled) => { void autoSave.saveNow({ written_exam_think: enabled }) }}
              label="深度思考 (Think)"
            />
            <p className="text-[11px] text-text-muted">用于笔试截图答题，需模型支持 Think。</p>
          </div>
        )}
      </Collapsible>
      </>)}

      {/* ── 3. 知识库 ── */}
      {variant === 'kb' && (<>
      <Collapsible
        alwaysOpen
        title="知识库与引用"
        icon={<BookOpen className="w-3.5 h-3.5" />}
        keywords="kb knowledge base 笔记 参考 rag retrieval 引用"
      >
        <Field
          label="主流程引用本地笔记"
          hint="开启后 LLM 会在回答前从本地知识库检索相关内容并以 [角标] 引用; 关闭仅 Drawer 内手动测试可用"
        >
          <Toggle
            checked={config?.kb_enabled ?? false}
            onChange={async (v) => {
              try {
                await autoSave.saveNow({ kb_enabled: v })
                useInterviewStore.getState().setToastMessage(v ? '已开启知识库' : '已关闭知识库')
              } catch (e) {
                useInterviewStore.getState().setToastMessage(e instanceof Error ? e.message : '保存失败')
              }
            }}
            label={config?.kb_enabled ? '已开启' : '已关闭'}
          />
        </Field>
        <Field
          label={`命中数 top_k: ${kbTopK}`}
          hint="一次检索最多返回的笔记片段数 (1-20)"
        >
          <input
            type="number"
            min={1}
            max={20}
            value={kbTopK}
            onChange={(e) => {
              const v = Math.max(1, Math.min(20, Number(e.target.value) || 4))
              setKbTopK(v)
              autoSave.saveDebounced({ kb_top_k: v })
            }}
            onBlur={() => {
              const v = Math.max(1, Math.min(20, kbTopK || 4))
              setKbTopK(v)
              void autoSave.flush({ kb_top_k: v })
            }}
            className="w-full max-w-[120px] bg-bg-tertiary border border-bg-hover rounded-lg px-3 py-2 text-sm text-text-primary"
          />
        </Field>
        <Field
          label={`手动模式 deadline: ${kbDeadlineMs} ms`}
          hint="手动输入 / 截图模式下检索的硬上限; 超时不阻塞首字, 直接返回空 (20-2000)"
        >
          <input
            type="number"
            min={20}
            max={2000}
            step={10}
            value={kbDeadlineMs}
            onChange={(e) => {
              const v = Math.max(20, Math.min(2000, Number(e.target.value) || 150))
              setKbDeadlineMs(v)
              autoSave.saveDebounced({ kb_deadline_ms: v })
            }}
            onBlur={() => {
              const v = Math.max(20, Math.min(2000, kbDeadlineMs || 150))
              setKbDeadlineMs(v)
              void autoSave.flush({ kb_deadline_ms: v })
            }}
            className="w-full max-w-[120px] bg-bg-tertiary border border-bg-hover rounded-lg px-3 py-2 text-sm text-text-primary"
          />
        </Field>
        <Field
          label={`ASR 模式 deadline: ${kbAsrDeadlineMs} ms`}
          hint="实时语音模式下更紧的检索上限, 优先保证首字延迟 (20-1000)"
        >
          <input
            type="number"
            min={20}
            max={1000}
            step={10}
            value={kbAsrDeadlineMs}
            onChange={(e) => {
              const v = Math.max(20, Math.min(1000, Number(e.target.value) || 80))
              setKbAsrDeadlineMs(v)
              autoSave.saveDebounced({ kb_asr_deadline_ms: v })
            }}
            onBlur={() => {
              const v = Math.max(20, Math.min(1000, kbAsrDeadlineMs || 80))
              setKbAsrDeadlineMs(v)
              void autoSave.flush({ kb_asr_deadline_ms: v })
            }}
            className="w-full max-w-[120px] bg-bg-tertiary border border-bg-hover rounded-lg px-3 py-2 text-sm text-text-primary"
          />
        </Field>
        <p className="text-[10px] text-text-muted leading-relaxed">
          点击右上角 📖 图标管理文档/手动检索测试; OCR、Vision 等高阶配置仍在 backend/config.json 中。
        </p>
      </Collapsible>
      </>)}

      {/* ── 4. 外观 ── */}
      {variant === 'appearance' && (<>
      <Collapsible alwaysOpen title="外观" icon={<Palette className="w-3.5 h-3.5" />} keywords="主题 配色方案 字体 流式跟滚阈值 滚动 theme color scheme font autoscroll">
        <div className="flex justify-end">
          <SaveStateBadge mode="auto" state="idle" label="本机即时生效" />
        </div>
        <Field label="配色方案">
          <div className="grid grid-cols-1 gap-1.5">
            {COLOR_SCHEME_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => {
                  setColorScheme(opt.id)
                  useInterviewStore.getState().setToastMessage(`已切换为 ${opt.label}`)
                }}
                className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-left transition-all ${
                  colorScheme === opt.id
                    ? 'border-accent-blue bg-accent-blue/10 ring-1 ring-accent-blue/30'
                    : 'border-bg-hover bg-bg-tertiary/30 hover:border-bg-hover'
                }`}
              >
                <Palette className="w-3 h-3 text-accent-blue flex-shrink-0" />
                <span className="text-xs font-medium text-text-primary">{opt.label}</span>
                <span className="text-[10px] text-text-muted ml-auto">{opt.hint}</span>
              </button>
            ))}
          </div>
        </Field>
        <Field label="流式跟滚阈值（像素）" hint="距底部小于该值时自动滚到底（4～400）">
          <input
            type="number" min={4} max={400} value={scrollBottomPx}
            onChange={(e) => {
              const v = Math.max(4, Math.min(400, Number(e.target.value) || 40))
              setScrollBottomPx(v)
              autoSave.saveDebounced({ answer_autoscroll_bottom_px: v })
            }}
            onBlur={() => {
              const v = Math.max(4, Math.min(400, scrollBottomPx || 40))
              setScrollBottomPx(v)
              void autoSave.flush({ answer_autoscroll_bottom_px: v })
            }}
            className="w-full max-w-[120px] bg-bg-tertiary border border-bg-hover rounded-lg px-3 py-2 text-sm text-text-primary"
          />
        </Field>
      </Collapsible>
      </>)}

    </div>
  )
}
