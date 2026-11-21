import React, { Suspense, lazy } from 'react'
import ReactDOM from 'react-dom/client'
import './lib/auth' // 抽取 URL 中的 ?t=<token>(必须先于其他 import 触发副作用)
import { ErrorBoundary } from './components/ErrorBoundary'
import { markStandalone } from './lib/wsLeader'
import './index.css'

const App = lazy(() => import('./App'))
const InterviewOverlay = lazy(() => import('./components/InterviewOverlay'))

const isOverlayView = new URLSearchParams(window.location.search).get('overlay') === '1'

// Overlay 窗口跳过 BroadcastChannel leader 选举，独立持有 WS。
// 见 lib/wsLeader.ts:markStandalone 注释。必须在 useInterviewWS 触发 init 之前。
if (isOverlayView) markStandalone()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary isOverlay={isOverlayView}>
      <Suspense fallback={null}>
        {isOverlayView ? <InterviewOverlay /> : <App />}
      </Suspense>
    </ErrorBoundary>
  </React.StrictMode>,
)
