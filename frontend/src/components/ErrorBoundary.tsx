import React from 'react'

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ReactNode
  isOverlay?: boolean
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    error: null,
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  private handleReload = () => {
    window.location.reload()
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }
    if (this.props.fallback !== undefined) {
      return this.props.fallback
    }

    const message = this.state.error?.message || '发生了未预期的渲染错误'
    return (
      <div
        role="alert"
        className={
          this.props.isOverlay
            ? 'flex h-full w-full items-center justify-center p-3'
            : 'flex h-full w-full items-center justify-center bg-bg-primary p-6 text-text-primary'
        }
      >
        <div className="w-full max-w-md rounded-xl border border-border-default bg-bg-secondary p-5 shadow-lg">
          <h2 className="text-base font-semibold text-text-primary">页面发生异常</h2>
          <p className="mt-2 break-words text-sm text-text-secondary">{message}</p>
          <div className="mt-4 flex items-center gap-2">
            <button
              type="button"
              onClick={this.handleReset}
              className="rounded-lg border border-border-default bg-bg-primary px-3 py-1.5 text-sm text-text-primary hover:bg-bg-tertiary"
            >
              尝试恢复
            </button>
            <button
              type="button"
              onClick={this.handleReload}
              className="rounded-lg bg-accent-blue px-3 py-1.5 text-sm font-medium text-white hover:opacity-90"
            >
              刷新页面
            </button>
          </div>
        </div>
      </div>
    )
  }
}
