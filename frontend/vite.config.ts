import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendAddress = process.env.IA_DEV_BACKEND_PORT
  ? `127.0.0.1:${process.env.IA_DEV_BACKEND_PORT}`
  : 'localhost:18080'

function manualChunks(id: string) {
  if (!id.includes('node_modules')) return

  if (
    id.includes('react-syntax-highlighter')
    || id.includes('/refractor/')
    || id.includes('/prismjs/')
  ) {
    return 'markdown-syntax'
  }

  if (
    id.includes('react-markdown')
    || id.includes('/remark-')
    || id.includes('/rehype-')
    || id.includes('/unified/')
    || id.includes('/mdast-')
    || id.includes('/hast-')
    || id.includes('/micromark')
    || id.includes('/vfile/')
    || id.includes('/property-information/')
    || id.includes('/space-separated-tokens/')
    || id.includes('/comma-separated-tokens/')
    || id.includes('/bail/')
  ) {
    return 'markdown-core'
  }

  if (
    id.includes('@dnd-kit')
    || id.includes('@tanstack/react-table')
  ) {
    return 'workspace-heavy'
  }

  if (id.includes('lucide-react')) {
    return 'icons'
  }
}

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: '@', replacement: path.resolve(__dirname, './src') },
    ],
  },
  server: {
    proxy: {
      // Keep the browser's Host header for the backend's same-origin check.
      '/api': { target: `http://${backendAddress}`, changeOrigin: false },
      '/ws': {
        target: `ws://${backendAddress}`,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    css: true,
    include: ['src/**/*.{test,spec}.{ts,tsx,js,jsx}'],
    exclude: ['node_modules', 'dist', 'e2e/**'],
  },
})
