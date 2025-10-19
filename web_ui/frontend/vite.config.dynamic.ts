import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:9002',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:9002',
        ws: true,
        changeOrigin: true,
      }
    }
  }
})
