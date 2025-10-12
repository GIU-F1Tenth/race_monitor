import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3005,
    host: '0.0.0.0', // Allow access from any network interface
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8082',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8082',
        ws: true,
        changeOrigin: true,
      }
    }
  }
})