#!/bin/bash
# Dynamic Frontend Startup Script

cd "$(dirname "$0")" || exit 1

# Read ports from environment variables (set by main script)
BACKEND_PORT=${BACKEND_PORT:-$(python3 port_manager.py get | grep "Backend:" | cut -d' ' -f2)}
FRONTEND_PORT=${FRONTEND_PORT:-$(python3 port_manager.py get | grep "Frontend:" | cut -d' ' -f2)}

if [ -z "$BACKEND_PORT" ] || [ -z "$FRONTEND_PORT" ]; then
    echo "❌ Port configuration not available"
    exit 1
fi

echo "🔗 Backend running on port: $BACKEND_PORT"
echo "🔗 Frontend will use port: $FRONTEND_PORT"

# Update Vite configuration dynamically
cd frontend || exit 1

echo "📝 Updating Vite configuration..."
cat > vite.config.dynamic.ts << EOF
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: $FRONTEND_PORT,
    host: '0.0.0.0', // Allow access from any network interface
    proxy: {
      '/api': {
        target: 'http://localhost:$BACKEND_PORT',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:$BACKEND_PORT',
        ws: true,
        changeOrigin: true,
      }
    }
  }
})
EOF

echo "🚀 Starting React frontend..."
echo "🌐 Local: http://localhost:$FRONTEND_PORT"
echo "🔗 API Backend: http://localhost:$BACKEND_PORT"

# Start with dynamic configuration
npm run dev -- --config ./vite.config.dynamic.ts --host 0.0.0.0 --port $FRONTEND_PORT