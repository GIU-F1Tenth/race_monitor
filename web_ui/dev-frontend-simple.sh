#!/bin/bash
cd frontend
echo "🚀 Starting React frontend..."
echo "🌐 Local: http://localhost:3000"
echo "🌐 Network: http://172.20.10.3:3000"
echo "📱 Anyone on your WiFi can access the network URL"
npm run dev -- --host 0.0.0.0
