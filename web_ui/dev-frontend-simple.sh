#!/bin/bash
cd frontend
echo "🚀 Starting React frontend..."
echo "🌐 Local: http://localhost:3005"
echo "🌐 Network: http://172.20.10.3:3005"
echo "📱 Anyone on your WiFi can access the network URL"
npm run dev -- --host 0.0.0.0 --port 3005
