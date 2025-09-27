#!/bin/bash
# Quick Start Script - Race Monitor Web UI

echo "🚀 Quick Start - Race Monitor Web UI"
echo "====================================="

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "🌐 Your machine IP: $LOCAL_IP"
echo ""

# Kill any existing processes
echo "🧹 Clearing ports..."
sudo fuser -k 8001/tcp 2>/dev/null || true
sudo fuser -k 3001/tcp 2>/dev/null || true

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend on port 8001
echo "🐍 Starting backend on port 8001..."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend on port 3001
echo "⚛️  Starting frontend on port 3001..."
cd ../frontend
VITE_API_URL=http://localhost:8001 npm run dev -- --host 0.0.0.0 --port 3001 &
FRONTEND_PID=$!

echo ""
echo "✅ Development servers started!"
echo ""
echo "🖥️  Local Access:"
echo "   Frontend: http://localhost:3001"
echo "   Backend:  http://localhost:8001"
echo "   API Docs: http://localhost:8001/docs"
echo ""
echo "📱 Network Access (WiFi users):"
echo "   Frontend: http://$LOCAL_IP:3001"
echo "   Backend:  http://$LOCAL_IP:8001"
echo ""
echo "🎯 Open your browser to: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for processes
wait