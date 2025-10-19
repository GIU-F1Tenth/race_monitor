#!/bin/bash
# Test the web UI basic functionality

echo "🧪 Testing Race Monitor Web UI"
echo "=============================="

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "🌐 Your machine IP: $LOCAL_IP"
echo ""

# Kill any existing processes
echo "🧹 Clearing ports..."
sudo fuser -k 8001/tcp 2>/dev/null || true
sudo fuser -k 3001/tcp 2>/dev/null || true

# Test backend first
echo "🐍 Testing backend..."
cd backend

# Use the configured Python environment
/home/mohammedazab/ws/src/race_stack/race_monitor/.venv/bin/python -c "
import fastapi
import uvicorn
print('✅ Backend dependencies OK')
" || {
    echo "❌ Backend dependencies missing"
    exit 1
}

echo "🚀 Starting backend on port 8001..."
/home/mohammedazab/ws/src/race_stack/race_monitor/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Test if backend is running
if curl -s http://localhost:8001/api/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend starting up..."
fi

# Start frontend
echo "⚛️  Starting frontend on port 3001..."
cd ../frontend

# Check if node modules exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

echo "🌐 Frontend will be available at:"
echo "   Local: http://localhost:3001"
echo "   Network: http://$LOCAL_IP:3001"
echo ""

# Start frontend
PORT=3001 npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers started!"
echo ""
echo "🎯 Open your browser to: http://localhost:3001"
echo "📱 Or from your phone/other devices: http://$LOCAL_IP:3001"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait