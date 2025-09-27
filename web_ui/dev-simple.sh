#!/bin/bash
echo "🚀 Starting Race Monitor Web UI (Simple Mode)"
echo "============================================="

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

# Replace placeholder in scripts
sed -i "s/LOCAL_IP/$LOCAL_IP/g" dev-backend-simple.sh
sed -i "s/LOCAL_IP/$LOCAL_IP/g" dev-frontend-simple.sh

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down development servers..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

echo "🌐 Your machine IP: $LOCAL_IP"
echo ""

# Start backend in background
echo "Starting backend..."
./dev-backend-simple.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "Starting frontend..."
./dev-frontend-simple.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Development servers started!"
echo ""
echo "🖥️  Local Access:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📱 Network Access (WiFi users):"
echo "   Frontend: http://$LOCAL_IP:3000"
echo "   Backend:  http://$LOCAL_IP:8000"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for both processes
wait
