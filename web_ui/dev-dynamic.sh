#!/bin/bash
# Dynamic Race Monitor Web UI Startup Script

echo "🏎️  Starting Race Monitor Web UI (Dynamic Mode)"
echo "================================================"

cd "$(dirname "$0")" || exit 1

# Clean up any existing port allocations
echo "🧹 Cleaning up previous port allocations..."
python3 port_manager.py cleanup

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down development servers..."
    jobs -p | xargs -r kill 2>/dev/null
    echo "🧹 Cleaning up port allocations..."
    python3 port_manager.py cleanup
    exit 0
}

# Set trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

echo "🔍 Allocating dynamic ports..."
PORTS=$(python3 port_manager.py allocate)
BACKEND_PORT=$(echo "$PORTS" | grep "Backend:" | cut -d' ' -f2)
FRONTEND_PORT=$(echo "$PORTS" | grep "Frontend:" | cut -d' ' -f2)

if [ -z "$BACKEND_PORT" ] || [ -z "$FRONTEND_PORT" ]; then
    echo "❌ Failed to allocate ports"
    exit 1
fi

echo "✅ Ports allocated:"
echo "   Backend:  $BACKEND_PORT"
echo "   Frontend: $FRONTEND_PORT"
echo ""

# Start backend in background
echo "🚀 Starting backend..."
export BACKEND_PORT=$BACKEND_PORT
export FRONTEND_PORT=$FRONTEND_PORT
./dev-backend-dynamic.sh &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is actually running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    cleanup
    exit 1
fi

# Start frontend in background
echo "🚀 Starting frontend..."
./dev-frontend-dynamic.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Development servers started!"
echo ""
echo "🖥️  Access URLs:"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   Backend:  http://localhost:$BACKEND_PORT"
echo "   API Docs: http://localhost:$BACKEND_PORT/docs"
echo ""
echo "🌐 Network Access:"
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "   Frontend: http://$LOCAL_IP:$FRONTEND_PORT"
echo "   Backend:  http://$LOCAL_IP:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
wait