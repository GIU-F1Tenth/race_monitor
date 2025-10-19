#!/bin/bash
# Dynamic Backend Startup Script

cd "$(dirname "$0")" || exit 1

# Read ports from environment variables (set by main script)
BACKEND_PORT=${BACKEND_PORT:-$(python3 port_manager.py get | grep "Backend:" | cut -d' ' -f2)}
FRONTEND_PORT=${FRONTEND_PORT:-$(python3 port_manager.py get | grep "Frontend:" | cut -d' ' -f2)}

if [ -z "$BACKEND_PORT" ]; then
    echo "❌ Backend port not available"
    exit 1
fi

cd backend || exit 1

echo "🚀 Starting FastAPI backend..."
echo "🌐 Local: http://localhost:$BACKEND_PORT"
echo "📚 API docs: http://localhost:$BACKEND_PORT/docs"
echo "🔗 Using allocated port: $BACKEND_PORT"

# Export port for the application to use
export BACKEND_PORT=$BACKEND_PORT
export FRONTEND_PORT=$FRONTEND_PORT

/home/mohammedazab/ws/src/race_stack/race_monitor/.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port $BACKEND_PORT