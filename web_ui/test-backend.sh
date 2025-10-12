#!/bin/bash
# Start backend and test connectivity

echo "Starting FastAPI backend..."
cd /home/mohammedazab/ws/src/race_stack/race_monitor/web_ui/backend

# Start backend in background
python3 main.py &
BACKEND_PID=$!

echo "Backend started with PID: $BACKEND_PID"
echo "Waiting for backend to start..."
sleep 5

echo "Testing backend connectivity..."
curl -s http://127.0.0.1:8082/api/health || echo "Health check failed"
curl -s http://127.0.0.1:8082/api/config/list || echo "Config list failed" 
curl -s http://127.0.0.1:8082/api/data/summary || echo "Data summary failed"

echo "Backend is running at http://127.0.0.1:8082"
echo "To stop: kill $BACKEND_PID"