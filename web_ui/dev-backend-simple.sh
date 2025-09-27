#!/bin/bash
cd backend
echo "🚀 Starting FastAPI backend..."
echo "🌐 Local: http://localhost:8000"
echo "🌐 Network: http://172.20.10.3:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo "📱 Anyone on your WiFi can access the network URL"
/home/mohammedazab/ws/src/race_stack/race_monitor/.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080
