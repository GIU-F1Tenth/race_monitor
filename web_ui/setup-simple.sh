#!/bin/bash
# Simple Development Setup - No Docker Required

set -e

echo "🚀 Race Monitor Web UI - Simple Development Setup"
echo "================================================"

# Get the machine's local IP for network access
LOCAL_IP=$(hostname -I | awk '{print $1}')

# Setup backend
echo ""
echo "🐍 Setting up Python backend..."
cd backend

echo "Installing Python dependencies globally..."
pip3 install --user -r requirements.txt

echo "✅ Backend dependencies installed"

# Setup frontend
echo ""
echo "⚛️  Setting up React frontend..."
cd ../frontend

echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend dependencies installed"

# Create simple dev scripts
echo ""
echo "📝 Creating development scripts..."

cd ..

# Simple backend script (no venv)
cat > dev-backend-simple.sh << 'EOF'
#!/bin/bash
cd backend
echo "🚀 Starting FastAPI backend..."
echo "🌐 Local: http://localhost:8000"
echo "🌐 Network: http://LOCAL_IP:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo "📱 Anyone on your WiFi can access the network URL"
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
EOF

# Simple frontend script
cat > dev-frontend-simple.sh << 'EOF'
#!/bin/bash
cd frontend
echo "🚀 Starting React frontend..."
echo "🌐 Local: http://localhost:3000"
echo "🌐 Network: http://LOCAL_IP:3000"
echo "📱 Anyone on your WiFi can access the network URL"
npm run dev -- --host 0.0.0.0
EOF

# Combined simple script
cat > dev-simple.sh << 'EOF'
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
EOF

# Make all scripts executable
chmod +x dev-backend-simple.sh
chmod +x dev-frontend-simple.sh
chmod +x dev-simple.sh

echo ""
echo "✅ Simple development setup complete!"
echo ""
echo "🎯 To start development servers:"
echo "   ./dev-simple.sh"
echo ""
echo "🌐 Network access enabled!"
echo "   Local IP: $LOCAL_IP"
echo "   Anyone on your WiFi can access: http://$LOCAL_IP:3000"
echo ""
echo "🚀 Ready to go!"