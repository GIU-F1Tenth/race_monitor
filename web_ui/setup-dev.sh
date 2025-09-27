#!/bin/bash
# Development Setup Script for Race Monitor Web UI

set -e

echo "🛠️  Setting up Race Monitor Web UI for Development"
echo "================================================="

# Check system requirements
echo "🔍 Checking system requirements..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed."
    exit 1
fi

echo "✅ System requirements met"

# Setup backend
echo ""
echo "🐍 Setting up Python backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete"

# Setup frontend
echo ""
echo "⚛️  Setting up React frontend..."
cd ../frontend

echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete"

# Create development scripts
echo ""
echo "📝 Creating development scripts..."

cd ..

# Backend dev script
cat > dev-backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
echo "🚀 Starting FastAPI backend..."
echo "🌐 Backend API: http://localhost:8000"
echo "🌐 Network API: http://$(hostname -I | awk '{print $1}'):8000"
echo "📚 API docs: http://localhost:8000/docs"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
EOF

# Frontend dev script
cat > dev-frontend.sh << 'EOF'
#!/bin/bash
cd frontend
echo "🚀 Starting React frontend..."
echo "🌐 Local: http://localhost:3000"
echo "🌐 Network: http://$(hostname -I | awk '{print $1}'):3000"
echo "📱 Anyone on your WiFi can access the network URL"
npm run dev -- --host 0.0.0.0
EOF

# Combined dev script
cat > dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Race Monitor Web UI in development mode"
echo "================================================="

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down development servers..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend in background
echo "Starting backend..."
./dev-backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "Starting frontend..."
./dev-frontend.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Development servers started!"
echo "📊 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for both processes
wait
EOF

# Make scripts executable
chmod +x dev-backend.sh
chmod +x dev-frontend.sh
chmod +x dev.sh

echo "✅ Development scripts created"

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo ""
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Dependencies
backend/venv/
frontend/node_modules/
frontend/dist/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Runtime
*.pid
*.seed
*.pid.lock

# Coverage
coverage/
.nyc_output/

# Cache
.cache/
.temp/

# Build outputs
build/
dist/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.coverage
.pytest_cache/

# Docker
.dockerignore
EOF
fi

echo ""
echo "🎉 Development setup complete!"
echo ""
echo "📋 Available commands:"
echo "   ./dev.sh           - Start both frontend and backend"
echo "   ./dev-backend.sh   - Start only backend"
echo "   ./dev-frontend.sh  - Start only frontend"
echo "   ./start.sh         - Start with Docker (production-like)"
echo ""
echo "🔧 For development:"
echo "   ./dev.sh"
echo ""
echo "🐳 For testing with Docker:"
echo "   ./start.sh"
echo ""
echo "Happy coding! 🚀"