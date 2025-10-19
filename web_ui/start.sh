#!/bin/bash
# Race Monitor Web UI Startup Script

set -e

echo "🏎️  Starting Race Monitor Web UI"
echo "================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker to continue."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose to continue."
    exit 1
fi

# Navigate to the docker directory
cd "$(dirname "$0")/docker"

# Check if .env file exists, create if not
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cat > .env << EOF
# Race Monitor Web UI Environment Variables
COMPOSE_PROJECT_NAME=race-monitor
BACKEND_PORT=8080
FRONTEND_PORT=3000
ENVIRONMENT=development

# Database (if needed in future)
# DATABASE_URL=sqlite:///./race_monitor.db

# Logging
LOG_LEVEL=INFO
EOF
fi

echo "🚀 Starting services with Docker Compose..."

# Start the services
docker-compose up --build -d

echo ""
echo "✅ Race Monitor Web UI is starting!"
echo ""
echo "📊 Frontend (React): http://localhost:3000"
echo "🔧 Backend API: http://localhost:8080"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "🔍 To view logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 To stop the services:"
echo "   docker-compose down"
echo ""
echo "⚡ Services are starting... Please wait a moment for them to be ready."

# Wait a bit and check if services are healthy
sleep 5

echo ""
echo "🏥 Checking service health..."

# Check backend health
if curl -f http://localhost:8000/api/health &> /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend is still starting up..."
fi

# Check if frontend is responding
if curl -f http://localhost:3000 &> /dev/null; then
    echo "✅ Frontend is healthy"
else
    echo "⚠️  Frontend is still starting up..."
fi

echo ""
echo "🎉 Setup complete! Open http://localhost:3000 in your browser."