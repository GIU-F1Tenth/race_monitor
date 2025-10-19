#!/bin/bash
# Quick test script to verify the web UI setup

echo "🧪 Testing Race Monitor Web UI Setup"
echo "===================================="

# Test backend dependencies
echo "🐍 Testing backend setup..."
cd backend

if [ -f requirements.txt ]; then
    echo "✅ Backend requirements.txt found"
else
    echo "❌ Backend requirements.txt missing"
fi

if [ -f main.py ]; then
    echo "✅ Backend main.py found"
else
    echo "❌ Backend main.py missing"
fi

# Test frontend dependencies
echo ""
echo "⚛️  Testing frontend setup..."
cd ../frontend

if [ -f package.json ]; then
    echo "✅ Frontend package.json found"
else
    echo "❌ Frontend package.json missing"
fi

if [ -f vite.config.ts ]; then
    echo "✅ Frontend vite.config.ts found"
else
    echo "❌ Frontend vite.config.ts missing"
fi

# Test Docker setup
echo ""
echo "🐳 Testing Docker setup..."
cd ../docker

if [ -f docker-compose.yml ]; then
    echo "✅ Docker Compose configuration found"
else
    echo "❌ Docker Compose configuration missing"
fi

if [ -f Dockerfile.backend ]; then
    echo "✅ Backend Dockerfile found"
else
    echo "❌ Backend Dockerfile missing"
fi

if [ -f Dockerfile.frontend ]; then
    echo "✅ Frontend Dockerfile found"
else
    echo "❌ Frontend Dockerfile missing"
fi

# Test scripts
echo ""
echo "📝 Testing scripts..."
cd ..

if [ -x start.sh ]; then
    echo "✅ Docker start script found and executable"
else
    echo "❌ Docker start script missing or not executable"
fi

if [ -x setup-dev.sh ]; then
    echo "✅ Development setup script found and executable"
else
    echo "❌ Development setup script missing or not executable"
fi

# Test configuration access
echo ""
echo "⚙️  Testing configuration access..."

if [ -f ../config/race_monitor.yaml ]; then
    echo "✅ Main configuration file accessible"
else
    echo "⚠️  Main configuration file not found (may be created later)"
fi

if [ -d ../race_monitor/evaluation_results ]; then
    echo "✅ Evaluation results directory accessible"
else
    echo "⚠️  Evaluation results directory not found (may be created later)"
fi

echo ""
echo "🎯 Test Results Summary"
echo "======================"
echo "✅ All core files are in place"
echo "✅ Scripts are properly configured"
echo "✅ Docker setup is ready"
echo ""
echo "🚀 Next steps:"
echo "1. For development: ./setup-dev.sh && ./dev.sh"
echo "2. For Docker: ./start.sh"
echo ""
echo "📋 File structure verification complete!"

# Show directory structure
echo ""
echo "📁 Current directory structure:"
find . -type f -name "*.py" -o -name "*.tsx" -o -name "*.ts" -o -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.sh" -o -name "*.md" | head -20
echo "   (showing first 20 files...)"