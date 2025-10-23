# Race Monitor Web UI

A comprehensive web interface for the F1Tenth Race Monitor system, providing real-time monitoring, configuration management, and advanced trajectory analysis with EVO integration.

## 🏎️ Features

### Core Functionality
- **Configuration Management**: Edit YAML configuration files with syntax highlighting and validation
- **Real-time Monitoring**: Live race data monitoring with WebSocket connections
- **Results Analysis**: Comprehensive race results and lap analysis
- **EVO Integration**: Advanced trajectory evaluation and comparison
- **Interactive Visualizations**: 2D/3D trajectory plots and performance metrics

### Key Components
- **Dashboard**: Overview of experiments, system health, and quick actions
- **Configuration Editor**: Tabbed interface for editing race monitor configs
- **Results Viewer**: Detailed race results and lap-by-lap analysis
- **Analysis Panel**: EVO trajectory evaluation and comparison tools
- **Live Monitor**: Real-time race monitoring and system status

## 🚀 Quick Start

### Option 1: Quick Development Start (Recommended)
```bash
cd web_ui
./quick-start.sh
```
- Fastest way to get started
- No Docker required
- Uses ports 3001 (frontend) and 8001 (backend)
- Access at http://localhost:3001

### Option 2: Docker Production Mode
```bash
cd web_ui
./start.sh
```
- Production-ready with Docker
- Ports 3000 (frontend) and 8080 (backend)
- Access at http://localhost:3000

### Option 3: Dynamic Port Allocation
```bash
cd web_ui
./dev-dynamic.sh
```
- Automatically finds free ports
- Useful when default ports are occupied
- Shows allocated ports on startup

### Access Points
- **Web Interface**: http://localhost:3001 (quick-start) or http://localhost:3000 (docker)
- **API Documentation**: http://localhost:8001/docs (quick-start) or http://localhost:8080/docs (docker)
- **Backend Health**: http://localhost:8001/api/health (quick-start) or http://localhost:8080/api/health (docker)

## 🏗️ Architecture

### Technology Stack
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + Python
- **Editor**: Monaco Editor (VS Code editor in browser)
- **Visualization**: Plotly.js for interactive charts
- **Real-time**: WebSockets for live data updates

### Directory Structure
```
web_ui/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── config_manager.py   # Configuration file management
│   ├── data_analyzer.py    # Race data analysis
│   ├── evo_integration.py  # EVO trajectory evaluation
│   ├── live_monitor.py     # Real-time monitoring
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable React components
│   │   ├── pages/         # Main application pages
│   │   └── App.tsx        # Main application component
│   ├── package.json       # Node.js dependencies
│   └── vite.config.ts     # Vite configuration
├── docker/                # Docker configuration
│   ├── docker-compose.yml # Multi-container setup
│   ├── Dockerfile.backend # Backend container
│   └── Dockerfile.frontend# Frontend container
└── start.sh              # Quick start script
```

## 📊 Configuration Management

### Features
- **Multi-file Support**: Manage multiple YAML configuration files
- **Tabbed Interface**: Easy switching between configurations
- **Syntax Validation**: Real-time YAML syntax checking
- **File Upload/Download**: Import and export configurations
- **Auto-save**: Automatic saving with change detection

### Configuration Files
The system automatically detects and loads:
- `race_monitor.yaml` - Main race configuration
- `template_*.yaml` - Configuration templates
- Custom configuration files

### Advanced Settings
Complex configuration sections are organized into:
- **Basic Settings**: Start line, laps, output files
- **Race Ending**: Crash detection, manual mode
- **Research Settings**: Controller info, experiment details
- **EVO Integration**: Trajectory evaluation parameters
- **Advanced Settings**: Hidden by default, expandable

## 📈 Data Analysis

### Supported Data Types
- **Trajectory Files**: `lap_*_trajectory.txt`
- **Evaluation Summary**: `evaluation_summary.csv`
- **Performance Data**: CPU, memory, timing metrics
- **EVO Results**: APE/RPE analysis results

### Analysis Features
- **Lap-by-lap Analysis**: Detailed performance breakdown
- **Consistency Metrics**: Statistical analysis of lap performance
- **Trajectory Visualization**: Interactive 2D/3D plots
- **Performance Trends**: Time-series analysis
- **Comparative Analysis**: Multi-experiment comparison

## 🔄 EVO Integration

### Capabilities
- **Automatic Format Conversion**: Trajectory to TUM format
- **APE Analysis**: Absolute Pose Error evaluation
- **RPE Analysis**: Relative Pose Error evaluation
- **Multi-trajectory Comparison**: Side-by-side analysis
- **Export Options**: Results in multiple formats

### Requirements
- Reference trajectory file (`horizon_reference_trajectory.txt`)
- EVO library installed and accessible
- Properly formatted trajectory data

## 🔴 Live Monitoring

### Real-time Features
- **Race Status**: Current lap, timing, position
- **System Health**: CPU, memory, ROS2 status
- **Topic Monitoring**: Live ROS2 topic data
- **File Activity**: Recent file changes
- **Emergency Controls**: Safety stop capabilities

### ROS2 Integration
- **Automatic Detection**: Checks for ROS2 availability
- **Topic Monitoring**: Key race monitor topics
- **Node Status**: Active ROS2 nodes
- **Health Checks**: System component status

## 🛠️ Development

### Initial Setup
```bash
# One-time setup
./setup-dev.sh
```

This installs all dependencies for both frontend and backend.

### Development Workflow

**Quick Start (No Docker):**
```bash
./quick-start.sh
# Frontend: http://localhost:3001
# Backend: http://localhost:8001
```

**Dynamic Ports (Auto-allocate):**
```bash
./dev-dynamic.sh
# Displays allocated ports on startup
```

**Manual Development:**
```bash
# Backend only
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Frontend only (separate terminal)
cd frontend
npm run dev
```

### API Endpoints
- `GET /api/health` - Service health check
- `GET /api/info/ports` - Port configuration info
- `GET /api/config/list` - List configuration files
- `GET /api/config/{filename}` - Get configuration content
- `POST /api/config/{filename}` - Save configuration
- `GET /api/data/experiments` - List experiments
- `GET /api/data/summary` - Data summary
- `POST /api/evo/analyze/{experiment_id}` - Run EVO analysis
- `WebSocket /ws/live` - Real-time data stream

### Available Scripts
- `./quick-start.sh` - Quick development start (ports 3001/8001)
- `./start.sh` - Docker production mode (ports 3000/8080)
- `./dev-dynamic.sh` - Development with dynamic port allocation
- `./setup-dev.sh` - One-time development environment setup
- `./port_manager.py` - Port allocation utility

### Configuration
Environment variables:
- `ENVIRONMENT` - Development/production mode
- `LOG_LEVEL` - Logging verbosity
- `BACKEND_PORT` - Backend port (default: 8000)
- `FRONTEND_PORT` - Frontend port (default: 3000)

## 🐳 Docker Deployment

### Production Deployment
```bash
# Use production docker-compose
docker-compose -f docker/docker-compose.prod.yml up -d

# Scale services if needed
docker-compose up --scale backend=2
```

### Custom Configuration
```yaml
# docker/.env
COMPOSE_PROJECT_NAME=race-monitor
BACKEND_PORT=8080
FRONTEND_PORT=3001
ENVIRONMENT=production
```

## 🔧 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Use dynamic port allocation
./dev-dynamic.sh

# Or manually kill processes
sudo fuser -k 3001/tcp 8001/tcp
```

**Backend not starting:**
```bash
# Check backend logs
cd backend
source venv/bin/activate
python main.py  # Run directly to see errors
```

**Frontend build errors:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Configuration not loading:**
- Ensure config files are in `../config/` directory
- Check file permissions
- Verify YAML syntax

**EVO integration issues:**
- Verify EVO library installation: `python -c "import evo"`
- Check trajectory file formats
- Ensure reference trajectory exists

### Performance Optimization
- Use `./start.sh` (Docker) for production deployment
- Configure proper resource limits in `docker/docker-compose.yml`
- Monitor memory usage for large datasets
- Enable caching for static assets

### Port Configuration
The system supports flexible port configuration:
- **Default Development**: 3001 (frontend), 8001 (backend)
- **Docker Production**: 3000 (frontend), 8080 (backend)
- **Dynamic Mode**: Auto-allocated free ports

## 📝 Contributing

### Code Style
- Frontend: ESLint + Prettier for TypeScript/React
- Backend: Black + isort for Python
- Commit messages: Conventional commits format

### Running Tests
```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && pytest
```

## 📄 License

This project is part of the F1Tenth Race Monitor system. See the main project license for details.