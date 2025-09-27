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

### Prerequisites
- Docker and Docker Compose
- Ports 3000 (frontend) and 8000 (backend) available

### Starting the Web UI
```bash
# Navigate to the web UI directory
cd web_ui

# Start all services
./start.sh
```

This will:
1. Build and start both frontend and backend services
2. Set up volume mounts for your configuration and data files
3. Configure automatic reloading for development

### Access Points
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Backend Health**: http://localhost:8000/api/health

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

### Local Development Setup
```bash
# Backend development
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend development (separate terminal)
cd frontend
npm install
npm run dev
```

### API Endpoints
- `GET /api/config/list` - List configuration files
- `GET /api/config/{filename}` - Get configuration content
- `POST /api/config/{filename}` - Save configuration
- `GET /api/data/experiments` - List experiments
- `GET /api/data/summary` - Data summary
- `POST /api/evo/analyze/{experiment_id}` - Run EVO analysis
- `WebSocket /ws/live` - Real-time data stream

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

**Backend not starting:**
```bash
# Check logs
docker-compose logs backend

# Common solutions
docker-compose down && docker-compose up --build
```

**Frontend build errors:**
```bash
# Clear node modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Configuration not loading:**
- Ensure config files are in the correct directory
- Check file permissions
- Verify YAML syntax

**EVO integration issues:**
- Verify EVO library installation
- Check trajectory file formats
- Ensure reference trajectory exists

### Performance Optimization
- Use production builds for deployment
- Configure proper resource limits
- Monitor memory usage for large datasets
- Enable caching for static assets

## 📝 Contributing

### Code Style
- Frontend: ESLint + Prettier for TypeScript/React
- Backend: Black + isort for Python
- Commit messages: Conventional commits format

### Testing
```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && pytest
```

## 📄 License

This project is part of the F1Tenth Race Monitor system. See the main project license for details.

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check container logs for errors
4. Ensure all dependencies are properly installed