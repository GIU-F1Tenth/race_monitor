"""
Race Monitor Web UI Backend
FastAPI application for race monitoring and analysis
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import yaml
import pandas as pd
import os
import shutil
from pathlib import Path
import asyncio
from datetime import datetime
import numpy as np

# Import our custom modules - simplified for initial testing
try:
    from config_manager import ConfigManager
    from data_analyzer import DataAnalyzer
    from evo_integration import EvoIntegration
    from live_monitor import LiveMonitor
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    # Create placeholder classes for testing
    class ConfigManager:
        def list_configs(self): return {"configs": [], "templates": {}, "config_dir": ""}
        def get_config(self, filename): return ""
        def save_config(self, filename, content): pass
        def delete_config(self, filename): pass
    
    class DataAnalyzer:
        def get_experiments(self, filter_params=None): return {"experiments": [], "total_count": 0}
        def get_experiment_details(self, experiment_id): return {}
        def get_summary(self): return {"experiments_count": 0, "total_laps": 0}
        def get_lap_analysis(self, experiment_id): return {}
        def get_trajectory_plot_data(self, experiment_id): return {}
        def get_performance_plot_data(self, experiment_id): return {}
        def get_comparison_plot_data(self, exp_list): return {}
        def export_experiment(self, experiment_id, format): return None
    
    class EvoIntegration:
        def get_metrics(self, experiment_id): return {"evo_available": False}
        async def run_analysis(self, experiment_id): return {"error": "EVO not available"}
        async def compare_experiments(self, exp1, exp2): return {"error": "EVO not available"}
    
    class LiveMonitor:
        async def get_live_data(self): return {"monitoring_active": False}
        def get_status(self): return {"monitoring_active": False}
        async def start(self): return {"status": "disabled"}
        async def stop(self): pass

app = FastAPI(title="Race Monitor Web UI", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
config_manager = ConfigManager()
data_analyzer = DataAnalyzer()
evo_integration = EvoIntegration()
live_monitor = LiveMonitor()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models
class ConfigUpdate(BaseModel):
    content: str
    filename: str

class ExperimentFilter(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    controller_name: Optional[str] = None
    experiment_id: Optional[str] = None

# API Routes

@app.get("/")
async def root():
    return {"message": "Race Monitor Web UI Backend"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Configuration Management
@app.get("/api/config/list")
async def list_configs():
    """Get list of available configuration files"""
    return config_manager.list_configs()

@app.get("/api/config/{filename}")
async def get_config(filename: str):
    """Get specific configuration file content"""
    try:
        content = config_manager.get_config(filename)
        return {"filename": filename, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Configuration file not found")

@app.post("/api/config/{filename}")
async def save_config(filename: str, config_update: ConfigUpdate):
    """Save configuration file"""
    try:
        config_manager.save_config(filename, config_update.content)
        return {"message": f"Configuration {filename} saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/upload")
async def upload_config(file: UploadFile = File(...)):
    """Upload new configuration file"""
    try:
        content = await file.read()
        filename = file.filename
        config_manager.save_config(filename, content.decode('utf-8'))
        return {"message": f"Configuration {filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/config/{filename}")
async def delete_config(filename: str):
    """Delete configuration file"""
    try:
        config_manager.delete_config(filename)
        return {"message": f"Configuration {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Data Analysis
@app.get("/api/data/experiments")
async def get_experiments(filter_params: Optional[ExperimentFilter] = None):
    """Get list of experiments with optional filtering"""
    return data_analyzer.get_experiments(filter_params)

@app.get("/api/data/experiment/{experiment_id}")
async def get_experiment_details(experiment_id: str):
    """Get detailed data for specific experiment"""
    try:
        return data_analyzer.get_experiment_details(experiment_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Experiment not found")

@app.get("/api/data/summary")
async def get_data_summary():
    """Get overall data summary and statistics"""
    return data_analyzer.get_summary()

@app.get("/api/data/lap-analysis/{experiment_id}")
async def get_lap_analysis(experiment_id: str):
    """Get detailed lap analysis for experiment"""
    try:
        return data_analyzer.get_lap_analysis(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# EVO Integration
@app.get("/api/evo/metrics/{experiment_id}")
async def get_evo_metrics(experiment_id: str):
    """Get EVO trajectory evaluation metrics"""
    try:
        return evo_integration.get_metrics(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evo/analyze/{experiment_id}")
async def run_evo_analysis(experiment_id: str):
    """Run EVO analysis on experiment data"""
    try:
        result = await evo_integration.run_analysis(experiment_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evo/compare")
async def compare_experiments(exp1: str, exp2: str):
    """Compare two experiments using EVO"""
    try:
        return await evo_integration.compare_experiments(exp1, exp2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Visualization
@app.get("/api/viz/trajectory/{experiment_id}")
async def get_trajectory_plot(experiment_id: str):
    """Get trajectory visualization data"""
    try:
        return data_analyzer.get_trajectory_plot_data(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/viz/performance/{experiment_id}")
async def get_performance_plot(experiment_id: str):
    """Get performance metrics visualization data"""
    try:
        return data_analyzer.get_performance_plot_data(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/viz/comparison")
async def get_comparison_plot(experiments: str):
    """Get comparison visualization for multiple experiments"""
    try:
        exp_list = experiments.split(',')
        return data_analyzer.get_comparison_plot_data(exp_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Live Monitoring
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket)
    try:
        while True:
            # Send live data updates
            live_data = await live_monitor.get_live_data()
            await manager.send_personal_message(json.dumps(live_data), websocket)
            await asyncio.sleep(0.5)  # 2Hz update rate
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/live/status")
async def get_live_status():
    """Get current live monitoring status"""
    return live_monitor.get_status()

@app.post("/api/live/start")
async def start_live_monitoring():
    """Start live monitoring"""
    try:
        result = await live_monitor.start()
        return {"message": "Live monitoring started", "status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/live/stop")
async def stop_live_monitoring():
    """Stop live monitoring"""
    try:
        await live_monitor.stop()
        return {"message": "Live monitoring stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File operations
@app.get("/api/files/graphs/{filename}")
async def get_graph_file(filename: str):
    """Serve graph files"""
    file_path = Path("../../race_monitor/evaluation_results/graphs") / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/files/export/{experiment_id}")
async def export_experiment_data(experiment_id: str, format: str = "csv"):
    """Export experiment data in various formats"""
    try:
        file_path = data_analyzer.export_experiment(experiment_id, format)
        return FileResponse(file_path, filename=f"{experiment_id}.{format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)