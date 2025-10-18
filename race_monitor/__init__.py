"""
Race Monitor Package

Professional autonomous racing monitoring and analysis system for ROS2.
Provides comprehensive lap timing, trajectory analysis, performance monitoring,
and research-grade data export capabilities.

Core Modules:
    - race_monitor: Main orchestrating node
    - lap_timing_monitor: Legacy comprehensive monitoring node
    - trajectory_analyzer: Advanced trajectory analysis using EVO library
    - race_evaluator: A-F grading and performance evaluation
    - lap_detector: Lap detection and timing algorithms
    - performance_monitor: Computational performance tracking
    - data_manager: Data storage and file management
    - metadata_manager: Experiment metadata and system information management
    - visualization_publisher: RViz visualization components
    - reference_trajectory_manager: Reference trajectory handling
    - visualization_engine: Advanced plotting and graphing

Key Features:
    - Modular architecture with clean interfaces
    - Professional logging with minimal verbosity
    - Research-grade trajectory analysis
    - Multi-format data export (JSON, CSV, TUM)
    - Real-time performance monitoring
    - Comprehensive visualization support

License: MIT
"""

__version__ = "2.0.0"
__author__ = "GIU F1Tenth Team"

# Main exports for external usage
from .race_monitor import RaceMonitor
from .trajectory_analyzer import ResearchTrajectoryEvaluator, create_research_evaluator
from .race_evaluator import RaceEvaluator, create_race_evaluator
from .data_manager import DataManager

__all__ = [
    "RaceMonitor",
    "ResearchTrajectoryEvaluator",
    "create_research_evaluator",
    "RaceEvaluator",
    "create_race_evaluator",
    "DataManager"
]
