# Race Monitor: Advanced Performance Analysis Framework for Roboracer Autonomous Racing

<div align="center">

[![CI](https://github.com/GIU-F1Tenth/race_monitor/workflows/Race%20Monitor%20CI/badge.svg)](https://github.com/GIU-F1Tenth/race_monitor/actions)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![EVO Integration](https://img.shields.io/badge/EVO-Integrated-orange)](https://github.com/MichaelGrupp/evo)
[![MIT License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

**A comprehensive ROS2-based framework for real-time performance monitoring, trajectory analysis, and controller evaluation in autonomous racing**

[Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start-guide) • [Results](#experimental-results) • [Documentation](#documentation) • [Citation](#citation)

</div>

---

## Abstract

Race Monitor is a research-grade performance analysis framework designed for Roboracer autonomous racing platforms. The system provides real-time lap timing, comprehensive trajectory evaluation using the EVO library, computational performance monitoring, and multi-format data export capabilities. Built on ROS2, Race Monitor enables systematic controller comparison, experimental reproducibility, and quantitative performance assessment essential for autonomous racing research.

**Key capabilities include:**
- Real-time race monitoring with configurable lap detection
- Trajectory analysis with 40+ performance metrics per lap
- Computational efficiency monitoring (latency, CPU, memory)
- Integration with the EVO trajectory evaluation library
- Multi-format data export (CSV, JSON, MATLAB, Pickle)
- Automated visualization and graph generation
- Structured experimental data organization

---

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Configuration](#configuration)
6. [Experimental Results](#experimental-results)
7. [Data Analysis & EVO Integration](#data-analysis--evo-integration)
8. [ROS2 Interface](#ros2-interface)
9. [Output Data Structure](#output-data-structure)
10. [Web UI (Coming Soon)](#web-ui-preview)
11. [Citation](#citation)
12. [License](#license)

---

## Key Features

### Race Management & Monitoring
- **Real-time Lap Detection**: Configurable start/finish line with geometric intersection detection
- **Multi-lap Sessions**: Support for extended race sessions with automatic lap counting
- **Debouncing Logic**: Prevents false lap triggers with configurable debounce timing
- **Race State Management**: Complete race lifecycle handling (pre-race, active, completed, aborted)
- **Flexible Race Endings**: Support for lap-based, distance-based, and manual race termination

### Trajectory Analysis
- **EVO Library Integration**: Full integration with the Evolution of Trajectory Evaluation (EVO) library
- **Comprehensive Metrics Suite**: 40+ performance indicators including:
  - Path geometry (length, efficiency, curvature)
  - Velocity statistics (mean, std, max, consistency)
  - Acceleration analysis (mean, std, max, jerk)
  - Angular motion (angular velocity, steering aggressiveness)
  - Trajectory quality (APE, RPE when reference available)
- **Multi-format Support**: TUM, KITTI, EuRoC trajectory formats
- **Reference Trajectory Comparison**: Ground truth trajectory alignment and error analysis

### Computational Performance Monitoring
- **Control Loop Latency**: Real-time measurement of control command timing
- **Resource Utilization**: CPU and memory usage tracking
- **Processing Efficiency**: Computational performance scoring
- **Threshold Alerting**: Configurable performance warnings
- **Multi-topic Support**: Simultaneous monitoring of multiple control topics

### Data Management & Export
- **Multi-format Export**: CSV, JSON, MATLAB (.mat), Pickle formats
- **Structured Organization**: Hierarchical data organization by controller/experiment
- **Trajectory Filtering**: Optional Savitzky-Golay filtering for noise reduction
- **Automated Visualization**: Publication-ready graphs in PNG and PDF
- **Metadata Tracking**: Complete experiment metadata and system information

### Advanced Research Features
- **Controller Comparison**: Side-by-side performance evaluation
- **Statistical Analysis**: Comprehensive statistical summaries and distributions
- **Reproducibility**: Complete experiment configuration logging
- **Batch Processing**: Multi-experiment analysis capabilities
- **Version Control**: Experiment versioning and tracking

---

## System Architecture

Race Monitor is built on a modular architecture designed for extensibility and research reproducibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    ROS2 Node: Race Monitor                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │  Lap Timing   │  │   Trajectory   │  │  Performance   │  │
│  │   Monitor     │  │    Analyzer    │  │    Monitor     │  │
│  └───────┬───────┘  └────────┬───────┘  └────────┬───────┘  │
│          │                   │                    │         │
│  ┌───────▼──────────────────▼────────────────────▼───────┐  │
│  │            Race Evaluator (Core Engine)               │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │                              │
│  ┌──────────────┬────────────▼─────────────┬─────────────┐  │
│  │ Visualization│  Data Manager            │  Metadata   │  │
│  │   Engine     │  (Export & Organization) │   Manager   │  │
│  └──────────────┴──────────────────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌─────────────┐  ┌──────────┐  ┌──────────┐
        │  RViz       │  │   Data   │  │   EVO    │
        │Visualization│  │  Export  │  │  Tools   │
        └─────────────┘  └──────────┘  └──────────┘
```

### Core Components

#### 1. **Race Monitor Node** (`race_monitor.py`)
Main ROS2 node coordinating all monitoring activities.

#### 2. **Lap Timing Monitor** (`lap_timing_monitor.py`)
- Geometric line intersection detection
- Lap timing with high-precision timestamps
- Debouncing and state management

#### 3. **Trajectory Analyzer** (`trajectory_analyzer.py`)
- Real-time trajectory recording and analysis
- 40+ performance metrics calculation
- Statistical analysis and smoothness evaluation

#### 4. **Performance Monitor** (`performance_monitor.py`)
- Control loop latency measurement
- System resource monitoring
- Computational efficiency scoring

#### 5. **Race Evaluator** (`race_evaluator.py`)
- Central coordination and data integration
- Race state management
- Multi-lap aggregation and analysis

#### 6. **Data Manager** (`data_manager.py`)
- Multi-format export (CSV, JSON, MATLAB, Pickle)
- Hierarchical data organization
- Experimental metadata tracking

#### 7. **Visualization Engine** (`visualization_engine.py`)
- Automated graph generation
- Publication-ready plots (PNG, PDF)
- Multiple visualization types (trajectories, errors, speeds)

#### 8. **EVO Integration**
- Trajectory format conversion
- APE/RPE analysis when reference available
- Advanced trajectory comparison tools

---

## Installation

### Prerequisites

- **Operating System**: Ubuntu 22.04 LTS (tested) or compatible
- **ROS2 Distribution**: Humble Hawksbill or later
- **Python**: 3.10 or higher
- **Build System**: Colcon

### Dependencies

#### System Dependencies
```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-pandas \
    ros-humble-desktop
```

#### Python Dependencies
```bash
pip install numpy scipy matplotlib pandas seaborn
```

### Installation Steps

#### 1. Clone the Repository
```bash
# Navigate to your ROS2 workspace source directory
cd ~/ros2_ws/src

# Clone with submodules (includes EVO)
git clone --recursive https://github.com/GIU-F1Tenth/race_monitor.git

# If already cloned without --recursive, initialize submodules:
cd race_monitor
git submodule update --init --recursive
```

#### 2. Install EVO Library
```bash
# Navigate to EVO directory
cd ~/ros2_ws/src/race_monitor/evo

# Install EVO in development mode
pip install -e .

# Verify installation
python -c "import evo; print(evo.__version__)"
```

#### 3. Build the Package
```bash
# Navigate to workspace root
cd ~/ros2_ws

# Build Race Monitor package
colcon build --packages-select race_monitor

# Source the workspace
source install/setup.bash
```

#### 4. Verify Installation
```bash
# Check if package is recognized
ros2 pkg list | grep race_monitor

# Verify node executable
ros2 pkg executables race_monitor
```

### Docker Installation (Optional)

A Docker container with all dependencies pre-installed is available:

```bash
# Pull the container
docker pull ghcr.io/giu-f1tenth/race_monitor:latest

# Run with ROS2 network
docker run -it --rm \
    --network=host \
    -v ~/ros2_ws:/workspace \
    ghcr.io/giu-f1tenth/race_monitor:latest
```

### Troubleshooting

**Issue: EVO import fails**
```bash
# Ensure EVO is in Python path
export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/race_monitor/evo
```

**Issue: Colcon build fails**
```bash
# Clean and rebuild
cd ~/ros2_ws
rm -rf build install log
colcon build --packages-select race_monitor --symlink-install
```

**Issue: Missing Python dependencies**
```bash
# Install from requirements file
pip install -r ~/ros2_ws/src/race_monitor/requirements.txt
```

---

## Quick Start Guide

### Basic Usage

#### 1. Configure the System

Edit the configuration file to match your setup:

```bash
# Open the main configuration file
nano ~/ros2_ws/src/race_monitor/config/race_monitor.yaml
```

Key parameters to configure:
- `start_line_p1` and `start_line_p2`: Define your start/finish line
- `controller_name`: Identifier for your controller
- `experiment_id`: Unique experiment identifier
- `required_laps`: Number of laps to complete

#### 2. Launch Race Monitor

```bash
# Source your workspace
source ~/ros2_ws/install/setup.bash

# Basic launch (default lap complete mode)
ros2 launch race_monitor race_monitor.launch.py

# With custom parameters
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=lqr_controller \
    required_laps:=10

# Note: experiment_id is now auto-generated based on timestamp
```

#### 3. Set Start/Finish Line (Interactive)

Using RViz:
1. Launch RViz with the Race Monitor configuration
2. Use the "Publish Point" tool (keyboard: `p`)
3. Click two points to define the start/finish line
4. Monitor console for confirmation

#### 4. Start Your Controller

Launch your autonomous racing controller in a separate terminal:

```bash
ros2 launch your_controller your_controller.launch.py
```

#### 5. Monitor Race Progress

```bash
# Watch lap times
ros2 topic echo /race_monitor/lap_time

# Monitor race status
ros2 topic echo /race_monitor/race_status

# View current lap
ros2 topic echo /race_monitor/lap_count
```

### Advanced Usage Examples

#### Research Experiment with Full Analysis

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=lqr_controller_node \
    required_laps:=8 \
    enable_trajectory_evaluation:=true \
    auto_generate_graphs:=true \
    enable_computational_monitoring:=true

# Note: Most analysis options are configured in race_monitor.yaml
# experiment_id is auto-generated with timestamp
```

#### Performance Benchmarking

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=benchmark_controller \
    enable_computational_monitoring:=true

# Configure performance thresholds in race_monitor.yaml:
# - max_acceptable_latency_ms
# - target_control_frequency_hz
# - max_acceptable_cpu_usage
```

#### Multi-lap Endurance Test

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=endurance_test \
    required_laps:=50 \
    save_trajectories:=true

# Configure evaluation interval in race_monitor.yaml:
# - evaluation_interval_laps: 5
```

### Real-time Monitoring

Monitor system performance in real-time:

```bash
# Performance statistics (JSON format)
ros2 topic echo /race_monitor/performance_stats

# Control loop latency (milliseconds)
ros2 topic echo /race_monitor/control_loop_latency

# CPU usage percentage
ros2 topic echo /race_monitor/cpu_usage

# Memory usage (MB)
ros2 topic echo /race_monitor/memory_usage
```

---

## Configuration

### Configuration File Structure

The primary configuration file is `config/race_monitor.yaml`. Below is a complete reference:

```yaml
race_monitor:
  ros__parameters:
    # ========================================
    # RACE SETUP
    # ========================================
    
    # Start/finish line definition (2D coordinates)
    start_line_p1: [0.0, -1.0]  # [x, y] in map frame
    start_line_p2: [0.0, 1.0]   # [x, y] in map frame
    
    # Race parameters
    required_laps: 7
    debounce_time: 2.0  # Minimum time between lap detections (seconds)
    frame_id: "map"     # TF frame for visualization and coordinates
    
    # ========================================
    # CONTROLLER & EXPERIMENT IDENTIFICATION
    # ========================================
    
    controller_name: ""  # Empty = auto-detect from ROS topics
    enable_smart_controller_detection: true  # Auto-detect controller name
    test_description: "Controller performance evaluation and analysis"
    
    # Note: experiment_id is auto-generated with timestamp
    
    # ========================================
    # LOGGING & DEBUGGING
    # ========================================
    
    log_level: "normal"  # Options: "minimal", "normal", "debug", "verbose"
    
    # ========================================
    # DIRECTORY & FILE MANAGEMENT
    # ========================================
    
    # Output directory (empty = package data directory)
    results_dir: ""
    
    # Export formats
    output_formats: ["csv", "json", "pickle", "mat"]
    
    # Trajectory formats
    save_trajectories: true
    trajectory_format: "tum"  # Options: tum, kitti, euroc
    
    # ========================================
    # RACE ENDING CONDITIONS
    # ========================================
    
    race_ending_mode: "lap_complete"  # Options: "lap_complete", "crash", "manual"
    auto_shutdown_on_race_complete: true
    shutdown_delay_seconds: 5.0
    
    # Crash detection (for crash mode)
    crash_detection:
      enable_crash_detection: true
      max_stationary_time: 5.0
      min_velocity_threshold: 0.1
    
    # ========================================
    # TRAJECTORY EVALUATION
    # ========================================
    
    enable_trajectory_evaluation: true
    evaluation_interval_laps: 1  # Evaluate every N laps
    
    # Trajectory filtering
    apply_trajectory_filtering: true
    filter_types: ["motion", "distance", "angle"]
    filter_parameters:
      motion_threshold: 0.1
      distance_threshold: 0.05
      angle_threshold: 0.1
    
    # Metrics to calculate
    evaluate_smoothness: true
    evaluate_consistency: true
    evaluate_efficiency: true
    evaluate_aggressiveness: true
    evaluate_stability: true
    
    # Advanced analysis
    enable_advanced_metrics: true
    calculate_all_statistics: true
    analyze_rotation_errors: true
    
    # ========================================
    # REFERENCE TRAJECTORY
    # ========================================
    
    # Reference trajectory file (for EVO analysis)
    reference_trajectory_file: "reference_track.csv"
    reference_trajectory_format: "csv"
    
    # ========================================
    # COMPUTATIONAL PERFORMANCE MONITORING
    # ========================================
    
    enable_computational_monitoring: false
    enable_performance_logging: true
    
    # Monitoring parameters
    cpu_monitoring_interval: 1.0
    performance_log_interval: 10.0
    
    # Performance thresholds
    max_acceptable_latency_ms: 100.0
    target_control_frequency_hz: 20.0
    max_acceptable_cpu_usage: 80.0
    max_acceptable_memory_mb: 500.0
    
    # ========================================
    # GRAPH GENERATION
    # ========================================
    
    auto_generate_graphs: true
    graph_formats: ["png", "pdf"]
    
    # Plot appearance
    plot_figsize: [12.0, 8.0]
    plot_dpi: 300
    plot_style: "seaborn"
    plot_color_scheme: "viridis"
    
    # Plot types
    generate_trajectory_plots: true
    generate_xyz_plots: true
    generate_rpy_plots: true
    generate_speed_plots: true
    generate_error_plots: true
    generate_error_mapped_plots: true
    generate_3d_vector_plots: true
```

### Path Configuration

Race Monitor supports flexible path configuration:

#### Reference Trajectory
```yaml
# Simple filename (searches in ref_trajectory/)
reference_trajectory_file: "track.csv"

# Relative path from package share directory
reference_trajectory_file: "custom/tracks/track.csv"

# Absolute path
reference_trajectory_file: "/home/user/tracks/track.csv"
```

Place reference trajectories in `race_monitor/ref_trajectory/` for automatic discovery.

#### Output Directory
```yaml
# Default: package data directory
results_dir: ""

# Relative to package share directory
results_dir: "experiments"

# Absolute path
results_dir: "/home/user/race_data"
```

### Launch File Parameters

Override configuration parameters at launch time:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=my_controller \
    required_laps:=10 \
    race_mode:=lap_complete
```

Available launch parameters:
- `race_mode`: Race ending mode ("lap_complete", "crash", "manual")
- `controller_name`: Controller identifier (empty = auto-detect)
- `required_laps`: Number of laps to complete
- `enable_trajectory_evaluation`: Enable EVO analysis
- `enable_computational_monitoring`: Enable performance monitoring
- `auto_generate_graphs`: Generate visualization graphs
- `enable_smart_controller_detection`: Enable auto-detection
- `auto_shutdown_on_race_complete`: Auto-shutdown after race
- `shutdown_delay_seconds`: Delay before shutdown

---

## Experimental Results

### LQR Controller Performance Analysis

The following results demonstrate Race Monitor's capabilities using an LQR (Linear Quadratic Regulator) controller evaluation.

**Experiment Details:**
- **Controller**: LQR Controller Node
- **Experiment ID**: exp_002
- **Date**: October 24, 2024
- **Laps Completed**: 8
- **Total Race Time**: 149.11 seconds
- **System**: Ubuntu 22.04, ROS2 Humble

#### Performance Summary

| Metric | Value |
|--------|-------|
| **Best Lap Time** | 19.47 seconds |
| **Average Lap Time** | 21.08 seconds |
| **Median Lap Time** | 19.57 seconds |
| **Lap Time Std Dev** | 3.97 seconds |
| **Consistency Score** | 81.19% |
| **Average Speed** | 3.14 m/s |
| **Best Lap Speed** | 3.44 m/s |
| **Total Distance** | 468.91 meters |
| **Average Lap Distance** | 66.99 meters |

#### Lap-by-Lap Analysis

```
Lap 1:  19.61s  ████████████████████ 
Lap 2:  19.47s  ███████████████████  (Best)
Lap 3:  19.57s  ████████████████████ 
Lap 4:  19.50s  ████████████████████ 
Lap 5:  19.57s  ████████████████████ 
Lap 6:  19.82s  ████████████████████▌
Lap 7:  19.54s  ████████████████████ 
Lap 8:  31.57s  ████████████████████████████████ (Anomaly)
```

*Note: Lap 8 shows an anomaly likely due to manual race termination or environmental factor.*

#### Vehicle Dynamics Analysis

**Velocity Statistics:**
- Mean Linear Velocity: 0.342 m/s
- Velocity Std Dev: 0.135 m/s
- Max Velocity (Mean): 1.073 m/s

**Acceleration Statistics:**
- Mean Acceleration: 0.280 m/s²
- Acceleration Std Dev: 1.240 m/s²
- Max Acceleration (Mean): 10.73 m/s²
- Mean Jerk: 0.533 m/s³

**Angular Motion:**
- Mean Angular Velocity: 0.080 rad/s

#### Path Quality Metrics

**Trajectory Characteristics:**
- Mean Curvature: 0.0081 rad/m
- Max Curvature (Mean): 0.0366 rad/m
- Path Efficiency: 4.67%
- Overall Consistency CV: 4.65%

**Trajectory Statistics:**
- Total Trajectory Points: 13,706
- Average Points per Lap: 1,958
- Sampling Rate: 10 Hz
- Average Lap Distance: 66.99 m
- Path Length Std Dev: 0.178 m

#### Visualization Gallery

The system automatically generates comprehensive visualizations:

**Available Graphs:**
- `trajectories.png/pdf` - 2D trajectory comparison (all laps)
- `xyz.png/pdf` - 3D position coordinates over time
- `rpy.png/pdf` - Roll, pitch, yaw orientation analysis
- `speeds.png/pdf` - Velocity profiles across laps
- `errors.png/pdf` - Tracking error analysis
- `best_lap_3d_trajectory_vectors.png/pdf` - 3D best lap with velocity vectors
- `best_lap_error_mapped_trajectory.png/pdf` - Error heat map on trajectory
- `Trajectory_Error_Distribution.png/pdf` - Statistical error distribution

*Graphs are located in:* `data/lqr_controller_node/exp_002/graphs/`

#### Data Exports

Complete experimental data is exported in multiple formats:

**Trajectories** (`trajectories/`):
- CSV format: `lap_00X_trajectory.csv`
- JSON format: `lap_00X_trajectory.json`

**Results** (`results/`):
- Race results: `race_results.csv`, `race_results.json`
- Race summary: `race_summary.csv`, `race_summary.json`
- Evaluation metrics: `race_evaluation.csv`, `race_evaluation.json`

**Metadata**:
- Complete experiment metadata with system info
- Configuration snapshot
- Timestamp and version information

---

## Data Analysis & EVO Integration

### EVO Trajectory Evaluation Library

Race Monitor includes the complete **EVO (Python package for the evaluation of odometry and SLAM)** library as a submodule, providing research-grade trajectory analysis capabilities.

#### EVO Components

**1. Absolute Pose Error (APE) Analysis**

APE measures the absolute deviation between estimated and reference trajectories:

```bash
# Navigate to experiment data
cd data/lqr_controller_node/exp_002/trajectories

# Basic APE analysis
evo_ape tum reference_trajectory.txt lap_001_trajectory.txt

# APE with alignment and visualization
evo_ape tum reference_trajectory.txt lap_001_trajectory.txt \
    --align --correct_scale \
    --plot --save_plot ../graphs/ape_analysis.pdf \
    --save_results ../results/ape_results.zip
```

**2. Relative Pose Error (RPE) Analysis**

RPE measures drift over specified distances or time intervals:

```bash
# RPE with 1.0 meter delta
evo_rpe tum reference_trajectory.txt lap_001_trajectory.txt \
    --delta 1.0 --delta_unit m \
    --plot --save_plot ../graphs/rpe_analysis.pdf \
    --save_results ../results/rpe_results.zip

# RPE translation only
evo_rpe tum reference_trajectory.txt lap_001_trajectory.txt \
    --pose_relation trans_part --plot
```

**3. Trajectory Visualization**

```bash
# Compare multiple trajectories
evo_traj tum lap_001_trajectory.txt lap_002_trajectory.txt \
    reference_trajectory.txt \
    --plot --save_plot trajectory_comparison.pdf

# 3D visualization with time-colored trajectory
evo_traj tum lap_001_trajectory.txt \
    --plot_mode xyz --plot_colormap hot
```

**4. Multi-Controller Comparison**

```bash
# Analyze multiple controllers
for controller in lqr mpc pid; do
    evo_ape tum reference.txt data/${controller}/exp_001/lap_001_trajectory.txt \
        --align --save_results results/ape_${controller}.zip
done

# Generate comparison report
evo_res results/ape_*.zip \
    --plot --save_plot controller_comparison.pdf \
    --save_table comparison_summary.csv
```

#### EVO Configuration

```bash
# View current configuration
evo_config show

# Set output preferences
evo_config set plot_export_format pdf
evo_config set plot_figsize "[12, 8]"
evo_config set plot_linewidth 2.0

# Reset to defaults
evo_config reset
```

### Research Workflows

#### Workflow 1: Single Lap Analysis

```bash
cd data/lqr_controller_node/exp_002/trajectories

# 1. Absolute pose error
evo_ape tum reference.txt lap_001_trajectory.txt \
    --align --save_results ../results/ape_lap001.zip

# 2. Relative pose error  
evo_rpe tum reference.txt lap_001_trajectory.txt \
    --delta 1.0 --save_results ../results/rpe_lap001.zip

# 3. Visualization
evo_traj tum lap_001_trajectory.txt reference.txt \
    --plot --save_plot ../graphs/lap001_comparison.pdf
```

#### Workflow 2: Multi-Lap Comparison

```bash
# Compare all laps from an experiment
evo_traj tum lap_00*.txt --plot --ref reference.txt \
    --save_plot ../graphs/all_laps_comparison.pdf

# Statistical analysis across laps
for lap in lap_00{1..7}_trajectory.txt; do
    evo_ape tum reference.txt $lap --align \
        --save_results ../results/ape_${lap%.txt}.zip
done

# Aggregate results
evo_res ../results/ape_lap_*.zip \
    --save_table ../results/lap_comparison.csv
```

#### Workflow 3: Controller Benchmarking

```bash
# Compare different controllers on same track
cd data/

# Analyze each controller's best lap
for ctrl in controller1 controller2 controller3; do
    best_lap=$(find ${ctrl}/ -name "lap_*.txt" | head -1)
    evo_ape tum reference.txt $best_lap --align \
        --save_results results/ape_${ctrl}.zip
done

# Generate benchmark report
evo_res results/ape_controller*.zip \
    --plot --save_plot benchmark_comparison.pdf \
    --save_table benchmark_results.csv
```

### Supported Trajectory Formats

Race Monitor exports trajectories in multiple formats compatible with EVO:

| Format | Description | Use Case |
|--------|-------------|----------|
| **TUM** | `timestamp x y z qx qy qz qw` | Standard for EVO, SLAM benchmarking |
| **KITTI** | KITTI odometry format | Autonomous driving datasets |
| **EuRoC** | EuRoC MAV dataset format | MAV/drone applications |
| **CSV** | Custom CSV with headers | General data analysis |
| **JSON** | Structured JSON format | Web applications, APIs |

### Advanced EVO Features

#### Batch Processing Script

```bash
#!/bin/bash
# analyze_all_experiments.sh

EXPERIMENT_DIR="data/lqr_controller_node"
REFERENCE="ref_trajectory/reference_track.txt"

for exp in ${EXPERIMENT_DIR}/exp_*/; do
    echo "Processing ${exp}"
    for lap in ${exp}trajectories/lap_*.txt; do
        lap_name=$(basename $lap .txt)
        evo_ape tum $REFERENCE $lap --align \
            --save_results ${exp}results/ape_${lap_name}.zip
    done
    
    # Generate summary
    evo_res ${exp}results/ape_*.zip \
        --save_table ${exp}results/ape_summary.csv
done
```

#### Custom Plotting

```python
# custom_evo_plot.py
from evo.core import trajectory, metrics
from evo.tools import file_interface
import matplotlib.pyplot as plt

# Load trajectories
ref_traj = file_interface.read_tum_trajectory_file("reference.txt")
est_traj = file_interface.read_tum_trajectory_file("lap_001_trajectory.txt")

# Compute APE
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data((ref_traj, est_traj))

# Custom visualization
fig, ax = plt.subplots(figsize=(12, 8))
ape_metric.plot(ax, show=False)
plt.title("Custom APE Analysis")
plt.savefig("custom_ape_plot.pdf")
```

---

## ROS2 Interface

### Published Topics

#### Race Status Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/race_monitor/lap_count` | `std_msgs/Int32` | Event | Current lap number |
| `/race_monitor/lap_time` | `std_msgs/Float32` | Event | Last completed lap time (s) |
| `/race_monitor/race_running` | `std_msgs/Bool` | 10 Hz | Race active status |
| `/race_monitor/race_status` | `std_msgs/String` | 1 Hz | Detailed race state (JSON) |
| `/race_monitor/total_distance` | `std_msgs/Float32` | 10 Hz | Total distance traveled (m) |

#### Performance Monitoring Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/race_monitor/control_loop_latency` | `std_msgs/Float32` | 10 Hz | Control latency (ms) |
| `/race_monitor/cpu_usage` | `std_msgs/Float32` | 1 Hz | CPU utilization (%) |
| `/race_monitor/memory_usage` | `std_msgs/Float32` | 1 Hz | Memory usage (MB) |
| `/race_monitor/processing_efficiency` | `std_msgs/Float32` | 1 Hz | Efficiency score (0-1) |
| `/race_monitor/performance_stats` | `std_msgs/String` | 1 Hz | Complete statistics (JSON) |

#### Trajectory Analysis Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/race_monitor/trajectory_metrics` | `std_msgs/String` | Event | EVO metrics per lap (JSON) |
| `/race_monitor/smoothness` | `std_msgs/Float32` | 10 Hz | Current trajectory smoothness |
| `/race_monitor/consistency` | `std_msgs/Float32` | Event | Lap consistency score |
| `/race_monitor/path_efficiency` | `std_msgs/Float32` | 10 Hz | Path efficiency metric |

#### Visualization Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/race_monitor/current_trajectory` | `nav_msgs/Path` | 10 Hz | Current lap trajectory |
| `/race_monitor/reference_trajectory` | `nav_msgs/Path` | 1 Hz | Reference trajectory |
| `/race_monitor/start_finish_line` | `visualization_msgs/Marker` | 1 Hz | Start/finish line marker |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `car_state/odom` | `nav_msgs/Odometry` | Vehicle odometry (required) |
| `/clicked_point` | `geometry_msgs/PointStamped` | Interactive line setup |

### Services (Future Release)

| Service | Type | Description |
|---------|------|-------------|
| `/race_monitor/reset_race` | `std_srvs/Trigger` | Reset race state |
| `/race_monitor/save_data` | `std_srvs/Trigger` | Force data export |
| `/race_monitor/get_statistics` | Custom | Retrieve current statistics |

### Parameters

Dynamic reconfiguration of key parameters (future release):

```bash
# Get current race mode
ros2 param get /race_monitor race_ending_mode

# Check controller name (if auto-detected)
ros2 param get /race_monitor controller_name

# View log level
ros2 param get /race_monitor log_level
```

Note: Most parameters are set via YAML config or launch file and cannot be changed at runtime.

---

## Output Data Structure

Race Monitor organizes experimental data in a hierarchical structure optimized for research reproducibility:

```
race_monitor/
└── data/
    └── {controller_name}/
        └── {experiment_id}/  # Auto-generated: exp_YYYYMMDD_HHMMSS
            ├── experiment_metadata.txt
            ├── trajectories/
            │   ├── csv/
            │   │   ├── lap_001_trajectory.csv
            │   │   ├── lap_002_trajectory.csv
            │   │   └── ...
            │   ├── json/
            │   │   ├── lap_001_trajectory.json
            │   │   └── ...
            │   ├── tum/
            │   │   └── lap_001_trajectory.txt
            │   ├── pickle/
            │   │   └── lap_001_trajectory.pkl
            │   └── mat/
            │       └── lap_001_trajectory.mat
            ├── filtered/
            │   └── (Filtered trajectories if enabled)
            ├── results/
            │   ├── csv/
            │   │   ├── race_results.csv
            │   │   ├── race_summary.csv
            │   │   └── race_evaluation.csv
            │   └── json/
            │       ├── race_results.json
            │       ├── race_summary.json
            │       └── race_evaluation.json
            ├── metrics/
            │   └── json/
            │       └── (Detailed per-lap metrics)
            └── graphs/
                ├── png/
                │   ├── trajectories.png
                │   ├── speeds.png
                │   ├── errors.png
                │   ├── xyz.png
                │   ├── rpy.png
                │   ├── best_lap_3d_trajectory_vectors.png
                │   ├── best_lap_error_mapped_trajectory.png
                │   └── Trajectory_Error_Distribution.png
                └── pdf/
                    └── (Same graphs in PDF format)
```

### Data Files Description

#### `experiment_metadata.txt`
Complete experiment information including:
- Experiment ID and controller name
- Timestamp and system information
- ROS2 distribution and package version
- Configuration snapshot
- Maintainer information

#### Trajectory Files
Each lap's trajectory in multiple formats:
- **CSV**: Comma-separated with headers
- **JSON**: Structured format for web/API use
- **TUM**: EVO-compatible format (`timestamp x y z qx qy qz qw`)
- **Pickle**: Python serialized format
- **MATLAB**: .mat format for MATLAB analysis

#### Results Files
- **race_results**: Raw race data (lap times, distance, status)
- **race_summary**: Statistical summary (averages, best/worst, consistency)
- **race_evaluation**: Comprehensive analysis with 40+ metrics

#### Graphs
Publication-ready visualizations automatically generated:
- 2D/3D trajectory plots
- Velocity and acceleration profiles
- Error distributions and heat maps
- Orientation analysis (roll, pitch, yaw)
- Best lap detailed analysis

### Data Access Examples

#### Python
```python
import json
import pandas as pd
from pathlib import Path

# Find most recent experiment
exp_dir = Path('data/lqr_controller_node')
latest_exp = sorted(exp_dir.glob('exp_*'))[-1]

# Load race summary
summary_file = latest_exp / 'results/json/race_summary.json'
with open(summary_file) as f:
    summary = json.load(f)
    
print(f"Best lap: {summary['lap_statistics']['best_lap_time']:.2f}s")

# Load trajectory as DataFrame
traj_file = latest_exp / 'trajectories/csv/lap_001_trajectory.csv'
df = pd.read_csv(traj_file)
print(df.describe())
```

#### MATLAB
```matlab
% Load trajectory data (use actual experiment ID)
data = load('data/lqr_controller_node/exp_20241024_150100/trajectories/mat/lap_001_trajectory.mat');

% Plot trajectory
plot(data.x, data.y);
title('Lap 1 Trajectory');
xlabel('X (m)'); ylabel('Y (m)');
```

---

## Web UI Preview

**Coming in Next Release (v1.1.0)**

A comprehensive web interface for Race Monitor is under active development and will be included in the next major release. The Web UI will provide:

### Planned Features

**Real-time Monitoring Dashboard**
- Live race status and lap timing
- Real-time performance metrics visualization
- WebSocket-based updates for zero-latency monitoring
- System health indicators (CPU, memory, ROS2 status)

**Configuration Management**
- Browser-based YAML editor with syntax highlighting
- Real-time configuration validation
- Template management and version control
- Multi-file configuration support

**Advanced Data Analysis**
- Interactive trajectory visualization (2D/3D)
- Lap-by-lap performance comparison
- Statistical analysis tools
- EVO integration for trajectory evaluation

**Results Explorer**
- Browse historical experiments
- Compare multiple controllers
- Export data in various formats
- Generate custom reports

**Visual Analytics**
- Interactive Plotly.js charts
- Customizable dashboards
- Performance trend analysis
- Heat maps and distribution plots

### Technology Stack

- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + Python
- **Visualization**: Plotly.js
- **Real-time**: WebSocket communication

### Early Access

The Web UI is currently in alpha testing. For early access or to contribute:

```bash
# Preview the Web UI (development mode)
cd web_ui
./quick-start.sh

# Access at http://localhost:3001
```

For more information, see [`web_ui/README.md`](web_ui/README.md).

---

## Documentation

### Available Documentation
- **[Data Structure](data/README.md)** - Output organization and data formats
- **[Reference Trajectories](ref_trajectory/README.md)** - Reference trajectory setup guide
- **[Testing](test/README.md)** - Testing procedures and guidelines
- **[Configuration Audit](CONFIG_AUDIT.md)** - Configuration parameter reference
- **[Parameter Audit Report](PARAMETER_AUDIT_REPORT.md)** - Detailed parameter documentation
- **[Refactor Summary](REFACTOR_SUMMARY.md)** - Recent architectural changes

### Additional Resources
- Configuration examples in `config/race_monitor.yaml`
- Launch file documentation in `launch/race_monitor.launch.py`
- Web UI documentation in `web_ui/README.md`

---

## Citation

If you use Race Monitor in your research, please cite:

```bibtex
@software{race_monitor2024,
  author       = {Azab, Mohammed},
  title        = {Race Monitor: Advanced Performance Analysis Framework for Roboracer Autonomous Racing},
  year         = {2024},
  publisher    = {GitHub},
  organization = {GIU Roboracer Team},
  howpublished = {\url{https://github.com/GIU-F1Tenth/race_monitor}},
  version      = {1.0.0}
}
```

### Related Publications

If you use the EVO trajectory evaluation component, please also cite:

```bibtex
@software{grupp2017evo,
  author       = {Grupp, Michael},
  title        = {evo: Python package for the evaluation of odometry and SLAM},
  year         = {2017},
  howpublished = {\url{https://github.com/MichaelGrupp/evo}}
}
```

---

## Troubleshooting

### Common Issues

#### Installation Issues

**Problem**: EVO import fails
```bash
# Solution: Add EVO to Python path
export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/race_monitor/evo
pip install -e ~/ros2_ws/src/race_monitor/evo
```

**Problem**: Colcon build fails
```bash
# Solution: Clean and rebuild
cd ~/ros2_ws
rm -rf build install log
colcon build --packages-select race_monitor --symlink-install
```

#### Runtime Issues

**Problem**: No odometry data received
```bash
# Verify odometry topic
ros2 topic list | grep odom
ros2 topic echo car_state/odom --once

# Check topic remapping in launch file
```

**Problem**: Lap detection not working
```bash
# Verify start line configuration
ros2 param get /race_monitor start_line_p1
ros2 param get /race_monitor start_line_p2

# Check vehicle position relative to line
ros2 topic echo car_state/odom --field pose.pose.position

# Monitor lap detection debug output
ros2 topic echo /race_monitor/lap_count
```

**Problem**: Performance issues
```bash
# Check system resources
top
# Verify odometry topic frequency
ros2 topic hz car_state/odom
# Enable performance logging in config
# Set log_level: "debug" in race_monitor.yaml
```

#### Data Export Issues

**Problem**: Graphs not generating
```bash
# Verify matplotlib backend
python3 -c "import matplotlib; print(matplotlib.get_backend())"

# Check output directory permissions
ls -la data/

# Enable debug logging
ros2 launch race_monitor race_monitor.launch.py log_level:=DEBUG
```

**Problem**: Missing trajectory files
```bash
# Verify save_trajectories is enabled
ros2 param get /race_monitor save_trajectories

# Check disk space
df -h

# Verify output directory path
ros2 param get /race_monitor results_dir

# Check if data directory exists
ls -la data/
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Edit race_monitor.yaml and set:
# log_level: "debug"  # or "verbose" for maximum detail

# Then launch normally
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=debug_test
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/GIU-F1Tenth/race_monitor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GIU-F1Tenth/race_monitor/discussions)
- **Email**: mohammed@azab.io

---

## Contributing

We welcome contributions! Race Monitor is an open-source project and benefits from community input.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/race_monitor.git
cd race_monitor

# Create development branch
git checkout -b dev

# Install development dependencies
pip install -c constraints.txt -r requirements.txt

# Run tests
python -m pytest test/
```

### Code Standards

- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Write unit tests for new features
- Update documentation as needed

### Areas for Contribution

- **Feature Development**: New analysis metrics, visualization types
- **Documentation**: Tutorials, examples, translations
- **Testing**: Unit tests, integration tests, CI/CD improvements
- **Bug Fixes**: Issue resolution and optimization
- **Web UI**: Frontend/backend development (upcoming)

---

## License

Race Monitor is released under the MIT License. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2024 Mohammed Azab - GIU Roboracer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

### Dependencies

Race Monitor builds upon excellent open-source projects:

- **[ROS2](https://www.ros.org/)** - Robot Operating System 2
- **[EVO](https://github.com/MichaelGrupp/evo)** - Trajectory evaluation library by Michael Grupp
- **[NumPy](https://numpy.org/)** - Numerical computing library
- **[SciPy](https://scipy.org/)** - Scientific computing library
- **[Matplotlib](https://matplotlib.org/)** - Visualization library
- **[Pandas](https://pandas.pydata.org/)** - Data analysis library

### Inspiration

Race Monitor was inspired by the need for systematic performance evaluation in the Roboracer autonomous racing community. 

### Support

Development supported by:
- **German International University (GIU)**
- **Roboracer Autonomous Racing Community**

---

## Contact

**Mohammed Azab**
- Email: mohammed@azab.io
- GitHub: [@mohammedazab](https://github.com/mohammedazab)
- Organization: [GIU F1Tenth Team](https://github.com/GIU-F1Tenth)

### Links

- **Repository**: https://github.com/GIU-F1Tenth/race_monitor
- **Issues**: https://github.com/GIU-F1Tenth/race_monitor/issues
- **Discussions**: https://github.com/GIU-F1Tenth/race_monitor/discussions
- **Releases**: https://github.com/GIU-F1Tenth/race_monitor/releases

---

<div align="center">

**Race Monitor v1.0.0** • *Updated October 2024*

*Advancing autonomous racing research through comprehensive performance analysis*

[![GitHub stars](https://img.shields.io/github/stars/GIU-F1Tenth/race_monitor?style=social)](https://github.com/GIU-F1Tenth/race_monitor)
[![GitHub forks](https://img.shields.io/github/forks/GIU-F1Tenth/race_monitor?style=social)](https://github.com/GIU-F1Tenth/race_monitor/fork)

Built with ❤️ by the GIU Roboracer Team

</div>