# Race Monitor

A ROS2 package for autonomous racing trajectory analysis and performance evaluation using the EVO trajectory evaluation library.

## Overview

The Race Monitor provides comprehensive trajectory analysis capabilities for autonomous racing applications. It integrates the EVO library to perform advanced trajectory evaluation, calculates performance metrics, and exports research-ready data for controller comparison and analysis.

## Features

- Real-time lap timing and trajectory recording
- Advanced trajectory analysis using EVO library (100+ metrics)
- Multi-format data export (JSON, CSV, TUM, Pickle)
- Configurable analysis parameters
- Research-grade statistical analysis
- Support for trajectory filtering and smoothing

## Installation

### Dependencies

- ROS2 Humble or later
- Python 3.8+
- numpy
- matplotlib
- EVO trajectory evaluation library

### Important: EVO Library Setup

**⚠️ Critical Requirement:** The EVO trajectory evaluation library must be available in your workspace. This package includes EVO as a submodule, but you need to ensure it's properly initialized.

#### Option 1: Initialize the EVO Submodule (Recommended)

If you cloned this repository, initialize the EVO submodule:

```bash
# From the race_monitor package directory
cd /path/to/your/workspace/src/race_monitor
git submodule init
git submodule update
```

#### Option 2: Clone EVO Separately

If the submodule approach doesn't work, clone EVO directly:

```bash
# From your workspace src directory
cd /path/to/your/workspace/src
git clone https://github.com/MichaelGrupp/evo.git
```

#### Verify EVO Installation

Ensure EVO is properly available:

```bash
# Check if EVO directory exists
ls -la /path/to/your/workspace/src/race_monitor/evo
# OR (if cloned separately)
ls -la /path/to/your/workspace/src/evo

# Install EVO dependencies
cd /path/to/evo/directory
pip install -e .
```

### Build

```bash
cd /path/to/your/workspace
colcon build --packages-select race_monitor
source install/setup.bash
```

## Usage

### Configuration

Edit `config/race_monitor.yaml` to configure the system for your controller:

```yaml
race_monitor:
  ros__parameters:
    # Controller identification
    controller_name: "your_controller_name"
    experiment_id: "session_001"
    
    # Analysis settings
    enable_trajectory_evaluation: true
    enable_advanced_metrics: true
    save_trajectories: true
```

### Running

```bash
# Start race monitor
ros2 run race_monitor race_monitor

# Or use launch file
ros2 launch race_monitor race_monitor.launch.py
```

### Data Output

The system automatically generates:

- Trajectory files in TUM format
- Detailed metrics per lap (JSON)
- Summary statistics (JSON/CSV)
- Research-ready exports

Output location: `trajectory_evaluation/research_data/{controller_name}/{experiment_id}/`

## Topics

### Subscribed

- `/car_state/odom` (nav_msgs/Odometry) - Vehicle odometry
- `/clicked_point` (geometry_msgs/PointStamped) - Manual start/finish line setup

### Published

- `/race_monitor/lap_count` (std_msgs/Int32) - Current lap number
- `/race_monitor/lap_time` (std_msgs/Float32) - Last lap time
- `/race_monitor/race_status` (std_msgs/String) - Race status

## Metrics Calculated

The system calculates 40+ performance metrics including:

- **Basic**: Path length, duration, average speed
- **Velocity**: Mean/std/max velocity, consistency
- **Acceleration**: Mean/std/max acceleration, jerk
- **Steering**: Angular velocity analysis, aggressiveness
- **Geometric**: Curvature analysis, path efficiency
- **Statistical**: Complete statistical breakdown per metric

## License

MIT License - see LICENSE file for details.