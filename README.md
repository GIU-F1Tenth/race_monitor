# Race Monitor

A ROS2 package for autonomous racing trajectory analysis and performance evaluation using the EVO trajectory evaluation library with advanced computational performance monitoring.

## Overview

The Race Monitor provides comprehensive trajectory analysis capabilities for autonomous racing applications. It integrates the EVO library to perform advanced trajectory evaluation, calculates performance metrics, monitors computational efficiency, and exports research-ready data for controller comparison and analysis.

## Features

### Trajectory Analysis
- Real-time lap timing and trajectory recording
- Advanced trajectory analysis using EVO library (100+ metrics)
- Multi-format data export (JSON, CSV, TUM, Pickle)
- Configurable analysis parameters
- Research-grade statistical analysis
- Support for trajectory filtering and smoothing

### 🆕 Computational Performance Monitoring
- **Real-time control loop latency measurement** (odometry → control command)
- **CPU and memory usage tracking** of the control process
- **Processing efficiency scoring** based on latency and resource usage
- **Control frequency analysis** and violation detection
- **Performance data logging** for post-race analysis
- **Comprehensive statistics** with configurable thresholds

### Key Performance Metrics
- Control loop latency (ms)
- CPU utilization (%)
- Memory consumption (MB)
- Processing efficiency score (0-1)
- Control command frequency (Hz)
- Real-time constraint violations

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

### Quick Start with Performance Monitoring

```bash
# Launch with computational performance monitoring (recommended)
ros2 launch race_monitor race_monitor_with_perf.launch.py \
    controller_name:=your_controller \
    control_command_topic:=/drive

# Monitor real-time performance
ros2 topic echo /race_monitor/control_loop_latency
ros2 topic echo /race_monitor/cpu_usage
ros2 topic echo /race_monitor/performance_stats
```

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
    
    # 🆕 Computational performance monitoring
    enable_computational_monitoring: true
    control_command_topic: "/drive"
    control_command_type: "ackermann"  # or "twist"
    max_acceptable_latency_ms: 50.0
    target_control_frequency_hz: 50.0
```

### Running

```bash
# Standard trajectory analysis
ros2 launch race_monitor race_monitor.launch.py

# With computational performance monitoring
ros2 launch race_monitor race_monitor_with_perf.launch.py

# Custom control topic
ros2 launch race_monitor race_monitor_with_perf.launch.py \
    control_command_topic:=/your_controller/cmd \
    control_command_type:=twist
```

### Data Output

The system automatically generates:

**Trajectory Analysis:**
- Trajectory files in TUM format
- Detailed metrics per lap (JSON)
- Summary statistics (JSON/CSV)
- Research-ready exports

**🆕 Performance Monitoring:**
- Real-time latency measurements
- CPU and memory usage logs
- Processing efficiency reports
- Performance statistics (CSV)

Output location: `trajectory_evaluation/research_data/{controller_name}/{experiment_id}/`

## Topics

### Subscribed Topics
- `car_state/odom` (nav_msgs/Odometry) - Vehicle odometry
- `/clicked_point` (geometry_msgs/PointStamped) - Start/finish line setup
- `{control_command_topic}` (AckermannDriveStamped/Twist) - Control commands for performance monitoring

### Published Topics

**Race Monitoring:**
- `/race_monitor/lap_count` (std_msgs/Int32) - Current lap number
- `/race_monitor/lap_time` (std_msgs/Float32) - Last lap time
- `/race_monitor/race_running` (std_msgs/Bool) - Race status

**🆕 Performance Monitoring:**
- `/race_monitor/control_loop_latency` (std_msgs/Float32) - Control latency (ms)
- `/race_monitor/cpu_usage` (std_msgs/Float32) - CPU usage (%)
- `/race_monitor/memory_usage` (std_msgs/Float32) - Memory usage (MB)
- `/race_monitor/processing_efficiency` (std_msgs/Float32) - Efficiency score (0-1)
- `/race_monitor/performance_stats` (std_msgs/String) - Comprehensive stats (JSON)

**Trajectory Analysis:**
- `/race_monitor/trajectory_metrics` (std_msgs/String) - EVO metrics (JSON)
- `/race_monitor/smoothness` (std_msgs/Float32) - Trajectory smoothness
- `/race_monitor/consistency` (std_msgs/Float32) - Trajectory consistency

## Documentation

### Quick Guides
- **[Quick Start Guide](QUICK_START_PERFORMANCE.md)** - Get started with performance monitoring
- **[Computational Monitoring](docs/COMPUTATIONAL_MONITORING.md)** - Detailed performance monitoring documentation

### Test and Examples
- **[Test Script](test_computational_monitoring.py)** - Example test for performance monitoring
- **[Configuration Examples](config/race_monitor.yaml)** - Sample configurations

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