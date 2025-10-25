# Usage Guide

Complete usage instructions for Race Monitor.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Race Modes](#race-modes)
- [Interactive Setup](#interactive-setup)
- [Real-time Monitoring](#real-time-monitoring)
- [Advanced Usage](#advanced-usage)
- [Workflows](#common-workflows)
- [Data Access](#data-access)
- [Examples](#usage-examples)

---

## Quick Start

### 1. Configure System

Edit the configuration file:

```bash
nano ~/ros2_ws/src/race_monitor/config/race_monitor.yaml
```

Set your start/finish line and required laps:
```yaml
start_line_p1: [0.0, -2.0]
start_line_p2: [0.0, 2.0]
required_laps: 7
```

### 2. Launch Race Monitor

```bash
# Source your workspace
source ~/ros2_ws/install/setup.bash

# Launch with default settings
ros2 launch race_monitor race_monitor.launch.py
```

### 3. Start Your Controller

In a separate terminal:

```bash
ros2 launch your_controller your_controller.launch.py
```

### 4. Monitor Progress

Watch the console output or check topics:

```bash
# Watch lap times
ros2 topic echo /race_monitor/lap_time

# Monitor race status
ros2 topic echo /race_monitor/race_status
```

---

## Basic Usage

### Launching Race Monitor

#### Default Launch

```bash
ros2 launch race_monitor race_monitor.launch.py
```

#### With Custom Parameters

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=lqr_controller \
    required_laps:=10
```

#### Available Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `controller_name` | `""` | Controller identifier (empty = auto-detect) |
| `required_laps` | `7` | Number of laps to complete |
| `race_mode` | `"lap_complete"` | Race ending mode |
| `enable_trajectory_evaluation` | `true` | Enable EVO analysis |
| `enable_computational_monitoring` | `false` | Enable performance monitoring |
| `auto_generate_graphs` | `true` | Generate visualization graphs |
| `enable_smart_controller_detection` | `true` | Auto-detect controller name |
| `auto_shutdown_on_race_complete` | `true` | Auto-shutdown after race |
| `shutdown_delay_seconds` | `5.0` | Delay before shutdown |

### Setting Start/Finish Line

#### Method 1: Configuration File

Edit `config/race_monitor.yaml`:

```yaml
start_line_p1: [0.0, -2.0]  # First point [x, y]
start_line_p2: [0.0, 2.0]   # Second point [x, y]
```

#### Method 2: RViz Interactive Setup

1. Launch RViz with Race Monitor configuration:
   ```bash
   rviz2 -d ~/ros2_ws/src/race_monitor/config/race_monitor.rviz
   ```

2. Select "Publish Point" tool (press `p` or click toolbar button)

3. Click two points on the track to define the line

4. The line will be visualized and confirmed in console

### Controller Detection

Race Monitor can automatically detect your controller:

```yaml
controller_name: ""                      # Empty = auto-detect
enable_smart_controller_detection: true  # Enable auto-detection
```

**Manual Override**:
```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=my_custom_controller
```

---

## Race Modes

Race Monitor supports three race ending modes:

### 1. Lap Complete Mode (Default)

Race ends after completing required laps:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    race_mode:=lap_complete \
    required_laps:=7
```

**Use Case**: Standard racing, performance testing

### 2. Crash Detection Mode

Race ends when vehicle crashes:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    race_mode:=crash
```

Configure crash detection in `race_monitor.yaml`:
```yaml
crash_detection:
  enable_crash_detection: true
  max_stationary_time: 5.0
  min_velocity_threshold: 0.1
```

**Use Case**: Safety testing, controller validation

### 3. Manual Mode

Race continues until manually stopped:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    race_mode:=manual
```

Stop the race:
```bash
# Press Ctrl+C in the Race Monitor terminal
```

**Use Case**: Development, testing, debugging

---

## Interactive Setup

### Using RViz for Start Line Setup

1. **Launch Race Monitor**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py
   ```

2. **Open RViz** (in separate terminal):
   ```bash
   rviz2 -d ~/ros2_ws/src/race_monitor/config/race_monitor.rviz
   ```

3. **Visualize Odometry**:
   - Check that `/car_state/odom` is being received
   - Vehicle should be visible on the map

4. **Set Start Line**:
   - Click "Publish Point" tool (or press `p`)
   - Click first point on track
   - Click second point on track
   - Line will appear in RViz

5. **Verify Line Position**:
   - Start line marker should be visible
   - Drive vehicle across line to test detection

### Monitoring in RViz

The Race Monitor RViz configuration includes:
- Current trajectory visualization
- Reference trajectory (if provided)
- Start/finish line marker
- Vehicle position

---

## Real-time Monitoring

### Console Output

Race Monitor provides real-time console output:

```
[INFO] Race started | Beginning lap 1
[INFO] Lap complete | Lap 1 in 19.67s
[INFO] Lap complete | Lap 2 in 19.73s
[INFO] Race completed!
[INFO] Total time: 138.23s
[INFO] Average lap: 19.75s
```

### ROS2 Topics

Monitor race progress via topics:

```bash
# Lap count
ros2 topic echo /race_monitor/lap_count

# Lap time (published on lap completion)
ros2 topic echo /race_monitor/lap_time

# Race running status
ros2 topic echo /race_monitor/race_running

# Race status (JSON)
ros2 topic echo /race_monitor/race_status

# Total distance
ros2 topic echo /race_monitor/total_distance
```

### Performance Monitoring

If computational monitoring is enabled:

```bash
# Control loop latency (ms)
ros2 topic echo /race_monitor/control_loop_latency

# CPU usage (%)
ros2 topic echo /race_monitor/cpu_usage

# Memory usage (MB)
ros2 topic echo /race_monitor/memory_usage

# Performance statistics (JSON)
ros2 topic echo /race_monitor/performance_stats
```

### Trajectory Monitoring

```bash
# Current trajectory path
ros2 topic echo /race_monitor/current_trajectory

# Trajectory metrics (JSON)
ros2 topic echo /race_monitor/trajectory_metrics

# Smoothness score
ros2 topic echo /race_monitor/smoothness

# Path efficiency
ros2 topic echo /race_monitor/path_efficiency
```

---

## Advanced Usage

### Research Experiment

Complete analysis with all features:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=lqr_controller_node \
    required_laps:=8 \
    enable_trajectory_evaluation:=true \
    auto_generate_graphs:=true \
    enable_computational_monitoring:=true
```

Configure in `race_monitor.yaml`:
```yaml
# Comprehensive analysis
enable_advanced_metrics: true
calculate_all_statistics: true
enable_geometric_analysis: true

# Reference comparison
reference_trajectory_file: "optimal_line.csv"

# Detailed output
save_trajectories: true
save_filtered_trajectories: true
output_formats: ["csv", "json", "pickle", "mat"]

# Visualization
generate_error_mapped_plots: true
generate_3d_vector_plots: true
```

### Performance Benchmarking

Focus on computational performance:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=benchmark_controller \
    enable_computational_monitoring:=true
```

Set performance thresholds in `race_monitor.yaml`:
```yaml
target_control_frequency_hz: 20.0
max_acceptable_latency_ms: 50.0
max_acceptable_cpu_usage: 70.0
max_acceptable_memory_mb: 500.0
```

### Multi-lap Endurance Test

Extended testing with periodic evaluation:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=endurance_test \
    required_laps:=50
```

Configure evaluation interval:
```yaml
evaluation_interval_laps: 5  # Evaluate every 5 laps
save_intermediate_results: true
```

### Controller Comparison

Testing multiple controllers:

```bash
# Test Controller A
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=controller_a \
    required_laps:=10

# Test Controller B
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=controller_b \
    required_laps:=10

# Compare results
# Results stored in:
# data/controller_a/exp_YYYYMMDD_HHMMSS/
# data/controller_b/exp_YYYYMMDD_HHMMSS/
```

---

## Common Workflows

### Workflow 1: Quick Performance Test

1. **Setup**:
   ```bash
   # Edit config if needed
   nano config/race_monitor.yaml
   ```

2. **Launch**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py \
       controller_name:=test_controller \
       required_laps:=5
   ```

3. **Start controller** in separate terminal

4. **Review results**:
   ```bash
   # Check latest experiment
   ls -lt data/test_controller/
   
   # View summary
   cat data/test_controller/exp_*/results/csv/race_summary.csv
   ```

### Workflow 2: Research Evaluation

1. **Prepare reference trajectory**:
   ```bash
   # Copy reference to ref_trajectory/
   cp /path/to/reference.csv ref_trajectory/
   ```

2. **Configure** in `race_monitor.yaml`:
   ```yaml
   reference_trajectory_file: "reference.csv"
   enable_trajectory_evaluation: true
   enable_advanced_metrics: true
   auto_generate_graphs: true
   ```

3. **Launch**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py \
       controller_name:=research_controller \
       required_laps:=10
   ```

4. **Analyze results**:
   - View graphs: `data/research_controller/exp_*/graphs/`
   - Check metrics: `data/research_controller/exp_*/results/`
   - Use EVO: See [EVO_INTEGRATION.md](EVO_INTEGRATION.md)

### Workflow 3: Safety Testing

1. **Configure crash detection**:
   ```yaml
   race_ending_mode: "crash"
   crash_detection:
     enable_crash_detection: true
     max_stationary_time: 3.0
     enable_collision_detection: true
   ```

2. **Launch**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py \
       race_mode:=crash \
       controller_name:=safety_test
   ```

3. **Monitor**:
   ```bash
   # Watch for crash detection
   ros2 topic echo /race_monitor/race_status
   ```

4. **Review crash data**:
   - Trajectory before crash
   - Velocity profile
   - Time to stationary

### Workflow 4: Development & Debugging

1. **Enable debug logging**:
   ```yaml
   log_level: "debug"
   race_ending_mode: "manual"
   save_intermediate_results: true
   ```

2. **Launch**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py \
       race_mode:=manual \
       controller_name:=dev_test
   ```

3. **Test your changes**

4. **Stop manually**: Press `Ctrl+C`

5. **Check logs and data**:
   ```bash
   # Recent logs
   less log/latest_list/race_monitor/stdout.log
   
   # Intermediate results
   ls data/dev_test/exp_*/results/
   ```

---

## Data Access

### Experiment Directory Structure

After a race, data is organized as:

```
data/
└── {controller_name}/
    └── exp_YYYYMMDD_HHMMSS/
        ├── experiment_metadata.txt
        ├── trajectories/
        │   ├── csv/
        │   ├── json/
        │   ├── tum/
        │   ├── pickle/
        │   └── mat/
        ├── results/
        │   ├── csv/
        │   │   ├── race_results.csv
        │   │   ├── race_summary.csv
        │   │   └── race_evaluation.csv
        │   └── json/
        └── graphs/
            ├── png/
            └── pdf/
```

### Accessing Results

#### Python

```python
import json
import pandas as pd
from pathlib import Path

# Find latest experiment
exp_dir = Path('data/lqr_controller_node')
latest_exp = sorted(exp_dir.glob('exp_*'))[-1]

# Load race summary
summary_file = latest_exp / 'results/json/race_summary.json'
with open(summary_file) as f:
    summary = json.load(f)
    
print(f"Best lap: {summary['lap_statistics']['best_lap_time']:.2f}s")
print(f"Average lap: {summary['lap_statistics']['average_lap_time']:.2f}s")

# Load trajectory as DataFrame
traj_file = latest_exp / 'trajectories/csv/lap_001_trajectory.csv'
df = pd.read_csv(traj_file)
print(df.describe())
```

#### MATLAB

```matlab
% Load experiment data
exp_dir = 'data/lqr_controller_node/exp_20250125_153000/';

% Load trajectory
data = load([exp_dir 'trajectories/mat/lap_001_trajectory.mat']);

% Plot trajectory
figure;
plot(data.x, data.y);
title('Lap 1 Trajectory');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on;

% Load race summary
summary = readtable([exp_dir 'results/csv/race_summary.csv']);
disp(summary);
```

#### Command Line

```bash
# Find latest experiment
LATEST=$(ls -td data/*/exp_* | head -1)

# View race summary
cat $LATEST/results/csv/race_summary.csv

# View lap times
awk -F, '/^[0-9]/ {print "Lap " $1 ": " $2 "s"}' \
    $LATEST/results/csv/race_results.csv

# Count trajectory points
wc -l $LATEST/trajectories/csv/lap_*.csv
```

### Viewing Graphs

```bash
# Open all graphs
LATEST=$(ls -td data/*/exp_* | head -1)
xdg-open $LATEST/graphs/png/*.png

# View specific graph
eog $LATEST/graphs/png/trajectories.png

# Convert to PDF presentation
cd $LATEST/graphs/png
convert *.png race_analysis.pdf
```

---

## Usage Examples

### Example 1: Standard Race

```bash
# Simple 7-lap race with auto-detection
ros2 launch race_monitor race_monitor.launch.py
```

### Example 2: Named Controller Test

```bash
# Test specific controller with 10 laps
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=mpc_controller \
    required_laps:=10
```

### Example 3: Research Analysis

```bash
# Full analysis with reference trajectory
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=research_lqr \
    required_laps:=8 \
    enable_trajectory_evaluation:=true \
    auto_generate_graphs:=true
```

### Example 4: Performance Benchmark

```bash
# Monitor computational performance
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=benchmark_test \
    enable_computational_monitoring:=true \
    required_laps:=20
```

### Example 5: Development Testing

```bash
# Manual mode for development
ros2 launch race_monitor race_monitor.launch.py \
    race_mode:=manual \
    controller_name:=dev_controller
# Stop with Ctrl+C when ready
```

---

## Tips & Best Practices

### For Racing
- Use clear controller names for easy identification
- Set appropriate debounce time (2-3 seconds)
- Enable auto-generation of graphs
- Use lap_complete mode

### For Development
- Use manual mode for iterative testing
- Enable debug logging
- Save intermediate results
- Test with fewer laps initially

### For Research
- Always use reference trajectories
- Enable all metrics and statistics
- Save in multiple formats
- Document experiments in metadata

### Troubleshooting Tips
- Check odometry topic: `ros2 topic hz /car_state/odom`
- Verify start line position in RViz
- Monitor lap detection with debug logging
- Ensure sufficient disk space for data

---

## Next Steps

- **Configuration Details**: [CONFIGURATION.md](CONFIGURATION.md)
- **EVO Analysis**: [EVO_INTEGRATION.md](EVO_INTEGRATION.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Example Results**: [RESULTS.md](RESULTS.md)

---

**Need Help?**
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Issues: https://github.com/GIU-F1Tenth/race_monitor/issues
- Discussions: https://github.com/GIU-F1Tenth/race_monitor/discussions
