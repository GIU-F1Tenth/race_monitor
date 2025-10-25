# Configuration Guide

Complete configuration reference for Race Monitor.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration File](#configuration-file)
- [Core Parameters](#core-parameters)
- [Race Ending Modes](#race-ending-modes)
- [Trajectory Evaluation](#trajectory-evaluation)
- [Analysis & Metrics](#analysis--metrics)
- [Data Export](#data-export)
- [Visualization](#visualization)
- [Performance Monitoring](#performance-monitoring)
- [Launch Parameters](#launch-parameters)
- [Examples](#configuration-examples)

---

## Quick Start

The primary configuration file is located at:
```
race_monitor/config/race_monitor.yaml
```

**Minimal configuration required**:
1. Define start/finish line: `start_line_p1` and `start_line_p2`
2. Set required laps: `required_laps`
3. All other parameters have sensible defaults

---

## Configuration File

### File Location

```bash
# Default location
~/ros2_ws/src/race_monitor/config/race_monitor.yaml
```

### File Structure

The configuration file uses YAML format with the following structure:

```yaml
race_monitor:
  ros__parameters:
    # Your parameters here
```

---

## Core Parameters

### Start/Finish Line Definition

Define the race start/finish line as two points in the map frame:

```yaml
start_line_p1: [0.0, -2.0]  # First endpoint [x, y] in meters
start_line_p2: [0.0, 2.0]   # Second endpoint [x, y] in meters
```

**Interactive Setup**: Use RViz to set these points:
1. Launch RViz with Race Monitor config
2. Use "Publish Point" tool (press `p`)
3. Click two points on the track
4. The line will be visualized in RViz

### Race Requirements

```yaml
required_laps: 7              # Number of laps to complete
debounce_time: 2.0            # Minimum time (s) between lap detections
frame_id: "map"               # TF frame for coordinates
```

### Lap Detection

```yaml
lap_detection:
  expected_direction: "any"           # "any", "positive", "negative"
  validate_heading_direction: false   # Validate based on vehicle heading
```

**Direction Options**:
- `"any"`: Count laps in both directions
- `"positive"`: Only clockwise/right-hand crossings
- `"negative"`: Only counterclockwise/left-hand crossings

### Logging & Debugging

```yaml
log_level: "normal"  # "minimal", "normal", "debug", "verbose"
```

**Log Levels**:
- `"minimal"`: Only critical events and final results
- `"normal"`: Standard operational messages (recommended)
- `"debug"`: Detailed diagnostic information
- `"verbose"`: Maximum detail for troubleshooting

### Directory Management

```yaml
results_dir: ""  # Empty = package's data/ directory
```

**Path Options**:
- `""` (empty): Uses `race_monitor/data/` (default)
- `"experiments"`: Relative path → `race_monitor/experiments/`
- `"/home/user/data"`: Absolute path

---

## Race Ending Modes

Race Monitor supports three race ending modes:

### 1. Lap Complete Mode (Default)

Race ends when required laps are finished:

```yaml
race_ending_mode: "lap_complete"
required_laps: 7
auto_shutdown_on_race_complete: true
shutdown_delay_seconds: 5.0
```

### 2. Crash Detection Mode

Race ends when vehicle crash is detected:

```yaml
race_ending_mode: "crash"
crash_detection:
  enable_crash_detection: true
  max_stationary_time: 5.0              # Max stationary time (s)
  min_velocity_threshold: 0.1           # Min velocity (m/s)
  max_odometry_timeout: 3.0             # Max time without odometry
  enable_collision_detection: true      # Detect sudden velocity changes
  collision_velocity_threshold: 2.0     # Velocity change indicating collision
  collision_detection_window: 0.5       # Time window for detection
```

### 3. Manual Mode

Race continues until manually stopped:

```yaml
race_ending_mode: "manual"
manual_mode:
  save_intermediate_results: true       # Save periodically during race
  save_interval: 30.0                   # Save interval (s)
  max_race_duration: 0                  # 0 = unlimited duration
```

---

## Trajectory Evaluation

### Enable/Disable Evaluation

```yaml
enable_trajectory_evaluation: true
```

### Evaluation Timing

Choose one interval type (set others to 0):

```yaml
# Time-based evaluation
evaluation_interval_seconds: 0.0    # Evaluate every X seconds

# Lap-based evaluation (recommended)
evaluation_interval_laps: 1         # Evaluate every X laps

# Distance-based evaluation
evaluation_interval_meters: 0.0     # Evaluate every X meters
```

### Reference Trajectory

For APE/RPE error analysis:

```yaml
reference_trajectory_file: "reference_track.csv"
reference_trajectory_format: "csv"  # "csv", "tum", or "kitti"
```

**Path Options**:
- Simple filename: `"track.csv"` → Searches in `ref_trajectory/`
- Relative path: `"custom/track.csv"` → Relative to package
- Absolute path: `"/home/user/tracks/track.csv"`

**Supported Formats**:
- **CSV**: Custom format with headers
- **TUM**: `timestamp x y z qx qy qz qw`
- **KITTI**: KITTI odometry format

---

## Analysis & Metrics

### Core Analysis Features

```yaml
enable_advanced_metrics: true           # Comprehensive metrics
calculate_all_statistics: true          # Mean/median/std/min/max
analyze_rotation_errors: true           # Orientation analysis
enable_geometric_analysis: true         # Arc length, curvature
enable_filtering_analysis: true         # Trajectory filtering
```

### Analysis Depth

```yaml
detailed_lap_analysis: true             # Per-lap breakdowns
comparative_analysis: true              # Cross-experiment comparison
statistical_significance: true          # Statistical tests
```

### Race Evaluation (Grading)

```yaml
enable_race_evaluation: true            # A-F performance grading
grading_strictness: "normal"            # "lenient", "normal", "strict"
enable_comparison: false                # Compare with previous experiments
enable_recommendations: false           # Improvement suggestions
```

### Trajectory Metrics

```yaml
# Quality metrics
evaluate_smoothness: true               # Jerk, acceleration
evaluate_consistency: true              # Lap-to-lap consistency
evaluate_efficiency: true               # Path efficiency
evaluate_aggressiveness: true           # Driving style
evaluate_stability: true                # Control stability
```

### EVO Metrics Configuration

```yaml
# Pose relations for error calculation
pose_relations: 
  - "translation_part"                  # Position errors
  - "rotation_part"                     # Orientation errors
  - "full_transformation"               # Combined errors

# Statistical measures
statistics_types: 
  - "rmse"                              # Root mean square error
  - "mean"                              # Mean error
  - "median"                            # Median error
  - "std"                               # Standard deviation
  - "min"                               # Minimum error
  - "max"                               # Maximum error
  - "sse"                               # Sum of squared errors
```

### Trajectory Filtering

```yaml
apply_trajectory_filtering: true

filter_types: 
  - "motion"                            # Motion-based filtering
  - "distance"                          # Distance-based filtering
  - "angle"                             # Angle-based filtering

filter_parameters:
  motion_threshold: 0.1                 # Min motion (m)
  distance_threshold: 0.05              # Min distance (m)
  angle_threshold: 0.1                  # Min angle (rad)
```

---

## Data Export

### Export Options

```yaml
save_trajectories: true                 # Raw trajectory data
save_filtered_trajectories: true        # Filtered trajectories
save_detailed_statistics: true          # Statistical analysis
export_research_summary: true           # Research-ready summary
export_to_pandas: true                  # Pandas-compatible formats
save_intermediate_results: true         # Save after each lap
include_timestamps: false               # Detailed timestamps
```

### Output Formats

```yaml
output_formats: 
  - "csv"                               # Excel, MATLAB
  - "json"                              # Python, JavaScript
  - "pickle"                            # Python pandas
  - "mat"                               # MATLAB
```

**Format Comparison**:

| Format | Best For | Size | Read Speed |
|--------|----------|------|------------|
| CSV | Excel, universal | Medium | Medium |
| JSON | Web apps, APIs | Large | Medium |
| Pickle | Python analysis | Small | Fast |
| MAT | MATLAB analysis | Medium | Medium |

---

## Visualization

### Graph Generation Control

```yaml
auto_generate_graphs: true              # Auto-generate after race
graph_formats: ["png", "pdf"]           # Output formats
```

**Supported Formats**:
- `"png"`: Web, presentations
- `"pdf"`: Publications, printing
- `"svg"`: Vector graphics, editing
- `"html"`: Interactive plots

### Plot Appearance

```yaml
plot_figsize: [12.0, 8.0]              # Size [width, height] in inches
plot_dpi: 300                           # Resolution (higher = better quality)
plot_style: "seaborn"                   # Matplotlib style
plot_color_scheme: "viridis"            # Color scheme
```

**Available Styles**:
- `"seaborn"`: Modern, clean (recommended)
- `"ggplot"`: R-style plots
- `"classic"`: Traditional matplotlib

**Color Schemes**:
- `"viridis"`: Perceptually uniform
- `"plasma"`: High contrast
- `"coolwarm"`: Diverging colors

### Plot Types

```yaml
# Standard plots
generate_trajectory_plots: true         # 2D/3D trajectories
generate_xyz_plots: true                # Position over time
generate_rpy_plots: true                # Orientation over time
generate_speed_plots: true              # Velocity/acceleration
generate_error_plots: true              # APE/RPE errors
generate_metrics_plots: true            # Performance metrics

# Advanced plots
generate_error_mapped_plots: true       # Error heatmap on trajectory
generate_violin_plots: true             # Error distributions
generate_3d_vector_plots: true          # 3D with velocity vectors
```

---

## Performance Monitoring

### Monitoring Control

```yaml
enable_computational_monitoring: false  # CPU/memory monitoring
enable_performance_logging: true        # Log metrics periodically
```

### Monitoring Parameters

```yaml
cpu_monitoring_interval: 1.0            # Monitoring interval (s)
performance_log_interval: 10.0          # Logging interval (s)
monitoring_window_size: 100             # Samples for statistics
```

### Performance Targets

```yaml
target_control_frequency_hz: 20.0       # Expected frequency (Hz)
max_acceptable_cpu_usage: 80.0          # Max CPU (%)
max_acceptable_memory_mb: 500.0         # Max memory (MB)
max_acceptable_latency_ms: 100.0        # Max latency (ms)
```

---

## Launch Parameters

Override configuration at launch time:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=my_controller \
    required_laps:=10 \
    race_mode:=lap_complete
```

**Available Launch Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `race_mode` | string | Race ending mode |
| `controller_name` | string | Controller identifier |
| `required_laps` | int | Number of laps |
| `enable_trajectory_evaluation` | bool | Enable EVO analysis |
| `enable_computational_monitoring` | bool | Enable performance monitoring |
| `auto_generate_graphs` | bool | Generate visualizations |
| `enable_smart_controller_detection` | bool | Auto-detect controller |
| `auto_shutdown_on_race_complete` | bool | Auto-shutdown after race |
| `shutdown_delay_seconds` | float | Delay before shutdown |

---

## Configuration Examples

### Example 1: Basic Racing

```yaml
race_monitor:
  ros__parameters:
    start_line_p1: [0.0, -2.0]
    start_line_p2: [0.0, 2.0]
    required_laps: 7
    race_ending_mode: "lap_complete"
    enable_trajectory_evaluation: true
    auto_generate_graphs: true
```

### Example 2: Research & Development

```yaml
race_monitor:
  ros__parameters:
    start_line_p1: [0.0, -2.0]
    start_line_p2: [0.0, 2.0]
    required_laps: 10
    race_ending_mode: "lap_complete"
    log_level: "debug"
    
    # Comprehensive analysis
    enable_trajectory_evaluation: true
    enable_advanced_metrics: true
    calculate_all_statistics: true
    enable_geometric_analysis: true
    
    # Reference comparison
    reference_trajectory_file: "optimal_line.csv"
    
    # Detailed output
    save_trajectories: true
    save_filtered_trajectories: true
    save_detailed_statistics: true
    output_formats: ["csv", "json", "pickle", "mat"]
    
    # Visualization
    auto_generate_graphs: true
    graph_formats: ["png", "pdf"]
    generate_error_mapped_plots: true
    generate_3d_vector_plots: true
```

### Example 3: Safety Testing

```yaml
race_monitor:
  ros__parameters:
    start_line_p1: [0.0, -2.0]
    start_line_p2: [0.0, 2.0]
    race_ending_mode: "crash"
    
    crash_detection:
      enable_crash_detection: true
      max_stationary_time: 3.0
      min_velocity_threshold: 0.05
      enable_collision_detection: true
      collision_velocity_threshold: 1.5
    
    save_trajectories: true
    auto_generate_graphs: true
```

### Example 4: Performance Benchmarking

```yaml
race_monitor:
  ros__parameters:
    start_line_p1: [0.0, -2.0]
    start_line_p2: [0.0, 2.0]
    required_laps: 20
    
    # Performance monitoring
    enable_computational_monitoring: true
    target_control_frequency_hz: 20.0
    max_acceptable_latency_ms: 50.0
    max_acceptable_cpu_usage: 70.0
    
    # Evaluation
    enable_trajectory_evaluation: true
    evaluation_interval_laps: 5
    
    # Efficiency analysis
    evaluate_efficiency: true
    evaluate_consistency: true
    evaluate_stability: true
```

---

## Dynamic Reconfiguration

Some parameters can be changed at runtime:

```bash
# Check current value
ros2 param get /race_monitor race_ending_mode

# Set new value (if supported)
ros2 param set /race_monitor log_level "debug"

# List all parameters
ros2 param list /race_monitor
```

**Note**: Most parameters require node restart to take effect.

---

## Best Practices

### For Racing
- Use `log_level: "normal"` or `"minimal"`
- Enable `auto_generate_graphs: true`
- Set appropriate `debounce_time` (2.0-3.0s)
- Use `race_ending_mode: "lap_complete"`

### For Development
- Use `log_level: "debug"` or `"verbose"`
- Enable all analysis features
- Save all data formats
- Use `race_ending_mode: "manual"`

### For Research
- Provide reference trajectory
- Enable comprehensive metrics
- Save detailed statistics
- Generate all plot types
- Use multiple output formats

---

## Next Steps

- **Installation**: See [INSTALLATION.md](INSTALLATION.md)
- **Usage Guide**: See [USAGE.md](USAGE.md)
- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)

---

**Need Help?**
- Configuration examples: `config/race_monitor.yaml`
- Parameter audit: [PARAMETER_AUDIT_REPORT.md](../PARAMETER_AUDIT_REPORT.md)
- Issues: https://github.com/GIU-F1Tenth/race_monitor/issues
