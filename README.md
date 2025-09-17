# 🏎️ Race Monitor - Advanced F1Tenth Performance Analysis

<div align="center">

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue?style=for-the-badge&logo=ros)](https://docs.ros.org/en/humble/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![EVO Integration](https://img.shields.io/badge/EVO-Integrated-orange?style=for-the-badge)](https://github.com/MichaelGrupp/evo)
[![MIT License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

**Professional-grade trajectory analysis and performance monitoring for autonomous racing 🏁**

*Transform your F1Tenth racing performance with real-time analysis, computational monitoring, and world-class trajectory evaluation powered by EVO*

</div>

---

## What is Race Monitor?

Race Monitor is a **comprehensive performance analysis system** for F1Tenth autonomous racing. It provides real-time lap timing, trajectory evaluation, computational performance monitoring, and advanced statistical analysis to help you optimize your racing algorithms.

**🎯 Perfect for:**
- **Racing Teams** - Optimize lap times and racing performance
- **Researchers** - Collect publication-ready trajectory data  
- **Algorithm Developers** - Debug and improve control algorithms
- **Students** - Learn trajectory analysis and performance optimization

---

## ✨ Key Features

### 🏁 **Race Management**
-  Real-time lap timing and race monitoring
-  Configurable start/finish line detection
-  Multi-lap race sessions with debouncing
-  Live race status and progress tracking

### 📊 **Trajectory Analysis (Powered by EVO)**
-  World-class trajectory evaluation using EVO library
-  40+ performance metrics per lap
-  TUM format compatibility for research
-  Smoothness, consistency, and efficiency analysis
-  Publication-ready statistical reports

### ⚡ **Computational Performance**
-  Real-time control loop latency monitoring
-  CPU and memory usage tracking
-  Processing efficiency scoring
-  Performance threshold alerting

### 📈 **Advanced Analytics**
-  Controller comparison capabilities
-  Multi-experiment analysis
-  Automated graph generation
-  Research data organization

### 🔬 **Research Tools**
-  Complete EVO library integration
-  Export data in multiple formats
-  Structured data organization
-  Comprehensive documentation

---

## 🛠️ Installation & Setup

### Prerequisites
- **ROS2 Humble** or later
- **Python 3.8+**
- **Colcon build system**

### Quick Install
```bash
# 1. Install Python dependencies
pip install numpy matplotlib pandas scipy seaborn

# 2. Clone and build
cd /path/to/your/workspace
git clone --recursive https://github.com/GIU-F1Tenth/race_monitor.git
colcon build --packages-select race_monitor

# 3. Source and go!
source install/setup.bash
```

> 💡 **Note**: The EVO library is included as a git submodule - no separate installation needed!

---

## Quick Start Guide

### Basic Racing Session
```bash
# Launch with default configuration
ros2 launch race_monitor race_monitor.launch.py

# Or with your controller name
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=my_awesome_controller \
    experiment_id:=speed_test_001
```

### Performance Monitoring Mode
```bash
# Enable all monitoring features
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=adaptive_lqr \
    experiment_id:=performance_test \
    enable_trajectory_evaluation:=true \
    auto_generate_graphs:=true \
    enable_computational_monitoring:=true
```

### Professional Analysis Mode
```bash
# Full analysis with comparison capabilities
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=advanced_mpc \
    experiment_id:=comprehensive_analysis \
    enable_trajectory_evaluation:=true \
    auto_generate_graphs:=true \
    enable_computational_monitoring:=true \
    evaluation_interval_laps:=1 \
    save_detailed_statistics:=true
```

---

## 🎛️ Configuration System

The main configuration file is `config/race_monitor.yaml`. Here's what you can customize:

### 🏁 Race Setup Parameters
```yaml
race_monitor:
  ros__parameters:
    # Start/finish line configuration
    start_line_p1: [0.0, -1.0]  # Bottom point [x, y]
    start_line_p2: [0.0, 1.0]   # Top point [x, y]
    required_laps: 5             # Number of laps for race completion
    debounce_time: 2.0           # Lap detection debounce (seconds)
```

### Controller Identification
```yaml
    # Controller identification
    controller_name: "my_controller"              # Name of controller being tested
    experiment_id: "exp_001"                      # Experiment identifier
    test_description: "Controller performance evaluation"
```

###  Analysis & Evaluation
```yaml
    # EVO Integration
    enable_trajectory_evaluation: true            # Enable EVO analysis
    evaluation_interval_laps: 1                   # Evaluate every N laps
    save_trajectories: true                       # Save trajectory files
    trajectory_output_directory: "trajectory_evaluation"
    
    # Metrics calculation
    evaluate_smoothness: true                     # Calculate trajectory smoothness
    evaluate_consistency: true                    # Calculate trajectory consistency
    evaluate_efficiency: true                     # Calculate path efficiency
    evaluate_aggressiveness: true                 # Calculate driving aggressiveness
```

### Visualization & Reporting
```yaml
    # Graph generation
    auto_generate_graphs: true                    # Auto-generate performance plots
    graph_output_directory: "trajectory_evaluation/graphs"
    graph_formats: ["png", "pdf"]                # Output formats
    save_detailed_statistics: true               # Export comprehensive data
```

### Computational Performance Monitoring
```yaml
    # Performance monitoring
    enable_computational_monitoring: true         # Monitor system performance
    control_command_topic: "/drive"               # Control topic to monitor
    control_command_type: "ackermann"             # "ackermann" or "twist"
    max_acceptable_latency_ms: 50.0              # Latency threshold
    target_control_frequency_hz: 50.0            # Target control frequency
```

---

## EVO Trajectory Analysis Suite

Race Monitor includes the complete **EVO (Evolution of Trajectory Evaluation)** library as a git submodule, providing world-class trajectory analysis capabilities!

### What is EVO?

EVO is a Python package for evaluating odometry and SLAM trajectories. It provides:
- **APE**: Absolute Pose Error metrics  
- **RPE**: Relative Pose Error metrics
- **Advanced plotting** and visualization tools
- **Statistical analysis** and report generation
- **Multiple trajectory format** support

### EVO Components Available

#### 1. **APE (Absolute Pose Error) Analysis**
```bash
# Basic APE analysis
evo_ape tum reference_trajectory.txt controller_trajectory.txt

# APE with custom settings
evo_ape tum reference_trajectory.txt controller_trajectory.txt \
    --align --correct_scale --plot --save_plot graphs/ape_plot.pdf

# APE with statistical output
evo_ape tum reference_trajectory.txt controller_trajectory.txt \
    --align --save_results results/ape_results.zip --verbose
```

#### 2. **📐 RPE (Relative Pose Error) Analysis**  
```bash
# Basic RPE analysis
evo_rpe tum reference_trajectory.txt controller_trajectory.txt

# RPE with pose relation (translation only)
evo_rpe tum reference_trajectory.txt controller_trajectory.txt \
    --pose_relation trans_part --plot --save_plot graphs/rpe_plot.pdf

# RPE with custom delta (1.0 meters)
evo_rpe tum reference_trajectory.txt controller_trajectory.txt \
    --delta 1.0 --delta_unit m --save_results results/rpe_results.zip
```

#### 3. **📊 Trajectory Visualization & Plotting**
```bash
# Plot trajectory comparison
evo_traj tum controller_trajectory.txt reference_trajectory.txt \
    --plot --save_plot graphs/trajectory_comparison.pdf

# 3D trajectory plot
evo_traj tum controller_trajectory.txt \
    --plot_mode xyz --save_plot graphs/3d_trajectory.pdf

# Plot with custom colors and labels
evo_traj tum controller_trajectory.txt reference_trajectory.txt \
    --plot --plot_colormap hot --save_plot graphs/colored_trajectories.pdf
```

#### 4. **🔧 Configuration Management**
```bash
# Show current EVO configuration
evo_config show

# Set global plot settings
evo_config set plot_export_format pdf
evo_config set plot_split True
evo_config set plot_usetex False

# Reset to defaults
evo_config reset
```

#### 5. **📈 Results Management & Comparison**
```bash
# Compare multiple result files
evo_res results/ape_controller1.zip results/ape_controller2.zip \
    --save_table results/comparison_table.csv

# Generate comparison plots
evo_res results/ape_*.zip --plot --save_plot graphs/ape_comparison.pdf

# Statistical summary
evo_res results/rpe_*.zip --statistics
```

#### 6. **🎨 Advanced Plotting Features**
```bash
# Create publication-ready plots
evo_fig trajectory_data.json --plot_collection \
    --save_plot graphs/publication_plot.pdf --figsize 12 8

# Interactive plots (if in Jupyter environment)
evo_ipython  # Launches IPython with EVO tools loaded
```

### 🛠️ Practical EVO Workflows

#### **Workflow 1: Single Controller Analysis**
```bash
# Navigate to your data directory
cd trajectory_evaluation/research_data/your_controller/exp_001/

# 1. Analyze absolute pose error
evo_ape tum horizon_reference_trajectory.txt lap_001_trajectory.txt \
    --align --plot --save_plot ../graphs/ape_analysis.pdf \
    --save_results ../results/ape_results.zip

# 2. Analyze relative pose error  
evo_rpe tum horizon_reference_trajectory.txt lap_001_trajectory.txt \
    --delta 1.0 --plot --save_plot ../graphs/rpe_analysis.pdf \
    --save_results ../results/rpe_results.zip

# 3. Create trajectory visualization
evo_traj tum lap_001_trajectory.txt horizon_reference_trajectory.txt \
    --plot --save_plot ../graphs/trajectory_comparison.pdf
```

#### **Workflow 2: Multi-Controller Comparison**
```bash
# Compare multiple controllers
cd trajectory_evaluation/

# Analyze each controller
for controller in mpc lqr pid; do
    evo_ape tum reference.txt research_data/${controller}/exp_001/lap_001_trajectory.txt \
        --align --save_results results/ape_${controller}.zip
done

# Generate comparison report
evo_res results/ape_*.zip --plot --save_plot graphs/controller_comparison.pdf \
    --save_table results/comparison_summary.csv
```

#### **Workflow 3: Statistical Analysis**
```bash
# Deep statistical analysis
evo_ape tum reference_trajectory.txt controller_trajectory.txt \
    --align --correct_scale \
    --save_results detailed_results.zip \
    --plot_mode xyz --save_plot detailed_plot.pdf

# Extract specific statistics
evo_res detailed_results.zip --statistics --save_table stats_summary.csv
```

### 📁 EVO File Formats Supported

- **TUM**: `timestamp x y z qx qy qz qw` (used by Race Monitor)
- **KITTI**: KITTI odometry format
- **EuRoC**: EuRoC MAV dataset format  
- **Bag**: ROS bag files (with trajectory extraction)

### Integration with Race Monitor

Race Monitor automatically:
1. **Saves trajectories** in TUM format for EVO compatibility
2. **Organizes data** in structured directories for easy EVO analysis
3. **Provides reference trajectories** for comparative analysis
4. **Generates basic plots** using EVO's plotting capabilities

### Pro Tips for EVO Usage

#### **🚀 Performance Optimization**
```bash
# For large datasets, use alignment algorithms efficiently
evo_ape tum ref.txt traj.txt --align --align_origin

# Reduce memory usage for long trajectories  
evo_ape tum ref.txt traj.txt --t_max 100.0  # Limit to first 100 seconds
```

#### **Customization Options**
```bash
# Custom plot styling
evo_config set plot_linewidth 2.0
evo_config set plot_fontsize 14
evo_config set plot_figsize "[12, 8]"

# Export settings for different use cases
evo_config set plot_export_format svg  # For web
evo_config set plot_export_format pdf  # For publications
```

#### **Batch Processing**
```bash
# Process multiple experiments
for exp in exp_001 exp_002 exp_003; do
    evo_ape tum reference.txt research_data/controller/${exp}/lap_001_trajectory.txt \
        --align --save_results results/ape_${exp}.zip
done

# Combine all results
evo_res results/ape_exp*.zip --merge --save_results combined_analysis.zip
```

### 🔗 EVO Resources

- **Official Documentation**: [GitHub - MichaelGrupp/evo](https://github.com/MichaelGrupp/evo)
- **Jupyter Notebooks**: Check `evo/notebooks/` for interactive examples
- **API Documentation**: `evo/doc/` contains detailed API references

---

## Real-Time Performance Topics

### 🏁 Race Monitoring Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/race_monitor/lap_count` | `std_msgs/Int32` | Current lap number |
| `/race_monitor/lap_time` | `std_msgs/Float32` | Last lap time (seconds) |
| `/race_monitor/race_running` | `std_msgs/Bool` | Race status flag |
| `/race_monitor/race_status` | `std_msgs/String` | Detailed race status |

### Performance Monitoring Topics  
| Topic | Type | Description |
|-------|------|-------------|
| `/race_monitor/control_loop_latency` | `std_msgs/Float32` | Control latency (ms) |
| `/race_monitor/cpu_usage` | `std_msgs/Float32` | CPU usage (%) |
| `/race_monitor/memory_usage` | `std_msgs/Float32` | Memory usage (MB) |
| `/race_monitor/processing_efficiency` | `std_msgs/Float32` | Efficiency score (0-1) |
| `/race_monitor/performance_stats` | `std_msgs/String` | Comprehensive stats (JSON) |

### Analysis Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/race_monitor/trajectory_metrics` | `std_msgs/String` | EVO metrics (JSON) |
| `/race_monitor/smoothness` | `std_msgs/Float32` | Trajectory smoothness |
| `/race_monitor/consistency` | `std_msgs/Float32` | Trajectory consistency |

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `car_state/odom` | `nav_msgs/Odometry` | Vehicle odometry |
| `/clicked_point` | `geometry_msgs/PointStamped` | Start/finish line setup |
| `{control_command_topic}` | `AckermannDriveStamped/Twist` | Control commands monitoring |

---

## 📂 Data Output Structure

The system automatically generates organized data in `trajectory_evaluation/`:

```
trajectory_evaluation/
├── evaluation_summary.csv          # High-level summary
├── horizon_reference_trajectory.txt # Reference trajectory  
├── graphs/                         # All visualizations
│   ├── trajectory_plots/
│   ├── performance_charts/
│   └── comparison_results/
├── performance_data/               # Raw performance logs
└── research_data/                  # Detailed analysis
    └── {controller_name}/
        └── {experiment_id}/
            ├── lap_001_trajectory.txt
            ├── detailed_metrics.json
            ├── performance_stats.csv
            └── summary_statistics.json
```

### 📊 Metrics Calculated (40+ Performance Indicators)

- **Basic Performance**: Path length, duration, average speed
- **Velocity Analysis**: Mean/std/max velocity, consistency scores  
- **Acceleration Metrics**: Mean/std/max acceleration, jerk analysis
- **Steering Analysis**: Angular velocity, aggressiveness indicators
- **Geometric Analysis**: Curvature analysis, path efficiency
- **Statistical Breakdown**: Complete statistical analysis per metric

---

## Advanced Usage Examples

### Professional Racing Analysis
```bash
# Complete analysis session
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=championship_mpc \
    experiment_id:=qualifying_session_$(date +%Y%m%d_%H%M%S) \
    required_laps:=20 \
    evaluation_interval_laps:=1 \
    enable_trajectory_evaluation:=true \
    enable_computational_monitoring:=true \
    auto_generate_graphs:=true \
    save_detailed_statistics:=true \
    max_acceptable_latency_ms:=25.0 \
    target_control_frequency_hz:=100.0
```

### Research & Development Mode  
```bash
# For algorithm development
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=experimental_rl_agent \
    experiment_id:=hyperparameter_sweep_v2_3 \
    required_laps:=50 \
    evaluation_interval_laps:=5 \
    enable_trajectory_evaluation:=true \
    enable_computational_monitoring:=true \
    trajectory_output_directory:=research_output \
    test_description:="RL agent with modified reward function"
```

### ⚡ Real-Time Performance Monitoring
```bash
# Monitor your controller's performance live
ros2 topic echo /race_monitor/performance_stats --once | jq .
ros2 topic hz /race_monitor/control_loop_latency
watch -n 1 "ros2 topic echo /race_monitor/cpu_usage --once"
```

---

## 🛠️ Troubleshooting Guide

### ❓ Common Issues

**Q: "No odometry data received"**  
A: Check that your odometry topic is publishing to `car_state/odom`

**Q: "EVO analysis failed"**  
A: Ensure trajectory files have sufficient data points (>10 poses)

**Q: "High control latency detected"**  
A: Check system load and optimize your controller's computational efficiency

**Q: "Graphs not generating"**  
A: Verify matplotlib backend and file permissions in output directory

### 🔍 Debug Mode
```bash
# Enable verbose logging
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=debug_controller \
    log_level:=DEBUG
```

---

## Documentation & Resources

- **[Usage Guide](USAGE_GUIDE.md)** - Comprehensive usage documentation
- **[Performance Monitoring](docs/COMPUTATIONAL_MONITORING.md)** - Advanced performance analysis
- **[Multi-Topic Monitoring](docs/multi_topic_monitoring.md)** - Multi-controller setup
- **[Test Examples](test_computational_monitoring.py)** - Testing and validation

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Ready to revolutionize your racing performance analysis? Get started now! 🏁**

*Built with ❤️ by the F1Tenth Racing Community*

Mohammed Abdelazim - mohammed@azab.io

</div>