# EVO Integration Guide

Complete guide to using the EVO trajectory evaluation library with Race Monitor.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Trajectory Formats](#trajectory-formats)
- [Basic Analysis](#basic-analysis)
- [Advanced Analysis](#advanced-analysis)
- [EVO Tools](#evo-tools)
- [Workflows](#research-workflows)
- [Python API](#python-api)
- [Examples](#examples)

---

## Overview

Race Monitor includes the **EVO (Python package for the evaluation of odometry and SLAM)** library as a submodule, providing research-grade trajectory analysis capabilities.

### What is EVO?

EVO is a Python package for evaluating odometry and SLAM algorithms. It provides:
- **APE (Absolute Pose Error)**: Global trajectory accuracy
- **RPE (Relative Pose Error)**: Local consistency and drift
- **Trajectory visualization**: 2D/3D plotting with various options
- **Result analysis**: Statistical comparisons and exports

### EVO in Race Monitor

Race Monitor automatically:
- Exports trajectories in EVO-compatible formats (TUM, KITTI, EuRoC)
- Computes APE/RPE when reference trajectory is provided
- Integrates EVO metrics into race evaluation
- Generates EVO-style plots automatically

---

## Installation

EVO is installed as part of Race Monitor installation:

```bash
# Navigate to EVO directory
cd ~/ros2_ws/src/race_monitor/evo

# Install in development mode
pip install -e .

# Verify installation
python3 -c "import evo; print(evo.__version__)"
```

Expected output: `1.x.x` (version number)

### Manual Installation

If EVO is not installed:

```bash
# Clone as submodule (if not done already)
cd ~/ros2_ws/src/race_monitor
git submodule update --init --recursive

# Install EVO
cd evo
pip install -e .
```

---

## Trajectory Formats

Race Monitor exports trajectories in multiple EVO-compatible formats:

### TUM Format (Recommended)

Standard format for SLAM/odometry evaluation:

```
# timestamp x y z qx qy qz qw
1729869839.220658 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1729869839.320658 0.1 0.0 0.0 0.0 0.0 0.0 1.0
```

**Location**: `trajectories/tum/lap_00X_trajectory.txt`

### KITTI Format

KITTI odometry benchmark format (3x4 transformation matrices):

```
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
```

**Location**: `trajectories/kitti/lap_00X_trajectory.txt`

### EuRoC Format

EuRoC MAV dataset CSV format:

```
# timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []
1729869839220658, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
```

**Location**: `trajectories/euroc/lap_00X_trajectory.csv`

### Format Configuration

Set in `race_monitor.yaml`:

```yaml
trajectory_format: "tum"  # Options: "tum", "kitti", "euroc"
```

---

## Basic Analysis

### Absolute Pose Error (APE)

APE measures the absolute deviation between estimated and reference trajectories:

```bash
# Navigate to experiment trajectories
cd data/lqr_controller_node/exp_20250125_153000/trajectories/tum

# Basic APE analysis
evo_ape tum reference_trajectory.txt lap_001_trajectory.txt

# APE with alignment and visualization
evo_ape tum reference_trajectory.txt lap_001_trajectory.txt \
    --align --correct_scale \
    --plot --save_plot ../../graphs/ape_lap_001.pdf
```

**Output Example**:
```
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max      4.518270
      mean      2.341567
    median      2.289456
       min      0.123456
      rmse      2.456789
       sse    12345.678
       std      0.987654
```

### Relative Pose Error (RPE)

RPE measures drift over specified distances or time intervals:

```bash
# RPE with 1.0 meter delta
evo_rpe tum reference_trajectory.txt lap_001_trajectory.txt \
    --delta 1.0 --delta_unit m \
    --plot --save_plot ../../graphs/rpe_lap_001.pdf

# RPE translation only
evo_rpe tum reference_trajectory.txt lap_001_trajectory.txt \
    --pose_relation trans_part --plot
```

**Output Example**:
```
RPE w.r.t. translation part (m)
for delta = 1.0 (m) using consecutive pairs

       max      0.234567
      mean      0.123456
    median      0.098765
       min      0.012345
      rmse      0.145678
       sse    123.456789
       std      0.067890
```

### Trajectory Visualization

```bash
# Visualize single trajectory
evo_traj tum lap_001_trajectory.txt --plot

# Compare multiple trajectories
evo_traj tum lap_001_trajectory.txt lap_002_trajectory.txt \
    reference_trajectory.txt \
    --plot --save_plot ../../graphs/trajectory_comparison.pdf

# 3D visualization with time coloring
evo_traj tum lap_001_trajectory.txt \
    --plot_mode xyz --plot_colormap hot
```

---

## Advanced Analysis

### Save Results for Comparison

```bash
# Compute and save APE results
evo_ape tum reference.txt lap_001_trajectory.txt \
    --align --save_results ../../results/ape_lap_001.zip

# Compute and save RPE results
evo_rpe tum reference.txt lap_001_trajectory.txt \
    --delta 1.0 --save_results ../../results/rpe_lap_001.zip
```

### Multi-Lap Comparison

```bash
# Analyze all laps
for lap in lap_00{1..7}_trajectory.txt; do
    evo_ape tum reference.txt $lap --align \
        --save_results ../../results/ape_${lap%.txt}.zip
done

# Generate comparison report
evo_res ../../results/ape_lap_*.zip \
    --plot --save_plot ../../graphs/all_laps_ape_comparison.pdf \
    --save_table ../../results/lap_comparison.csv
```

### Multi-Controller Comparison

```bash
# Analyze multiple controllers
cd data/

for controller in lqr mpc pid; do
    best_lap=$(find ${controller}/exp_*/trajectories/tum -name "lap_*.txt" | head -1)
    evo_ape tum reference.txt $best_lap --align \
        --save_results results/ape_${controller}.zip
done

# Generate comparison report
evo_res results/ape_*.zip \
    --plot --save_plot controller_comparison.pdf \
    --save_table controller_comparison.csv
```

---

## EVO Tools

### evo_ape

Absolute Pose Error evaluation:

```bash
# Basic usage
evo_ape {tum,kitti,euroc} <ref_file> <est_file> [options]

# Common options:
--align                # Align trajectories (SE(3) Umeyama)
--correct_scale        # Correct scale difference
--pose_relation {full_transformation,translation_part,rotation_part}
--plot                 # Show plot
--save_plot <file>     # Save plot to file
--save_results <file>  # Save results (ZIP format)
```

### evo_rpe

Relative Pose Error evaluation:

```bash
# Basic usage
evo_rpe {tum,kitti,euroc} <ref_file> <est_file> [options]

# Common options:
--delta <value>        # Delta for relative poses
--delta_unit {f,d,r,m} # Unit: frames, degrees, radians, meters
--all_pairs            # All pairs instead of consecutive
--pose_relation {full_transformation,translation_part,rotation_part}
--plot                 # Show plot
--save_plot <file>     # Save plot to file
--save_results <file>  # Save results (ZIP format)
```

### evo_traj

Trajectory visualization and manipulation:

```bash
# Basic usage
evo_traj {tum,kitti,euroc} <file1> [<file2> ...] [options]

# Common options:
--plot                 # Show plot
--plot_mode {xy,xz,yz,xyz} # Plot mode
--plot_colormap <name> # Color map (viridis, plasma, etc.)
--save_plot <file>     # Save plot to file
--ref <file>           # Reference trajectory
--align                # Align to reference
```

### evo_res

Result comparison and analysis:

```bash
# Basic usage
evo_res <result1.zip> [<result2.zip> ...] [options]

# Common options:
--plot                 # Show plot
--save_plot <file>     # Save plot to file
--save_table <file>    # Save comparison table (CSV)
--use_filenames        # Use filenames as labels
```

### evo_config

EVO configuration management:

```bash
# View current configuration
evo_config show

# Set configuration value
evo_config set <key> <value>

# Reset to defaults
evo_config reset

# Common settings:
evo_config set plot_export_format pdf
evo_config set plot_figsize "[12, 8]"
evo_config set plot_linewidth 2.0
evo_config set plot_fontsize 12
```

---

## Research Workflows

### Workflow 1: Single Lap Analysis

Complete analysis of a single lap:

```bash
cd data/lqr_controller_node/exp_20250125_153000/trajectories/tum

# 1. Absolute pose error
evo_ape tum reference.txt lap_001_trajectory.txt \
    --align --save_results ../../results/ape_lap_001.zip

# 2. Relative pose error
evo_rpe tum reference.txt lap_001_trajectory.txt \
    --delta 1.0 --save_results ../../results/rpe_lap_001.zip

# 3. Visualization
evo_traj tum lap_001_trajectory.txt reference.txt \
    --plot --save_plot ../../graphs/lap_001_comparison.pdf
```

### Workflow 2: Multi-Lap Comparison

Compare all laps from an experiment:

```bash
cd data/lqr_controller_node/exp_20250125_153000/trajectories/tum

# Compare all trajectories
evo_traj tum lap_00*.txt --plot --ref reference.txt \
    --save_plot ../../graphs/all_laps_comparison.pdf

# Statistical analysis across laps
for lap in lap_00{1..7}_trajectory.txt; do
    evo_ape tum reference.txt $lap --align \
        --save_results ../../results/ape_${lap%.txt}.zip
done

# Aggregate results
evo_res ../../results/ape_lap_*.zip \
    --plot --save_plot ../../graphs/lap_ape_comparison.pdf \
    --save_table ../../results/lap_comparison.csv
```

### Workflow 3: Controller Benchmarking

Compare different controllers on same track:

```bash
cd data/

# Analyze each controller's best lap
for ctrl in controller_a controller_b controller_c; do
    exp=$(ls -td ${ctrl}/exp_* | head -1)
    best_lap="${exp}/trajectories/tum/lap_001_trajectory.txt"
    evo_ape tum reference.txt $best_lap --align \
        --save_results results/ape_${ctrl}.zip
done

# Generate benchmark report
evo_res results/ape_controller*.zip \
    --plot --save_plot benchmark_comparison.pdf \
    --save_table benchmark_results.csv --use_filenames
```

### Workflow 4: Batch Processing

Process all experiments:

```bash
#!/bin/bash
# analyze_all_experiments.sh

EXPERIMENT_DIR="data/lqr_controller_node"
REFERENCE="ref_trajectory/reference_track.txt"

for exp in ${EXPERIMENT_DIR}/exp_*/; do
    echo "Processing ${exp}"
    
    cd ${exp}trajectories/tum/
    
    for lap in lap_*.txt; do
        lap_name=$(basename $lap .txt)
        evo_ape tum $REFERENCE $lap --align \
            --save_results ../../results/ape_${lap_name}.zip
    done
    
    # Generate summary
    evo_res ../../results/ape_*.zip \
        --save_table ../../results/ape_summary.csv
    
    cd -
done
```

---

## Python API

### Loading Trajectories

```python
from evo.core import trajectory
from evo.tools import file_interface

# Load TUM format trajectory
ref_traj = file_interface.read_tum_trajectory_file("reference.txt")
est_traj = file_interface.read_tum_trajectory_file("lap_001_trajectory.txt")

# Access trajectory data
print(f"Reference trajectory length: {len(ref_traj.timestamps)}")
print(f"Positions shape: {ref_traj.positions_xyz.shape}")
```

### Computing APE

```python
from evo.core import metrics
from evo.core.trajectory import PoseRelation

# Create APE metric
ape_metric = metrics.APE(PoseRelation.translation_part)

# Process data (with alignment)
ape_metric.process_data((ref_traj, est_traj), align=True)

# Get statistics
stats = ape_metric.get_all_statistics()
print(f"APE RMSE: {stats['rmse']:.4f} m")
print(f"APE Mean: {stats['mean']:.4f} m")
print(f"APE Std:  {stats['std']:.4f} m")
```

### Computing RPE

```python
# Create RPE metric (1.0 meter delta)
rpe_metric = metrics.RPE(
    PoseRelation.translation_part,
    delta=1.0,
    delta_unit=metrics.Unit.meters
)

# Process data
rpe_metric.process_data((ref_traj, est_traj))

# Get statistics
stats = rpe_metric.get_all_statistics()
print(f"RPE RMSE: {stats['rmse']:.4f} m")
print(f"RPE Mean: {stats['mean']:.4f} m")
```

### Visualization

```python
import matplotlib.pyplot as plt
from evo.tools import plot

# Plot trajectory
fig, ax = plt.subplots(figsize=(10, 10))
plot.trajectories(
    ax, [ref_traj, est_traj],
    names=['Reference', 'Estimated'],
    style='seaborn'
)
plt.savefig('trajectory_comparison.pdf')

# Plot APE
fig, ax = plt.subplots(figsize=(12, 6))
ape_metric.plot(ax, name='Lap 1 APE')
plt.savefig('ape_analysis.pdf')
```

### Custom Analysis

```python
import numpy as np
from evo.core import sync
from evo.tools import plot

# Load trajectories
ref = file_interface.read_tum_trajectory_file("reference.txt")
est = file_interface.read_tum_trajectory_file("lap_001_trajectory.txt")

# Synchronize trajectories
max_diff = 0.01  # 10ms
ref_sync, est_sync = sync.associate_trajectories(ref, est, max_diff)

# Compute custom metrics
position_errors = np.linalg.norm(
    ref_sync.positions_xyz - est_sync.positions_xyz,
    axis=1
)

print(f"Mean position error: {np.mean(position_errors):.4f} m")
print(f"Max position error: {np.max(position_errors):.4f} m")

# Plot custom metric
plt.figure(figsize=(12, 6))
plt.plot(est_sync.timestamps - est_sync.timestamps[0], position_errors)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)
plt.savefig('custom_error_plot.pdf')
```

---

## Examples

### Example 1: Quick APE Analysis

```bash
cd data/lqr_controller_node/exp_20250125_153000/trajectories/tum
evo_ape tum reference.txt lap_001_trajectory.txt --align --plot
```

### Example 2: Detailed RPE with Export

```bash
evo_rpe tum reference.txt lap_001_trajectory.txt \
    --delta 1.0 --delta_unit m \
    --plot --save_plot rpe_detailed.pdf \
    --save_results rpe_results.zip
```

### Example 3: Multi-Trajectory Visualization

```bash
evo_traj tum lap_00*.txt --plot \
    --ref reference.txt \
    --plot_mode xy \
    --save_plot all_laps_2d.pdf
```

### Example 4: Statistical Comparison

```bash
# Compute for all laps
for i in {1..7}; do
    evo_ape tum reference.txt lap_00${i}_trajectory.txt \
        --align --save_results ape_lap_${i}.zip
done

# Compare
evo_res ape_lap_*.zip \
    --save_table lap_statistics.csv \
    --save_plot lap_comparison.pdf
```

---

## Best Practices

### For Racing
- Use TUM format for compatibility
- Always align trajectories (`--align`)
- Save results for later comparison
- Use 1.0m delta for RPE

### For Research
- Document all analysis parameters
- Save both plots and numerical results
- Use consistent delta values for RPE
- Compare multiple controllers systematically

### For Publications
- Use high DPI for plots
- Export to PDF format
- Include statistical tables
- Document alignment methods

---

## Additional Resources

- **EVO Documentation**: https://github.com/MichaelGrupp/evo
- **EVO Wiki**: https://github.com/MichaelGrupp/evo/wiki
- **Tutorial Notebooks**: `evo/notebooks/`
- **Examples**: `evo/doc/examples/`

---

## Citation

If you use EVO in your research, please cite:

```bibtex
@software{grupp2017evo,
  author       = {Grupp, Michael},
  title        = {evo: Python package for the evaluation of odometry and SLAM},
  year         = {2017},
  howpublished = {\url{https://github.com/MichaelGrupp/evo}}
}
```

---

**Need Help?**
- Configuration: [CONFIGURATION.md](CONFIGURATION.md)
- Usage Guide: [USAGE.md](USAGE.md)
- API Reference: [API_REFERENCE.md](API_REFERENCE.md)
- Issues: https://github.com/GIU-F1Tenth/race_monitor/issues
