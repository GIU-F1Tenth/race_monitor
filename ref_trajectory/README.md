# Reference Trajectory Directory

This directory contains reference trajectory files used by the Race Monitor for trajectory evaluation and analysis.

## Usage

**Reference trajectories are optional!** Leave `reference_trajectory_file` empty to skip reference-based evaluation. The Race Monitor will still generate trajectory plots and metrics.

### To Use a Reference Trajectory:

Place your reference trajectory file in this directory and configure it in `race_monitor.yaml`:

```yaml
reference_trajectory_file: "your_trajectory.csv"  # Just the filename
reference_trajectory_format: "csv"  # or "tum" or "kitti"
```

The Race Monitor will automatically look for the file in this directory and use it for:
- Error-mapped trajectory visualization
- APE/RPE statistical comparison plots
- Lap-by-lap deviation analysis

### To Skip Reference Trajectory:

Simply leave it empty:

```yaml
reference_trajectory_file: ""  # No reference trajectory
```

Graphs will not show reference trajectory lines or error comparisons.

## Supported Formats

### 1. CSV Format (`.csv`)

A simple CSV file with pose information:

```csv
x,y,z,qx,qy,qz,qw
0.0,0.0,0.0,0.0,0.0,0.0,1.0
1.0,0.5,0.0,0.0,0.0,0.1,0.995
...
```

**Columns:**
- `x, y, z`: Position in meters
- `qx, qy, qz, qw`: Orientation as quaternion

Optional columns:
- `timestamp`: Time in seconds (optional)
- `vx, vy, vz`: Linear velocity (optional)

### 2. TUM Format (`.txt` or `.tum`)

TUM RGB-D SLAM dataset format:

```
timestamp x y z qx qy qz qw
1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1.1 0.1 0.0 0.0 0.0 0.0 0.0 1.0
...
```

**Format:** Space-separated values
- Column 1: Timestamp (seconds)
- Columns 2-4: Position (x, y, z)
- Columns 5-8: Orientation quaternion (qx, qy, qz, qw)

### 3. KITTI Format (`.txt` or `.kitti`)

KITTI odometry benchmark format (3x4 transformation matrices):

```
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
...
```

**Format:** Space-separated 12 values representing a 3x4 transformation matrix
- First 9 values: 3x3 rotation matrix (row-major)
- Last 3 values: translation vector (tx, ty, tz)

## Example Files

Example reference trajectories for common tracks:

- `levine_loop.csv` - Levine building loop
- `head_to_head.csv` - Head-to-head track
- `porto.csv` - Porto track

## Creating Your Own Reference Trajectory

### From ROS2 Bag

1. Record your trajectory:
   ```bash
   ros2 bag record /odom -o my_reference
   ```

2. Convert to CSV using the Race Monitor's trajectory export feature or use `evo_traj`:
   ```bash
   evo_traj bag my_reference.bag /odom --save_as_tum
   ```

### From Race Monitor Output

After a successful run, Race Monitor saves trajectories in the `data/` directory. You can copy any `actual_trajectory_*.csv` file here and use it as a reference.

## Path Resolution

The Race Monitor supports flexible path configuration:

| Configuration | Resolved Path |
|--------------|---------------|
| `"ref_trajectory.csv"` | `<package>/ref_trajectory/ref_trajectory.csv` |
| `"custom/path.csv"` | `<package>/custom/path.csv` |
| `"/absolute/path.csv"` | `/absolute/path.csv` (used as-is) |

## Notes

- Files in this directory are installed with the ROS2 package
- Large trajectory files (>10MB) should be excluded from version control
- The `.gitkeep` file ensures this directory is tracked by Git even when empty
- Reference trajectories are optional - leave `reference_trajectory_file` empty to skip evaluation

## Troubleshooting

### "Reference trajectory file not found"

Make sure:
1. File exists in this directory
2. Filename in config matches exactly (case-sensitive)
3. File has correct format and readable permissions

### "Unsupported reference trajectory format"

Check that `reference_trajectory_format` in config is one of: `csv`, `tum`, or `kitti`

### File parsing errors

Verify your file follows the correct format specification above. Common issues:
- CSV: Missing header row or incorrect column names
- TUM: Non-numeric values or incorrect number of columns
- KITTI: Incorrect matrix format or non-numeric values
