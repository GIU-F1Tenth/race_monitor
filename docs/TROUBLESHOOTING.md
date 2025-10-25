# Troubleshooting Guide

Solutions to common issues with Race Monitor.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Launch Issues](#launch-issues)
- [Runtime Issues](#runtime-issues)
- [Data Issues](#data-issues)
- [Performance Issues](#performance-issues)
- [Topic Issues](#topic-issues)
- [Visualization Issues](#visualization-issues)
- [EVO Integration Issues](#evo-integration-issues)
- [Debug Mode](#debug-mode)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: EVO Import Fails

**Symptom**:
```
ImportError: No module named 'evo'
ModuleNotFoundError: No module named 'evo'
```

**Solutions**:

1. **Add EVO to Python path**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/race_monitor/evo
   ```

2. **Make permanent** (add to `~/.bashrc`):
   ```bash
   echo 'export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/race_monitor/evo' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Reinstall EVO**:
   ```bash
   cd ~/ros2_ws/src/race_monitor/evo
   pip install -e .
   ```

4. **Verify installation**:
   ```bash
   python3 -c "import evo; print(evo.__version__)"
   ```

---

### Issue: Colcon Build Fails

**Symptom**:
```
Failed to find package 'race_monitor'
CMake Error: ...
```

**Solutions**:

1. **Clean build artifacts**:
   ```bash
   cd ~/ros2_ws
   rm -rf build install log
   ```

2. **Rebuild with verbose output**:
   ```bash
   colcon build --packages-select race_monitor --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
   ```

3. **Source workspace**:
   ```bash
   source install/setup.bash
   ```

4. **Check for missing dependencies**:
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

---

### Issue: Missing Python Dependencies

**Symptom**:
```
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'scipy'
```

**Solutions**:

1. **Install from requirements**:
   ```bash
   cd ~/ros2_ws/src/race_monitor
   pip install -r requirements.txt
   ```

2. **Install with constraints**:
   ```bash
   pip install -c constraints.txt -r requirements.txt
   ```

3. **Install individually**:
   ```bash
   pip install numpy scipy matplotlib pandas seaborn
   ```

---

### Issue: ROS2 Not Sourced

**Symptom**:
```
ros2: command not found
bash: ros2: command not found
```

**Solutions**:

1. **Source ROS2 installation**:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Source workspace**:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

3. **Add to ~/.bashrc**:
   ```bash
   echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
   echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
   source ~/.bashrc
   ```

---

### Issue: Submodule Not Initialized

**Symptom**:
- Empty `evo/` directory
- Missing EVO library files

**Solution**:
```bash
cd ~/ros2_ws/src/race_monitor
git submodule update --init --recursive
```

---

## Launch Issues

### Issue: Node Fails to Start

**Symptom**:
```
[ERROR] Failed to create node
[ERROR] Node 'race_monitor' could not be started
```

**Solutions**:

1. **Check if node is already running**:
   ```bash
   ros2 node list | grep race_monitor
   # If found, kill it first
   ```

2. **Verify package installation**:
   ```bash
   ros2 pkg list | grep race_monitor
   ```

3. **Check executables**:
   ```bash
   ros2 pkg executables race_monitor
   ```

4. **Check for errors in config file**:
   ```bash
   cat config/race_monitor.yaml
   # Look for syntax errors
   ```

---

### Issue: Configuration Not Loaded

**Symptom**:
- Node starts but uses default values
- Parameters not applied

**Solutions**:

1. **Verify config file location**:
   ```bash
   ls -la config/race_monitor.yaml
   ```

2. **Check YAML syntax**:
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('config/race_monitor.yaml'))"
   ```

3. **Launch with explicit config**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py \
       config_file:=/full/path/to/race_monitor.yaml
   ```

---

### Issue: Launch Arguments Not Working

**Symptom**:
- Parameters not overridden
- Default values used instead

**Solution**:

Use correct syntax:
```bash
# Correct
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=my_controller

# Incorrect
ros2 launch race_monitor race_monitor.launch.py \
    controller_name=my_controller  # Missing colon
```

---

## Runtime Issues

### Issue: No Odometry Data Received

**Symptom**:
```
[WARN] No odometry data received
[ERROR] Waiting for odometry...
```

**Solutions**:

1. **Verify odometry topic**:
   ```bash
   ros2 topic list | grep odom
   ros2 topic echo /car_state/odom --once
   ```

2. **Check topic rate**:
   ```bash
   ros2 topic hz /car_state/odom
   ```

3. **Remap if needed**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py \
       /car_state/odom:=/your_robot/odom
   ```

4. **Check message type**:
   ```bash
   ros2 topic type /car_state/odom
   # Should be: nav_msgs/msg/Odometry
   ```

---

### Issue: Lap Detection Not Working

**Symptom**:
- No lap count increments
- Start/finish line not detected

**Solutions**:

1. **Verify start line configuration**:
   ```bash
   ros2 param get /race_monitor start_line_p1
   ros2 param get /race_monitor start_line_p2
   ```

2. **Check vehicle position**:
   ```bash
   ros2 topic echo /car_state/odom --field pose.pose.position
   ```

3. **Visualize in RViz**:
   ```bash
   rviz2 -d config/race_monitor.rviz
   # Check if start line and vehicle are visible
   ```

4. **Enable debug logging**:
   ```yaml
   # In race_monitor.yaml
   log_level: "debug"
   ```

5. **Adjust debounce time**:
   ```yaml
   debounce_time: 3.0  # Increase if detecting too quickly
   ```

---

### Issue: Race Not Starting

**Symptom**:
- `race_running` stays `false`
- No lap detection

**Solutions**:

1. **Check if vehicle has crossed line**:
   - Drive vehicle across start/finish line

2. **Verify lap detection direction**:
   ```yaml
   lap_detection:
     expected_direction: "any"  # Allow both directions
   ```

3. **Check race mode**:
   ```bash
   ros2 param get /race_monitor race_ending_mode
   ```

---

### Issue: Race Ends Prematurely

**Symptom**:
- Race stops before required laps
- Unexpected shutdown

**Solutions**:

1. **Check crash detection**:
   ```yaml
   crash_detection:
     enable_crash_detection: false  # Disable if not needed
   ```

2. **Verify required laps**:
   ```bash
   ros2 param get /race_monitor required_laps
   ```

3. **Check for manual termination**:
   - Look for Ctrl+C or external kill signals

4. **Review logs**:
   ```bash
   less log/latest_list/race_monitor/stdout.log
   ```

---

## Data Issues

### Issue: No Data Files Generated

**Symptom**:
- Empty `data/` directory
- No results files

**Solutions**:

1. **Check output directory**:
   ```bash
   ros2 param get /race_monitor results_dir
   ```

2. **Verify write permissions**:
   ```bash
   ls -la data/
   chmod -R u+w data/
   ```

3. **Check disk space**:
   ```bash
   df -h
   ```

4. **Verify race completed**:
   - Data is saved on race completion
   - Check if race finished properly

---

### Issue: Missing Trajectory Files

**Symptom**:
- No trajectory files in output
- Empty trajectories directory

**Solutions**:

1. **Check if saving is enabled**:
   ```bash
   ros2 param get /race_monitor save_trajectories
   ```

2. **Enable in config**:
   ```yaml
   save_trajectories: true
   ```

3. **Check trajectory format**:
   ```yaml
   trajectory_format: "tum"  # or "kitti", "euroc"
   ```

---

### Issue: Graphs Not Generated

**Symptom**:
- No PNG/PDF files in graphs directory
- Visualization failed

**Solutions**:

1. **Check if enabled**:
   ```yaml
   auto_generate_graphs: true
   ```

2. **Verify matplotlib backend**:
   ```bash
   python3 -c "import matplotlib; print(matplotlib.get_backend())"
   ```

3. **Install display backend** (if needed):
   ```bash
   pip install matplotlib pillow
   ```

4. **Check for errors in logs**:
   ```bash
   grep -i "error.*plot\|error.*graph" log/latest_list/race_monitor/stdout.log
   ```

---

## Performance Issues

### Issue: High CPU Usage

**Symptom**:
- System sluggish
- High CPU reported

**Solutions**:

1. **Disable computational monitoring**:
   ```yaml
   enable_computational_monitoring: false
   ```

2. **Reduce evaluation frequency**:
   ```yaml
   evaluation_interval_laps: 5  # Instead of 1
   ```

3. **Disable unnecessary features**:
   ```yaml
   generate_3d_vector_plots: false
   generate_error_mapped_plots: false
   ```

4. **Limit output formats**:
   ```yaml
   output_formats: ["csv"]  # Instead of ["csv", "json", "pickle", "mat"]
   ```

---

### Issue: Slow Performance

**Symptom**:
- Laggy response
- Delayed topic updates

**Solutions**:

1. **Check topic rates**:
   ```bash
   ros2 topic hz /car_state/odom
   ```

2. **Monitor system resources**:
   ```bash
   top
   # Look for CPU/memory usage
   ```

3. **Reduce visualization**:
   ```yaml
   auto_generate_graphs: false  # Generate manually after race
   ```

4. **Simplify analysis**:
   ```yaml
   enable_advanced_metrics: false
   calculate_all_statistics: false
   ```

---

### Issue: Memory Usage Growing

**Symptom**:
- Increasing memory consumption
- System slowdown over time

**Solutions**:

1. **Limit trajectory points**:
   ```yaml
   apply_trajectory_filtering: true
   filter_parameters:
     distance_threshold: 0.1  # Increase to reduce points
   ```

2. **Disable intermediate saving**:
   ```yaml
   save_intermediate_results: false
   ```

3. **Monitor memory**:
   ```bash
   ros2 topic echo /race_monitor/memory_usage
   ```

---

## Topic Issues

### Issue: Topics Not Publishing

**Symptom**:
- No data on Race Monitor topics
- Echo shows nothing

**Solutions**:

1. **Verify node is running**:
   ```bash
   ros2 node list | grep race_monitor
   ```

2. **Check topic list**:
   ```bash
   ros2 topic list | grep race_monitor
   ```

3. **Check for errors**:
   ```bash
   ros2 node info /race_monitor
   ```

4. **Restart node**:
   ```bash
   # Ctrl+C and relaunch
   ros2 launch race_monitor race_monitor.launch.py
   ```

---

### Issue: Topic Rate Too Low

**Symptom**:
- Slow updates
- Delayed information

**Solutions**:

1. **Check source topic rate**:
   ```bash
   ros2 topic hz /car_state/odom
   ```

2. **Verify QoS settings**:
   - Race Monitor adapts to source topic QoS

3. **Check system load**:
   ```bash
   top
   ```

---

## Visualization Issues

### Issue: RViz Not Showing Trajectory

**Symptom**:
- No trajectory visible in RViz
- Empty visualization

**Solutions**:

1. **Add topic to RViz**:
   - Add `Path` display
   - Set topic to `/race_monitor/current_trajectory`
   - Set frame to `map`

2. **Check frame ID**:
   ```bash
   ros2 param get /race_monitor frame_id
   ```

3. **Verify topic publishing**:
   ```bash
   ros2 topic echo /race_monitor/current_trajectory --once
   ```

---

### Issue: Start Line Not Visible

**Symptom**:
- Start/finish line marker missing in RViz

**Solutions**:

1. **Add Marker display**:
   - Add `Marker` display type
   - Set topic to `/race_monitor/start_finish_line`

2. **Check marker publishing**:
   ```bash
   ros2 topic echo /race_monitor/start_finish_line --once
   ```

3. **Verify frame matches**:
   - RViz fixed frame should match `frame_id` parameter

---

## EVO Integration Issues

### Issue: EVO Commands Not Found

**Symptom**:
```
bash: evo_ape: command not found
bash: evo_rpe: command not found
```

**Solutions**:

1. **Install EVO**:
   ```bash
   cd ~/ros2_ws/src/race_monitor/evo
   pip install -e .
   ```

2. **Add to PATH** (if needed):
   ```bash
   export PATH=$PATH:~/.local/bin
   ```

3. **Verify installation**:
   ```bash
   which evo_ape
   evo_ape --help
   ```

---

### Issue: EVO Trajectory Format Error

**Symptom**:
```
Error: Could not parse trajectory file
ValueError: Invalid format
```

**Solutions**:

1. **Check file format**:
   ```bash
   head trajectories/tum/lap_001_trajectory.txt
   ```

2. **Verify format specification**:
   ```bash
   # TUM format should be:
   # timestamp x y z qx qy qz qw
   ```

3. **Use correct format argument**:
   ```bash
   evo_ape tum reference.txt estimated.txt  # Not csv, kitti
   ```

---

### Issue: Reference Trajectory Not Found

**Symptom**:
```
FileNotFoundError: reference_trajectory.csv not found
```

**Solutions**:

1. **Check file location**:
   ```bash
   ls -la ref_trajectory/
   ```

2. **Use absolute path**:
   ```yaml
   reference_trajectory_file: "/full/path/to/reference.csv"
   ```

3. **Copy to correct location**:
   ```bash
   cp /path/to/reference.csv ref_trajectory/
   ```

---

## Debug Mode

### Enable Debug Logging

1. **In configuration file**:
   ```yaml
   log_level: "debug"  # or "verbose"
   ```

2. **Relaunch**:
   ```bash
   ros2 launch race_monitor race_monitor.launch.py
   ```

3. **View detailed output**:
   - All logging will appear in console
   - Check `log/latest_list/race_monitor/stdout.log`

### Debug Information to Collect

When reporting issues, include:

1. **System information**:
   ```bash
   uname -a
   ros2 --version
   python3 --version
   ```

2. **Package version**:
   ```bash
   cd ~/ros2_ws/src/race_monitor
   git describe --tags
   ```

3. **Topic information**:
   ```bash
   ros2 topic list
   ros2 topic info /car_state/odom
   ros2 topic hz /car_state/odom
   ```

4. **Parameter dump**:
   ```bash
   ros2 param dump /race_monitor
   ```

5. **Log files**:
   ```bash
   cat log/latest_list/race_monitor/stdout.log
   cat log/latest_list/race_monitor/stderr.log
   ```

---

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Review documentation**:
   - [INSTALLATION.md](INSTALLATION.md)
   - [CONFIGURATION.md](CONFIGURATION.md)
   - [USAGE.md](USAGE.md)
3. **Search existing issues**: https://github.com/GIU-F1Tenth/race_monitor/issues
4. **Enable debug logging** and review output

### How to Report Issues

When opening an issue, provide:

1. **Clear description** of the problem
2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **System information** (OS, ROS2 version, Python version)
5. **Error messages** (full output)
6. **Log files** (if applicable)
7. **Configuration used** (`race_monitor.yaml`)

### Support Channels

- **GitHub Issues**: https://github.com/GIU-F1Tenth/race_monitor/issues
- **GitHub Discussions**: https://github.com/GIU-F1Tenth/race_monitor/discussions
- **Email**: mohammed@azab.io

---

## Common Solutions Summary

### Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Module not found | `pip install -r requirements.txt` |
| Build fails | `rm -rf build install log && colcon build` |
| Node won't start | `source ~/ros2_ws/install/setup.bash` |
| No odometry | `ros2 topic echo /car_state/odom` |
| No lap detection | Check start line in RViz |
| No data saved | Check `results_dir` and permissions |
| EVO not working | `cd evo && pip install -e .` |

---

## Additional Resources

- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Configuration Guide**: [CONFIGURATION.md](CONFIGURATION.md)
- **Usage Guide**: [USAGE.md](USAGE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **EVO Integration**: [EVO_INTEGRATION.md](EVO_INTEGRATION.md)

---

**Still Having Issues?**

Open an issue with detailed information:
https://github.com/GIU-F1Tenth/race_monitor/issues/new
