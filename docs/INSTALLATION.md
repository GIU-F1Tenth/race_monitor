# Installation Guide

Complete installation instructions for Race Monitor on ROS2 Humble.

## Table of Contents
- [Prerequisites](#prerequisites)
- [System Dependencies](#system-dependencies)
- [Python Dependencies](#python-dependencies)
- [Installation Steps](#installation-steps)
- [Docker Installation](#docker-installation-optional)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (tested) or compatible
- **ROS2 Distribution**: Humble Hawksbill or later
- **Python**: 3.10 or higher
- **Build System**: Colcon
- **Git**: For cloning repositories and submodules

### Hardware Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2GB free space
- **CPU**: Multi-core processor recommended for real-time performance

---

## System Dependencies

Install required system packages:

```bash
# Update package lists
sudo apt update

# Install ROS2 Humble (if not already installed)
sudo apt install -y ros-humble-desktop

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-pandas \
    python3-yaml \
    git
```

---

## Python Dependencies

### Core Dependencies

```bash
# Install core Python packages
pip install numpy scipy matplotlib pandas seaborn pyyaml
```

### Optional Dependencies

For advanced features:

```bash
# For MATLAB file support
pip install scipy

# For enhanced plotting
pip install seaborn plotly

# For data processing
pip install pandas
```

---

## Installation Steps

### 1. Create ROS2 Workspace (if needed)

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. Clone the Repository

```bash
# Clone with submodules (includes EVO library)
git clone --recursive https://github.com/GIU-F1Tenth/race_monitor.git

# If already cloned without --recursive, initialize submodules:
cd race_monitor
git submodule update --init --recursive
```

### 3. Install EVO Library

The EVO trajectory evaluation library is included as a submodule:

```bash
# Navigate to EVO directory
cd ~/ros2_ws/src/race_monitor/evo

# Install EVO in development mode
pip install -e .

# Verify installation
python3 -c "import evo; print(evo.__version__)"
```

Expected output: `1.x.x` (version number)

### 4. Install Race Monitor Dependencies

```bash
# Navigate to race_monitor directory
cd ~/ros2_ws/src/race_monitor

# Install from requirements file with constrains
pip install -c constraints.txt -r requirements.txt
```

### 5. Build the Package

```bash
# Navigate to workspace root
cd ~/ros2_ws

# Build Race Monitor package
colcon build --packages-select race_monitor

# Source the workspace
source install/setup.bash
```

### 6. Verify Installation

```bash
# Check if package is recognized
ros2 pkg list | grep race_monitor

# Expected output: race_monitor

# Verify node executable
ros2 pkg executables race_monitor

# Expected output: race_monitor race_monitor_node
```

---

## Docker Installation (Optional)

A Docker container with all dependencies pre-installed is available:

### Pull Docker Image

```bash
# Pull the latest container
docker pull ghcr.io/giu-f1tenth/race_monitor:latest
```

### Run with ROS2 Network

```bash
# Run container with host network
docker run -it --rm \
    --network=host \
    -v ~/ros2_ws:/workspace \
    ghcr.io/giu-f1tenth/race_monitor:latest
```

### Build Docker Image Locally

```bash
# Navigate to race_monitor directory
cd ~/ros2_ws/src/race_monitor

# Build Docker image
docker build -t race_monitor:local .

# Run the container
docker run -it --rm \
    --network=host \
    -v ~/ros2_ws:/workspace \
    race_monitor:local
```

---

## Verification

### Test Import

Verify Python modules are accessible:

```bash
python3 << EOF
import evo
import numpy as np
import matplotlib
import pandas as pd
print("✓ All imports successful!")
EOF
```

### Test ROS2 Integration

```bash
# Source workspace
source ~/ros2_ws/install/setup.bash

# Launch Race Monitor (will start and wait for odometry)
ros2 launch race_monitor race_monitor.launch.py

# In another terminal, check topics
ros2 topic list | grep race_monitor
```

Expected topics:
- `/race_monitor/lap_count`
- `/race_monitor/lap_time`
- `/race_monitor/race_running`
- `/race_monitor/race_status`

Press `Ctrl+C` to stop the test.

---

## Troubleshooting

### Issue: EVO Import Fails

**Symptom**: `ImportError: No module named 'evo'`

**Solution**:
```bash
# Add EVO to Python path
export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/race_monitor/evo

# Make permanent (add to ~/.bashrc)
echo 'export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/race_monitor/evo' >> ~/.bashrc
source ~/.bashrc

# Reinstall EVO
cd ~/ros2_ws/src/race_monitor/evo
pip install -e .
```

### Issue: Colcon Build Fails

**Symptom**: Build errors or package not found

**Solution**:
```bash
# Clean build artifacts
cd ~/ros2_ws
rm -rf build install log

# Rebuild with verbose output
colcon build --packages-select race_monitor --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace
source install/setup.bash
```

### Issue: Missing Python Dependencies

**Symptom**: `ModuleNotFoundError` for various packages

**Solution**:
```bash
# Install from requirements file
cd ~/ros2_ws/src/race_monitor
pip install -r requirements.txt

# Or install with constraints
pip install -c constraints.txt -r requirements.txt
```

### Issue: ROS2 Not Sourced

**Symptom**: `ros2: command not found`

**Solution**:
```bash
# Source ROS2 installation
source /opt/ros/humble/setup.bash

# Source workspace
source ~/ros2_ws/install/setup.bash

# Add to ~/.bashrc for automatic sourcing
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
```

### Issue: Permission Denied

**Symptom**: Cannot write to data directory

**Solution**:
```bash
# Fix permissions for data directory
cd ~/ros2_ws/src/race_monitor
chmod -R u+w data/

# Ensure output directory is writable
mkdir -p data
chmod 755 data
```

### Issue: Submodule Not Initialized

**Symptom**: Empty `evo/` directory

**Solution**:
```bash
cd ~/ros2_ws/src/race_monitor
git submodule update --init --recursive
```

---

## Next Steps

After successful installation:

1. **Configure the system**: See [CONFIGURATION.md](CONFIGURATION.md)
2. **Run your first race**: See [USAGE.md](USAGE.md)
3. **Explore examples**: Check `resource/sample_output_data/`

---

## Additional Resources

- **ROS2 Installation**: https://docs.ros.org/en/humble/Installation.html
- **EVO Documentation**: https://github.com/MichaelGrupp/evo
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Need Help?**
- Open an issue: https://github.com/GIU-F1Tenth/race_monitor/issues
- Discussions: https://github.com/GIU-F1Tenth/race_monitor/discussions
