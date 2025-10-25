# API Reference

Complete ROS2 interface reference for Race Monitor.

## Table of Contents
- [Overview](#overview)
- [Published Topics](#published-topics)
- [Subscribed Topics](#subscribed-topics)
- [Services](#services)
- [Parameters](#parameters)
- [Message Types](#message-types)
- [Launch Files](#launch-files)
- [Examples](#examples)

---

## Overview

Race Monitor provides a comprehensive ROS2 interface for monitoring race progress, trajectory analysis, and performance evaluation. All topics use standard ROS2 message types for maximum compatibility.

### Namespace

All Race Monitor topics are published under the `/race_monitor/` namespace.

### Quality of Service (QoS)

Default QoS profiles:
- **Status topics**: RELIABLE, VOLATILE
- **Data topics**: RELIABLE, TRANSIENT_LOCAL
- **High-frequency topics**: BEST_EFFORT, VOLATILE

---

## Published Topics

### Race Status Topics

#### `/race_monitor/lap_count`

Current lap number.

- **Type**: `std_msgs/Int32`
- **Rate**: Event-based (published on lap completion)
- **Description**: Increments each time vehicle crosses start/finish line

**Example**:
```bash
ros2 topic echo /race_monitor/lap_count
```

**Output**:
```
data: 3
---
```

#### `/race_monitor/lap_time`

Time taken for last completed lap.

- **Type**: `std_msgs/Float32`
- **Rate**: Event-based (published on lap completion)
- **Units**: Seconds
- **Description**: Duration of most recently completed lap

**Example**:
```bash
ros2 topic echo /race_monitor/lap_time
```

**Output**:
```
data: 19.67
---
```

#### `/race_monitor/race_running`

Boolean flag indicating if race is active.

- **Type**: `std_msgs/Bool`
- **Rate**: 10 Hz
- **Description**: `true` when race is in progress, `false` otherwise

**Example**:
```bash
ros2 topic echo /race_monitor/race_running
```

**Output**:
```
data: true
---
```

#### `/race_monitor/race_status`

Comprehensive race status information.

- **Type**: `std_msgs/String`
- **Rate**: 1 Hz
- **Format**: JSON
- **Description**: Complete race state including laps, time, distance, status

**JSON Structure**:
```json
{
  "race_active": true,
  "current_lap": 3,
  "required_laps": 7,
  "elapsed_time": 59.12,
  "total_distance": 201.5,
  "current_status": "racing",
  "controller": "lqr_controller_node",
  "experiment_id": "exp_20250125_153000"
}
```

**Example**:
```bash
ros2 topic echo /race_monitor/race_status
```

#### `/race_monitor/total_distance`

Cumulative distance traveled.

- **Type**: `std_msgs/Float32`
- **Rate**: 10 Hz
- **Units**: Meters
- **Description**: Total distance covered since race start

**Example**:
```bash
ros2 topic echo /race_monitor/total_distance
```

**Output**:
```
data: 201.45
---
```

---

### Performance Monitoring Topics

#### `/race_monitor/control_loop_latency`

Control command processing latency.

- **Type**: `std_msgs/Float32`
- **Rate**: 10 Hz
- **Units**: Milliseconds
- **Description**: Time delay in control loop execution
- **Enabled**: When `enable_computational_monitoring: true`

**Example**:
```bash
ros2 topic echo /race_monitor/control_loop_latency
```

**Output**:
```
data: 12.34
---
```

#### `/race_monitor/cpu_usage`

CPU utilization percentage.

- **Type**: `std_msgs/Float32`
- **Rate**: 1 Hz
- **Units**: Percentage (0-100)
- **Description**: Current CPU usage by Race Monitor process
- **Enabled**: When `enable_computational_monitoring: true`

**Example**:
```bash
ros2 topic echo /race_monitor/cpu_usage
```

**Output**:
```
data: 15.6
---
```

#### `/race_monitor/memory_usage`

Memory consumption.

- **Type**: `std_msgs/Float32`
- **Rate**: 1 Hz
- **Units**: Megabytes (MB)
- **Description**: Current memory usage by Race Monitor process
- **Enabled**: When `enable_computational_monitoring: true`

**Example**:
```bash
ros2 topic echo /race_monitor/memory_usage
```

**Output**:
```
data: 245.8
---
```

#### `/race_monitor/processing_efficiency`

Computational efficiency score.

- **Type**: `std_msgs/Float32`
- **Rate**: 1 Hz
- **Units**: Score (0.0-1.0)
- **Description**: Overall processing efficiency metric
- **Enabled**: When `enable_computational_monitoring: true`

**Example**:
```bash
ros2 topic echo /race_monitor/processing_efficiency
```

**Output**:
```
data: 0.92
---
```

#### `/race_monitor/performance_stats`

Comprehensive performance statistics.

- **Type**: `std_msgs/String`
- **Rate**: 1 Hz
- **Format**: JSON
- **Description**: Complete performance metrics
- **Enabled**: When `enable_computational_monitoring: true`

**JSON Structure**:
```json
{
  "latency": {
    "current": 12.34,
    "mean": 11.45,
    "std": 2.15,
    "max": 18.92
  },
  "cpu": {
    "current": 15.6,
    "mean": 14.2,
    "max": 22.1
  },
  "memory": {
    "current": 245.8,
    "mean": 240.5,
    "max": 260.3
  },
  "efficiency": 0.92
}
```

---

### Trajectory Analysis Topics

#### `/race_monitor/trajectory_metrics`

Comprehensive trajectory metrics per lap.

- **Type**: `std_msgs/String`
- **Rate**: Event-based (on lap completion)
- **Format**: JSON
- **Description**: EVO-based trajectory quality metrics
- **Enabled**: When `enable_trajectory_evaluation: true`

**JSON Structure**:
```json
{
  "lap": 1,
  "ape": {
    "rmse": 4.52,
    "mean": 4.12,
    "std": 1.23,
    "max": 8.45
  },
  "rpe": {
    "rmse": 0.34,
    "mean": 0.28,
    "std": 0.15,
    "max": 0.67
  },
  "smoothness": 0.89,
  "efficiency": 0.95
}
```

#### `/race_monitor/smoothness`

Current trajectory smoothness score.

- **Type**: `std_msgs/Float32`
- **Rate**: 10 Hz
- **Units**: Score (0.0-1.0)
- **Description**: Real-time trajectory smoothness metric

#### `/race_monitor/consistency`

Lap-to-lap consistency score.

- **Type**: `std_msgs/Float32`
- **Rate**: Event-based (on lap completion)
- **Units**: Score (0.0-1.0)
- **Description**: Consistency between current and previous laps

#### `/race_monitor/path_efficiency`

Path efficiency metric.

- **Type**: `std_msgs/Float32`
- **Rate**: 10 Hz
- **Units**: Score (0.0-1.0)
- **Description**: Efficiency of current path vs. optimal

---

### Visualization Topics

#### `/race_monitor/current_trajectory`

Current lap trajectory path.

- **Type**: `nav_msgs/Path`
- **Rate**: 10 Hz
- **Frame**: Configured `frame_id` (default: "map")
- **Description**: Real-time trajectory visualization

**Example**:
```bash
ros2 topic echo /race_monitor/current_trajectory
```

#### `/race_monitor/reference_trajectory`

Reference trajectory path.

- **Type**: `nav_msgs/Path`
- **Rate**: 1 Hz (or latched)
- **Frame**: Configured `frame_id` (default: "map")
- **Description**: Optimal/reference trajectory for comparison
- **Enabled**: When reference trajectory is provided

#### `/race_monitor/start_finish_line`

Start/finish line visualization marker.

- **Type**: `visualization_msgs/Marker`
- **Rate**: 1 Hz (or latched)
- **Frame**: Configured `frame_id` (default: "map")
- **Description**: Visual marker for start/finish line in RViz

---

## Subscribed Topics

### Required Topics

#### `/car_state/odom`

Vehicle odometry data.

- **Type**: `nav_msgs/Odometry`
- **Required**: Yes
- **Description**: Primary input for race monitoring and trajectory analysis
- **Frame**: Typically "odom" or "map"

**Expected Fields**:
- `pose.pose.position`: Vehicle position (x, y, z)
- `pose.pose.orientation`: Vehicle orientation (quaternion)
- `twist.twist.linear`: Linear velocity
- `twist.twist.angular`: Angular velocity

**Remapping Example**:
```xml
<remap from="/car_state/odom" to="/your_robot/odom"/>
```

### Optional Topics

#### `/clicked_point`

Interactive point selection for start line setup.

- **Type**: `geometry_msgs/PointStamped`
- **Required**: No
- **Description**: Used in RViz for interactive start/finish line definition
- **Usage**: Click two points in RViz to define line

---

## Services

### Current Services

Currently, Race Monitor operates without services. Future releases will include:

### Planned Services (v1.1.0)

#### `/race_monitor/reset_race`

Reset race state to initial conditions.

- **Type**: `std_srvs/Trigger`
- **Status**: Planned
- **Description**: Reset all race data and prepare for new race

#### `/race_monitor/save_data`

Force immediate data export.

- **Type**: `std_srvs/Trigger`
- **Status**: Planned
- **Description**: Save current race data without stopping race

#### `/race_monitor/get_statistics`

Retrieve current race statistics.

- **Type**: Custom service type
- **Status**: Planned
- **Description**: Query current race metrics programmatically

---

## Parameters

### Runtime Parameters

Get parameters at runtime:

```bash
# List all parameters
ros2 param list /race_monitor

# Get specific parameter
ros2 param get /race_monitor race_ending_mode

# Set parameter (if supported)
ros2 param set /race_monitor log_level "debug"
```

### Key Parameters

#### `race_ending_mode`

- **Type**: string
- **Default**: "lap_complete"
- **Options**: "lap_complete", "crash", "manual"
- **Dynamic**: No (requires restart)

#### `controller_name`

- **Type**: string
- **Default**: "" (auto-detect)
- **Dynamic**: No (set at launch)

#### `required_laps`

- **Type**: integer
- **Default**: 7
- **Dynamic**: No (requires restart)

#### `enable_trajectory_evaluation`

- **Type**: boolean
- **Default**: true
- **Dynamic**: No (requires restart)

#### `log_level`

- **Type**: string
- **Default**: "normal"
- **Options**: "minimal", "normal", "debug", "verbose"
- **Dynamic**: Yes

For complete parameter list, see [CONFIGURATION.md](CONFIGURATION.md).

---

## Message Types

### Standard ROS2 Messages

Race Monitor uses standard ROS2 message types:

| Type | Package | Usage |
|------|---------|-------|
| `std_msgs/Int32` | std_msgs | Lap count |
| `std_msgs/Float32` | std_msgs | Numeric values |
| `std_msgs/Bool` | std_msgs | Boolean flags |
| `std_msgs/String` | std_msgs | JSON data |
| `nav_msgs/Odometry` | nav_msgs | Vehicle state |
| `nav_msgs/Path` | nav_msgs | Trajectories |
| `geometry_msgs/PointStamped` | geometry_msgs | Point selection |
| `visualization_msgs/Marker` | visualization_msgs | RViz markers |

### Custom Messages

Currently, Race Monitor does not define custom message types. All data is transmitted using standard messages, with complex data encoded as JSON strings.

---

## Launch Files

### Primary Launch File

**File**: `launch/race_monitor.launch.py`

```bash
ros2 launch race_monitor race_monitor.launch.py [arguments]
```

#### Launch Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `race_mode` | string | "lap_complete" | Race ending mode |
| `controller_name` | string | "" | Controller identifier |
| `required_laps` | int | 7 | Number of laps |
| `enable_trajectory_evaluation` | bool | true | Enable EVO analysis |
| `enable_computational_monitoring` | bool | false | Enable performance monitoring |
| `auto_generate_graphs` | bool | true | Generate visualizations |
| `enable_smart_controller_detection` | bool | true | Auto-detect controller |
| `auto_shutdown_on_race_complete` | bool | true | Auto-shutdown |
| `shutdown_delay_seconds` | float | 5.0 | Shutdown delay |

#### Example Launch

```bash
ros2 launch race_monitor race_monitor.launch.py \
    controller_name:=my_controller \
    required_laps:=10 \
    race_mode:=lap_complete
```

---

## Examples

### Monitoring Lap Progress

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32

class LapMonitor(Node):
    def __init__(self):
        super().__init__('lap_monitor')
        self.lap_sub = self.create_subscription(
            Int32, '/race_monitor/lap_count', self.lap_callback, 10)
        self.time_sub = self.create_subscription(
            Float32, '/race_monitor/lap_time', self.time_callback, 10)
    
    def lap_callback(self, msg):
        self.get_logger().info(f'Completed lap {msg.data}')
    
    def time_callback(self, msg):
        self.get_logger().info(f'Lap time: {msg.data:.2f}s')

def main():
    rclpy.init()
    node = LapMonitor()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Reading Race Status

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class RaceStatusMonitor(Node):
    def __init__(self):
        super().__init__('race_status_monitor')
        self.sub = self.create_subscription(
            String, '/race_monitor/race_status',
            self.status_callback, 10)
    
    def status_callback(self, msg):
        status = json.loads(msg.data)
        self.get_logger().info(
            f"Lap {status['current_lap']}/{status['required_laps']} - "
            f"Time: {status['elapsed_time']:.1f}s - "
            f"Distance: {status['total_distance']:.1f}m"
        )

def main():
    rclpy.init()
    node = RaceStatusMonitor()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Performance Monitoring

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        self.sub = self.create_subscription(
            String, '/race_monitor/performance_stats',
            self.perf_callback, 10)
    
    def perf_callback(self, msg):
        stats = json.loads(msg.data)
        latency = stats['latency']['current']
        cpu = stats['cpu']['current']
        efficiency = stats['efficiency']
        
        self.get_logger().info(
            f"Latency: {latency:.2f}ms | "
            f"CPU: {cpu:.1f}% | "
            f"Efficiency: {efficiency:.2%}"
        )

def main():
    rclpy.init()
    node = PerformanceMonitor()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Command Line Monitoring

```bash
# Watch lap count
ros2 topic echo /race_monitor/lap_count

# Monitor lap times
ros2 topic echo /race_monitor/lap_time

# View race status (JSON)
ros2 topic echo /race_monitor/race_status

# Check control latency
ros2 topic echo /race_monitor/control_loop_latency

# Monitor CPU usage
ros2 topic echo /race_monitor/cpu_usage
```

### Topic Frequency Check

```bash
# Check update rate of race status
ros2 topic hz /race_monitor/race_status

# Check trajectory update rate
ros2 topic hz /race_monitor/current_trajectory

# Verify odometry input
ros2 topic hz /car_state/odom
```

---

## Topic Remapping

### Remap Odometry Topic

If your odometry topic has a different name:

```bash
ros2 launch race_monitor race_monitor.launch.py \
    /car_state/odom:=/your_robot/odom
```

Or in a custom launch file:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='race_monitor',
            executable='race_monitor_node',
            name='race_monitor',
            remappings=[
                ('/car_state/odom', '/your_robot/odom'),
            ],
        ),
    ])
```

---

## Debugging Topics

### View All Topics

```bash
# List all Race Monitor topics
ros2 topic list | grep race_monitor
```

### Topic Information

```bash
# Get topic info
ros2 topic info /race_monitor/race_status

# Get message type
ros2 topic type /race_monitor/lap_count

# View topic data rate
ros2 topic hz /race_monitor/race_status
```

### Monitor All Topics

```bash
# Echo all status topics
ros2 topic echo /race_monitor/lap_count &
ros2 topic echo /race_monitor/lap_time &
ros2 topic echo /race_monitor/race_running &
```

---

## Best Practices

### For Integration

1. **Subscribe to `/race_monitor/race_status`** for comprehensive race information
2. **Monitor `/race_monitor/lap_count`** for lap-based logic
3. **Use `/race_monitor/race_running`** to check if race is active
4. **Check topic rates** to ensure proper data flow

### For Development

1. **Enable debug logging** for detailed topic information
2. **Monitor performance topics** to identify bottlenecks
3. **Use RViz** to visualize trajectory topics
4. **Check odometry input** first when troubleshooting

### For Performance

1. **Disable unnecessary features** to reduce topic load
2. **Use appropriate QoS profiles** for your use case
3. **Monitor system resources** via performance topics
4. **Adjust update rates** in configuration if needed

---

## Next Steps

- **Configuration**: [CONFIGURATION.md](CONFIGURATION.md)
- **Usage Examples**: [USAGE.md](USAGE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Need Help?**
- Issues: https://github.com/GIU-F1Tenth/race_monitor/issues
- Discussions: https://github.com/GIU-F1Tenth/race_monitor/discussions
