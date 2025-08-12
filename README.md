# Race Monitor

A ROS2 Python node for autonomous racing robots (F1TENTH) that tracks laps, lap times, and race statistics with dynamic start/finish line updates from RViz.

## Features

- **Dynamic Start/Finish Line**: Update start/finish line during race using RViz "Publish Point" tool
- **Visual Line Display**: Bright green line marker shows start/finish line position on map
- **Accurate Lap Detection**: Line segment intersection with heading verification and debounce protection
- **Real-time Statistics**: Track lap times, best/worst/average laps, and total race time
- **Live Publishing**: Publishes lap count, lap times, and race status in real-time
- **CSV Export**: Automatically saves detailed race results to CSV file
- **Configurable Parameters**: Customizable lap requirements, debounce time, and output settings

## Topics

### Subscriptions
- `/odom` (nav_msgs/Odometry) - Vehicle position and heading
- `/clicked_point` (geometry_msgs/PointStamped) - Points clicked in RViz for start/finish line

### Publications
- `/race_monitor/lap_count` (std_msgs/Int32) - Current lap count
- `/race_monitor/lap_time` (std_msgs/Float32) - Last completed lap time
- `/race_monitor/best_lap_time` (std_msgs/Float32) - Best lap time so far
- `/race_monitor/race_running` (std_msgs/Bool) - Race status (running/stopped)
- `/race_monitor/start_line_marker` (visualization_msgs/Marker) - Visual marker for start/finish line

## Parameters

- `start_line_p1` (float array, default: [0.0, 0.0]) - First point of start/finish line
- `start_line_p2` (float array, default: [1.0, 0.0]) - Second point of start/finish line
- `required_laps` (int, default: 5) - Number of laps to complete the race
- `debounce_time` (float, default: 2.0) - Debounce time in seconds to avoid false detections
- `output_file` (string, default: "race_results.csv") - Output CSV filename

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/your_ws/src
# Package should already be in race_stack
```

2. Build the package:
```bash
cd ~/your_ws
colcon build --packages-select race_monitor
source install/setup.bash
```

## Usage

### Basic Launch
```bash
ros2 launch race_monitor race_monitor_launch.py
```

### Launch with Custom Parameters
```bash
ros2 launch race_monitor race_monitor_launch.py \
    required_laps:=10 \
    debounce_time:=1.5 \
    start_line_p1:="[2.0, 1.0]" \
    start_line_p2:="[2.0, -1.0]"
```

### Setting Start/Finish Line in RViz

1. **Open RViz** and make sure you can see your robot and the track

2. **Add Publish Point Tool**:
   - In RViz, go to the toolbar
   - Click "Add" button or use the "2D Nav Goal" dropdown
   - Select "Publish Point" tool

3. **Set Start/Finish Line**:
   - Click the first point where you want the start/finish line to begin
   - Click the second point where you want the start/finish line to end
   - The race monitor will automatically update the line and log the change

4. **Monitor Race Progress**:
   - Watch the terminal for lap completion messages
   - Use `ros2 topic echo /race_monitor/lap_count` to see live lap count
   - Use `ros2 topic echo /race_monitor/lap_time` to see lap times

### Example RViz Workflow

1. **Setup RViz**:
```bash
# Terminal 1: Launch your robot/simulation
ros2 launch your_robot_package robot_launch.py

# Terminal 2: Launch race monitor
ros2 launch race_monitor race_monitor_launch.py

# Terminal 3: Launch RViz
rviz2
```

2. **Configure RViz**:
   - Add Map display (if using SLAM/localization)
   - Add RobotModel display
   - Add Odometry display for `/odom`
   - Add Marker display for `/race_monitor/start_line_marker` to see the start/finish line
   - Add the "Publish Point" tool to toolbar

3. **Set Start/Finish Line**:
   - Use "Publish Point" tool to click two points
   - First click = P1, second click = P2
   - Line is immediately updated and visualized as a bright green line
   - Yellow "START/FINISH" text appears at the midpoint

4. **Start Racing**:
   - Drive your robot across the start/finish line to begin
   - Monitor progress in terminal and via ROS topics
   - The green line shows exactly where the detection zone is

## Output Files

Race results are automatically saved to CSV files in `data/race_monitor/` directory:

```csv
lap_number,lap_time
1,45.2340
2,43.1250
3,44.5670
total_time,132.9260
best_lap,43.1250
worst_lap,45.2340
average_lap,44.3087
```

## Visualization Features

### Start/Finish Line Display
- **Green Line Marker**: Shows the exact position of the start/finish line on the map
- **Yellow Text Label**: "START/FINISH" text appears at the midpoint of the line
- **Real-time Updates**: Visualization updates immediately when line is changed via RViz
- **Persistent Display**: Line remains visible throughout the race for reference

### RViz Integration
- Publishes to `/race_monitor/start_line_marker` topic using `visualization_msgs/Marker`
- Compatible with standard RViz Marker display
- Line width and colors optimized for visibility
- Text positioned above ground level for clear viewing

## Algorithm Details

### Lap Detection
- Uses line segment intersection between vehicle path and start/finish line
- Verifies crossing direction using heading check
- Applies debounce time to prevent false positives from oscillations

### Line Intersection
- Implements counter-clockwise (CCW) algorithm for robust intersection detection
- Handles edge cases and ensures accurate crossing detection

### Heading Verification
- Calculates line normal vector to determine "forward" direction
- Compares vehicle heading with line normal using dot product
- Only counts crossings in the positive direction

## Troubleshooting

### Common Issues

1. **No lap detection**:
   - Check that `/odom` topic is publishing
   - Verify start/finish line is set correctly
   - Ensure vehicle crosses the line completely

2. **False lap detections**:
   - Increase `debounce_time` parameter
   - Check for noisy odometry data
   - Verify start/finish line placement

3. **CSV file not saved**:
   - Check write permissions in current directory
   - Verify `data/race_monitor/` directory creation
   - Check terminal for error messages

### Debug Commands

```bash
# Check if topics are publishing
ros2 topic list
ros2 topic echo /odom
ros2 topic echo /clicked_point

# Monitor race status
ros2 topic echo /race_monitor/lap_count
ros2 topic echo /race_monitor/race_running

# Check node status
ros2 node list
ros2 node info /race_monitor
```

## Contributing

1. Follow ROS2 Python style guidelines
2. Add tests for new functionality
3. Update documentation for new features
4. Test with real robot hardware when possible

## License

MIT License - see LICENSE file for details.