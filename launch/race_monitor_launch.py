#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    start_line_p1_arg = DeclareLaunchArgument(
        'start_line_p1',
        default_value='[0.0, -1.0]',
        description='First point of start/finish line [x, y] - bottom of vertical line'
    )

    start_line_p2_arg = DeclareLaunchArgument(
        'start_line_p2',
        default_value='[0.0, 1.0]',
        description='Second point of start/finish line [x, y] - top of vertical line'
    )

    required_laps_arg = DeclareLaunchArgument(
        'required_laps',
        default_value='5',
        description='Number of laps required to complete the race'
    )

    debounce_time_arg = DeclareLaunchArgument(
        'debounce_time',
        default_value='2.0',
        description='Debounce time in seconds to avoid false lap detections'
    )

    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value='race_results.csv',
        description='Output CSV filename for race results'
    )

    # Race monitor node
    race_monitor_node = Node(
        package='race_monitor',
        executable='race_monitor',
        name='race_monitor',
        output='screen',
        parameters=[{
            'start_line_p1': LaunchConfiguration('start_line_p1'),
            'start_line_p2': LaunchConfiguration('start_line_p2'),
            'required_laps': LaunchConfiguration('required_laps'),
            'debounce_time': LaunchConfiguration('debounce_time'),
            'output_file': LaunchConfiguration('output_file'),
        }],
        remappings=[
            # Add any topic remappings here if needed
            # ('/odom', '/robot/odom'),  # Example remapping
        ]
    )

    return LaunchDescription([
        start_line_p1_arg,
        start_line_p2_arg,
        required_laps_arg,
        debounce_time_arg,
        output_file_arg,
        race_monitor_node,
    ])
