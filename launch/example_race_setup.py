#!/usr/bin/env python3
"""
Example launch file showing how to set up a complete race monitoring setup.
This demonstrates launching the race monitor with custom parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Log info about the race setup
        LogInfo(msg="Starting F1TENTH Race Monitor Example Setup"),
        
        # Example: 5-lap race with 1.5s debounce time
        # Start/finish line from (0,0) to (0,2) - a vertical line
        DeclareLaunchArgument(
            'track_name',
            default_value='example_track',
            description='Name of the track for file naming'
        ),
        
        # Race monitor with example parameters for a typical F1TENTH track
        Node(
            package='race_monitor',
            executable='race_monitor',
            name='race_monitor',
            output='screen',
            parameters=[{
                'start_line_p1': [0.0, 0.0],      # Start of finish line
                'start_line_p2': [0.0, 2.0],      # End of finish line (2m vertical line)
                'required_laps': 5,                # 5 lap race
                'debounce_time': 1.5,              # 1.5 second debounce
                'output_file': 'example_race_results.csv',
            }],
            remappings=[
                # Example remappings if your topics have different names
                # ('/odom', '/car/odom'),
                # ('/clicked_point', '/rviz/clicked_point'),
            ]
        ),
        
        LogInfo(msg="Race Monitor started. Use RViz 'Publish Point' tool to set start/finish line."),
        LogInfo(msg="Drive across the line to start the race!"),
    ])
