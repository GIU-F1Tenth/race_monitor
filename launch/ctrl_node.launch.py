#!/usr/bin/env python3

"""
ctrl_node standalone launch file

Launches the keyboard/joystick control node for race_monitor services.
Run this separately from race_monitor.launch.py in its own terminal:

    ros2 launch race_monitor ctrl_node.launch.py

Or run directly (also works):

    ros2 run race_monitor ctrl_node

Key bindings (default):
    r  - reset race
    f  - force lap complete
    p  - pause race
    u  - resume race
    t  - reset lap time

Author: Mohammed S. Azab Abdelazim (mohammed@azab.io)
License: MIT License
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file_path = PathJoinSubstitution([
        FindPackageShare('race_monitor'),
        'config',
        'ctrl_node.yaml'
    ])

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file_path,
        description='Path to ctrl_node config file'
    )

    ctrl_node = Node(
        package='race_monitor',
        executable='ctrl_node',
        name='ctrl_node',
        parameters=[LaunchConfiguration('config')],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        config_arg,
        ctrl_node,
    ])
