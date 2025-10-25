#!/usr/bin/env python3

"""
Race Monitor - Unified Launch File

A single launch file that supports all three race ending modes:
1. lap_complete - Race ends after completing required laps (default)
2. crash - Race ends when crash is detected
3. manual - Race continues until manually killed

Usage Examples:
  # Default lap complete mode (uses config file: 20 laps)
  ros2 launch race_monitor race_monitor.launch.py

  # Lap complete mode with custom laps (overrides config file)
  ros2 launch race_monitor race_monitor.launch.py race_mode:=lap_complete required_laps:=10

  # Crash detection mode
  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash

  # Manual endurance mode
  ros2 launch race_monitor race_monitor.launch.py race_mode:=manual save_interval:=60.0

  # Custom crash detection parameters
  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash max_stationary_time:=10.0 enable_collision_detection:=false

Note: This launch file now fully respects the configuration file (race_monitor.yaml) as the
source of truth. Launch arguments ONLY override config values when explicitly provided
on the command line. You can change any default value in the config file without
needing to update this launch file.Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate unified launch description supporting all race ending modes."""

    # Get package directory and config file
    config_file_path = PathJoinSubstitution([
        FindPackageShare('race_monitor'),
        'config',
        'race_monitor.yaml'
    ])

    # Common launch arguments
    race_mode_arg = DeclareLaunchArgument(
        'race_mode',
        default_value='lap_complete',
        description='Race ending mode: lap_complete, crash, or manual',
        choices=['lap_complete', 'crash', 'manual']
    )

    controller_name_arg = DeclareLaunchArgument(
        'controller_name',
        default_value='custom_controller',
        description='Name of the controller being tested'
    )

    # Note: experiment_id is now fully auto-generated and not configurable

    # Lap complete mode arguments
    required_laps_arg = DeclareLaunchArgument(
        'required_laps',
        default_value='20',
        description='Number of laps required to complete the race (lap_complete mode only)'
    )

    # Crash detection mode arguments
    max_stationary_time_arg = DeclareLaunchArgument(
        'max_stationary_time',
        default_value='5.0',
        description='Maximum time vehicle can be stationary before crash detection (seconds)'
    )

    min_velocity_threshold_arg = DeclareLaunchArgument(
        'min_velocity_threshold',
        default_value='0.1',
        description='Minimum velocity threshold to consider vehicle as moving (m/s)'
    )

    max_odometry_timeout_arg = DeclareLaunchArgument(
        'max_odometry_timeout',
        default_value='3.0',
        description='Maximum time without odometry updates before crash detection (seconds)'
    )

    enable_collision_detection_arg = DeclareLaunchArgument(
        'enable_collision_detection',
        default_value='true',
        description='Enable collision detection based on sudden velocity changes'
    )

    collision_velocity_threshold_arg = DeclareLaunchArgument(
        'collision_velocity_threshold',
        default_value='2.0',
        description='Velocity change threshold for collision detection (m/s)'
    )

    collision_detection_window_arg = DeclareLaunchArgument(
        'collision_detection_window',
        default_value='0.5',
        description='Time window for collision detection (seconds)'
    )

    # Manual mode arguments
    save_interval_arg = DeclareLaunchArgument(
        'save_interval',
        default_value='30.0',
        description='Interval for saving intermediate results in manual mode (seconds)'
    )

    max_duration_arg = DeclareLaunchArgument(
        'max_duration',
        default_value='0',
        description='Maximum race duration for safety in manual mode (seconds, 0 = no limit)'
    )

    enable_intermediate_saves_arg = DeclareLaunchArgument(
        'enable_intermediate_saves',
        default_value='true',
        description='Enable periodic intermediate result saves in manual mode'
    )

    # Advanced configuration arguments
    enable_crash_detection_arg = DeclareLaunchArgument(
        'enable_crash_detection',
        default_value='true',
        description='Enable crash detection (used in crash and manual modes)'
    )

    enable_trajectory_evaluation_arg = DeclareLaunchArgument(
        'enable_trajectory_evaluation',
        default_value='true',
        description='Enable trajectory evaluation and analysis'
    )

    enable_computational_monitoring_arg = DeclareLaunchArgument(
        'enable_computational_monitoring',
        default_value='false',
        description='Enable computational performance monitoring'
    )

    save_trajectories_arg = DeclareLaunchArgument(
        'save_trajectories',
        default_value='true',
        description='Save trajectory data to files'
    )

    auto_generate_graphs_arg = DeclareLaunchArgument(
        'auto_generate_graphs',
        default_value='true',
        description='Automatically generate analysis graphs'
    )

    # Smart controller detection arguments
    enable_smart_controller_detection_arg = DeclareLaunchArgument(
        'enable_smart_controller_detection',
        default_value='true',
        description='Enable automatic controller detection from topic publishers (only when controller_name is empty)'
    )

    # Auto-shutdown arguments
    auto_shutdown_on_race_complete_arg = DeclareLaunchArgument(
        'auto_shutdown_on_race_complete',
        default_value='true',
        description='Automatically shutdown node when race is completed'
    )

    shutdown_delay_seconds_arg = DeclareLaunchArgument(
        'shutdown_delay_seconds',
        default_value='5.0',
        description='Delay before shutdown to allow final data processing'
    )

    def launch_setup(context, *args, **kwargs):
        """Setup launch configuration based on selected mode."""

        import sys

        # Get parameter values
        race_mode = LaunchConfiguration('race_mode').perform(context)

        # Build parameter overrides only for explicitly provided arguments
        override_parameters = {}

        # Always set race mode since it controls behavior
        override_parameters['race_ending_mode'] = race_mode

        # Check command line arguments to see what was explicitly provided
        cmd_args = sys.argv

        # Helper function to check if an argument was explicitly provided
        def was_explicitly_provided(arg_name):
            """Check if a launch argument was explicitly provided on command line."""
            return any(f'{arg_name}:=' in arg for arg in cmd_args)

        # Only override parameters that were explicitly provided via command line
        if was_explicitly_provided('controller_name'):
            override_parameters['controller_name'] = LaunchConfiguration('controller_name').perform(context)

        # Note: experiment_id is auto-generated, so we don't accept it from command line

        if was_explicitly_provided('required_laps'):
            override_parameters['required_laps'] = int(LaunchConfiguration('required_laps').perform(context))

        # Mode-specific overrides
        if race_mode == 'crash':
            # For crash mode, override lap count to prevent normal completion
            override_parameters['required_laps'] = 999

            # Only override crash detection parameters if explicitly provided
            if was_explicitly_provided('enable_crash_detection'):
                override_parameters['crash_detection.enable_crash_detection'] = LaunchConfiguration(
                    'enable_crash_detection')
            if was_explicitly_provided('max_stationary_time'):
                override_parameters['crash_detection.max_stationary_time'] = LaunchConfiguration('max_stationary_time')
            if was_explicitly_provided('min_velocity_threshold'):
                override_parameters['crash_detection.min_velocity_threshold'] = LaunchConfiguration(
                    'min_velocity_threshold')
            if was_explicitly_provided('max_odometry_timeout'):
                override_parameters['crash_detection.max_odometry_timeout'] = LaunchConfiguration(
                    'max_odometry_timeout')
            if was_explicitly_provided('enable_collision_detection'):
                override_parameters['crash_detection.enable_collision_detection'] = LaunchConfiguration(
                    'enable_collision_detection')
            if was_explicitly_provided('collision_velocity_threshold'):
                override_parameters['crash_detection.collision_velocity_threshold'] = LaunchConfiguration(
                    'collision_velocity_threshold')
            if was_explicitly_provided('collision_detection_window'):
                override_parameters['crash_detection.collision_detection_window'] = LaunchConfiguration(
                    'collision_detection_window')

        elif race_mode == 'manual':
            # For manual mode, override lap count to prevent normal completion
            override_parameters['required_laps'] = 999

            # Only override manual mode parameters if explicitly provided
            if was_explicitly_provided('enable_intermediate_saves'):
                override_parameters['manual_mode.save_intermediate_results'] = LaunchConfiguration(
                    'enable_intermediate_saves')
            if was_explicitly_provided('save_interval'):
                override_parameters['manual_mode.save_interval'] = LaunchConfiguration('save_interval')
            if was_explicitly_provided('max_duration'):
                override_parameters['manual_mode.max_race_duration'] = LaunchConfiguration('max_duration')

        # For lap_complete mode, only use config file + any explicitly provided overrides

        # Create race monitor node with config file as primary source
        race_monitor_node = Node(
            package='race_monitor',
            executable='race_monitor',
            name='race_monitor',
            parameters=[config_file_path, override_parameters],
            output='screen',
            emulate_tty=True,
        )

        return [race_monitor_node]

    return LaunchDescription([
        race_mode_arg,
        controller_name_arg,

        # Lap complete mode arguments
        required_laps_arg,

        # Crash detection arguments
        max_stationary_time_arg,
        min_velocity_threshold_arg,
        max_odometry_timeout_arg,
        enable_collision_detection_arg,
        collision_velocity_threshold_arg,
        collision_detection_window_arg,

        # Manual mode arguments
        save_interval_arg,
        max_duration_arg,
        enable_intermediate_saves_arg,

        # Advanced arguments
        enable_crash_detection_arg,
        enable_trajectory_evaluation_arg,
        enable_computational_monitoring_arg,
        save_trajectories_arg,
        auto_generate_graphs_arg,

        # Smart controller detection arguments
        enable_smart_controller_detection_arg,

        # Auto-shutdown arguments
        auto_shutdown_on_race_complete_arg,
        shutdown_delay_seconds_arg,

        # Setup function
        OpaqueFunction(function=launch_setup)
    ])


if __name__ == '__main__':
    print("Race Monitor - Unified Launch File")
    print("=" * 40)
    print()
    print("Available race modes:")
    print("  lap_complete - Race ends after completing required laps (default)")
    print("  crash        - Race ends when crash is detected")
    print("  manual       - Race continues until manually killed")
    print()
    print("Quick start examples:")
    print()
    print("Standard 5-lap race:")
    print("  ros2 launch race_monitor race_monitor.launch.py")
    print()
    print("10-lap race:")
    print("  ros2 launch race_monitor race_monitor.launch.py required_laps:=10")
    print()
    print("Crash detection mode:")
    print("  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash")
    print()
    print("With smart controller detection disabled:")
    print("  ros2 launch race_monitor race_monitor.launch.py enable_smart_controller_detection:=false")
    print()
    print("With auto-shutdown disabled:")
    print("  ros2 launch race_monitor race_monitor.launch.py auto_shutdown_on_race_complete:=false")
    print()
    print("Set custom controller name (disables auto-detection):")
    print("  ros2 launch race_monitor race_monitor.launch.py controller_name:=my_controller")
    print()
    print("Manual endurance mode:")
    print("  ros2 launch race_monitor race_monitor.launch.py race_mode:=manual")
    print()
    print("Custom crash detection:")
    print("  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash \\")
    print("    max_stationary_time:=10.0 enable_collision_detection:=false")
    print()
    print("For full argument list:")
    print("  ros2 launch race_monitor race_monitor.launch.py --show-args")
