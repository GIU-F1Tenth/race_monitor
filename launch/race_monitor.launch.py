#!/usr/bin/env python3

"""
Race Monitor - Unified Launch File

A single launch file that supports all three race ending modes:
1. lap_complete - Race ends after completing required laps (default)
2. crash - Race ends when crash is detected
3. manual - Race continues until manually killed

Usage Examples:
  # Default lap complete mode (5 laps)
  ros2 launch race_monitor race_monitor.launch.py
  
  # Lap complete mode with custom laps
  ros2 launch race_monitor race_monitor.launch.py race_mode:=lap_complete required_laps:=10
  
  # Crash detection mode
  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash
  
  # Manual endurance mode
  ros2 launch race_monitor race_monitor.launch.py race_mode:=manual save_interval:=60.0
  
  # Custom crash detection parameters
  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash max_stationary_time:=10.0 enable_collision_detection:=false

Author: Mohammed Abdelazim (mohammed@azab.io)
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
        default_value='test_controller',
        description='Name of the controller being tested'
    )
    
    experiment_id_arg = DeclareLaunchArgument(
        'experiment_id',
        default_value='exp_001',
        description='Experiment identifier'
    )
    
    # Lap complete mode arguments
    required_laps_arg = DeclareLaunchArgument(
        'required_laps',
        default_value='5',
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
        default_value='true',
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

    def launch_setup(context, *args, **kwargs):
        """Setup launch configuration based on selected mode."""
        
        # Get parameter values
        race_mode = LaunchConfiguration('race_mode').perform(context)
        controller_name = LaunchConfiguration('controller_name').perform(context)
        experiment_id = LaunchConfiguration('experiment_id').perform(context)
        
        # Base parameters (common to all modes)
        base_parameters = {
            'race_ending_mode': race_mode,
            'controller_name': controller_name,
            'experiment_id': experiment_id,
            'enable_trajectory_evaluation': LaunchConfiguration('enable_trajectory_evaluation'),
            'enable_computational_monitoring': LaunchConfiguration('enable_computational_monitoring'),
            'save_trajectories': LaunchConfiguration('save_trajectories'),
            'auto_generate_graphs': LaunchConfiguration('auto_generate_graphs'),
        }
        
        # Mode-specific parameters
        if race_mode == 'lap_complete':
            # Lap complete mode parameters
            mode_parameters = {
                'required_laps': LaunchConfiguration('required_laps'),
            }
            
        elif race_mode == 'crash':
            # Crash detection mode parameters
            mode_parameters = {
                'required_laps': 999,  # High number for crash mode (won't be reached)
                'crash_detection.enable_crash_detection': LaunchConfiguration('enable_crash_detection'),
                'crash_detection.max_stationary_time': LaunchConfiguration('max_stationary_time'),
                'crash_detection.min_velocity_threshold': LaunchConfiguration('min_velocity_threshold'),
                'crash_detection.max_odometry_timeout': LaunchConfiguration('max_odometry_timeout'),
                'crash_detection.enable_collision_detection': LaunchConfiguration('enable_collision_detection'),
                'crash_detection.collision_velocity_threshold': LaunchConfiguration('collision_velocity_threshold'),
                'crash_detection.collision_detection_window': LaunchConfiguration('collision_detection_window'),
            }
            
        elif race_mode == 'manual':
            # Manual mode parameters
            mode_parameters = {
                'required_laps': 999,  # High number for manual mode (won't be reached)
                'crash_detection.enable_crash_detection': LaunchConfiguration('enable_crash_detection'),
                'crash_detection.max_stationary_time': LaunchConfiguration('max_stationary_time'),
                'crash_detection.min_velocity_threshold': LaunchConfiguration('min_velocity_threshold'),
                'crash_detection.max_odometry_timeout': LaunchConfiguration('max_odometry_timeout'),
                'crash_detection.enable_collision_detection': LaunchConfiguration('enable_collision_detection'),
                'crash_detection.collision_velocity_threshold': LaunchConfiguration('collision_velocity_threshold'),
                'crash_detection.collision_detection_window': LaunchConfiguration('collision_detection_window'),
                'manual_mode.save_intermediate_results': LaunchConfiguration('enable_intermediate_saves'),
                'manual_mode.save_interval': LaunchConfiguration('save_interval'),
                'manual_mode.max_race_duration': LaunchConfiguration('max_duration'),
            }
        
        else:
            raise ValueError(f"Unknown race mode: {race_mode}")
        
        # Combine all parameters
        all_parameters = {**base_parameters, **mode_parameters}
        
        # Create race monitor node
        race_monitor_node = Node(
            package='race_monitor',
            executable='race_monitor',
            name='race_monitor',
            parameters=[config_file_path, all_parameters],
            output='screen',
            emulate_tty=True,
        )
        
        return [race_monitor_node]
    
    return LaunchDescription([
        # Common arguments
        race_mode_arg,
        controller_name_arg,
        experiment_id_arg,
        
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
    print("Manual endurance mode:")
    print("  ros2 launch race_monitor race_monitor.launch.py race_mode:=manual")
    print()
    print("Custom crash detection:")
    print("  ros2 launch race_monitor race_monitor.launch.py race_mode:=crash \\")
    print("    max_stationary_time:=10.0 enable_collision_detection:=false")
    print()
    print("For full argument list:")
    print("  ros2 launch race_monitor race_monitor.launch.py --show-args")
