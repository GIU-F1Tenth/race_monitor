#!/usr/bin/env python3

"""
Race Monitor Launch File

Launches the race monitor node with comprehensive trajectory analysis and computational 
performance monitoring capabilities. This unified launch file provides configurable 
parameters for autonomous racing performance evaluation including real-time computational 
efficiency monitoring.

Usage:
    ros2 launch race_monitor race_monitor.launch.py
    
    # With custom parameters:
    ros2 launch race_monitor race_monitor.launch.py \
        controller_name:=my_controller \
        experiment_id:=session_001 \
        enable_trajectory_evaluation:=true \
        enable_computational_monitoring:=true

    # With multiple odometry sources:
    ros2 launch race_monitor race_monitor.launch.py \
        controller_name:=my_controller \
        experiment_id:=session_001 \
        odometry_topics:='["car_state/odom", "ekf/odometry"]' \
        control_command_topics:='["drive", "cmd_vel"]' \
        enable_computational_monitoring:=true

Available Parameters:
    controller_name (string): Name of the controller being tested
    experiment_id (string): Unique identifier for this experiment
    start_line_p1 (list): First point of start/finish line [x, y]
    start_line_p2 (list): Second point of start/finish line [x, y]
    required_laps (int): Number of laps required to complete race
    enable_trajectory_evaluation (bool): Enable advanced trajectory analysis
    enable_computational_monitoring (bool): Enable performance monitoring
    enable_horizon_mapper_reference (bool): Enable horizon mapper reference trajectory
    odometry_topics (list): Array of odometry topic names to monitor
    control_command_topics (list): Array of control command topics to monitor
    
Computational Monitoring Features:
    - Real-time control loop latency measurement
    - CPU and memory usage tracking
    - Processing efficiency calculation
    - Multi-topic support for complex systems
    - Performance statistics logging
    - Control frequency analysis
    
Output:
    The system generates comprehensive analysis data in:
    evaluation_results/research_data/{controller_name}/{experiment_id}/
    
    Performance monitoring topics:
    /race_monitor/control_loop_latency - Control loop latency in milliseconds
    /race_monitor/cpu_usage - Current CPU usage percentage
    /race_monitor/memory_usage - Current memory usage in MB
    /race_monitor/processing_efficiency - Processing efficiency score (0-1)
    
Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License 
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """
    Generate launch description for race monitor with comprehensive analysis capabilities.
    
    Returns:
        LaunchDescription: Configured launch description with all necessary parameters
    """

    # Race monitoring parameters
    start_line_p1_arg = DeclareLaunchArgument(
        'start_line_p1',
        default_value='[0.0, -1.0]',
        description='Start line point 1 [x, y]'
    )

    start_line_p2_arg = DeclareLaunchArgument(
        'start_line_p2',
        default_value='[0.0, 1.0]',
        description='Start line point 2 [x, y]'
    )

    required_laps_arg = DeclareLaunchArgument(
        'required_laps',
        default_value='5',
        description='Number of required laps'
    )

    debounce_time_arg = DeclareLaunchArgument(
        'debounce_time',
        default_value='2.0',
        description='Debounce time for lap detection'
    )

    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value='race_results.csv',
        description='Output CSV file name'
    )

    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='map',
        description='Frame ID for visualization'
    )

    # EVO integration parameters
    enable_trajectory_evaluation_arg = DeclareLaunchArgument(
        'enable_trajectory_evaluation',
        default_value='true',
        description='Whether to enable trajectory evaluation with evo'
    )

    # Evaluation timing options
    evaluation_interval_seconds_arg = DeclareLaunchArgument(
        'evaluation_interval_seconds',
        default_value='1.0',
        description='Time-based evaluation interval in seconds (0 = disable)'
    )

    evaluation_interval_laps_arg = DeclareLaunchArgument(
        'evaluation_interval_laps',
        default_value='0',
        description='Lap-based evaluation interval (0 = disable)'
    )

    evaluation_interval_meters_arg = DeclareLaunchArgument(
        'evaluation_interval_meters',
        default_value='0.0',
        description='Distance-based evaluation interval in meters (0 = disable)'
    )

    # Reference trajectory configuration
    reference_trajectory_file_arg = DeclareLaunchArgument(
        'reference_trajectory_file',
        default_value='horizon_mapper/horizon_mapper/ref_trajectory.csv',
        description='Path to reference trajectory file'
    )

    reference_trajectory_format_arg = DeclareLaunchArgument(
        'reference_trajectory_format',
        default_value='csv',
        description='Reference trajectory format (csv, tum, kitti)'
    )

    # Controller and experiment identification
    controller_name_arg = DeclareLaunchArgument(
        'controller_name',
        default_value='unknown_controller',
        description='Name of the controller being tested'
    )

    experiment_id_arg = DeclareLaunchArgument(
        'experiment_id',
        default_value='exp_001',
        description='Unique identifier for this experiment session'
    )

    # Trajectory analysis settings
    save_trajectories_arg = DeclareLaunchArgument(
        'save_trajectories',
        default_value='true',
        description='Whether to save trajectory files'
    )

    trajectory_output_directory_arg = DeclareLaunchArgument(
        'trajectory_output_directory',
        default_value='evaluation_results',
        description='Directory for trajectory evaluation output'
    )

    evaluate_smoothness_arg = DeclareLaunchArgument(
        'evaluate_smoothness',
        default_value='true',
        description='Whether to evaluate trajectory smoothness'
    )

    evaluate_consistency_arg = DeclareLaunchArgument(
        'evaluate_consistency',
        default_value='true',
        description='Whether to evaluate trajectory consistency'
    )

    # Graph generation settings
    auto_generate_graphs_arg = DeclareLaunchArgument(
        'auto_generate_graphs',
        default_value='true',
        description='Whether to automatically generate EVO plots'
    )

    graph_output_directory_arg = DeclareLaunchArgument(
        'graph_output_directory',
        default_value='evaluation_results/graphs',
        description='Directory for graph output'
    )

    graph_formats_arg = DeclareLaunchArgument(
        'graph_formats',
        default_value='["png", "pdf"]',
        description='Graph output formats'
    )

    # Plot settings
    plot_figsize_arg = DeclareLaunchArgument(
        'plot_figsize',
        default_value='[12.0, 8.0]',
        description='Figure size [width, height] for plots'
    )

    plot_dpi_arg = DeclareLaunchArgument(
        'plot_dpi',
        default_value='300',
        description='DPI for saved plot images'
    )

    plot_style_arg = DeclareLaunchArgument(
        'plot_style',
        default_value='seaborn',
        description='Matplotlib style for plots'
    )

    plot_color_scheme_arg = DeclareLaunchArgument(
        'plot_color_scheme',
        default_value='viridis',
        description='Color scheme for multiple trajectories'
    )

    # Graph types to generate
    generate_trajectory_plots_arg = DeclareLaunchArgument(
        'generate_trajectory_plots',
        default_value='true',
        description='Generate 2D/3D trajectory visualization'
    )

    generate_xyz_plots_arg = DeclareLaunchArgument(
        'generate_xyz_plots',
        default_value='true',
        description='Generate X, Y, Z position over time plots'
    )

    generate_rpy_plots_arg = DeclareLaunchArgument(
        'generate_rpy_plots',
        default_value='true',
        description='Generate Roll, Pitch, Yaw over time plots'
    )

    generate_speed_plots_arg = DeclareLaunchArgument(
        'generate_speed_plots',
        default_value='true',
        description='Generate velocity analysis plots'
    )

    generate_error_plots_arg = DeclareLaunchArgument(
        'generate_error_plots',
        default_value='true',
        description='Generate APE/RPE error analysis plots'
    )

    generate_metrics_plots_arg = DeclareLaunchArgument(
        'generate_metrics_plots',
        default_value='true',
        description='Generate smoothness/consistency over time plots'
    )

    # Horizon mapper reference trajectory settings
    enable_horizon_mapper_reference_arg = DeclareLaunchArgument(
        'enable_horizon_mapper_reference',
        default_value='true',
        description='Whether to enable horizon mapper reference trajectory interface'
    )

    horizon_mapper_reference_topic_arg = DeclareLaunchArgument(
        'horizon_mapper_reference_topic',
        default_value='/horizon_mapper/reference_trajectory',
        description='Topic for horizon mapper reference trajectory messages'
    )



    # Computational performance monitoring parameters
    enable_computational_monitoring_arg = DeclareLaunchArgument(
        'enable_computational_monitoring',
        default_value='false',
        description='Enable computational performance monitoring'
    )

    odometry_topics_arg = DeclareLaunchArgument(
        'odometry_topics',
        default_value='["car_state/odom"]',
        description='Array of odometry topic names to monitor'
    )

    control_command_topics_arg = DeclareLaunchArgument(
        'control_command_topics',
        default_value='["drive"]',
        description='Array of control command topic names to monitor'
    )

    monitoring_window_size_arg = DeclareLaunchArgument(
        'monitoring_window_size',
        default_value='100',
        description='Number of performance samples to keep in memory'
    )

    cpu_monitoring_interval_arg = DeclareLaunchArgument(
        'cpu_monitoring_interval',
        default_value='0.1',
        description='CPU monitoring interval in seconds'
    )

    enable_performance_logging_arg = DeclareLaunchArgument(
        'enable_performance_logging',
        default_value='true',
        description='Enable periodic performance logging'
    )

    performance_log_interval_arg = DeclareLaunchArgument(
        'performance_log_interval',
        default_value='5.0',
        description='Performance logging interval in seconds'
    )

    # Configuration file path
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('race_monitor'),
            'config',
            'race_monitor.yaml'
        ]),
        description='Path to the race monitor configuration file'
    )

    # Single integrated Race Monitor Node
    race_monitor_node = Node(
        package='race_monitor',
        executable='race_monitor',
        name='race_monitor',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                # Allow command line overrides
                'controller_name': LaunchConfiguration('controller_name'),
                'experiment_id': LaunchConfiguration('experiment_id'),
            }
        ]
    )

    # Create launch description
    return LaunchDescription([
        # Configuration file argument
        config_file_arg,
        
        # Most commonly overridden arguments (can override config file values)
        controller_name_arg,
        experiment_id_arg,

        # Single integrated node
        race_monitor_node,
    ])
