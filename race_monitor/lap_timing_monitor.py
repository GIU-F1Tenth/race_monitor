#!/usr/bin/env python3

"""
Lap Timing Monitor

Professional race monitoring system for autonomous vehicle trajectory analysis and
performance evaluation. Provides real-time lap timing, comprehensive trajectory
recording, and research-grade performance metrics using the EVO library.

Core Features:
    - Real-time lap detection and precise timing
    - Comprehensive trajectory recording and analysis
    - Advanced metrics calculation using EVO library
    - Multi-format data export (JSON, CSV, TUM, Pickle)
    - Research-ready statistical analysis
    - Computational performance monitoring
    - Trajectory filtering and smoothing algorithms

ROS2 Interface:
    Subscribed Topics:
        - /car_state/odom (nav_msgs/Odometry): Vehicle odometry data
        - /clicked_point (geometry_msgs/PointStamped): Manual start/finish line setup

    Published Topics:
        - /race_monitor/lap_count (std_msgs/Int32): Current lap number
        - /race_monitor/lap_time (std_msgs/Float32): Last completed lap time
        - /race_monitor/race_status (std_msgs/String): Current race status
        - /race_monitor/race_running (std_msgs/Bool): Race active status
        - /visualization_marker (visualization_msgs/Marker): RViz visualization

Key Parameters:
    - controller_name (string): Controller identifier for experiments
    - experiment_id (string): Unique experiment session identifier
    - enable_trajectory_evaluation (bool): Enable/disable trajectory analysis
    - enable_advanced_metrics (bool): Enable comprehensive metric calculation
    - save_trajectories (bool): Save trajectory data to files

License: MIT
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, Bool, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from giu_f1t_interfaces.msg import VehicleState, ConstrainedVehicleState, VehicleStateArray, ConstrainedVehicleStateArray

import numpy as np
import tf_transformations
import csv
import os
import time
from datetime import datetime
import math
import json
import sys
import time
import psutil
import threading
from collections import deque


def time_to_nanoseconds(time_obj):
    """Convert a time object to nanoseconds."""
    if hasattr(time_obj, 'nanoseconds'):
        # rclpy.time.Time object
        return time_obj.nanoseconds
    elif hasattr(time_obj, 'sec') and hasattr(time_obj, 'nanosec'):
        # builtin_interfaces.msg.Time object
        return time_obj.sec * 1e9 + time_obj.nanosec
    else:
        raise ValueError(f"Unknown time object type: {type(time_obj)}")


# Please ensure the 'evo' library is installed and available in your PYTHONPATH.
# Add EVO library to Python path
evo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'evo')
if os.path.exists(evo_path) and evo_path not in sys.path:
    sys.path.insert(0, evo_path)

# Import evo modules
try:
    from evo.core import trajectory, metrics, sync, transformations, filters, geometry, lie_algebra, result
    from evo.tools import file_interface, pandas_bridge, settings
    from evo.core.units import Unit, METER_SCALE_FACTORS, ANGLE_UNITS
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False

# Import our enhanced EVO plotter and research evaluator
try:
    from .visualization_engine import EVOPlotter
    EVO_PLOTTER_AVAILABLE = True
except ImportError:
    EVO_PLOTTER_AVAILABLE = False

try:
    from .trajectory_analyzer import ResearchTrajectoryEvaluator, create_research_evaluator
    RESEARCH_EVALUATOR_AVAILABLE = True
except ImportError:
    RESEARCH_EVALUATOR_AVAILABLE = False

try:
    from .race_evaluator import RaceEvaluator, create_race_evaluator
    RACE_EVALUATOR_AVAILABLE = True
except ImportError:
    RACE_EVALUATOR_AVAILABLE = False


class RaceMonitor(Node):
    """Race monitoring node with integrated trajectory evaluation

    Provides lap counting, timing, and comprehensive trajectory analysis
    for autonomous racing systems using the EVO library.
    """

    def __init__(self):
        super().__init__('race_monitor',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.get_logger().info('Race Monitor node started')

        # Declare all parameters matching the config file structure
        self._declare_all_parameters()

        # Read parameters
        self.start_line_p1 = np.array(self.get_parameter('start_line_p1').value, dtype=float)
        self.start_line_p2 = np.array(self.get_parameter('start_line_p2').value, dtype=float)
        self.required_laps = int(self.get_parameter('required_laps').value)
        self.debounce_time = float(self.get_parameter('debounce_time').value)
        self.output_file = str(self.get_parameter('output_file').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        # Controller and experiment identification
        self.controller_name = str(self.get_parameter('controller_name').value)
        self.experiment_id = str(self.get_parameter('experiment_id').value)

        # EVO parameters
        self.enable_trajectory_evaluation = self.get_parameter('enable_trajectory_evaluation').value

        # Evaluation timing options
        self.evaluation_interval_seconds = float(self.get_parameter('evaluation_interval_seconds').value)
        self.evaluation_interval_laps = int(self.get_parameter('evaluation_interval_laps').value)
        self.evaluation_interval_meters = float(self.get_parameter('evaluation_interval_meters').value)

        # Reference trajectory configuration
        self.reference_trajectory_file = str(self.get_parameter('reference_trajectory_file').value)
        self.reference_trajectory_format = str(self.get_parameter('reference_trajectory_format').value)

        # Trajectory analysis settings
        self.save_trajectories = self.get_parameter('save_trajectories').value
        self.trajectory_output_directory = str(self.get_parameter('trajectory_output_directory').value)

        self.save_horizon_reference = self.get_parameter('save_horizon_reference').value
        self.evaluate_smoothness = self.get_parameter('evaluate_smoothness').value
        self.evaluate_consistency = self.get_parameter('evaluate_consistency').value

        # Graph generation settings
        self.auto_generate_graphs = self.get_parameter('auto_generate_graphs').value
        self.graph_output_directory = str(self.get_parameter('graph_output_directory').value)
        self.graph_formats = self.get_parameter('graph_formats').value

        # Graph types to generate
        self.generate_trajectory_plots = self.get_parameter('generate_trajectory_plots').value
        self.generate_xyz_plots = self.get_parameter('generate_xyz_plots').value
        self.generate_rpy_plots = self.get_parameter('generate_rpy_plots').value
        self.generate_speed_plots = self.get_parameter('generate_speed_plots').value
        self.generate_error_plots = self.get_parameter('generate_error_plots').value
        self.generate_metrics_plots = self.get_parameter('generate_metrics_plots').value

        # Plot configuration settings
        self.plot_figsize = self.get_parameter('plot_figsize').value
        self.plot_dpi = self.get_parameter('plot_dpi').value
        self.plot_style = str(self.get_parameter('plot_style').value)
        self.plot_color_scheme = str(self.get_parameter('plot_color_scheme').value)

        # Computational Performance Monitoring parameters
        self.enable_computational_monitoring = self.get_parameter('enable_computational_monitoring').value

        # Race Evaluation System parameters
        self.enable_race_evaluation = self.get_parameter('enable_race_evaluation').value
        self.grading_strictness = str(self.get_parameter('grading_strictness').value)
        self.enable_recommendations = self.get_parameter('enable_recommendations').value
        self.enable_comparison = self.get_parameter('enable_comparison').value
        self.auto_increment_experiment = self.get_parameter('auto_increment_experiment').value

        # Get multiple topics configuration with simplified string-based approach
        try:
            odometry_topics_param = self.get_parameter('odometry_topics').value
            if isinstance(odometry_topics_param, list):
                # Simple string list approach
                self.odometry_topics_config = []
                for topic in odometry_topics_param:
                    if isinstance(topic, str):
                        # Determine message type based on topic name
                        if 'pose' in topic.lower() and 'cov' in topic.lower():
                            msg_type = 'geometry_msgs/PoseWithCovarianceStamped'
                        else:
                            msg_type = 'nav_msgs/Odometry'

                        self.odometry_topics_config.append({
                            'topic': topic,
                            'type': msg_type,
                            'enabled': True
                        })
                    elif isinstance(topic, dict):
                        # Handle dict format if provided
                        topic_name = topic.get('name', topic.get('topic', ''))
                        msg_type = topic.get('message_type', topic.get('type', 'nav_msgs/Odometry'))
                        enabled = topic.get('enabled', True)

                        self.odometry_topics_config.append({
                            'topic': topic_name,
                            'type': msg_type,
                            'enabled': enabled
                        })
            else:
                # Use default configuration
                self.odometry_topics_config = [
                    {'topic': 'car_state/odom', 'type': 'nav_msgs/Odometry', 'enabled': True}
                ]
        except Exception as e:
            self.get_logger().error(f"Error parsing odometry_topics parameter: {e}")
            self.odometry_topics_config = [
                {'topic': 'car_state/odom', 'type': 'nav_msgs/Odometry', 'enabled': True}
            ]

        try:
            control_topics_param = self.get_parameter('control_command_topics').value
            if isinstance(control_topics_param, list):
                # Simple string list approach
                self.control_command_topics_config = []
                for topic in control_topics_param:
                    if isinstance(topic, str):
                        # Determine message type based on topic name
                        if 'cmd_vel' in topic.lower() or 'twist' in topic.lower():
                            msg_type = 'twist'
                        else:
                            msg_type = 'ackermann'

                        self.control_command_topics_config.append({
                            'topic': topic,
                            'type': msg_type,
                            'enabled': True
                        })
                    elif isinstance(topic, dict):
                        # Handle dict format if provided
                        topic_name = topic.get('name', topic.get('topic', ''))
                        msg_type = topic.get('message_type', topic.get('type', 'ackermann'))
                        enabled = topic.get('enabled', True)

                        # Convert message type to simple format
                        if 'AckermannDriveStamped' in msg_type:
                            simple_type = 'ackermann'
                        elif 'Twist' in msg_type:
                            simple_type = 'twist'
                        else:
                            simple_type = msg_type

                        self.control_command_topics_config.append({
                            'topic': topic_name,
                            'type': simple_type,
                            'enabled': enabled
                        })
            else:
                # Use default configuration
                self.control_command_topics_config = [
                    {'topic': '/drive', 'type': 'ackermann', 'enabled': True}
                ]
        except Exception as e:
            self.get_logger().error(f"Error parsing control_command_topics parameter: {e}")
            self.control_command_topics_config = [
                {'topic': '/drive', 'type': 'ackermann', 'enabled': True}
            ]

        # Legacy single topic support (for backward compatibility)
        self.control_command_topic = str(self.get_parameter('control_command_topic').value)
        self.control_command_type = str(self.get_parameter('control_command_type').value)
        self.monitoring_window_size = int(self.get_parameter('monitoring_window_size').value)
        self.cpu_monitoring_interval = float(self.get_parameter('cpu_monitoring_interval').value)
        self.enable_performance_logging = self.get_parameter('enable_performance_logging').value
        self.performance_log_interval = float(self.get_parameter('performance_log_interval').value)

        self.get_logger().info(f"Initial start line: P1={self.start_line_p1}, P2={self.start_line_p2}")

        if self.enable_trajectory_evaluation:
            if not EVO_AVAILABLE:
                self.get_logger().warn("EVO library not available - trajectory evaluation disabled")
                self.enable_trajectory_evaluation = False

        # Race state
        self.lap_count = 0
        self.race_running = False
        self.race_started = False
        self.lap_times = []
        self.race_start_time = None
        self.lap_start_time = None
        self.last_crossing_time = None
        self.car_running = False
        self.race_finished = False

        # Position tracking
        self.current_position = np.array([0.0, 0.0])
        self.last_position = np.array([0.0, 0.0])
        self.current_heading = 0.0
        self.position_initialized = False

        # Computational Performance Monitoring (only if enabled)
        self.computational_monitoring_initialized = False
        if self.enable_computational_monitoring:
            try:
                # Initialize computational monitoring variables
                self.odom_receive_times = deque(maxlen=self.monitoring_window_size)
                self.control_send_times = deque(maxlen=self.monitoring_window_size)
                self.control_loop_latencies = deque(maxlen=self.monitoring_window_size)
                self.cpu_usage_history = deque(maxlen=self.monitoring_window_size)
                self.memory_usage_history = deque(maxlen=self.monitoring_window_size)

                # Control loop timing tracking (support multiple topics)
                self.pending_odom_timestamps = {}  # topic_name -> timestamp
                self.last_odom_timestamp = None
                self.last_control_timestamp = None

                # Performance statistics
                self.total_control_cycles = 0
                self.total_processing_time = 0.0
                self.max_latency = 0.0
                self.min_latency = float('inf')
                self.latency_violations = 0  # Count of cycles exceeding threshold

                # CPU/Memory monitoring
                self.process = psutil.Process()
                self.system_cpu_count = psutil.cpu_count()

                # Performance logging
                self.performance_log_data = []
                self.last_performance_log_time = time.time()

                # Topic tracking
                self.active_odom_topics = []
                self.active_control_topics = []

                self.computational_monitoring_initialized = True

            except Exception as e:
                self.get_logger().error(f"Failed to initialize computational monitoring: {e}")
                self.enable_computational_monitoring = False

        # EVO trajectory tracking (only if enabled)
        if self.enable_trajectory_evaluation and EVO_AVAILABLE:
            # Create trajectory output directory
            os.makedirs(self.trajectory_output_directory, exist_ok=True)

            # Trajectory storage
            self.current_lap_trajectory = []
            self.lap_trajectories = {}  # lap_number -> trajectory
            self.reference_trajectory = None
            self.lap_metrics = {}  # lap_number -> metrics
            self.last_lap_number = 0  # Initialize last lap number
            self.evo_plots_generated = False  # Flag to prevent re-generating EVO plots
            self.horizon_reference_saved = False  # Flag to track if horizon reference has been saved

            # Initialize EVO plotter if available
            if EVO_PLOTTER_AVAILABLE and self.auto_generate_graphs:
                self.get_logger().info("Initializing EVO plotter...")
                # Create configuration dict for plotter
                plotter_config = {
                    'auto_generate_graphs': self.auto_generate_graphs,
                    'graph_output_directory': self.graph_output_directory,
                    'graph_formats': self.graph_formats,
                    'generate_trajectory_plots': self.generate_trajectory_plots,
                    'generate_xyz_plots': self.generate_xyz_plots,
                    'generate_rpy_plots': self.generate_rpy_plots,
                    'generate_speed_plots': self.generate_speed_plots,
                    'generate_error_plots': self.generate_error_plots,
                    'generate_metrics_plots': self.generate_metrics_plots,
                    'plot_figsize': self.plot_figsize,
                    'plot_dpi': self.plot_dpi,
                    'plot_style': self.plot_style,
                    'plot_color_scheme': self.plot_color_scheme
                }
                self.evo_plotter = EVOPlotter(plotter_config)
            else:
                self.evo_plotter = None

            # Load reference trajectory if provided (regardless of plotter availability)
            if self.reference_trajectory_file and os.path.exists(self.reference_trajectory_file):
                if self.evo_plotter:
                    self.evo_plotter.load_reference_trajectory(
                        self.reference_trajectory_file,
                        self.reference_trajectory_format
                    )
                # Also load for legacy compatibility
                self.load_reference_trajectory()

            # Initialize Research Evaluator for comprehensive analysis
            if RESEARCH_EVALUATOR_AVAILABLE:
                research_config = {
                    'controller_name': self.get_parameter('controller_name').get_parameter_value().string_value if self.has_parameter('controller_name') else 'unknown_controller',
                    'experiment_id': self.get_parameter('experiment_id').get_parameter_value().string_value if self.has_parameter('experiment_id') else 'exp_001',
                    'test_description': self.get_parameter('test_description').get_parameter_value().string_value if self.has_parameter('test_description') else 'Research experiment',
                    'trajectory_output_directory': self.trajectory_output_directory,
                    'enable_advanced_metrics': self.get_parameter('enable_advanced_metrics').get_parameter_value().bool_value if self.has_parameter('enable_advanced_metrics') else True,
                    'enable_geometric_analysis': self.get_parameter('enable_geometric_analysis').get_parameter_value().bool_value if self.has_parameter('enable_geometric_analysis') else True,
                    'calculate_all_statistics': self.get_parameter('calculate_all_statistics').get_parameter_value().bool_value if self.has_parameter('calculate_all_statistics') else True,
                    'apply_trajectory_filtering': self.get_parameter('apply_trajectory_filtering').get_parameter_value().bool_value if self.has_parameter('apply_trajectory_filtering') else True,
                    'save_intermediate_results': self.get_parameter('save_intermediate_results').get_parameter_value().bool_value if self.has_parameter('save_intermediate_results') else True,
                    'pose_relations': ['translation_part', 'rotation_part', 'full_transformation'],
                    'statistics_types': ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'],
                    'filter_types': ['motion', 'distance'],
                    'filter_parameters': {'motion_threshold': 0.1, 'angle_threshold': 0.1, 'distance_threshold': 0.05},
                    'output_formats': ['json', 'csv', 'pickle']
                }

                try:
                    self.research_evaluator = create_research_evaluator(research_config)

                    # Set reference trajectory if available
                    if self.reference_trajectory_file and os.path.exists(self.reference_trajectory_file):
                        self.research_evaluator.set_reference_trajectory(
                            self.reference_trajectory_file,
                            self.reference_trajectory_format
                        )
                except Exception as e:
                    self.get_logger().error(f"Failed to initialize Research Evaluator: {e}")
                    self.research_evaluator = None
            else:
                self.research_evaluator = None

        # Initialize Race Evaluator (only if enabled)
        self.race_evaluator = None
        if self.enable_race_evaluation and RACE_EVALUATOR_AVAILABLE:
            race_eval_config = {
                'controller_name': self.controller_name,
                'trajectory_output_directory': self.trajectory_output_directory,
                'grading_strictness': self.grading_strictness,
                'enable_recommendations': self.enable_recommendations,
                'enable_comparison': self.enable_comparison,
                'auto_increment_experiment': self.auto_increment_experiment
            }

            try:
                self.race_evaluator = create_race_evaluator(race_eval_config)
            except Exception as e:
                self.get_logger().error(f"Failed to initialize Race Evaluator: {e}")
                self.race_evaluator = None

        # Clicked point handling
        self.pending_point = None  # holds first clicked point until second is given

        # Subscribers
        # Create multiple odometry subscribers based on configuration
        self.odom_subscribers = {}

        # Always create the default odometry subscriber
        self.odom_sub = self.create_subscription(Odometry, 'car_state/odom', self.odom_callback, 20)
        self.odom_subscribers['car_state/odom'] = self.odom_sub

        # Create additional odometry subscribers if computational monitoring is enabled
        if self.enable_computational_monitoring and self.computational_monitoring_initialized:
            for odom_config in self.odometry_topics_config:
                if odom_config.get('enabled', False):
                    topic = odom_config['topic']
                    msg_type = odom_config['type']

                    # Skip if already created
                    if topic in self.odom_subscribers:
                        continue

                    if msg_type == 'nav_msgs/Odometry':
                        def callback(msg, topic_name=topic): return self.odom_callback(msg, topic_name)
                        sub = self.create_subscription(Odometry, topic, callback, 20)
                    elif msg_type == 'geometry_msgs/PoseWithCovarianceStamped':
                        def callback(msg, topic_name=topic): return self.pose_with_cov_callback(msg, topic_name)
                        sub = self.create_subscription(PoseWithCovarianceStamped, topic, callback, 20)
                    else:
                        self.get_logger().warn(f"Unsupported odometry message type: {msg_type} for topic {topic}")
                        continue

                    self.odom_subscribers[topic] = sub
                    self.active_odom_topics.append(topic)
                    self.get_logger().info(f"Subscribed to odometry topic: {topic} ({msg_type})")

        # Clicked point subscriber
        self.clicked_point_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.clicked_point_callback, 10)

        # Control command subscribers for computational monitoring
        self.control_subscribers = {}
        if self.enable_computational_monitoring and self.computational_monitoring_initialized:
            # Create subscribers for multiple control topics
            for control_config in self.control_command_topics_config:
                if control_config.get('enabled', False):
                    topic = control_config['topic']
                    cmd_type = control_config['type']

                    if cmd_type == 'ackermann':
                        def callback(msg, topic_name=topic): return self.control_command_callback(msg, topic_name)
                        sub = self.create_subscription(AckermannDriveStamped, topic, callback, 20)
                    elif cmd_type == 'twist':
                        def callback(msg, topic_name=topic): return self.control_command_callback(msg, topic_name)
                        sub = self.create_subscription(Twist, topic, callback, 20)
                    else:
                        self.get_logger().warn(f"Unsupported control command type: {cmd_type} for topic {topic}")
                        continue

                    self.control_subscribers[topic] = sub
                    self.active_control_topics.append(topic)
                    self.get_logger().info(f"Subscribed to control topic: {topic} ({cmd_type})")

            # Fallback to legacy single topic if no topics are configured
            if not self.active_control_topics:
                if self.control_command_type == 'ackermann':
                    self.control_sub = self.create_subscription(
                        AckermannDriveStamped, self.control_command_topic,
                        lambda msg: self.control_command_callback(msg, self.control_command_topic), 20)
                elif self.control_command_type == 'twist':
                    self.control_sub = self.create_subscription(
                        Twist, self.control_command_topic,
                        lambda msg: self.control_command_callback(msg, self.control_command_topic), 20)
                else:
                    self.get_logger().warn(f"Unsupported control command type: {self.control_command_type}")
                    self.enable_computational_monitoring = False

                if self.enable_computational_monitoring:
                    self.active_control_topics.append(self.control_command_topic)
                    self.get_logger().info(
                        f"Using legacy control topic: {self.control_command_topic} ({self.control_command_type})")

        # Horizon mapper reference trajectory subscriber (for EVO analysis)
        if not self.has_parameter('enable_horizon_mapper_reference'):
            self.declare_parameter('enable_horizon_mapper_reference', True)
        if not self.has_parameter('horizon_mapper_reference_topic'):
            self.declare_parameter('horizon_mapper_reference_topic', '/horizon_mapper/reference_trajectory')
        if not self.has_parameter('use_complete_reference_path'):
            self.declare_parameter('use_complete_reference_path', True)
        if not self.has_parameter('horizon_mapper_path_topic'):
            self.declare_parameter('horizon_mapper_path_topic', '/horizon_mapper/reference_path')

        self.enable_horizon_mapper_reference = self.get_parameter('enable_horizon_mapper_reference').value
        self.horizon_mapper_reference_topic = str(self.get_parameter('horizon_mapper_reference_topic').value)
        self.use_complete_reference_path = self.get_parameter('use_complete_reference_path').value
        self.horizon_mapper_path_topic = str(self.get_parameter('horizon_mapper_path_topic').value)

        if self.enable_horizon_mapper_reference:
            if self.use_complete_reference_path:
                # Subscribe to complete reference path (nav_msgs/Path)
                from nav_msgs.msg import Path
                self.reference_path_sub = self.create_subscription(
                    Path, self.horizon_mapper_path_topic, self.reference_path_callback, 20)
                self.get_logger().info(
                    f"Horizon mapper reference enabled - subscribing to complete path: {self.horizon_mapper_path_topic}")
            else:
                # Subscribe to reference trajectory points (VehicleStateArray)
                self.reference_trajectory_sub = self.create_subscription(
                    VehicleStateArray, self.horizon_mapper_reference_topic, self.reference_trajectory_callback, 20)
                self.get_logger().info(
                    f"Horizon mapper reference enabled - subscribing to trajectory points: {self.horizon_mapper_reference_topic}")

        # Publishers
        self.lap_count_pub = self.create_publisher(Int32, '/race_monitor/lap_count', 10)
        self.lap_time_pub = self.create_publisher(Float32, '/race_monitor/lap_time', 10)
        self.best_lap_time_pub = self.create_publisher(Float32, '/race_monitor/best_lap_time', 10)
        self.race_running_pub = self.create_publisher(Bool, '/race_monitor/race_running', 10)
        self.start_line_marker_pub = self.create_publisher(Marker, '/race_monitor/start_line_marker', 10)
        self.race_state_pub = self.create_publisher(String, '/race_monitor/state', 10)

        # Add raceline visualization publisher
        from visualization_msgs.msg import MarkerArray
        self.raceline_marker_pub = self.create_publisher(MarkerArray, '/race_monitor/raceline_markers', 10)

        # EVO publishers (only if enabled)
        if self.enable_trajectory_evaluation and EVO_AVAILABLE:
            self.trajectory_metrics_pub = self.create_publisher(
                String, '/race_monitor/trajectory_metrics', 10)
            self.smoothness_score_pub = self.create_publisher(
                Float32, '/race_monitor/smoothness', 10)
            self.consistency_score_pub = self.create_publisher(
                Float32, '/race_monitor/consistency', 10)

        # Computational Performance Monitoring publishers (only if enabled)
        if self.enable_computational_monitoring and self.computational_monitoring_initialized:
            self.control_loop_latency_pub = self.create_publisher(
                Float32, '/race_monitor/control_loop_latency', 10)
            self.cpu_usage_pub = self.create_publisher(
                Float32, '/race_monitor/cpu_usage', 10)
            self.memory_usage_pub = self.create_publisher(
                Float32, '/race_monitor/memory_usage', 10)
            self.processing_efficiency_pub = self.create_publisher(
                Float32, '/race_monitor/processing_efficiency', 10)
            self.performance_stats_pub = self.create_publisher(
                String, '/race_monitor/performance_stats', 10)

        # Timers
        self.status_timer = self.create_timer(0.1, self.publish_race_status)
        self.marker_timer = self.create_timer(1.0, self.publish_start_line_marker)
        self.raceline_marker_timer = self.create_timer(2.0, self.publish_raceline_markers)

        # EVO evaluation timer (only if enabled)
        if self.enable_trajectory_evaluation and EVO_AVAILABLE:
            # Determine evaluation interval based on configuration
            if self.evaluation_interval_seconds > 0:
                # Time-based evaluation
                self.evaluation_timer = self.create_timer(
                    self.evaluation_interval_seconds, self.evaluate_current_trajectory)
                self.get_logger().info(f"Time-based evaluation every {self.evaluation_interval_seconds} seconds")
            elif self.evaluation_interval_laps > 0:
                # Lap-based evaluation (will be triggered in lap completion)
                self.evaluation_timer = None
                self.get_logger().info(f"Lap-based evaluation every {self.evaluation_interval_laps} laps")
            elif self.evaluation_interval_meters > 0:
                # Distance-based evaluation (will be triggered in odom callback)
                self.evaluation_timer = None
                self.get_logger().info(f"Distance-based evaluation every {self.evaluation_interval_meters} meters")
            else:
                # No real-time evaluation, only on lap completion
                self.evaluation_timer = None
                self.get_logger().info("Evaluation only on lap completion")

        # Computational Performance Monitoring timer (only if enabled)
        if self.enable_computational_monitoring and self.computational_monitoring_initialized:
            self.cpu_monitoring_timer = self.create_timer(
                self.cpu_monitoring_interval, self.monitor_computational_performance)
            if self.enable_performance_logging:
                self.performance_logging_timer = self.create_timer(
                    self.performance_log_interval, self.log_performance_stats)

        # Remember last published line to avoid re-publishing identical markers unnecessarily
        self._last_line = (None, None)

        self.get_logger().info('Enhanced Race Monitor initialized. Use RViz Publish Point to set start/finish line (click two points).')

    def _declare_all_parameters(self):
        """Declare all parameters matching the config file structure."""
        # ========================================
        # RACE MONITORING PARAMETERS
        # ========================================
        self.declare_parameter('start_line_p1', [0.0, -1.0])
        self.declare_parameter('start_line_p2', [1.0, 0.0])
        self.declare_parameter('required_laps', 20)
        self.declare_parameter('debounce_time', 2.0)
        self.declare_parameter('output_file', 'race_results.csv')
        self.declare_parameter('frame_id', 'map')

        # ========================================
        # RACE ENDING CONDITIONS
        # ========================================
        self.declare_parameter('race_ending_mode', 'lap_complete')

        # Crash detection parameters
        self.declare_parameter('crash_detection.enable_crash_detection', True)
        self.declare_parameter('crash_detection.max_stationary_time', 5.0)
        self.declare_parameter('crash_detection.min_velocity_threshold', 0.1)
        self.declare_parameter('crash_detection.max_odometry_timeout', 3.0)
        self.declare_parameter('crash_detection.enable_collision_detection', True)
        self.declare_parameter('crash_detection.collision_velocity_threshold', 2.0)
        self.declare_parameter('crash_detection.collision_detection_window', 0.5)

        # Manual mode parameters
        self.declare_parameter('manual_mode.save_intermediate_results', True)
        self.declare_parameter('manual_mode.save_interval', 30.0)
        self.declare_parameter('manual_mode.max_race_duration', 0)

        # ========================================
        # RESEARCH & ANALYSIS PARAMETERS
        # ========================================
        self.declare_parameter('controller_name', 'custom_controller')
        self.declare_parameter('experiment_id', 'exp_001')
        self.declare_parameter('test_description', 'Controller performance evaluation and analysis')

        # Advanced metrics parameters
        self.declare_parameter('enable_advanced_metrics', True)
        self.declare_parameter('calculate_all_statistics', True)
        self.declare_parameter('analyze_rotation_errors', True)
        self.declare_parameter('enable_geometric_analysis', True)
        self.declare_parameter('enable_filtering_analysis', True)

        # Analysis evaluation modes
        self.declare_parameter('detailed_lap_analysis', True)
        self.declare_parameter('comparative_analysis', True)
        self.declare_parameter('statistical_significance', True)

        # ========================================
        # EVO INTEGRATION PARAMETERS
        # ========================================
        self.declare_parameter('enable_trajectory_evaluation', True)
        self.declare_parameter('evaluation_interval_seconds', 0)
        self.declare_parameter('evaluation_interval_laps', 1)
        self.declare_parameter('evaluation_interval_meters', 0.0)

        # ========================================
        # REFERENCE TRAJECTORY CONFIGURATION
        # ========================================
        self.declare_parameter('reference_trajectory_file', 'horizon_mapper/horizon_mapper/ref_trajectory.csv')
        self.declare_parameter('reference_trajectory_format', 'csv')
        self.declare_parameter('enable_horizon_mapper_reference', False)
        self.declare_parameter('horizon_mapper_reference_topic', '/horizon_mapper/reference_trajectory')
        self.declare_parameter('use_complete_reference_path', True)
        self.declare_parameter('horizon_mapper_path_topic', '/horizon_mapper/reference_path')

        # ========================================
        # ADVANCED FILE OUTPUT FOR RESEARCH
        # ========================================
        self.declare_parameter('export_to_pandas', True)
        self.declare_parameter('save_detailed_statistics', True)
        self.declare_parameter('save_filtered_trajectories', True)
        self.declare_parameter('export_research_summary', True)
        self.declare_parameter('output_formats', ['csv', 'json', 'pickle', 'mat'])
        self.declare_parameter('include_timestamps', True)
        self.declare_parameter('save_intermediate_results', True)

        # ========================================
        # TRAJECTORY ANALYSIS SETTINGS
        # ========================================
        self.declare_parameter('save_trajectories', True)
        self.declare_parameter('trajectory_output_directory',
                               '/home/mohammedazab/ws/src/race_stack/race_monitor/race_monitor/evaluation_results')

        # Metrics to calculate
        self.declare_parameter('evaluate_smoothness', True)
        self.declare_parameter('evaluate_consistency', True)
        self.declare_parameter('evaluate_efficiency', True)
        self.declare_parameter('evaluate_aggressiveness', True)
        self.declare_parameter('evaluate_stability', True)

        # Advanced EVO metrics
        self.declare_parameter('pose_relations', ['translation_part', 'rotation_part', 'full_transformation'])
        self.declare_parameter('statistics_types', ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'])

        # Filtering options
        self.declare_parameter('apply_trajectory_filtering', True)
        self.declare_parameter('filter_types', ['motion', 'distance', 'angle'])
        self.declare_parameter('filter_parameters.motion_threshold', 0.1)
        self.declare_parameter('filter_parameters.distance_threshold', 0.05)
        self.declare_parameter('filter_parameters.angle_threshold', 0.1)

        # ========================================
        # GRAPH GENERATION SETTINGS
        # ========================================
        self.declare_parameter('auto_generate_graphs', True)
        self.declare_parameter(
            'graph_output_directory',
            '/home/mohammedazab/ws/src/race_stack/race_monitor/race_monitor/evaluation_results/graphs')
        self.declare_parameter('graph_formats', ['png', 'pdf'])

        # Plot appearance settings
        self.declare_parameter('plot_figsize', [12.0, 8.0])
        self.declare_parameter('plot_dpi', 300)
        self.declare_parameter('plot_style', 'seaborn')
        self.declare_parameter('plot_color_scheme', 'viridis')

        # Types of graphs to generate
        self.declare_parameter('generate_trajectory_plots', True)
        self.declare_parameter('generate_xyz_plots', True)
        self.declare_parameter('generate_rpy_plots', True)
        self.declare_parameter('generate_speed_plots', True)
        self.declare_parameter('generate_error_plots', True)
        self.declare_parameter('generate_metrics_plots', True)

        # ========================================
        # F1TENTH INTERFACE SETTINGS
        # ========================================
        self.declare_parameter('enable_f1tenth_interface', True)
        self.declare_parameter('f1tenth_vehicle_state_topic', '/vehicle_state')
        self.declare_parameter('f1tenth_constrained_state_topic', '/constrained_vehicle_state')

        # ========================================
        # COMPUTATIONAL PERFORMANCE MONITORING
        # ========================================
        self.declare_parameter('enable_computational_monitoring', False)

        # Odometry input topics
        self.declare_parameter('odometry_topics', ['car_state/odom'])

        # Control command output topics
        self.declare_parameter('control_command_topics', ['/drive'])

        # Performance monitoring configuration
        self.declare_parameter('monitoring_window_size', 100)
        self.declare_parameter('cpu_monitoring_interval', 0.1)
        self.declare_parameter('enable_performance_logging', True)
        self.declare_parameter('performance_log_interval', 5.0)

        # Performance thresholds
        self.declare_parameter('max_acceptable_latency_ms', 50.0)
        self.declare_parameter('target_control_frequency_hz', 50.0)
        self.declare_parameter('max_acceptable_cpu_usage', 80.0)
        self.declare_parameter('max_acceptable_memory_mb', 512.0)

        # ========================================
        # CUSTOM RACE EVALUATION SYSTEM
        # ========================================
        self.declare_parameter('enable_race_evaluation', True)

        # Race evaluation settings
        self.declare_parameter('race_evaluation.enable_export', True)
        self.declare_parameter('race_evaluation.auto_increment_experiment', True)
        self.declare_parameter('race_evaluation.include_recommendations', True)
        self.declare_parameter('race_evaluation.enable_comparison', True)
        self.declare_parameter('race_evaluation.grading_strictness', 'normal')

        # Evaluation focus areas
        self.declare_parameter('race_evaluation.focus_on_consistency', True)
        self.declare_parameter('race_evaluation.focus_on_trajectory_accuracy', True)
        self.declare_parameter('race_evaluation.focus_on_racing_efficiency', True)

        # Output preferences
        self.declare_parameter('race_evaluation.include_detailed_metrics', True)
        self.declare_parameter('race_evaluation.max_recommendations', 5)

    def load_reference_trajectory(self):
        """Load reference trajectory from file using evo"""
        if not self.enable_trajectory_evaluation or not EVO_AVAILABLE:
            return

        try:
            if self.reference_trajectory_format == 'tum':
                self.reference_trajectory = file_interface.read_tum_trajectory_file(
                    self.reference_trajectory_file)
            elif self.reference_trajectory_format == 'kitti':
                self.reference_trajectory = file_interface.read_kitti_poses_file(
                    self.reference_trajectory_file)
            elif self.reference_trajectory_format == 'csv':
                # Load CSV format (x, y, v, theta) and convert to EVO format
                data = np.loadtxt(self.reference_trajectory_file, delimiter=',', skiprows=1)
                x = data[:, 0]
                y = data[:, 1]
                v = data[:, 2] if data.shape[1] > 2 else np.zeros_like(x)
                theta = data[:, 3] if data.shape[1] > 3 else np.zeros_like(x)

                # Convert to EVO trajectory format
                timestamps = np.arange(len(x)) * 0.1  # Assuming 10Hz sampling

                # Convert to quaternions (assuming 2D motion)
                z = np.zeros_like(x)
                qx = np.zeros_like(x)
                qy = np.zeros_like(x)
                qz = np.sin(theta / 2)
                qw = np.cos(theta / 2)

                # Create EVO trajectory
                self.reference_trajectory = trajectory.PoseTrajectory3D(
                    positions_xyz=np.column_stack([x, y, z]),
                    orientations_quat_wxyz=np.column_stack([qw, qx, qy, qz]),
                    timestamps=timestamps
                )
            else:
                self.get_logger().warn(f"Unsupported reference trajectory format: {self.reference_trajectory_format}")
                return

            # Check the actual attribute name for EVO trajectory
            if hasattr(self.reference_trajectory, 'poses'):
                num_poses = len(self.reference_trajectory.poses)
            elif hasattr(self.reference_trajectory, 'positions_xyz'):
                num_poses = len(self.reference_trajectory.positions_xyz)
            else:
                num_poses = 'unknown'
            self.get_logger().info(f"Loaded reference trajectory with {num_poses} poses")
        except Exception as e:
            self.get_logger().error(f"Failed to load reference trajectory: {e}")

    def reference_trajectory_callback(self, msg):
        """Callback for horizon mapper reference trajectory messages"""
        if not self.enable_horizon_mapper_reference:
            return

        # Store the latest reference trajectory for EVO comparison
        if hasattr(self, 'evo_plotter') and self.evo_plotter:
            # Convert VehicleStateArray to EVO format
            reference_trajectory_data = []
            for vehicle_state in msg.states:
                reference_trajectory_data.append({
                    'x': vehicle_state.x,
                    'y': vehicle_state.y,
                    'theta': vehicle_state.theta,
                    'v': vehicle_state.v
                })

            # Update the reference trajectory in EVO plotter
            self.evo_plotter.update_reference_trajectory_from_horizon_mapper(reference_trajectory_data)

            # Save horizon mapper reference trajectory as TUM file for EVO analysis
            if (self.save_horizon_reference and not self.horizon_reference_saved and
                    len(reference_trajectory_data) > 0):
                self._save_horizon_reference_trajectory(reference_trajectory_data)
                self.horizon_reference_saved = True

    def reference_path_callback(self, msg):
        """Callback for horizon mapper complete reference path messages (nav_msgs/Path)"""
        if not self.enable_horizon_mapper_reference:
            return

        # Store the latest reference path for EVO comparison
        if hasattr(self, 'evo_plotter') and self.evo_plotter:
            # Convert nav_msgs/Path to EVO format
            reference_trajectory_data = []
            for pose_stamped in msg.poses:
                pose = pose_stamped.pose
                # Extract orientation (convert quaternion to yaw angle)
                orientation_q = pose.orientation
                import tf_transformations
                euler = tf_transformations.euler_from_quaternion([
                    orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
                theta = euler[2]  # yaw angle

                reference_trajectory_data.append({
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'theta': theta,
                    'v': 0.0  # Velocity not available in Path message
                })

            # Update the reference trajectory in EVO plotter
            self.evo_plotter.update_reference_trajectory_from_horizon_mapper(reference_trajectory_data)

            # Save horizon mapper reference path as TUM file for EVO analysis
            if (self.save_horizon_reference and not self.horizon_reference_saved and
                    len(reference_trajectory_data) > 0):
                self._save_horizon_reference_trajectory(reference_trajectory_data)
                self.horizon_reference_saved = True

    def odom_callback(self, msg, topic_name='car_state/odom'):
        """Callback for odometry messages"""
        # Computational monitoring: Record odometry receive time
        current_time = time.time()
        if self.enable_computational_monitoring and self.computational_monitoring_initialized:
            self.odom_receive_times.append(current_time)
            # Store timestamp per topic to handle multiple odometry sources
            self.pending_odom_timestamps[topic_name] = current_time

        # Update position tracking for lap detection (use primary topic)
        if topic_name == 'car_state/odom' or not self.position_initialized:
            self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

            if not self.position_initialized:
                self.last_position = self.current_position.copy()
                self.position_initialized = True
                return

            # Calculate heading
            dx = self.current_position[0] - self.last_position[0]
            dy = self.current_position[1] - self.last_position[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only update if there's significant movement
                self.current_heading = math.atan2(dy, dx)
                self.last_position = self.current_position.copy()

            # EVO trajectory tracking (only if enabled)
            if self.enable_trajectory_evaluation and EVO_AVAILABLE:
                # Add to current lap trajectory
                self.current_lap_trajectory.append({
                    'header': msg.header,
                    'pose': msg.pose.pose
                })

                # Check for distance-based evaluation
                if self.evaluation_interval_meters > 0:
                    distance_traveled = np.linalg.norm(self.current_position - self.last_position)
                    if distance_traveled >= self.evaluation_interval_meters:
                        self.get_logger().info(f"Distance-based evaluation triggered after {distance_traveled:.2f}m")
                        self.evaluate_current_trajectory()

            # Lap detection logic
            if self.detect_lap_crossing():
                self.handle_lap_crossing()

    def pose_with_cov_callback(self, msg, topic_name):
        """Callback for PoseWithCovarianceStamped messages"""
        # Convert to Odometry format for consistent processing
        odom_msg = Odometry()
        odom_msg.header = msg.header
        odom_msg.pose = msg.pose

        # Call the regular odometry callback
        self.odom_callback(odom_msg, topic_name)

    def detect_lap_crossing(self):
        """Detect if the vehicle has crossed the start/finish line"""
        if not self.position_initialized:
            return False

        # Calculate distance to start line
        line_vector = self.start_line_p2 - self.start_line_p1
        line_length = np.linalg.norm(line_vector)

        if line_length < 0.01:  # Line too short
            return False

        # Normalize line vector
        line_unit = line_vector / line_length

        # Vector from start line point 1 to current position
        to_position = self.current_position - self.start_line_p1

        # Project current position onto the line
        projection_length = np.dot(to_position, line_unit)

        # Check if projection is within line bounds
        if projection_length < 0 or projection_length > line_length:
            return False

        # Calculate perpendicular distance to line
        projected_point = self.start_line_p1 + projection_length * line_unit
        perpendicular_distance = np.linalg.norm(self.current_position - projected_point)

        # Check if we're close enough to the line (within 1 meter)
        if perpendicular_distance > 1.0:
            return False

        # Check if we're moving in the right direction (crossing from one side to the other)
        # We need to determine which side of the line we were on before
        if not hasattr(self, 'last_side_of_line'):
            self.last_side_of_line = self.get_side_of_line(self.last_position)
            return False

        current_side = self.get_side_of_line(self.current_position)

        # Check if we crossed from one side to the other
        if current_side != self.last_side_of_line and current_side != 0:
            self.last_side_of_line = current_side
            return True

        self.last_side_of_line = current_side
        return False

    def get_side_of_line(self, position):
        """Determine which side of the line a position is on"""
        # Vector from start line point 1 to position
        to_position = position - self.start_line_p1

        # Vector from start line point 1 to end line point 2
        line_vector = self.start_line_p2 - self.start_line_p1

        # Cross product to determine side
        cross_product = np.cross(line_vector, to_position)

        if abs(cross_product) < 0.01:  # On the line
            return 0
        elif cross_product > 0:  # Right side
            return 1
        else:  # Left side
            return -1

    def handle_lap_crossing(self):
        """Handle when a lap crossing is detected"""
        current_time = self.get_clock().now()

        # Debounce check
        if (self.last_crossing_time is not None and (time_to_nanoseconds(current_time) -
                                                     time_to_nanoseconds(self.last_crossing_time)) / 1e9 < self.debounce_time):
            return

        self.last_crossing_time = current_time

        if not self.race_started:
            # First crossing - start the race
            self.race_started = True
            self.race_start_time = current_time
            self.lap_start_time = current_time
            self.race_running = True
            self.lap_count = 1
            self.get_logger().info(f"Race started! Lap {self.lap_count}")

            # EVO: Start new lap trajectory
            if self.enable_trajectory_evaluation and EVO_AVAILABLE:
                self.start_new_lap_trajectory(1)

        else:
            # Subsequent crossings - complete lap
            if self.lap_start_time is not None:
                lap_time = (time_to_nanoseconds(current_time) - time_to_nanoseconds(self.lap_start_time)) / 1e9
                self.lap_times.append(lap_time)

                # EVO: Complete and evaluate lap trajectory
                if self.enable_trajectory_evaluation and EVO_AVAILABLE:
                    self.complete_lap_trajectory(self.lap_count, lap_time)

                    # Check if we should evaluate based on lap interval
                    if (self.evaluation_interval_laps > 0 and
                            self.lap_count % self.evaluation_interval_laps == 0):
                        self.get_logger().info(f"Lap-based evaluation triggered for lap {self.lap_count}")
                        self.evaluate_current_trajectory()

                self.get_logger().info(f"Lap {self.lap_count} completed in {lap_time:.2f}s")

                # Check if race is finished (only if not already finished)
                if self.lap_count >= self.required_laps and not self.race_finished:
                    self.race_finished = True
                    self.race_running = False
                    total_time = (time_to_nanoseconds(current_time) - time_to_nanoseconds(self.race_start_time)) / 1e9
                    self.get_logger().info(f"Race finished! Total time: {total_time:.2f}s")

                    # EVO: Save final evaluation summary and generate plots
                    if self.enable_trajectory_evaluation and EVO_AVAILABLE:
                        self.save_trajectory_evaluation_summary()

                        # Export comprehensive research data
                        if hasattr(self, 'research_evaluator') and self.research_evaluator:
                            self.get_logger().info("Exporting comprehensive research data...")
                            try:
                                self.research_evaluator.export_research_data()
                                summary = self.research_evaluator.generate_research_summary()
                                self.get_logger().info(
                                    f"Research analysis complete - analyzed {summary['experiment_info']['total_laps']} laps")
                                self.get_logger().info(f"Controller: {summary['experiment_info']['controller_name']}")
                            except Exception as e:
                                self.get_logger().error(f"Error exporting research data: {e}")

                        # Generate custom race evaluation
                        if hasattr(self, 'race_evaluator') and self.race_evaluator:
                            self.get_logger().info("Generating custom race evaluation...")
                            try:
                                # Get research data if available
                                research_data = {}
                                if hasattr(self, 'research_evaluator') and self.research_evaluator:
                                    research_data = self.research_evaluator.generate_research_summary()

                                # Pass actual lap times to the race evaluator
                                self.race_evaluator.lap_times_from_monitor = self.lap_times

                                # Get EVO metrics if available
                                evo_metrics = {}
                                if hasattr(self, 'lap_metrics') and self.lap_metrics:
                                    # Combine all lap metrics for overall evaluation
                                    for lap_num, lap_data in self.lap_metrics.items():
                                        for key, value in lap_data.items():
                                            if key not in evo_metrics:
                                                evo_metrics[key] = []
                                            evo_metrics[key].append(value)

                                    # Calculate averages for EVO metrics
                                    for key, values in evo_metrics.items():
                                        if values and isinstance(values[0], (int, float)):
                                            evo_metrics[f'{key}_mean'] = sum(values) / len(values)
                                            evo_metrics[f'{key}_std'] = (
                                                sum((x - evo_metrics[f'{key}_mean'])**2 for x in values) / len(values)) ** 0.5
                                            evo_metrics[f'{key}_max'] = max(values)
                                            evo_metrics[f'{key}_min'] = min(values)

                                # Create and save race evaluation
                                race_evaluation = self.race_evaluator.create_race_evaluation(research_data, evo_metrics)
                                evaluation_filepath = self.race_evaluator.save_race_evaluation(race_evaluation)

                                # Log summary
                                eval_data = race_evaluation['race_evaluation']
                                overall_grade = eval_data['performance_summary']['overall_grade']
                                score = eval_data['performance_summary']['numerical_score']
                                experiment_id = eval_data['metadata']['experiment_id']

                                self.get_logger().info(f"Race evaluation saved: {evaluation_filepath}")
                                self.get_logger().info(f"Overall Grade: {overall_grade} (Score: {score:.1f}/100)")
                                self.get_logger().info(f"Experiment ID: {experiment_id}")

                                # Log top recommendations
                                recommendations = eval_data.get('recommendations', [])
                                if recommendations:
                                    self.get_logger().info("Top recommendations:")
                                    for i, rec in enumerate(recommendations[:3], 1):
                                        self.get_logger().info(f"  {i}. {rec}")

                            except Exception as e:
                                self.get_logger().error(f"Error generating race evaluation: {e}")

                        # Generate all EVO plots if plotter is available (only once when race finishes)
                        if self.evo_plotter and not self.evo_plots_generated:
                            self.get_logger().info("Generating comprehensive EVO analysis plots...")
                            self.evo_plotter.generate_all_plots()
                            self.evo_plots_generated = True  # Mark as generated to prevent re-generation
                else:
                    # Start next lap
                    self.lap_count += 1
                    self.lap_start_time = current_time
                    self.get_logger().info(f"Starting lap {self.lap_count}")

                    # EVO: Start new lap trajectory
                    if self.enable_trajectory_evaluation and EVO_AVAILABLE:
                        self.start_new_lap_trajectory(self.lap_count)

    def start_new_lap_trajectory(self, lap_number):
        """Start tracking trajectory for a new lap"""
        if not self.enable_trajectory_evaluation or not EVO_AVAILABLE:
            return

        if self.current_lap_trajectory:
            # Save previous lap trajectory if it exists
            if hasattr(self, 'last_lap_number') and self.last_lap_number in self.lap_trajectories:
                self.lap_trajectories[self.last_lap_number].extend(self.current_lap_trajectory)
            else:
                self.lap_trajectories[lap_number] = self.current_lap_trajectory.copy()

            # Evaluate the completed lap
            self.evaluate_lap_trajectory(self.last_lap_number)

        # Start new trajectory
        self.current_lap_trajectory = []
        self.last_lap_number = lap_number

        self.get_logger().info(f"Started tracking trajectory for lap {lap_number}")

    def complete_lap_trajectory(self, lap_number, lap_time):
        """Complete trajectory tracking for a lap"""
        if not self.enable_trajectory_evaluation or not EVO_AVAILABLE:
            return

        # Store the completed lap trajectory
        self.lap_trajectories[lap_number] = self.current_lap_trajectory.copy()

        # Add to Research Evaluator for comprehensive analysis
        if hasattr(self, 'research_evaluator') and self.research_evaluator:
            self.get_logger().info(
                f"Adding lap {lap_number} to Research Evaluator for comprehensive analysis (data points: {len(self.current_lap_trajectory)})")
            try:
                self.research_evaluator.add_trajectory(lap_number, self.current_lap_trajectory.copy(), lap_time)
                self.get_logger().info(f"Research analysis completed for lap {lap_number}")
            except Exception as e:
                self.get_logger().error(f"Error in research evaluation for lap {lap_number}: {e}")

        # Add to EVO plotter if available
        if self.evo_plotter:
            self.get_logger().info(
                f"Adding lap {lap_number} trajectory to EVO plotter (data points: {len(self.current_lap_trajectory)})")
            self.evo_plotter.add_lap_trajectory(lap_number, self.current_lap_trajectory.copy())

        # Evaluate the completed lap
        self.evaluate_lap_trajectory(lap_number)

        # Reset for next lap
        self.current_lap_trajectory = []

    def evaluate_lap_trajectory(self, lap_number):
        """Evaluate trajectory for a specific lap using evo"""
        if not self.enable_trajectory_evaluation or not EVO_AVAILABLE or lap_number not in self.lap_trajectories:
            return

        try:
            # Convert to evo trajectory format
            poses = self.lap_trajectories[lap_number]
            if len(poses) < 2:
                return

            # Extract raw data
            positions = []
            orientations = []
            timestamps = []

            for pose in poses:
                timestamp = pose['header'].stamp.sec + pose['header'].stamp.nanosec * 1e-9
                timestamps.append(timestamp)
                positions.append([pose['pose'].position.x, pose['pose'].position.y, pose['pose'].position.z])
                orientations.append([pose['pose'].orientation.w, pose['pose'].orientation.x,
                                     pose['pose'].orientation.y, pose['pose'].orientation.z])

            # Remove duplicate timestamps by keeping only unique consecutive timestamps
            filtered_positions = []
            filtered_orientations = []
            filtered_timestamps = []

            for i in range(len(timestamps)):
                # Keep first point or points with different timestamps
                if i == 0 or timestamps[i] != timestamps[i - 1]:
                    filtered_timestamps.append(timestamps[i])
                    filtered_positions.append(positions[i])
                    filtered_orientations.append(orientations[i])

            if len(filtered_timestamps) < 2:
                self.get_logger().warn(f"Insufficient unique timestamps for lap {lap_number} trajectory evaluation")
                return

            self.get_logger().info(
                f"Filtered trajectory: {len(poses)} -> {len(filtered_timestamps)} poses (removed {len(poses) - len(filtered_timestamps)} duplicates)")

            # Create trajectory object with filtered data
            traj = trajectory.PoseTrajectory3D(
                positions_xyz=np.array(filtered_positions),
                orientations_quat_wxyz=np.array(filtered_orientations),
                timestamps=np.array(filtered_timestamps)
            )

            # Calculate metrics
            metrics_dict = {}

            # Path length
            path_length = self.calculate_path_length(traj)
            metrics_dict['path_length'] = path_length

            # Smoothness (if enabled)
            if self.evaluate_smoothness:
                smoothness = self.calculate_smoothness(traj)
                metrics_dict['smoothness'] = smoothness
                self.smoothness_score_pub.publish(Float32(data=float(smoothness)))

            # Consistency (if enabled)
            if self.evaluate_consistency:
                consistency = self.calculate_consistency(traj)
                metrics_dict['consistency'] = consistency
                self.consistency_score_pub.publish(Float32(data=float(consistency)))

            # Compare with reference if available
            if self.reference_trajectory is not None:
                ref_metrics = self.compare_with_reference(traj)
                metrics_dict.update(ref_metrics)

                # Store metrics
            self.lap_metrics[lap_number] = metrics_dict

            # Add metrics to EVO plotter if available
            if self.evo_plotter:
                self.evo_plotter.add_lap_metrics(lap_number, metrics_dict)

            # Publish metrics
            metrics_msg = String()
            metrics_msg.data = json.dumps(metrics_dict)
            self.trajectory_metrics_pub.publish(metrics_msg)

            # Save trajectory if enabled
            if self.save_trajectories:
                self.save_trajectory_to_file(traj, lap_number)

            self.get_logger().info(f"Evaluated lap {lap_number} trajectory: {metrics_dict}")

        except Exception as e:
            self.get_logger().error(f"Error evaluating lap {lap_number} trajectory: {e}")

    def calculate_path_length(self, traj):
        """Calculate total path length of trajectory"""
        if len(traj.positions_xyz) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(traj.positions_xyz)):
            p1 = traj.positions_xyz[i - 1]
            p2 = traj.positions_xyz[i]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]

            segment_length = np.sqrt(dx * dx + dy * dy + dz * dz)
            total_length += segment_length

        return total_length

    def calculate_smoothness(self, traj):
        """Calculate trajectory smoothness based on curvature"""
        if len(traj.positions_xyz) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(traj.positions_xyz) - 1):
            p1 = traj.positions_xyz[i - 1]
            p2 = traj.positions_xyz[i]
            p3 = traj.positions_xyz[i + 1]

            # Calculate curvature using three points
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            # Cross product for curvature
            cross_product = np.cross(v1, v2)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                curvature = abs(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures.append(curvature)

        if not curvatures:
            return 0.0

        # Lower values indicate smoother trajectories
        return np.mean(curvatures)

    def calculate_consistency(self, traj):
        """Calculate trajectory consistency based on velocity variations"""
        if len(traj.positions_xyz) < 2:
            return 0.0

        velocities = []
        for i in range(1, len(traj.positions_xyz)):
            p1 = traj.positions_xyz[i - 1]
            p2 = traj.positions_xyz[i]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]

            velocity = np.sqrt(dx * dx + dy * dy + dz * dz)
            velocities.append(velocity)

        if not velocities:
            return 0.0

        # Calculate coefficient of variation (lower is more consistent)
        mean_vel = np.mean(velocities)
        std_vel = np.std(velocities)

        if mean_vel > 0:
            return std_vel / mean_vel
        return 0.0

    def compare_with_reference(self, traj):
        """Compare trajectory with reference trajectory using evo metrics"""
        if self.reference_trajectory is None:
            return {}

        try:
            # Synchronize trajectories
            traj_ref, traj_est = sync.associate_trajectories(
                self.reference_trajectory, traj, max_diff=0.01)

            # Calculate APE (Absolute Pose Error)
            pose_relation = metrics.PoseRelation.translation_part
            ape_metric = metrics.APE(pose_relation)
            ape_metric.process_data((traj_ref, traj_est))
            ape_stats = ape_metric.get_statistic(metrics.StatisticsType.rmse)

            # Calculate RPE (Relative Pose Error)
            rpe_metric = metrics.RPE(pose_relation)
            rpe_metric.process_data((traj_ref, traj_est))
            rpe_stats = rpe_metric.get_statistic(metrics.StatisticsType.rmse)

            return {
                'ape_rmse': ape_stats,
                'rpe_rmse': rpe_stats
            }

        except Exception as e:
            self.get_logger().warn(f"Could not compare with reference: {e}")
            return {}

    def save_trajectory_to_file(self, traj, lap_number):
        """Save trajectory to file in TUM format"""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.txt"
            filepath = os.path.join(self.trajectory_output_directory, filename)

            with open(filepath, 'w') as f:
                for i in range(len(traj.positions_xyz)):
                    timestamp = traj.timestamps[i]
                    x, y, z = traj.positions_xyz[i]
                    qw, qx, qy, qz = traj.orientations_quat_wxyz[i]

                    f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

            self.get_logger().info(f"Saved trajectory for lap {lap_number} to {filepath}")

        except Exception as e:
            self.get_logger().error(f"Error saving trajectory for lap {lap_number}: {e}")

    def _save_horizon_reference_trajectory(self, reference_data):
        """Save horizon mapper reference trajectory as TUM file for EVO analysis"""
        try:
            # Create reference trajectory filename
            reference_filename = "horizon_reference_trajectory.txt"
            reference_filepath = os.path.join(self.trajectory_output_directory, reference_filename)

            # Ensure directory exists
            os.makedirs(self.trajectory_output_directory, exist_ok=True)

            # Convert horizon mapper data to TUM format
            with open(reference_filepath, 'w') as f:
                # Create synthetic timestamps (assuming 10Hz)
                base_timestamp = time.time()

                for i, point in enumerate(reference_data):
                    # Generate timestamp
                    timestamp = base_timestamp + (i * 0.1)

                    # Extract position
                    x = point['x']
                    y = point['y']
                    z = 0.0  # Assume 2D trajectory

                    # Convert theta to quaternion (2D rotation around z-axis)
                    theta = point['theta']
                    qx = 0.0
                    qy = 0.0
                    qz = np.sin(theta / 2.0)
                    qw = np.cos(theta / 2.0)

                    # Write TUM format: timestamp x y z qx qy qz qw
                    f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

            self.get_logger().info(
                f"Saved horizon mapper reference trajectory to {reference_filepath} with {len(reference_data)} points")
            self.get_logger().info(
                f"Use this file for EVO comparisons: evo_ape tum {reference_filename} lap_001_trajectory.txt --plot")

        except Exception as e:
            self.get_logger().error(f"Error saving horizon reference trajectory: {e}")

    def evaluate_current_trajectory(self):
        """Periodic evaluation of current trajectory"""
        if not self.enable_trajectory_evaluation or not EVO_AVAILABLE or not self.current_lap_trajectory:
            return

        # Calculate current trajectory metrics
        if len(self.current_lap_trajectory) >= 2:
            # Create temporary trajectory for current lap with timestamp deduplication
            poses = self.current_lap_trajectory

            # Extract raw data
            positions = []
            orientations = []
            timestamps = []

            for pose in poses:
                timestamp = pose['header'].stamp.sec + pose['header'].stamp.nanosec * 1e-9
                timestamps.append(timestamp)
                positions.append([pose['pose'].position.x, pose['pose'].position.y, pose['pose'].position.z])
                orientations.append([pose['pose'].orientation.w, pose['pose'].orientation.x,
                                     pose['pose'].orientation.y, pose['pose'].orientation.z])

            # Remove duplicate timestamps
            filtered_positions = []
            filtered_orientations = []
            filtered_timestamps = []

            for i in range(len(timestamps)):
                if i == 0 or timestamps[i] != timestamps[i - 1]:
                    filtered_timestamps.append(timestamps[i])
                    filtered_positions.append(positions[i])
                    filtered_orientations.append(orientations[i])

            if len(filtered_timestamps) >= 2:
                traj = trajectory.PoseTrajectory3D(
                    positions_xyz=np.array(filtered_positions),
                    orientations_quat_wxyz=np.array(filtered_orientations),
                    timestamps=np.array(filtered_timestamps)
                )

                # Calculate and publish current metrics
                if self.evaluate_smoothness:
                    smoothness = self.calculate_smoothness(traj)
                    self.smoothness_score_pub.publish(Float32(data=float(smoothness)))

    def save_trajectory_evaluation_summary(self):
        """Save summary of all lap trajectory evaluations"""
        if not self.enable_trajectory_evaluation or not EVO_AVAILABLE:
            return

        try:
            summary_file = os.path.join(self.trajectory_output_directory, 'evaluation_summary.csv')

            with open(summary_file, 'w', newline='') as csvfile:
                if not self.lap_metrics:
                    return

                # Get all metric keys
                all_keys = set()
                for metrics in self.lap_metrics.values():
                    all_keys.update(metrics.keys())

                # Write header
                writer = csv.writer(csvfile)
                header = ['lap_number'] + sorted(all_keys)
                writer.writerow(header)

                # Write data
                for lap_num in sorted(self.lap_metrics.keys()):
                    row = [lap_num]
                    for key in sorted(all_keys):
                        value = self.lap_metrics[lap_num].get(key, '')
                        row.append(value)
                    writer.writerow(row)

            self.get_logger().info(f"Saved trajectory evaluation summary to {summary_file}")

        except Exception as e:
            self.get_logger().error(f"Error saving trajectory evaluation summary: {e}")

    # --------------------------- Computational Performance Monitoring ---------------------------------
    def monitor_computational_performance(self):
        """Monitor CPU and memory usage for computational performance analysis"""
        if not self.enable_computational_monitoring or not self.computational_monitoring_initialized:
            return

        try:
            # Get CPU usage
            cpu_percent = self.process.cpu_percent()
            self.cpu_usage_history.append(cpu_percent)

            # Get memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            self.memory_usage_history.append(memory_mb)

            # Publish current CPU usage
            cpu_msg = Float32()
            cpu_msg.data = float(cpu_percent)
            self.cpu_usage_pub.publish(cpu_msg)

            # Publish current memory usage
            memory_msg = Float32()
            memory_msg.data = float(memory_mb)
            self.memory_usage_pub.publish(memory_msg)

            # Calculate and publish processing efficiency
            if len(self.control_loop_latencies) > 0:
                avg_latency = sum(self.control_loop_latencies) / len(self.control_loop_latencies)
                # Efficiency = 1 / (1 + normalized_latency + normalized_cpu_usage)
                # This gives higher values for lower latency and CPU usage
                normalized_latency = min(avg_latency * 20, 1.0)  # Normalize 50ms to 1.0
                normalized_cpu = min(cpu_percent / 100.0, 1.0)
                efficiency = 1.0 / (1.0 + normalized_latency + normalized_cpu)

                efficiency_msg = Float32()
                efficiency_msg.data = float(efficiency)
                self.processing_efficiency_pub.publish(efficiency_msg)

        except Exception as e:
            self.get_logger().error(f"Error monitoring computational performance: {e}")

    def log_performance_stats(self):
        """Log comprehensive performance statistics"""
        if not self.enable_computational_monitoring or not self.computational_monitoring_initialized:
            return

        try:
            current_time = time.time()

            # Calculate statistics
            stats = self.calculate_performance_statistics()

            # Log to console
            self.get_logger().info(f"Performance Stats - Avg Latency: {stats['avg_latency_ms']:.2f}ms, "
                                   f"Max Latency: {stats['max_latency_ms']:.2f}ms, "
                                   f"CPU: {stats['avg_cpu_usage']:.1f}%, "
                                   f"Memory: {stats['avg_memory_usage']:.1f}MB, "
                                   f"Violations: {stats['latency_violations']}")

            # Publish comprehensive stats as JSON
            stats_msg = String()
            stats_msg.data = json.dumps(stats, indent=2)
            self.performance_stats_pub.publish(stats_msg)

            # Store for CSV export
            stats['timestamp'] = current_time
            self.performance_log_data.append(stats)

            self.last_performance_log_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error logging performance stats: {e}")

    def calculate_performance_statistics(self):
        """Calculate comprehensive performance statistics"""
        stats = {
            'total_control_cycles': self.total_control_cycles,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'min_latency_ms': 0.0,
            'latency_std_ms': 0.0,
            'latency_violations': self.latency_violations,
            'violation_rate': 0.0,
            'avg_cpu_usage': 0.0,
            'max_cpu_usage': 0.0,
            'avg_memory_usage': 0.0,
            'max_memory_usage': 0.0,
            'processing_efficiency': 0.0,
            'control_frequency_hz': 0.0
        }

        # Latency statistics
        if len(self.control_loop_latencies) > 0:
            latencies_ms = [l * 1000 for l in self.control_loop_latencies]
            stats['avg_latency_ms'] = sum(latencies_ms) / len(latencies_ms)
            stats['max_latency_ms'] = max(latencies_ms)
            stats['min_latency_ms'] = min(latencies_ms)

            # Calculate standard deviation
            mean_lat = stats['avg_latency_ms']
            variance = sum((x - mean_lat) ** 2 for x in latencies_ms) / len(latencies_ms)
            stats['latency_std_ms'] = math.sqrt(variance)

            stats['violation_rate'] = self.latency_violations / len(self.control_loop_latencies)

        # CPU statistics
        if len(self.cpu_usage_history) > 0:
            stats['avg_cpu_usage'] = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
            stats['max_cpu_usage'] = max(self.cpu_usage_history)

        # Memory statistics
        if len(self.memory_usage_history) > 0:
            stats['avg_memory_usage'] = sum(self.memory_usage_history) / len(self.memory_usage_history)
            stats['max_memory_usage'] = max(self.memory_usage_history)

        # Control frequency
        if len(self.control_send_times) > 1:
            time_span = self.control_send_times[-1] - self.control_send_times[0]
            if time_span > 0:
                stats['control_frequency_hz'] = (len(self.control_send_times) - 1) / time_span

        # Processing efficiency
        if stats['avg_latency_ms'] > 0 and stats['avg_cpu_usage'] > 0:
            normalized_latency = min(stats['avg_latency_ms'] / 50.0, 1.0)  # 50ms = 1.0
            normalized_cpu = min(stats['avg_cpu_usage'] / 100.0, 1.0)
            stats['processing_efficiency'] = 1.0 / (1.0 + normalized_latency + normalized_cpu)

        return stats

    def save_performance_data_to_csv(self):
        """Save computational performance data to CSV file"""
        if not self.enable_computational_monitoring or not self.performance_log_data:
            return

        try:
            # Create performance data directory
            perf_dir = os.path.join(self.trajectory_output_directory, 'performance_data')
            os.makedirs(perf_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(perf_dir, f'computational_performance_{timestamp}.csv')

            # Write CSV
            with open(filename, 'w', newline='') as csvfile:
                if self.performance_log_data:
                    fieldnames = self.performance_log_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in self.performance_log_data:
                        writer.writerow(row)

            self.get_logger().info(f"Computational performance data saved to: {filename}")

        except Exception as e:
            self.get_logger().error(f"Failed to save performance data: {e}")

    # --------------------------- Callbacks ---------------------------------
    def clicked_point_callback(self, msg: PointStamped):
        """Receive points from RViz Publish Point tool and set start/finish line.

        Behavior:
        - First click sets pending_point (P1)
        - Second click sets P2 and updates stored line immediately
        - If user wants to set only one point and keep previous other point, they may click once and wait
        """
        point = np.array([msg.point.x, msg.point.y], dtype=float)

        if self.pending_point is None:
            self.pending_point = point
            self.get_logger().info(
                f"Pending start/finish point set to ({point[0]:.3f}, {point[1]:.3f}). Click second point to complete the line.")
        else:
            # Use previous P2 if pending_point is only update for P1 -- but here we set both
            new_p1 = self.pending_point
            new_p2 = point

            # Update the start line
            self.start_line_p1 = new_p1
            self.start_line_p2 = new_p2

            # Reset lap detection state when start line changes
            if hasattr(self, 'last_side_of_line'):
                delattr(self, 'last_side_of_line')

            # Optionally reset race state for a fresh start with new line
            # Uncomment the following lines if you want to reset the race when line changes
            # self.race_started = False
            # self.race_running = False
            # self.lap_count = 0
            # self.lap_times = []
            # self.last_crossing_time = None

            # Clear pending point
            self.pending_point = None

            # Log the new line
            line_length = np.linalg.norm(new_p2 - new_p1)
            self.get_logger().info(
                f"Start/finish line updated: P1=({new_p1[0]:.3f}, {new_p1[1]:.3f}), "
                f"P2=({new_p2[0]:.3f}, {new_p2[1]:.3f}), length={line_length:.3f}m")
            self.get_logger().info("Lap detection state reset for new start line")

    def control_command_callback(self, msg, topic_name='drive'):
        """Callback for control command messages (computational monitoring)"""
        if not self.enable_computational_monitoring or not self.computational_monitoring_initialized:
            return

        current_time = time.time()
        self.control_send_times.append(current_time)
        self.total_control_cycles += 1

        # Calculate control loop latency from any pending odometry timestamps
        latencies_calculated = []
        for odom_topic, odom_timestamp in self.pending_odom_timestamps.items():
            latency = (time_to_nanoseconds(current_time) - time_to_nanoseconds(odom_timestamp)) / 1e9
            self.control_loop_latencies.append(latency)
            latencies_calculated.append(latency)

            # Update statistics
            self.total_processing_time += latency
            self.max_latency = max(self.max_latency, latency)
            self.min_latency = min(self.min_latency, latency)

            # Check for latency violations (e.g., > 50ms for real-time control)
            if latency > 0.05:  # 50ms threshold
                self.latency_violations += 1

            self.get_logger().debug(f"Control latency from {odom_topic} to {topic_name}: {latency*1000:.2f}ms")

        # Publish average latency if we calculated any
        if latencies_calculated:
            avg_latency = sum(latencies_calculated) / len(latencies_calculated)
            latency_msg = Float32()
            latency_msg.data = float(avg_latency * 1000)  # Convert to milliseconds
            self.control_loop_latency_pub.publish(latency_msg)

        # Clear all pending timestamps after processing
        self.pending_odom_timestamps.clear()

    def publish_race_status(self):
        """Publish current race status"""
        # Publish lap count
        lap_msg = Int32()
        lap_msg.data = self.lap_count
        self.lap_count_pub.publish(lap_msg)

        # Publish race running status
        running_msg = Bool()
        running_msg.data = self.race_running
        self.race_running_pub.publish(running_msg)

        # Publish race state
        state_msg = String()
        if not self.race_started:
            state_msg.data = "waiting"
        elif self.race_finished:
            state_msg.data = "finished"
        else:
            state_msg.data = "racing"
        self.race_state_pub.publish(state_msg)

        # Publish best lap time if available
        if len(self.lap_times) > 0:
            best_time_msg = Float32()
            best_time_msg.data = min(self.lap_times)
            self.best_lap_time_pub.publish(best_time_msg)

        # Publish last lap time if available
        if len(self.lap_times) > 0:
            last_time_msg = Float32()
            last_time_msg.data = self.lap_times[-1]
            self.lap_time_pub.publish(last_time_msg)

    def publish_start_line_marker(self):
        """Publish start/finish line visualization marker"""
        # Check if line has changed or if it's time to republish
        current_line = (tuple(self.start_line_p1), tuple(self.start_line_p2))

        # Republish every 10 seconds even if unchanged (for new subscribers)
        should_republish = False
        if not hasattr(self, '_last_marker_publish_time'):
            self._last_marker_publish_time = 0

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self._last_marker_publish_time > 10.0:  # 10 seconds
            should_republish = True

        if current_line == self._last_line and not should_republish:
            return  # No change and not time to republish

        self._last_line = current_line
        self._last_marker_publish_time = current_time

        # Create line strip marker
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = self.frame_id
        line_marker.ns = 'race_monitor'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        # Line properties
        line_marker.scale.x = 0.1  # Line width
        line_marker.scale.y = 0.1
        line_marker.scale.z = 0.1

        # Bright green color for start/finish line with reduced opacity
        line_marker.color.r = 0.0
        line_marker.color.g = 0.8
        line_marker.color.b = 0.2
        line_marker.color.a = 0.6  # More transparent for subtle appearance

        # Add two points (endpoints) to the line strip
        from geometry_msgs.msg import Point
        pt1 = Point()
        pt1.x = float(self.start_line_p1[0])
        pt1.y = float(self.start_line_p1[1])
        pt1.z = 0.02  # Lower to ground for more realistic appearance

        pt2 = Point()
        pt2.x = float(self.start_line_p2[0])
        pt2.y = float(self.start_line_p2[1])
        pt2.z = 0.02  # Lower to ground for more realistic appearance

        line_marker.points = [pt1, pt2]
        self.start_line_marker_pub.publish(line_marker)

        # Add subtle endpoint markers for better visibility
        for i, point in enumerate([self.start_line_p1, self.start_line_p2]):
            endpoint_marker = Marker()
            endpoint_marker.header.stamp = self.get_clock().now().to_msg()
            endpoint_marker.header.frame_id = self.frame_id
            endpoint_marker.ns = 'race_monitor'
            endpoint_marker.id = i + 1
            endpoint_marker.type = Marker.SPHERE
            endpoint_marker.action = Marker.ADD

            # Position
            endpoint_marker.pose.position.x = float(point[0])
            endpoint_marker.pose.position.y = float(point[1])
            endpoint_marker.pose.position.z = 0.03
            endpoint_marker.pose.orientation.w = 1.0

            # Small sphere size
            endpoint_marker.scale.x = 0.1
            endpoint_marker.scale.y = 0.1
            endpoint_marker.scale.z = 0.1

            # Slightly darker green for endpoints with reduced opacity
            endpoint_marker.color.r = 0.0
            endpoint_marker.color.g = 0.6
            endpoint_marker.color.b = 0.1
            endpoint_marker.color.a = 0.5  # More transparent

            self.start_line_marker_pub.publish(endpoint_marker)

        # Calculate line length for logging
        line_length = np.linalg.norm(self.start_line_p2 - self.start_line_p1)
        self.get_logger().info(f"Published start line (length: {line_length:.2f}m) in frame '{self.frame_id}'")

    def publish_raceline_markers(self):
        """Publish raceline visualization markers from reference trajectory"""
        if not hasattr(self, 'raceline_marker_pub'):
            return

        # Try to get reference trajectory from different sources
        ref_trajectory = None

        # First, try to get from evo_plotter
        if hasattr(self, 'evo_plotter') and self.evo_plotter and hasattr(self.evo_plotter, 'reference_trajectory'):
            ref_trajectory = self.evo_plotter.reference_trajectory

        # If not available, try to load from file
        elif hasattr(self, 'reference_trajectory_file') and self.reference_trajectory_file:
            if os.path.exists(self.reference_trajectory_file):
                try:
                    # Load reference trajectory directly
                    if self.reference_trajectory_file.endswith('.csv'):
                        # Load CSV format (x, y, v, theta)
                        data = np.loadtxt(self.reference_trajectory_file, delimiter=',', skiprows=1)
                        if data.size > 0 and len(data.shape) == 2 and data.shape[1] >= 2:
                            self._publish_raceline_from_csv_data(data)
                            return
                    elif EVO_AVAILABLE:
                        # Try EVO loading
                        if self.reference_trajectory_file.endswith('.txt'):
                            ref_trajectory = file_interface.read_tum_trajectory_file(self.reference_trajectory_file)
                        elif self.reference_trajectory_file.endswith('.csv'):
                            ref_trajectory = file_interface.read_kitti_poses_file(self.reference_trajectory_file)
                except Exception as e:
                    # Don't log error too frequently
                    if not hasattr(self, '_last_raceline_error_time') or time.time() - \
                            self._last_raceline_error_time > 10.0:
                        self.get_logger().warn(f"Could not load reference trajectory for visualization: {e}")
                        self._last_raceline_error_time = time.time()
                    return

        # Publish markers from EVO trajectory
        if ref_trajectory and hasattr(ref_trajectory, 'positions_xyz'):
            self._publish_raceline_from_evo_trajectory(ref_trajectory)

    def _publish_raceline_from_csv_data(self, data):
        """Publish raceline markers from CSV data (x, y, v, theta)"""
        from visualization_msgs.msg import MarkerArray
        from geometry_msgs.msg import Point

        marker_array = MarkerArray()

        # Create line strip marker for the raceline
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = self.frame_id
        line_marker.ns = 'raceline'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        # Line properties
        line_marker.scale.x = 0.05  # Line width
        line_marker.scale.y = 0.05
        line_marker.scale.z = 0.05

        # Blue color for raceline
        line_marker.color.r = 0.0
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0
        line_marker.color.a = 0.8

        # Add points to the line strip
        for i in range(len(data)):
            if len(data[i]) >= 2:  # Ensure we have at least x, y
                pt = Point()
                pt.x = float(data[i][0])
                pt.y = float(data[i][1])
                pt.z = 0.01  # Slightly above ground
                line_marker.points.append(pt)

        marker_array.markers.append(line_marker)

        # Note: Start point marker removed to avoid visualization confusion

        self.raceline_marker_pub.publish(marker_array)

        # Log only once or every 30 seconds
        if not hasattr(self, '_last_raceline_log_time') or time.time() - self._last_raceline_log_time > 30.0:
            self.get_logger().info(f"Published raceline with {len(data)} points")
            self._last_raceline_log_time = time.time()

    def _publish_raceline_from_evo_trajectory(self, trajectory):
        """Publish raceline markers from EVO trajectory"""
        from visualization_msgs.msg import MarkerArray
        from geometry_msgs.msg import Point

        marker_array = MarkerArray()

        # Create line strip marker for the raceline
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = self.frame_id
        line_marker.ns = 'raceline'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        # Line properties
        line_marker.scale.x = 0.05  # Line width
        line_marker.scale.y = 0.05
        line_marker.scale.z = 0.05

        # Blue color for raceline
        line_marker.color.r = 0.0
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0
        line_marker.color.a = 0.8

        # Add points to the line strip
        positions = trajectory.positions_xyz
        for pos in positions:
            pt = Point()
            pt.x = float(pos[0])
            pt.y = float(pos[1])
            pt.z = 0.01  # Slightly above ground
            line_marker.points.append(pt)

        marker_array.markers.append(line_marker)

        # Note: Start point marker removed to avoid visualization confusion

        self.raceline_marker_pub.publish(marker_array)

        # Log only once or every 30 seconds
        if not hasattr(self, '_last_raceline_log_time') or time.time() - self._last_raceline_log_time > 30.0:
            self.get_logger().info(f"Published raceline with {len(positions)} points")
            self._last_raceline_log_time = time.time()

    def save_results_to_csv(self, total_race_time):
        try:
            data_dir = os.path.join(os.getcwd(), 'race_monitor', 'data')
            os.makedirs(data_dir, exist_ok=True)
            filename = self.output_file
            filepath = os.path.join(data_dir, filename)

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['lap_number', 'lap_time_s'])
                for i, t in enumerate(self.lap_times, start=1):
                    writer.writerow([i, f"{t:.4f}"])

                writer.writerow(['total_time_s', f"{total_race_time:.4f}"])
                if len(self.lap_times) > 0:
                    writer.writerow(['best_lap_s', f"{min(self.lap_times):.4f}"])
                    writer.writerow(['worst_lap_s', f"{max(self.lap_times):.4f}"])
                    writer.writerow(['average_lap_s', f"{np.mean(self.lap_times):.4f}"])

            self.get_logger().info(f"Saved race results to: {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error saving CSV: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RaceMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received — shutting down')
        # Save partial results if running or any laps recorded
        if node.race_started and len(node.lap_times) > 0:
            total = 0.0
            if node.race_start_time is not None:
                total = (time_to_nanoseconds(node.get_clock().now()) - time_to_nanoseconds(node.race_start_time)) / 1e9
            node.save_results_to_csv(total)

            # Save trajectory evaluation summary if enabled
            if node.enable_trajectory_evaluation and EVO_AVAILABLE:
                node.save_trajectory_evaluation_summary()

            # Save computational performance data if enabled
            if node.enable_computational_monitoring and node.computational_monitoring_initialized:
                node.save_performance_data_to_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
