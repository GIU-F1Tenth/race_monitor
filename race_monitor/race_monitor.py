#!/usr/bin/env python3

"""
Race Monitor Node

Main ROS2 node that integrates all race monitoring components into a unified system.
Orchestrates lap detection, trajectory analysis, performance monitoring, visualization,
and data management for comprehensive autonomous racing evaluation.

Architecture:
    - Modular design with clear separation of concerns
    - Easy maintenance and testing of individual components
    - Flexible configuration and extensibility
    - Clean interfaces between components

Core Components:
    - LapDetector: Lap detection and timing
    - ReferenceTrajectoryManager: Reference trajectory management
    - PerformanceMonitor: Computational performance monitoring
    - VisualizationPublisher: RViz visualization
    - DataManager: Data storage and file operations
    - TrajectoryAnalyzer: Advanced trajectory analysis
    - RaceEvaluator: Custom race evaluation system

License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from std_msgs.msg import Int32, Float32, Bool, String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PointStamped, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
# F1Tenth interface imports moved to conditional sections where used

import numpy as np
import tf_transformations
import math
import os
import sys
import json
from datetime import datetime

# Import our modular components
from .lap_detector import LapDetector
from .reference_trajectory_manager import ReferenceTrajectoryManager
from .performance_monitor import PerformanceMonitor
from .visualization_publisher import VisualizationPublisher
from .logger_utils import RaceMonitorLogger, LogLevel
from .data_manager import DataManager

# Import existing analysis components
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

try:
    from .visualization_engine import EVOPlotter
    EVO_PLOTTER_AVAILABLE = True
except ImportError:
    EVO_PLOTTER_AVAILABLE = False

# EVO library availability
try:
    import sys
    evo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'evo')
    if os.path.exists(evo_path) and evo_path not in sys.path:
        sys.path.insert(0, evo_path)

    from evo.core import trajectory, metrics, sync
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


class RaceMonitor(Node):
    """
    Main race monitoring node that orchestrates all components.

    Provides modular architecture for race monitoring with
    clear separation of concerns and easy extensibility.
    """

    def __init__(self):
        super().__init__('race_monitor')

        # Load configuration parameters first (needed for logger)
        self._load_parameters()
        
        # Initialize centralized logger with configured log level
        self.logger = RaceMonitorLogger(
            self, 
            "RaceMonitor", 
            self.config['log_level']
        )
        
        self.logger.startup("🏁 Starting Race Monitor")

        # Initialize core components
        self._initialize_components()

        # Set up ROS interfaces
        self._setup_ros_interfaces()

        # Configure components with loaded parameters
        self._configure_components()

        # Set up component interactions
        self._setup_component_callbacks()

        # Start monitoring systems
        self._start_monitoring_systems()

        self.logger.success("Race Monitor initialized successfully!", LogLevel.NORMAL)

    def _load_parameters(self):
        """Load ROS2 parameters and configuration."""
        # ========================================
        # GLOBAL LOGGING CONFIGURATION
        # ========================================
        self.declare_parameter('log_level', 'normal')
        
        # ========================================
        # RACE MONITORING PARAMETERS
        # ========================================
        self.declare_parameter('start_line_p1', [-2.0, 2.0])
        self.declare_parameter('start_line_p2', [2.0, 2.0])
        self.declare_parameter('required_laps', 20)
        self.declare_parameter('debounce_time', 2.0)
        self.declare_parameter('output_file', 'race_results.csv')
        self.declare_parameter('frame_id', 'map')
        
        # Lap detection parameters
        self.declare_parameter('lap_detection.expected_direction', 'any')
        self.declare_parameter('lap_detection.validate_heading_direction', False)
        
        # ========================================
        # RACE ENDING CONDITIONS
        # ========================================
        self.declare_parameter('race_ending_mode', 'lap_complete')

        # Crash detection parameters
        self.declare_parameter('crash_detection.enable_crash_detection', True)
        self.declare_parameter('crash_detection.max_stationary_time', 5.0)
        self.declare_parameter('crash_detection.min_velocity_threshold', 0.1)
        self.declare_parameter('crash_detection.max_odometry_timeout', 3.0)
        self.declare_parameter(
            'crash_detection.enable_collision_detection', True)
        self.declare_parameter(
            'crash_detection.collision_velocity_threshold', 2.0)
        self.declare_parameter(
            'crash_detection.collision_detection_window', 0.5)

        # Manual mode parameters
        self.declare_parameter('manual_mode.save_intermediate_results', True)
        self.declare_parameter('manual_mode.save_interval', 30.0)
        self.declare_parameter('manual_mode.max_race_duration', 0)

        # ========================================
        # RESEARCH & ANALYSIS PARAMETERS
        # ========================================
        self.declare_parameter('controller_name', '')

        # Smart controller detection parameters
        self.declare_parameter('enable_smart_controller_detection', True)

        # Note: experiment_id is now fully auto-generated and not configurable
        self.declare_parameter(
            'test_description', 'Controller performance evaluation and analysis')

        # Auto-shutdown parameters
        self.declare_parameter('auto_shutdown_on_race_complete', True)
        self.declare_parameter('shutdown_delay_seconds', 5.0)

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
        self.declare_parameter('evaluation_interval_seconds', 0.0)
        self.declare_parameter('evaluation_interval_laps', 1)
        self.declare_parameter('evaluation_interval_meters', 0.0)

        # ========================================
        # REFERENCE TRAJECTORY CONFIGURATION
        # ========================================
        self.declare_parameter('reference_trajectory_file', '')
        self.declare_parameter('reference_trajectory_format', 'csv')

        # ========================================
        # ADVANCED FILE OUTPUT FOR RESEARCH
        # ========================================
        self.declare_parameter('export_to_pandas', True)
        self.declare_parameter('save_detailed_statistics', True)
        self.declare_parameter('save_filtered_trajectories', True)
        self.declare_parameter('export_research_summary', True)
        self.declare_parameter(
            'output_formats', ['csv', 'tum', 'json', 'pickle', 'mat'])
        self.declare_parameter('include_timestamps', False)
        self.declare_parameter('save_intermediate_results', True)

        # ========================================
        # DIRECTORY CONFIGURATION
        # ========================================
        # Base directory for all results (with backward compatibility)
        self.declare_parameter('results_dir', '')
        # Deprecated: kept for backward compatibility
        self.declare_parameter('trajectory_output_directory', '')

        # ========================================
        # TRAJECTORY ANALYSIS SETTINGS
        # ========================================
        self.declare_parameter('save_trajectories', True)

        # Metrics to calculate
        self.declare_parameter('evaluate_smoothness', True)
        self.declare_parameter('evaluate_consistency', True)
        self.declare_parameter('evaluate_efficiency', True)
        self.declare_parameter('evaluate_aggressiveness', True)
        self.declare_parameter('evaluate_stability', True)

        # Advanced EVO metrics
        self.declare_parameter(
            'pose_relations', ['translation_part', 'rotation_part', 'full_transformation'])
        self.declare_parameter('statistics_types', [
                               'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'])

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
        self.declare_parameter('generate_error_mapped_plots', True)
        self.declare_parameter('generate_violin_plots', True)
        self.declare_parameter('generate_3d_vector_plots', True)

        # ========================================
        # Roboracer INTERFACE SETTINGS
        # ========================================
        self.declare_parameter('enable_f1tenth_interface', True)
        self.declare_parameter('f1tenth_vehicle_state_topic', '/vehicle_state')
        self.declare_parameter(
            'f1tenth_constrained_state_topic', '/constrained_vehicle_state')

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
        self.declare_parameter(
            'race_evaluation.auto_increment_experiment', True)
        self.declare_parameter('race_evaluation.include_recommendations', True)
        self.declare_parameter('race_evaluation.enable_comparison', True)
        self.declare_parameter('race_evaluation.grading_strictness', 'normal')

        # Evaluation focus areas
        self.declare_parameter('race_evaluation.focus_on_consistency', True)
        self.declare_parameter(
            'race_evaluation.focus_on_trajectory_accuracy', True)
        self.declare_parameter(
            'race_evaluation.focus_on_racing_efficiency', True)

        # Output preferences
        self.declare_parameter(
            'race_evaluation.include_detailed_metrics', True)
        self.declare_parameter('race_evaluation.max_recommendations', 5)

        # Store comprehensive configuration
        self.config = {
            # Global logging
            'log_level': self.get_parameter('log_level').value,
            
            # Basic race monitoring
            'start_line_p1': self.get_parameter('start_line_p1').value,
            'start_line_p2': self.get_parameter('start_line_p2').value,
            'required_laps': self.get_parameter('required_laps').value,
            'debounce_time': self.get_parameter('debounce_time').value,
            'output_file': self.get_parameter('output_file').value,
            'frame_id': self.get_parameter('frame_id').value,
            
            # Lap detection
            'lap_detection_expected_direction': self.get_parameter('lap_detection.expected_direction').value,
            'lap_detection_validate_heading': self.get_parameter('lap_detection.validate_heading_direction').value,
            
            # Directory configuration
            'results_dir': self.get_parameter('results_dir').value,

            # Research & analysis
            'controller_name': self.get_parameter('controller_name').value,
            'enable_smart_controller_detection': self.get_parameter('enable_smart_controller_detection').value,
            # experiment_id will be auto-generated dynamically, not from config
            'test_description': self.get_parameter('test_description').value,
            'auto_shutdown_on_race_complete': self.get_parameter('auto_shutdown_on_race_complete').value,
            'shutdown_delay_seconds': self.get_parameter('shutdown_delay_seconds').value,
            'enable_advanced_metrics': self.get_parameter('enable_advanced_metrics').value,
            'calculate_all_statistics': self.get_parameter('calculate_all_statistics').value,
            'analyze_rotation_errors': self.get_parameter('analyze_rotation_errors').value,
            'enable_geometric_analysis': self.get_parameter('enable_geometric_analysis').value,
            'enable_filtering_analysis': self.get_parameter('enable_filtering_analysis').value,
            'detailed_lap_analysis': self.get_parameter('detailed_lap_analysis').value,
            'comparative_analysis': self.get_parameter('comparative_analysis').value,
            'statistical_significance': self.get_parameter('statistical_significance').value,

            # EVO integration
            'enable_trajectory_evaluation': self.get_parameter('enable_trajectory_evaluation').value,
            'evaluation_interval_seconds': self.get_parameter('evaluation_interval_seconds').value,
            'evaluation_interval_laps': self.get_parameter('evaluation_interval_laps').value,
            'evaluation_interval_meters': self.get_parameter('evaluation_interval_meters').value,

            # Reference trajectory
            'reference_trajectory_file': self.get_parameter('reference_trajectory_file').value,
            'reference_trajectory_format': self.get_parameter('reference_trajectory_format').value,

            # Advanced file output
            'export_to_pandas': self.get_parameter('export_to_pandas').value,
            'save_detailed_statistics': self.get_parameter('save_detailed_statistics').value,
            'save_filtered_trajectories': self.get_parameter('save_filtered_trajectories').value,
            'export_research_summary': self.get_parameter('export_research_summary').value,
            'output_formats': self.get_parameter('output_formats').value,
            'include_timestamps': self.get_parameter('include_timestamps').value,
            'save_intermediate_results': self.get_parameter('save_intermediate_results').value,

            # Trajectory analysis
            'save_trajectories': self.get_parameter('save_trajectories').value,
            'results_dir': self.get_parameter('results_dir').value or self.get_parameter('trajectory_output_directory').value,
            'trajectory_output_directory': self.get_parameter('results_dir').value or self.get_parameter('trajectory_output_directory').value,
            'evaluate_smoothness': self.get_parameter('evaluate_smoothness').value,
            'evaluate_consistency': self.get_parameter('evaluate_consistency').value,
            'evaluate_efficiency': self.get_parameter('evaluate_efficiency').value,
            'evaluate_aggressiveness': self.get_parameter('evaluate_aggressiveness').value,
            'evaluate_stability': self.get_parameter('evaluate_stability').value,
            'pose_relations': self.get_parameter('pose_relations').value,
            'statistics_types': self.get_parameter('statistics_types').value,
            'apply_trajectory_filtering': self.get_parameter('apply_trajectory_filtering').value,
            'filter_types': self.get_parameter('filter_types').value,
            'filter_parameters': {
                'motion_threshold': self.get_parameter('filter_parameters.motion_threshold').value,
                'distance_threshold': self.get_parameter('filter_parameters.distance_threshold').value,
                'angle_threshold': self.get_parameter('filter_parameters.angle_threshold').value,
            },

            # Graph generation
            'auto_generate_graphs': self.get_parameter('auto_generate_graphs').value,
            'graph_formats': self.get_parameter('graph_formats').value,
            'plot_figsize': self.get_parameter('plot_figsize').value,
            'plot_dpi': self.get_parameter('plot_dpi').value,
            'plot_style': self.get_parameter('plot_style').value,
            'plot_color_scheme': self.get_parameter('plot_color_scheme').value,
            'generate_trajectory_plots': self.get_parameter('generate_trajectory_plots').value,
            'generate_xyz_plots': self.get_parameter('generate_xyz_plots').value,
            'generate_rpy_plots': self.get_parameter('generate_rpy_plots').value,
            'generate_speed_plots': self.get_parameter('generate_speed_plots').value,
            'generate_error_plots': self.get_parameter('generate_error_plots').value,
            'generate_metrics_plots': self.get_parameter('generate_metrics_plots').value,

            # Roboracer interface
            'enable_f1tenth_interface': self.get_parameter('enable_f1tenth_interface').value,
            'f1tenth_vehicle_state_topic': self.get_parameter('f1tenth_vehicle_state_topic').value,
            'f1tenth_constrained_state_topic': self.get_parameter('f1tenth_constrained_state_topic').value,

            # Performance monitoring
            'enable_computational_monitoring': self.get_parameter('enable_computational_monitoring').value,
            'odometry_topics': self.get_parameter('odometry_topics').value,
            'control_command_topics': self.get_parameter('control_command_topics').value,
            'monitoring_window_size': self.get_parameter('monitoring_window_size').value,
            'cpu_monitoring_interval': self.get_parameter('cpu_monitoring_interval').value,
            'enable_performance_logging': self.get_parameter('enable_performance_logging').value,
            'performance_log_interval': self.get_parameter('performance_log_interval').value,
            'max_acceptable_latency_ms': self.get_parameter('max_acceptable_latency_ms').value,
            'target_control_frequency_hz': self.get_parameter('target_control_frequency_hz').value,
            'max_acceptable_cpu_usage': self.get_parameter('max_acceptable_cpu_usage').value,
            'max_acceptable_memory_mb': self.get_parameter('max_acceptable_memory_mb').value,

            # Race ending mode
            'race_ending_mode': self.get_parameter('race_ending_mode').value,
            'crash_detection': {
                'enable_crash_detection': self.get_parameter('crash_detection.enable_crash_detection').value,
                'max_stationary_time': self.get_parameter('crash_detection.max_stationary_time').value,
                'min_velocity_threshold': self.get_parameter('crash_detection.min_velocity_threshold').value,
                'max_odometry_timeout': self.get_parameter('crash_detection.max_odometry_timeout').value,
                'enable_collision_detection': self.get_parameter('crash_detection.enable_collision_detection').value,
                'collision_velocity_threshold': self.get_parameter('crash_detection.collision_velocity_threshold').value,
                'collision_detection_window': self.get_parameter('crash_detection.collision_detection_window').value,
            },
            'manual_mode': {
                'save_intermediate_results': self.get_parameter('manual_mode.save_intermediate_results').value,
                'save_interval': self.get_parameter('manual_mode.save_interval').value,
                'max_race_duration': self.get_parameter('manual_mode.max_race_duration').value,
            },

            # Race evaluation
            'enable_race_evaluation': self.get_parameter('enable_race_evaluation').value,
            'race_evaluation': {
                'enable_export': self.get_parameter('race_evaluation.enable_export').value,
                'auto_increment_experiment': self.get_parameter('race_evaluation.auto_increment_experiment').value,
                'include_recommendations': self.get_parameter('race_evaluation.include_recommendations').value,
                'enable_comparison': self.get_parameter('race_evaluation.enable_comparison').value,
                'grading_strictness': self.get_parameter('race_evaluation.grading_strictness').value,
                'focus_on_consistency': self.get_parameter('race_evaluation.focus_on_consistency').value,
                'focus_on_trajectory_accuracy': self.get_parameter('race_evaluation.focus_on_trajectory_accuracy').value,
                'focus_on_racing_efficiency': self.get_parameter('race_evaluation.focus_on_racing_efficiency').value,
                'include_detailed_metrics': self.get_parameter('race_evaluation.include_detailed_metrics').value,
                'max_recommendations': self.get_parameter('race_evaluation.max_recommendations').value,
            }
        }

    def _initialize_components(self):
        """Initialize all monitoring components."""
        # Core components
        self.lap_detector = LapDetector(self.logger, self.get_clock())
        self.reference_manager = ReferenceTrajectoryManager(self.logger)
        self.performance_monitor = PerformanceMonitor(
            self.logger, self.get_clock())
        self.visualization_publisher = VisualizationPublisher(
            self.logger, self._publish_marker)
        self.data_manager = DataManager(self.logger)

        # Smart controller detection variables
        self.detected_controller_names = set()
        self.controller_detection_timer = None
        self.controller_first_detection_time = None

        # Analysis components (will be initialized after run directory is created)
        self.research_evaluator = None
        self.race_evaluator = None
        self.evo_plotter = None

    def _setup_ros_interfaces(self):
        """Set up ROS2 publishers, subscribers, and services."""
        # Publishers
        self.lap_count_pub = self.create_publisher(
            Int32, '/race_monitor/lap_count', 10)
        self.lap_time_pub = self.create_publisher(
            Float32, '/race_monitor/lap_time', 10)
        self.race_status_pub = self.create_publisher(
            String, '/race_monitor/race_status', 10)
        self.race_running_pub = self.create_publisher(
            Bool, '/race_monitor/race_running', 10)
        self.start_line_marker_pub = self.create_publisher(
            Marker, '/race_monitor/start_line_marker', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/car_state/odom', self._odometry_callback, 10
        )
        self.clicked_point_sub = self.create_subscription(
            PointStamped, '/clicked_point', self._clicked_point_callback, 10
        )

        # Optional subscribers based on configuration

        # Control command subscriber for performance monitoring
        if self.config['enable_computational_monitoring']:
            self.control_cmd_sub = self.create_subscription(
                AckermannDriveStamped, '/drive', self._control_command_callback, 10
            )

    def _configure_components(self):
        """Configure all components with loaded parameters."""
        self.lap_detector.configure(self.config)
        self.reference_manager.configure(self.config)
        self.performance_monitor.configure(self.config)
        self.visualization_publisher.configure(self.config)

        # Configure data manager and create run directory
        self.data_manager.configure(self.config)

        self.original_base_output_dir = self.data_manager.trajectory_output_directory
        self.logger.config("📂 Results directory", self.original_base_output_dir, LogLevel.NORMAL)

        # Create dedicated run directory for this session
        controller_name = self.config.get(
            'controller_name', 'unknown_controller')

        # Only auto-generate experiment ID if we have a valid controller name
        # If controller name is empty, smart detection will handle this in _on_race_start
        if controller_name and controller_name.strip() and controller_name != 'unknown_controller':
            experiment_id = self._get_next_experiment_id(controller_name)
            self.config['experiment_id'] = experiment_id
            self.experiment_id_generated = True
            self.logger.info(f"Auto-generated experiment ID: {experiment_id}", LogLevel.NORMAL)

            run_directory = self.data_manager.create_run_directory(
                controller_name, experiment_id)

            # Update configuration paths for all components to use the run directory
            if self.config.get('auto_generate_graphs', False):
                self.config['graph_output_directory'] = os.path.join(
                    run_directory, "graphs")

            # Update trajectory output directory for research evaluator to use the run directory directly
            self.config['trajectory_output_directory'] = run_directory

            # Store run directory for other components
            self.run_directory = run_directory

            # Initialize analysis components now that paths are configured
            self._initialize_analysis_components()
        else:
            # No controller name yet, delay analysis components initialization until race start
            self.experiment_id_generated = False  # Will be generated during race start

        # Load reference trajectory if specified
        if self.config['reference_trajectory_file']:
            self.reference_manager.load_reference_trajectory()

    def _initialize_analysis_components(self):
        """Initialize analysis components after run directory is configured."""
        if RESEARCH_EVALUATOR_AVAILABLE and self.config['enable_trajectory_evaluation']:
            try:
                self.research_evaluator = create_research_evaluator(
                    self.config)

                # Connect data manager for proper directory handling
                if hasattr(self.research_evaluator, 'set_data_manager'):
                    self.research_evaluator.set_data_manager(self.data_manager)

                # Set up reference trajectory for APE/RPE calculations
                if self.reference_manager.is_reference_available():
                    reference_trajectory = self.reference_manager.get_reference_trajectory()
                    if reference_trajectory:
                        # Pass the EVO trajectory object directly to research evaluator
                        self.research_evaluator.reference_trajectory = reference_trajectory

            except Exception as e:
                self.logger.error(f"Failed to initialize research evaluator: {e}", LogLevel.MINIMAL)

        if RACE_EVALUATOR_AVAILABLE and self.config.get('enable_race_evaluation', True):
            try:
                self.race_evaluator = create_race_evaluator(self.config)
            except Exception as e:
                self.logger.error(f"Failed to initialize race evaluator: {e}", LogLevel.MINIMAL)

        if EVO_PLOTTER_AVAILABLE and self.config['auto_generate_graphs']:
            try:
                self.evo_plotter = EVOPlotter(self.config)
            except Exception as e:
                self.logger.warn(f"EVO plotter initialization failed: {e}", LogLevel.NORMAL)
                self.evo_plotter = None
                # Disable auto_generate_graphs to prevent further attempts
                self.config['auto_generate_graphs'] = False

    def _setup_component_callbacks(self):
        """Set up callbacks between components."""
        # Lap detector callbacks
        self.lap_detector.set_callbacks(
            on_race_start=self._on_race_start,
            on_lap_complete=self._on_lap_complete,
            on_race_complete=self._on_race_complete,
            on_race_crash=self._on_race_crash
        )

        # Update visualization publisher with current start line configuration
        self.visualization_publisher.start_line_p1 = self.config['start_line_p1']
        self.visualization_publisher.start_line_p2 = self.config['start_line_p2']

    def _start_monitoring_systems(self):
        """Start active monitoring systems."""
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.start_monitoring()

        # Set up intermediate save timer for manual mode
        if (self.config['race_ending_mode'] == 'manual' and
                self.config['manual_mode']['save_intermediate_results']):
            save_interval = self.config['manual_mode']['save_interval']
            self.intermediate_save_timer = self.create_timer(
                save_interval, self._save_intermediate_results
            )

        # Set up periodic visualization timer to ensure raceline stays visible
        self.visualization_timer = self.create_timer(
            5.0, self._periodic_visualization_update)

        # Set up smart controller detection timer
        if self.config['enable_smart_controller_detection']:
            self.controller_detection_timer = self.create_timer(
                2.0, self._detect_active_controller)

        # Initial enhanced visualization
        self.visualization_publisher.publish_start_line_marker()

    # ROS2 Callback Methods
    def _odometry_callback(self, msg: Odometry):
        """Handle odometry messages."""
        try:
            # Extract position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            # Extract heading from quaternion
            orientation = msg.pose.pose.orientation
            heading = self._quaternion_to_yaw(orientation)

            # Calculate velocity
            linear_velocity = (msg.twist.twist.linear.x**2 +
                               msg.twist.twist.linear.y**2 +
                               msg.twist.twist.linear.z**2) ** 0.5

            # Update lap detector with position, velocity, and heading
            self.lap_detector.update_position(
                x, y, msg.header.stamp, velocity=linear_velocity, heading=heading)

            # Add point to current trajectory
            if self.lap_detector.is_race_active():
                additional_data = {
                    'qx': msg.pose.pose.orientation.x,
                    'qy': msg.pose.pose.orientation.y,
                    'qz': msg.pose.pose.orientation.z,
                    'qw': msg.pose.pose.orientation.w,
                    'linear_velocity': linear_velocity,
                    'angular_velocity': msg.twist.twist.angular.z
                }
                self.data_manager.add_trajectory_point(
                    x, y, z, msg.header.stamp, additional_data)

            # Performance monitoring
            if self.config['enable_computational_monitoring']:
                self.performance_monitor.record_odometry_timestamp(
                    'car_state/odom', msg.header.stamp)

            # Publish race status
            self._publish_race_status()

        except Exception as e:
            self.logger.error(f"Error in odometry callback: {e}", LogLevel.MINIMAL)

    def _control_command_callback(self, msg: AckermannDriveStamped):
        """Handle control command messages for performance monitoring."""
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.record_control_command_timestamp(
                '/drive', msg.header.stamp)

    def _detect_active_controller(self):
        """Detect which controller is actively publishing to control topics."""
        try:
            # Only detect if enabled and conditions are met
            if not self.config['enable_smart_controller_detection']:
                return

            current_name = self.config['controller_name']

            # Only detect when controller name is empty
            if current_name not in ['', None]:
                return

            # Get topic info for control command topics
            topic_names_and_types = self.get_topic_names_and_types()
            control_topics = self.config.get(
                'control_command_topics', ['/drive'])

            for topic in control_topics:
                # Check if topic exists
                topic_exists = any(name == topic for name,
                                   _ in topic_names_and_types)

                if topic_exists:
                    # Get publishers for this topic
                    try:
                        publishers_info = self.get_publishers_info_by_topic(
                            topic)

                        for publisher_info in publishers_info:
                            node_name = publisher_info.node_name

                            # Skip system/simulation nodes
                            if node_name in ['race_monitor', 'rqt_gui', 'rviz2', '/sim_time_monitor']:
                                continue

                            # Add to detected controllers
                            if node_name not in self.detected_controller_names:
                                self.detected_controller_names.add(node_name)

                                # Set first detection time
                                if self.controller_first_detection_time is None:
                                    self.controller_first_detection_time = self.get_clock().now()

                                self.logger.info(
                                    f"Detected controller: {node_name} publishing to {topic}", LogLevel.NORMAL)

                                # Update controller name if this is the first one or if multiple detected
                                if len(self.detected_controller_names) == 1:
                                    new_name = node_name
                                    if new_name.startswith('/'):
                                        # Remove leading slash
                                        new_name = new_name[1:]

                                    self.config['controller_name'] = new_name

                                    # Update the parameter as well
                                    self.set_parameters([Parameter('controller_name',
                                                                   Parameter.Type.STRING,
                                                                   new_name)])

                                    self.logger.info(
                                        f"Controller name automatically updated to: {new_name}", LogLevel.NORMAL)

                                elif len(self.detected_controller_names) > 1:
                                    # Multiple controllers detected
                                    controller_list = sorted(
                                        list(self.detected_controller_names))
                                    combined_name = "_".join(
                                        [name.lstrip('/') for name in controller_list])

                                    self.config['controller_name'] = combined_name
                                    self.set_parameters([Parameter('controller_name',
                                                                   Parameter.Type.STRING,
                                                                   combined_name)])

                                    self.logger.info(
                                        f"Multiple controllers detected. Name updated to: {combined_name}", LogLevel.NORMAL)

                    except Exception as e:
                        self.logger.debug(
                            f"Could not get publisher info for {topic}: {e}", LogLevel.DEBUG)

            # Stop detection after 30 seconds to avoid continuous polling
            if (self.controller_first_detection_time and
                    (self.get_clock().now() - self.controller_first_detection_time).nanoseconds / 1e9 > 30.0):

                if self.controller_detection_timer:
                    self.controller_detection_timer.cancel()
                    self.controller_detection_timer = None
                    self.logger.info("Controller detection stopped after 30 seconds", LogLevel.NORMAL)

        except Exception as e:
            self.logger.error(f"Error in controller detection: {e}", LogLevel.MINIMAL)

    def _clicked_point_callback(self, msg: PointStamped):
        """Handle clicked point messages for manual start line setup."""
        point = [msg.point.x, msg.point.y]
        self.logger.info(
            f"Clicked point received: ({point[0]:.2f}, {point[1]:.2f})", LogLevel.NORMAL)

        if not hasattr(self, 'pending_start_point'):
            self.pending_start_point = None

        if self.pending_start_point is None:
            # First click - store as pending point
            self.pending_start_point = point
            self.logger.info(
                f"Pending start/finish point set. Click second point to complete the line.", LogLevel.NORMAL)
        else:
            # Second click - update start line
            self.config['start_line_p1'] = self.pending_start_point
            self.config['start_line_p2'] = point

            # Reconfigure lap detector with new start line
            self.lap_detector.configure(self.config)

            # Reset pending point
            self.pending_start_point = None

            self.logger.info(f"Start/finish line updated:", LogLevel.NORMAL)
            self.logger.info(
                f"  P1: ({self.config['start_line_p1'][0]:.2f}, {self.config['start_line_p1'][1]:.2f})", LogLevel.NORMAL)
            self.logger.info(
                f"  P2: ({self.config['start_line_p2'][0]:.2f}, {self.config['start_line_p2'][1]:.2f})", LogLevel.NORMAL)

            # Update visualization with enhanced start line markers
            self.visualization_publisher.publish_start_line_markers(
                self.config['start_line_p1'], self.config['start_line_p2'], self.config['frame_id']
            )

    def _publish_marker(self, marker: Marker):
        """Publish visualization marker."""
        self.start_line_marker_pub.publish(marker)

    # Race Event Handlers
    def _on_race_start(self, timestamp):
        """Handle race start event."""
        self.logger.event("🏁 Race started", "Race has begun", LogLevel.MINIMAL)

        # Create run directory if not created yet (in case controller was detected after initialization)
        controller_name = self.config.get('controller_name', '')

        # If no controller name, it might be detected now by smart detection
        if not controller_name and hasattr(self, 'detected_controller_names') and self.detected_controller_names:
            controller_name = list(self.detected_controller_names)[0]
            self.config['controller_name'] = controller_name
            self.logger.info(
                f"Using detected controller: {controller_name}", LogLevel.NORMAL)

        if controller_name and not getattr(self, 'experiment_id_generated', False):
            experiment_id = self._get_next_experiment_id(controller_name)
            self.config['experiment_id'] = experiment_id
            self.experiment_id_generated = True
            self.logger.info(
                f"Auto-generated experiment ID for race start: {experiment_id}", LogLevel.NORMAL)
        else:
            experiment_id = self.config.get('experiment_id', 'exp_001')

        # Create experiment directory if we have a controller name and haven't created one yet
        if controller_name and not hasattr(self.data_manager, 'run_directory_created'):
            run_directory = self.data_manager.create_run_directory(
                controller_name, experiment_id)
            self.data_manager.run_directory_created = True
            # self.get_logger().info(f"Created experiment directory for {controller_name}: {run_directory}")

            # Update config paths to point to the experiment directory
            self.config['trajectory_output_directory'] = run_directory
            if self.config.get('auto_generate_graphs', False):
                self.config['graph_output_directory'] = os.path.join(
                    run_directory, "graphs")

            # Initialize analysis components now that we have the run directory
            self._initialize_analysis_components()

        # Start new lap trajectory
        self.data_manager.start_new_lap_trajectory(1)

        # Update visualization
        self.visualization_publisher.publish_race_status_marker(
            "RACING", 1, self.config['required_laps'])

        # Reset performance monitoring data
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.reset_performance_data()

    def _on_lap_complete(self, lap_number: int, lap_time: float):
        """Handle lap completion event."""
        self.logger.event("Lap complete", f"Lap {lap_number} in {lap_time:.3f}s", LogLevel.NORMAL)

        # Complete current lap trajectory
        self.data_manager.complete_lap_trajectory(lap_number, lap_time)

        # Start next lap trajectory if race not complete
        if lap_number < self.config['required_laps']:
            self.data_manager.start_new_lap_trajectory(lap_number + 1)

        # Update visualization
        race_stats = self.lap_detector.get_race_stats()
        self.visualization_publisher.publish_lap_completion_marker(
            lap_number, lap_time)

        if not race_stats['race_completed']:
            self.visualization_publisher.publish_race_status_marker(
                "RACING", race_stats['current_lap'], self.config['required_laps']
            )

        # Perform trajectory evaluation if enabled
        if self.config['enable_trajectory_evaluation'] and EVO_AVAILABLE:
            self._evaluate_lap_trajectory(lap_number)

    def _on_race_complete(self, total_time: float, lap_times: list):
        """Handle race completion event."""
        race_stats = self.lap_detector.get_race_stats()
        reason = race_stats['race_ending_reason']

        self.logger.event("Race complete", f"Finished in {total_time:.3f}s - {reason}", LogLevel.MINIMAL)

        # Update visualization based on ending reason
        status_text = "FINISHED" if reason == "Laps completed" else f"ENDED-{reason.upper()}"
        required_laps = self.config['required_laps'] if self.config['race_ending_mode'] == 'lap_complete' else len(
            lap_times)
        self.visualization_publisher.publish_race_status_marker(
            status_text, len(lap_times), required_laps)

        # Save race results
        race_data = {
            'total_race_time': total_time,
            'lap_times': lap_times,
            'controller_name': self.config['controller_name'],
            'experiment_id': self.config['experiment_id'],
            'race_ending_mode': self.config['race_ending_mode'],
            'race_ending_reason': reason,
            'laps_completed': len(lap_times),
            'timestamp': datetime.now().isoformat()
        }
        self.data_manager.save_race_results_to_csv(race_data)

        # Perform comprehensive race analysis
        self._perform_comprehensive_analysis(race_data)

        # Save performance data
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.save_performance_data_to_csv(
                os.path.join(
                    self.config['trajectory_output_directory'], 'cpu_performance_data')
            )

        # Auto-shutdown after race completion if enabled
        if self.config['auto_shutdown_on_race_complete']:
            delay = self.config['shutdown_delay_seconds']
            self.logger.info(
                f"Race complete! Shutting down in {delay} seconds...", LogLevel.NORMAL)

            # Create a timer for delayed shutdown
            self.shutdown_timer = self.create_timer(delay, self._shutdown_node)

    def _shutdown_node(self):
        """Shutdown the race monitor node."""
        self.logger.shutdown("Race Monitor", LogLevel.MINIMAL)

        # Cancel the shutdown timer
        if hasattr(self, 'shutdown_timer'):
            self.shutdown_timer.cancel()

        # Cancel other timers if they exist
        if hasattr(self, 'controller_detection_timer') and self.controller_detection_timer:
            self.controller_detection_timer.cancel()

        if hasattr(self, 'visualization_timer'):
            self.visualization_timer.cancel()

        if hasattr(self, 'intermediate_save_timer'):
            self.intermediate_save_timer.cancel()

        # Initiate shutdown
        rclpy.shutdown()

    def _on_race_crash(self, crash_reason: str, total_time: float, lap_times: list):
        """Handle race crash event."""
        self.logger.warn(f"Race ended due to crash: {crash_reason}", LogLevel.MINIMAL)
        self.logger.info(
            f"  Duration: {total_time:.3f}s, Laps: {len(lap_times)}", LogLevel.MINIMAL)

        # Update visualization
        self.visualization_publisher.publish_race_status_marker(
            "CRASHED", len(lap_times), len(lap_times))

        # Save race results with crash information
        race_data = {
            'total_race_time': total_time,
            'lap_times': lap_times,
            'controller_name': self.config['controller_name'],
            'experiment_id': self.config['experiment_id'],
            'race_ending_mode': 'crash',
            'race_ending_reason': crash_reason,
            'laps_completed': len(lap_times),
            'crashed': True,
            'timestamp': datetime.now().isoformat()
        }
        self.data_manager.save_race_results_to_csv(race_data)

        # Perform analysis even for crashed races
        self._perform_comprehensive_analysis(race_data)

        # Save performance data
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.save_performance_data_to_csv(
                os.path.join(
                    self.config['trajectory_output_directory'], 'cpu_performance_data')
            )

    def _evaluate_lap_trajectory(self, lap_number: int):
        """Evaluate individual lap trajectory."""
        try:
            # Create EVO trajectory for the lap
            evo_trajectory = self.data_manager.create_evo_trajectory(
                lap_number)
            if not evo_trajectory:
                return

            # Perform evaluation using research evaluator if available
            if self.research_evaluator and self.reference_manager.is_reference_available():
                reference_trajectory = self.reference_manager.get_reference_trajectory()
                if reference_trajectory:
                    # TODO: Implement single lap evaluation
                    pass

        except Exception as e:
            self.logger.error(f"Error evaluating lap {lap_number}: {e}", LogLevel.NORMAL)

    def _perform_comprehensive_analysis(self, race_data: dict):
        """Perform comprehensive race analysis after completion."""
        try:
            self.logger.section("Performing comprehensive race analysis", LogLevel.NORMAL)

            # Compile all trajectory data for analysis
            all_trajectories = self.data_manager.get_trajectory_data()

            # Generate comprehensive race summary
            race_summary = self._generate_race_summary(
                race_data, all_trajectories)

            # Save comprehensive race summary
            self.data_manager.save_race_summary(race_summary)

            # Save APE/RPE metrics as individual files (plots disabled)
            advanced_metrics = race_summary.get('advanced_metrics', {})
            if advanced_metrics:
                try:
                    # Save APE/RPE metrics files in metrics directory
                    if self.data_manager.save_ape_rpe_metrics_files(advanced_metrics):
                        self.logger.success("APE/RPE metrics files saved successfully", LogLevel.NORMAL)
                    else:
                        self.logger.warn("Failed to save APE/RPE metrics files", LogLevel.NORMAL)
                    # Note: APE/RPE plot generation has been removed to reduce storage overhead
                except Exception as e:
                    self.logger.error(
                        f"Error saving APE/RPE metrics files: {e}", LogLevel.NORMAL)

            # Initialize variables for consolidated save
            race_evaluation = None

            if self.research_evaluator and all_trajectories:
                # Perform research-grade analysis
                self.logger.info("Running research trajectory evaluation...", LogLevel.NORMAL)
                try:
                    # Add trajectory data to research evaluator
                    for lap_num, trajectory_data in all_trajectories.items():
                        if 'points' in trajectory_data and 'lap_time' in trajectory_data:
                            # Convert data format for research evaluator
                            converted_points = []
                            for i, point in enumerate(trajectory_data['points']):
                                # Convert to nested dictionary format that research evaluator expects
                                converted_point = {
                                    'pose': {
                                        'position': {
                                            'x': point['x'],
                                            'y': point['y'],
                                            'z': point.get('z', 0.0)
                                        },
                                        'orientation': {
                                            'x': point.get('qx', 0.0),
                                            'y': point.get('qy', 0.0),
                                            'z': point.get('qz', 0.0),
                                            'w': point.get('qw', 1.0)
                                        }
                                    },
                                    'header': {
                                        'stamp': {
                                            'sec': int(point.get('timestamp', i * 0.1)),
                                            'nanosec': int((point.get('timestamp', i * 0.1) - int(point.get('timestamp', i * 0.1))) * 1e9)
                                        }
                                    }
                                }
                                converted_points.append(converted_point)

                            self.research_evaluator.add_trajectory(
                                lap_num,
                                converted_points,
                                trajectory_data['lap_time']
                            )

                    # Generate research summary
                    evaluation_results = self.research_evaluator.generate_research_summary()
                    if evaluation_results:
                        self.data_manager.save_evaluation_summary(
                            evaluation_results)
                        self.logger.success("Research evaluation completed successfully", LogLevel.NORMAL)
                except Exception as e:
                    self.logger.error(f"Research evaluation failed: {e}", LogLevel.NORMAL)

            if self.race_evaluator and all_trajectories:
                # Perform custom race evaluation
                self.logger.info("Running custom race evaluation...", LogLevel.NORMAL)
                try:
                    # Create research data summary for race evaluator
                    research_summary = self._generate_race_summary(
                        race_data, all_trajectories)

                    # Extract advanced metrics for EVO evaluation
                    evo_metrics = {}
                    if 'advanced_metrics' in research_summary:
                        advanced_metrics = research_summary['advanced_metrics']

                        # Map advanced metrics to EVO format
                        evo_metrics = {
                            'ape_rmse': advanced_metrics.get('overall_ape_mean', 0),
                            'ape_mean': advanced_metrics.get('overall_ape_mean', 0),
                            'ape_std': next((v for k, v in advanced_metrics.items() if k.startswith('ape_') and k.endswith('_std')), 0),
                            'ape_max': next((v for k, v in advanced_metrics.items() if k.startswith('ape_') and k.endswith('_max')), 0),
                            'rpe_rmse': advanced_metrics.get('overall_rpe_mean', 0),
                            'rpe_mean': advanced_metrics.get('overall_rpe_mean', 0),
                            'rpe_std': next((v for k, v in advanced_metrics.items() if k.startswith('rpe_') and k.endswith('_std')), 0),
                            'rpe_max': next((v for k, v in advanced_metrics.items() if k.startswith('rpe_') and k.endswith('_max')), 0),
                            'reference_available': True,
                            'reference_deviation': advanced_metrics.get('overall_ape_mean', 0)
                        }
                        self.logger.metric(
                            "APE/RPE", f"APE={evo_metrics['ape_mean']:.4f}, RPE={evo_metrics['rpe_mean']:.4f}", "", LogLevel.NORMAL)

                    # Pass lap times from race_data to race_evaluator
                    if 'lap_times' in race_data:
                        self.race_evaluator.lap_times_from_monitor = race_data['lap_times']
                        self.logger.info(
                            f"Set lap times for evaluator: {len(race_data['lap_times'])} laps", LogLevel.DEBUG)

                    race_evaluation = self.race_evaluator.create_race_evaluation(
                        research_summary, evo_metrics)
                    if race_evaluation:
                        self.data_manager.save_race_evaluation(race_evaluation)
                        self.logger.success("Custom race evaluation completed successfully", LogLevel.NORMAL)
                except Exception as e:
                    self.logger.error(
                        f"Custom race evaluation failed: {e}", LogLevel.NORMAL)

            if self.evo_plotter and all_trajectories:
                # Generate visualization plots
                self.logger.info("Generating analysis plots...", LogLevel.NORMAL)
                try:
                    # Add trajectory data to the plotter
                    for lap_num, trajectory_data in all_trajectories.items():
                        if 'points' in trajectory_data:
                            # Convert trajectory points to proper format for plotting
                            poses = []
                            for i, point in enumerate(trajectory_data['points']):
                                # Create a simple object-like structure that the visualization engine expects
                                class MockPose:
                                    def __init__(self, x, y, z, qx, qy, qz, qw, theta=None):
                                        self.x = x
                                        self.y = y
                                        self.z = z
                                        if theta is not None:
                                            self.theta = theta
                                        # Also store quaternion data if available
                                        if qx is not None:
                                            self.qx = qx
                                            self.qy = qy
                                            self.qz = qz
                                            self.qw = qw

                                class MockHeader:
                                    def __init__(self, timestamp):
                                        # Create a proper stamp object that the visualization engine expects
                                        class MockStamp:
                                            def __init__(self, ts):
                                                if ts is not None:
                                                    self.sec = int(ts)
                                                    self.nanosec = int(
                                                        (ts - int(ts)) * 1e9)
                                                else:
                                                    self.sec = 0
                                                    self.nanosec = 0

                                        self.stamp = MockStamp(timestamp)
                                        self.timestamp = timestamp

                                # Calculate theta from quaternion if not provided
                                qz = point.get('qz', 0.0)
                                qw = point.get('qw', 1.0)
                                theta = 2.0 * \
                                    np.arctan2(qz, qw) if qw != 0 else 0.0

                                pose_data = {
                                    'pose': MockPose(
                                        x=point['x'],
                                        y=point['y'],
                                        z=point.get('z', 0.0),
                                        qx=point.get('qx', 0.0),
                                        qy=point.get('qy', 0.0),
                                        qz=point.get('qz', 0.0),
                                        qw=point.get('qw', 1.0),
                                        theta=theta
                                    ),
                                    'header': MockHeader(point.get('timestamp', i * 0.1))
                                }
                                poses.append(pose_data)

                            if poses:
                                self.evo_plotter.add_lap_trajectory(
                                    lap_num, poses)

                    # Generate all plots
                    plot_success = self.evo_plotter.generate_all_plots()
                    if plot_success:
                        self.logger.success("Visualization plots generated successfully", LogLevel.NORMAL)
                    else:
                        self.logger.warn("Some visualization plots failed to generate", LogLevel.NORMAL)
                except Exception as e:
                    self.logger.error(f"Plot generation failed: {e}", LogLevel.NORMAL)

            # Save consolidated race_results.json file with all important information
            try:
                self.data_manager.save_consolidated_race_results(
                    race_summary, race_evaluation)
                self.logger.success("Saved consolidated race_results.json file", LogLevel.NORMAL)
            except Exception as e:
                self.logger.error(
                    f"Error saving consolidated results: {e}", LogLevel.NORMAL)

            self.logger.success("Comprehensive analysis completed!", LogLevel.NORMAL)

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}", LogLevel.MINIMAL)

    def _generate_race_summary(self, race_data: dict, trajectory_data: dict) -> dict:
        """Generate comprehensive race summary with detailed statistics."""
        try:
            lap_times = race_data.get('lap_times', [])

            # Basic race statistics
            summary = {
                'race_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'controller_name': self.config['controller_name'],
                    'experiment_id': self.config['experiment_id'],
                    'test_description': self.config.get('test_description', ''),
                    'total_race_time': race_data.get('total_race_time', 0.0),
                    'laps_completed': len(lap_times),
                    'race_ending_reason': race_data.get('race_ending_reason', 'Unknown'),
                    'crashed': race_data.get('crashed', False)
                },
                'lap_statistics': {
                    'lap_times': lap_times,
                    'best_lap_time': min(lap_times) if lap_times else 0.0,
                    'worst_lap_time': max(lap_times) if lap_times else 0.0,
                    'average_lap_time': np.mean(lap_times) if lap_times else 0.0,
                    'median_lap_time': np.median(lap_times) if lap_times else 0.0,
                    'lap_time_std': np.std(lap_times) if lap_times else 0.0,
                    'consistency_score': 1.0 - (np.std(lap_times) / np.mean(lap_times)) if lap_times and np.mean(lap_times) > 0 else 0.0
                },
                'trajectory_statistics': {},
                'performance_metrics': {},
                'advanced_metrics': {}
            }

            # Calculate trajectory statistics if available
            if trajectory_data:
                total_distance = 0.0
                total_points = 0

                for lap_num, traj_data in trajectory_data.items():
                    if 'points' in traj_data:
                        points = traj_data['points']
                        total_points += len(points)

                        # Calculate lap distance
                        lap_distance = 0.0
                        for i in range(1, len(points)):
                            dx = points[i]['x'] - points[i - 1]['x']
                            dy = points[i]['y'] - points[i - 1]['y']
                            lap_distance += np.sqrt(dx * dx + dy * dy)

                        total_distance += lap_distance

                summary['trajectory_statistics'] = {
                    'total_distance': total_distance,
                    'average_lap_distance': total_distance / len(trajectory_data) if trajectory_data else 0.0,
                    'total_trajectory_points': total_points,
                    'average_points_per_lap': total_points / len(trajectory_data) if trajectory_data else 0.0
                }

                # Calculate performance metrics
                if lap_times and total_distance > 0:
                    summary['performance_metrics'] = {
                        'average_speed': total_distance / race_data.get('total_race_time', 1.0),
                        'best_lap_speed': (total_distance / len(trajectory_data)) / min(lap_times) if lap_times else 0.0,
                        'distance_per_second': total_distance / race_data.get('total_race_time', 1.0)
                    }

            # Calculate averaged advanced metrics from research evaluator or lap files
            try:
                advanced_metrics = self._calculate_averaged_metrics()
                if advanced_metrics:
                    summary['advanced_metrics'] = advanced_metrics
                    self.logger.debug(
                        f"Added {len(advanced_metrics)} averaged advanced metrics to race summary", LogLevel.DEBUG)
                else:
                    # No advanced metrics available - this is normal for basic monitoring
                    summary['advanced_metrics'] = {}
            except Exception as e:
                self.logger.error(
                    f"Error calculating averaged metrics: {e}", LogLevel.NORMAL)
                summary['advanced_metrics'] = {}

            return summary

        except Exception as e:
            self.logger.error(f"Error generating race summary: {e}", LogLevel.MINIMAL)
            return {}

    def _calculate_averaged_metrics(self) -> dict:
        """Calculate averaged metrics from all lap measurements."""
        try:
            detailed_metrics = {}

            # First try to get metrics from research evaluator in-memory storage
            if (hasattr(self, 'research_evaluator') and self.research_evaluator and
                    hasattr(self.research_evaluator, 'detailed_metrics') and self.research_evaluator.detailed_metrics):
                detailed_metrics = self.research_evaluator.detailed_metrics
                self.logger.debug(
                    f"Using in-memory detailed metrics from research evaluator: {len(detailed_metrics)} laps", LogLevel.DEBUG)
            else:
                # Fallback: Load metrics from saved lap files
                self.logger.debug(
                    "Research evaluator not available or no in-memory metrics, loading from saved lap files", LogLevel.DEBUG)
                detailed_metrics = self._load_metrics_from_files()

            if not detailed_metrics:
                self.logger.debug(
                    "No detailed metrics available from research evaluator or saved files", LogLevel.DEBUG)
                return {}

            # Collect all metrics from all laps
            all_metrics = {}
            lap_count = len(detailed_metrics)

            if lap_count == 0:
                return {}

            self.logger.info(
                f"Processing {lap_count} laps for advanced metrics calculation", LogLevel.DEBUG)

            # Aggregate metrics across all laps
            for lap_num, lap_metrics in detailed_metrics.items():
                for metric_name, metric_value in lap_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)

            # Calculate statistics for each metric
            averaged_metrics = {}
            for metric_name, values in all_metrics.items():
                if len(values) > 0:
                    try:
                        averaged_metrics[f'{metric_name}_mean'] = float(
                            np.mean(values))
                        averaged_metrics[f'{metric_name}_std'] = float(
                            np.std(values))
                        averaged_metrics[f'{metric_name}_min'] = float(
                            np.min(values))
                        averaged_metrics[f'{metric_name}_max'] = float(
                            np.max(values))
                        averaged_metrics[f'{metric_name}_median'] = float(
                            np.median(values))

                        # Add coefficient of variation for key metrics
                        if metric_name.startswith(('ape_', 'rpe_', 'velocity_', 'acceleration_')):
                            mean_val = np.mean(values)
                            if mean_val > 0:
                                averaged_metrics[f'{metric_name}_cv'] = float(
                                    np.std(values) / mean_val * 100)
                    except Exception as e:
                        self.logger.warn(
                            f"Error calculating statistics for {metric_name}: {e}", LogLevel.DEBUG)
                        continue

            # Calculate overall performance indicators
            if averaged_metrics:
                # APE/RPE consistency scores
                ape_keys = [k for k in averaged_metrics.keys(
                ) if k.startswith('ape_') and k.endswith('_mean')]
                rpe_keys = [k for k in averaged_metrics.keys(
                ) if k.startswith('rpe_') and k.endswith('_mean')]

                if ape_keys:
                    ape_mean = np.mean([averaged_metrics[k] for k in ape_keys])
                    averaged_metrics['overall_ape_mean'] = float(ape_mean)

                if rpe_keys:
                    rpe_mean = np.mean([averaged_metrics[k] for k in rpe_keys])
                    averaged_metrics['overall_rpe_mean'] = float(rpe_mean)

                # Overall consistency score
                cv_keys = [k for k in averaged_metrics.keys()
                           if k.endswith('_cv')]
                if cv_keys:
                    overall_cv = np.mean([averaged_metrics[k]
                                         for k in cv_keys])
                    averaged_metrics['overall_consistency_cv'] = float(
                        overall_cv)

            self.logger.debug(
                f"Calculated {len(averaged_metrics)} averaged metrics from {lap_count} laps", LogLevel.DEBUG)
            return averaged_metrics

        except Exception as e:
            self.logger.error(f"Error calculating averaged metrics: {e}", LogLevel.NORMAL)
            return {}

    def _load_metrics_from_files(self) -> dict:
        """Load detailed metrics from saved lap files as fallback."""
        try:
            detailed_metrics = {}

            # Get the metrics directory path
            if (hasattr(self, 'research_evaluator') and self.research_evaluator and
                    hasattr(self.research_evaluator, 'experiment_dir') and self.research_evaluator.experiment_dir):
                metrics_dir = os.path.join(
                    self.research_evaluator.experiment_dir, 'metrics')
                self.logger.debug(
                    f"Using research evaluator experiment_dir for metrics: {metrics_dir}", LogLevel.VERBOSE)
            else:
                # Fallback to data manager's path
                metrics_dir = self.data_manager.get_metrics_directory()
                self.logger.debug(
                    f"Using data manager metrics directory: {metrics_dir}", LogLevel.VERBOSE)

            if not os.path.exists(metrics_dir):
                self.logger.warn(
                    f"Metrics directory not found: {metrics_dir}", LogLevel.DEBUG)
                return {}

            # Load all lap metric files
            lap_files = [f for f in os.listdir(metrics_dir) if f.startswith(
                'lap_') and f.endswith('_metrics.json')]
            lap_files.sort()

            self.logger.debug(
                f"Found {len(lap_files)} lap metric files in {metrics_dir}", LogLevel.VERBOSE)

            for lap_file in lap_files:
                try:
                    # Extract lap number from filename like "lap_001_metrics.json"
                    lap_num_str = lap_file.split('_')[1]
                    lap_num = int(lap_num_str)

                    file_path = os.path.join(metrics_dir, lap_file)
                    with open(file_path, 'r') as f:
                        lap_metrics = json.load(f)

                    detailed_metrics[lap_num] = lap_metrics
                    self.logger.debug(
                        f"Loaded {len(lap_metrics)} metrics for lap {lap_num}", LogLevel.VERBOSE)

                except Exception as e:
                    self.logger.warn(
                        f"Error loading lap file {lap_file}: {e}", LogLevel.DEBUG)
                    continue

            self.logger.debug(
                f"Successfully loaded metrics for {len(detailed_metrics)} laps from files", LogLevel.DEBUG)
            return detailed_metrics

        except Exception as e:
            self.logger.error(f"Error loading metrics from files: {e}", LogLevel.NORMAL)
            return {}

    def _experiment_directory_exists(self, controller_name: str, experiment_id: str) -> bool:
        """Check if an experiment directory already exists for this controller and experiment ID."""
        try:
            # Use the resolved results directory
            base_output_dir = getattr(
                self, 'original_base_output_dir', 'data')
            controller_dir = os.path.join(base_output_dir, controller_name)

            if not os.path.exists(controller_dir):
                self.logger.debug(
                    f"Controller directory does not exist: {controller_dir}", LogLevel.DEBUG)
                return False

            # Look for any directory that starts with the experiment_id
            import glob
            exp_pattern = os.path.join(controller_dir, f'{experiment_id}_*')
            existing_dirs = glob.glob(exp_pattern)

            self.logger.debug(
                f"Checking for existing experiment {experiment_id} in {controller_dir}", LogLevel.VERBOSE)
            self.logger.verbose(f"Pattern: {exp_pattern}")
            self.logger.verbose(f"Found: {existing_dirs}")

            return len(existing_dirs) > 0
        except Exception as e:
            self.logger.warn(
                f"Error checking experiment directory existence: {e}", LogLevel.DEBUG)
            return False

    def _get_next_experiment_id(self, controller_name: str) -> str:
        """Get the next available experiment ID for this controller."""
        try:
            import re
            import glob

            base_output_dir = getattr(
                self, 'original_base_output_dir', 'data')
            controller_dir = os.path.join(base_output_dir, controller_name)

            max_exp_num = 0

            if os.path.exists(controller_dir):
                exp_pattern = os.path.join(controller_dir, 'exp_*')
                exp_dirs = glob.glob(exp_pattern)

                # Extract experiment numbers from existing directories
                for path in exp_dirs:
                    # Look for pattern: exp_XXX_timestamp or exp_XXX
                    basename = os.path.basename(path)
                    match = re.search(r'exp_(\d+)', basename)
                    if match:
                        exp_num = int(match.group(1))
                        self.logger.verbose(f"  - Found: {basename} (exp #{exp_num})")
                        max_exp_num = max(max_exp_num, exp_num)

            next_exp_num = max_exp_num + 1
            next_id = f'exp_{next_exp_num:03d}'
            return next_id

        except Exception as e:
            self.logger.error(
                f"Error generating next experiment ID: {e}", LogLevel.NORMAL)
            # Fallback to timestamp-based ID
            from datetime import datetime
            timestamp = datetime.now().strftime("%H%M%S")
            return f'exp_{timestamp}'

    def _publish_race_status(self):
        """Publish current race status."""
        race_stats = self.lap_detector.get_race_stats()

        # Publish lap count
        lap_count_msg = Int32()
        lap_count_msg.data = race_stats['current_lap']
        self.lap_count_pub.publish(lap_count_msg)

        # Publish last lap time
        if race_stats['lap_times']:
            lap_time_msg = Float32()
            lap_time_msg.data = race_stats['lap_times'][-1]
            self.lap_time_pub.publish(lap_time_msg)

        # Publish race status based on ending mode and current state
        if race_stats['race_completed']:
            if race_stats['race_ended_by_crash']:
                status = "CRASHED"
            elif race_stats['race_ended_manually']:
                status = "TIMEOUT"
            else:
                status = "FINISHED"
        elif race_stats['race_started']:
            status = f"RACING-{race_stats['race_ending_mode'].upper()}"
        else:
            status = "WAITING"

        race_status_msg = String()
        race_status_msg.data = status
        self.race_status_pub.publish(race_status_msg)

        # Publish race running status
        race_running_msg = Bool()
        race_running_msg.data = race_stats['race_started'] and not race_stats['race_completed']
        self.race_running_pub.publish(race_running_msg)

    def _periodic_visualization_update(self):
        """Periodically update visualization to ensure start line stays visible."""
        try:
            # Re-publish enhanced start line marker
            self.visualization_publisher.publish_start_line_marker()

        except Exception as e:
            self.logger.error(
                f"Error in periodic visualization update: {e}", LogLevel.NORMAL)

    def _save_intermediate_results(self):
        """Save intermediate results for manual mode."""
        race_stats = self.lap_detector.get_race_stats()

        if race_stats['race_started'] and not race_stats['race_completed']:
            current_time = self.get_clock().now()
            elapsed_time = (current_time.nanoseconds -
                            race_stats.get('race_start_time', 0)) / 1e9

            self.logger.info(f"💾 Saving intermediate results (Manual mode) - "
                                   f"Elapsed: {elapsed_time:.1f}s, Laps: {len(race_stats['lap_times'])}", LogLevel.NORMAL)

            # Save intermediate race data
            race_data = {
                'total_race_time': elapsed_time,
                'lap_times': race_stats['lap_times'],
                'controller_name': self.config['controller_name'],
                'experiment_id': self.config['experiment_id'],
                'race_ending_mode': self.config['race_ending_mode'],
                'race_ending_reason': 'Intermediate save',
                'laps_completed': len(race_stats['lap_times']),
                'intermediate_save': True,
                'timestamp': datetime.now().isoformat()
            }

            # Save intermediate results with clean filename
            filename = "intermediate_results.csv"
            self.data_manager.save_race_results_to_csv(
                race_data, filename_override=filename)

    def destroy_node(self):
        """Clean up resources before node destruction."""
        self.logger.shutdown("Race Monitor", LogLevel.MINIMAL)

        # Stop monitoring systems
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.stop_monitoring()

        # Clean up timers
        if hasattr(self, 'intermediate_save_timer'):
            self.intermediate_save_timer.cancel()
        if hasattr(self, 'visualization_timer'):
            self.visualization_timer.cancel()

        # Save any remaining data
        race_stats = self.lap_detector.get_race_stats()
        if race_stats['race_started']:
            current_time = self.get_clock().now()

            # Calculate elapsed time if race is still active
            if not race_stats['race_completed'] and race_stats.get('race_start_time'):
                elapsed_time = (current_time.nanoseconds -
                                race_stats['race_start_time']) / 1e9
            else:
                elapsed_time = race_stats['total_race_time']

            race_data = {
                'total_race_time': elapsed_time,
                'lap_times': race_stats['lap_times'],
                'controller_name': self.config['controller_name'],
                'experiment_id': self.config['experiment_id'],
                'race_ending_mode': self.config['race_ending_mode'],
                'race_ending_reason': race_stats.get('race_ending_reason', 'Node shutdown'),
                'laps_completed': len(race_stats['lap_times']),
                'forced_shutdown': not race_stats['race_completed'],
                'timestamp': datetime.now().isoformat()
            }
            self.data_manager.save_race_results_to_csv(race_data)


            if not race_stats['race_completed'] and self.config['race_ending_mode'] in ['manual', 'crash']:
                if len(race_stats['lap_times']) > 0: 
                    try:
                        self._perform_comprehensive_analysis(race_data)
                        
                        if self.config['enable_computational_monitoring']:
                            self.performance_monitor.save_performance_data_to_csv(
                                os.path.join(
                                    self.config['trajectory_output_directory'], 'cpu_performance_data')
                            )
                        
                        self.logger.success("✓ Comprehensive analysis completed successfully", LogLevel.NORMAL)
                    except Exception as e:
                        self.logger.error(f"Failed to complete comprehensive analysis: {e}", LogLevel.MINIMAL)

        super().destroy_node()

    def _quaternion_to_yaw(self, q):
        """
        Convert quaternion to yaw angle in radians.

        Args:
            q: Quaternion object with x, y, z, w attributes

        Returns:
            float: Yaw angle in radians
        """
        x, y, z, w = q.x, q.y, q.z, q.w
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)


def main(args=None):
    """Main entry point for the race monitor node."""
    rclpy.init(args=args)

    node = None
    try:
        node = RaceMonitor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except ExternalShutdownException:
        # This happens when the process is killed externally (e.g., timeout)
        pass
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Safe cleanup - only if node was created and RCL is still valid
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass  # Ignore cleanup errors

        # Only shutdown if RCL context is still valid
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass  # Ignore shutdown errors if already shut down


if __name__ == '__main__':
    main()
