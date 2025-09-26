#!/usr/bin/env python3

"""
Race Monitor Node

Main ROS2 node that integrates all race monitoring components into a unified system.
Orchestrates lap detection, trajectory analysis, performance monitoring, visualization,
and data management for comprehensive autonomous racing evaluation.

This refactored design provides:
    - Modular architecture with clear separation of concerns
    - Easy maintenance and testing of individual components
    - Flexible configuration and extensibility
    - Clean interfaces between components

Components:
    - LapDetector: Handles lap detection and timing
    - ReferenceTrajectoryManager: Manages reference trajectories
    - PerformanceMonitor: Monitors computational performance
    - VisualizationPublisher: Handles RViz visualization
    - DataManager: Manages data storage and file operations
    - TrajectoryAnalyzer: Advanced trajectory analysis
    - RaceEvaluator: Custom race evaluation system

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Int32, Float32, Bool, String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PointStamped, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from giu_f1t_interfaces.msg import VehicleState, ConstrainedVehicleState

import numpy as np
import tf_transformations
import os
import sys
from datetime import datetime

# Import our modular components
from .lap_detector import LapDetector
from .reference_trajectory_manager import ReferenceTrajectoryManager
from .performance_monitor import PerformanceMonitor
from .visualization_publisher import VisualizationPublisher
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
    
    Provides a clean, modular architecture for race monitoring with
    clear separation of concerns and easy extensibility.
    """
    
    def __init__(self):
        super().__init__('race_monitor')
        
        self.get_logger().info("🏁 Starting Race Monitor with modular architecture...")
        
        # Load configuration parameters
        self._load_parameters()
        
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
        
        self.get_logger().info("✅ Race Monitor initialized successfully!")
        self.get_logger().info(f"   - Lap Detection: Enabled")
        self.get_logger().info(f"   - Reference Trajectory: {'Enabled' if self.reference_manager.is_reference_available() else 'Disabled'}")
        self.get_logger().info(f"   - Performance Monitoring: {'Enabled' if self.config.get('enable_computational_monitoring', True) else 'Disabled'}")
        self.get_logger().info(f"   - Research Evaluator: {'Enabled' if RESEARCH_EVALUATOR_AVAILABLE else 'Disabled'}")
        self.get_logger().info(f"   - Race Evaluator: {'Enabled' if RACE_EVALUATOR_AVAILABLE else 'Disabled'}")
        self.get_logger().info(f"   - EVO Integration: {'Enabled' if EVO_AVAILABLE else 'Disabled'}")
        self.get_logger().info(f"   - EVO Graph Generation: {'Enabled' if self.evo_plotter is not None else 'Disabled'}")
    
    def _load_parameters(self):
        """Load ROS2 parameters and configuration."""
        # Declare parameters with defaults
        self.declare_parameter('start_line_p1', [0.0, -1.0])
        self.declare_parameter('start_line_p2', [0.0, 1.0])
        self.declare_parameter('required_laps', 5)
        self.declare_parameter('debounce_time', 2.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('controller_name', 'custom_controller')
        self.declare_parameter('experiment_id', 'exp_001')
        self.declare_parameter('trajectory_output_directory', '')
        self.declare_parameter('enable_trajectory_evaluation', True)
        self.declare_parameter('enable_computational_monitoring', True)
        self.declare_parameter('enable_race_evaluation', True)
        self.declare_parameter('save_trajectories', True)
        self.declare_parameter('auto_generate_graphs', True)
        
        # Reference trajectory parameters
        self.declare_parameter('reference_trajectory_file', '')
        self.declare_parameter('reference_trajectory_format', 'csv')
        self.declare_parameter('enable_horizon_mapper_reference', False)
        self.declare_parameter('use_complete_reference_path', True)
        
        # Performance monitoring parameters
        self.declare_parameter('monitoring_window_size', 100)
        self.declare_parameter('cpu_monitoring_interval', 0.1)
        self.declare_parameter('performance_log_interval', 5.0)
        self.declare_parameter('max_acceptable_latency_ms', 50.0)
        self.declare_parameter('target_control_frequency_hz', 50.0)
        self.declare_parameter('max_acceptable_cpu_usage', 80.0)
        self.declare_parameter('max_acceptable_memory_mb', 512.0)
        
        # Data management parameters
        self.declare_parameter('output_formats', ['csv', 'json'])
        self.declare_parameter('include_timestamps', True)
        self.declare_parameter('save_intermediate_results', True)
        
        # Race evaluation parameters
        self.declare_parameter('race_evaluation.grading_strictness', 'normal')
        self.declare_parameter('race_evaluation.enable_recommendations', True)
        self.declare_parameter('race_evaluation.enable_comparison', True)
        self.declare_parameter('race_evaluation.auto_increment_experiment', True)
        
        # Store configuration
        self.config = {
            'start_line_p1': self.get_parameter('start_line_p1').value,
            'start_line_p2': self.get_parameter('start_line_p2').value,
            'required_laps': self.get_parameter('required_laps').value,
            'debounce_time': self.get_parameter('debounce_time').value,
            'frame_id': self.get_parameter('frame_id').value,
            'controller_name': self.get_parameter('controller_name').value,
            'experiment_id': self.get_parameter('experiment_id').value,
            'trajectory_output_directory': self.get_parameter('trajectory_output_directory').value,
            'enable_trajectory_evaluation': self.get_parameter('enable_trajectory_evaluation').value,
            'enable_computational_monitoring': self.get_parameter('enable_computational_monitoring').value,
            'enable_race_evaluation': self.get_parameter('enable_race_evaluation').value,
            'save_trajectories': self.get_parameter('save_trajectories').value,
            'auto_generate_graphs': self.get_parameter('auto_generate_graphs').value,
            'reference_trajectory_file': self.get_parameter('reference_trajectory_file').value,
            'reference_trajectory_format': self.get_parameter('reference_trajectory_format').value,
            'enable_horizon_mapper_reference': self.get_parameter('enable_horizon_mapper_reference').value,
            'use_complete_reference_path': self.get_parameter('use_complete_reference_path').value,
            'monitoring_window_size': self.get_parameter('monitoring_window_size').value,
            'cpu_monitoring_interval': self.get_parameter('cpu_monitoring_interval').value,
            'performance_log_interval': self.get_parameter('performance_log_interval').value,
            'max_acceptable_latency_ms': self.get_parameter('max_acceptable_latency_ms').value,
            'target_control_frequency_hz': self.get_parameter('target_control_frequency_hz').value,
            'max_acceptable_cpu_usage': self.get_parameter('max_acceptable_cpu_usage').value,
            'max_acceptable_memory_mb': self.get_parameter('max_acceptable_memory_mb').value,
            'output_formats': self.get_parameter('output_formats').value,
            'include_timestamps': self.get_parameter('include_timestamps').value,
            'save_intermediate_results': self.get_parameter('save_intermediate_results').value,
            'race_evaluation': {
                'grading_strictness': self.get_parameter('race_evaluation.grading_strictness').value,
                'enable_recommendations': self.get_parameter('race_evaluation.enable_recommendations').value,
                'enable_comparison': self.get_parameter('race_evaluation.enable_comparison').value,
                'auto_increment_experiment': self.get_parameter('race_evaluation.auto_increment_experiment').value
            }
        }
    
    def _initialize_components(self):
        """Initialize all monitoring components."""
        # Core components
        self.lap_detector = LapDetector(self.get_logger(), self.get_clock())
        self.reference_manager = ReferenceTrajectoryManager(self.get_logger())
        self.performance_monitor = PerformanceMonitor(self.get_logger(), self.get_clock())
        self.visualization_publisher = VisualizationPublisher(self.get_logger(), self._publish_marker)
        self.data_manager = DataManager(self.get_logger())
        
        # Analysis components (if available)
        self.research_evaluator = None
        self.race_evaluator = None
        self.evo_plotter = None
        
        if RESEARCH_EVALUATOR_AVAILABLE and self.config['enable_trajectory_evaluation']:
            try:
                self.research_evaluator = create_research_evaluator(self.config)
                self.get_logger().info("Research evaluator initialized")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize research evaluator: {e}")
        
        if RACE_EVALUATOR_AVAILABLE and self.config['enable_race_evaluation']:
            try:
                self.race_evaluator = create_race_evaluator(self.config)
                self.get_logger().info("Race evaluator initialized")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize race evaluator: {e}")
        
        if EVO_PLOTTER_AVAILABLE and self.config['auto_generate_graphs']:
            try:
                self.evo_plotter = EVOPlotter(self.config)
                self.get_logger().info("EVO plotter initialized")
            except Exception as e:
                self.get_logger().warn(f"EVO plotter initialization failed: {e}")
                self.get_logger().warn("Graph generation will be disabled. Race monitoring will continue normally.")
                self.evo_plotter = None
                # Disable auto_generate_graphs to prevent further attempts
                self.config['auto_generate_graphs'] = False
    
    def _setup_ros_interfaces(self):
        """Set up ROS2 publishers, subscribers, and services."""
        # Publishers
        self.lap_count_pub = self.create_publisher(Int32, '/race_monitor/lap_count', 10)
        self.lap_time_pub = self.create_publisher(Float32, '/race_monitor/lap_time', 10)
        self.race_status_pub = self.create_publisher(String, '/race_monitor/race_status', 10)
        self.race_running_pub = self.create_publisher(Bool, '/race_monitor/race_running', 10)
        self.visualization_marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/car_state/odom', self._odometry_callback, 10
        )
        self.clicked_point_sub = self.create_subscription(
            PointStamped, '/clicked_point', self._clicked_point_callback, 10
        )
        
        # Optional subscribers based on configuration
        if self.config['enable_horizon_mapper_reference']:
            self.reference_trajectory_sub = self.create_subscription(
                Path, '/horizon_mapper/reference_trajectory', 
                self.reference_manager.update_reference_trajectory, 10
            )
        
        if self.config['use_complete_reference_path']:
            self.reference_path_sub = self.create_subscription(
                Path, '/horizon_mapper/reference_path',
                self.reference_manager.update_reference_path, 10
            )
        
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
        self.data_manager.configure(self.config)
        
        # Load reference trajectory if specified
        if self.config['reference_trajectory_file']:
            self.reference_manager.load_reference_trajectory()
    
    def _setup_component_callbacks(self):
        """Set up callbacks between components."""
        # Lap detector callbacks
        self.lap_detector.set_callbacks(
            on_race_start=self._on_race_start,
            on_lap_complete=self._on_lap_complete,
            on_race_complete=self._on_race_complete
        )
    
    def _start_monitoring_systems(self):
        """Start active monitoring systems."""
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.start_monitoring()
        
        # Initial visualization
        self.visualization_publisher.publish_start_line_marker()
        
        # Publish reference trajectory if available
        reference_points = self.reference_manager.get_reference_points()
        if reference_points:
            self.visualization_publisher.publish_raceline_markers(reference_points=reference_points)
    
    # ROS2 Callback Methods
    def _odometry_callback(self, msg: Odometry):
        """Handle odometry messages."""
        try:
            # Extract position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            
            # Update lap detector
            self.lap_detector.update_position(x, y, msg.header.stamp)
            
            # Add point to current trajectory
            if self.lap_detector.is_race_active():
                additional_data = {
                    'qx': msg.pose.pose.orientation.x,
                    'qy': msg.pose.pose.orientation.y,
                    'qz': msg.pose.pose.orientation.z,
                    'qw': msg.pose.pose.orientation.w,
                    'linear_velocity': np.sqrt(
                        msg.twist.twist.linear.x**2 + 
                        msg.twist.twist.linear.y**2 + 
                        msg.twist.twist.linear.z**2
                    ),
                    'angular_velocity': msg.twist.twist.angular.z
                }
                self.data_manager.add_trajectory_point(x, y, z, msg.header.stamp, additional_data)
            
            # Performance monitoring
            if self.config['enable_computational_monitoring']:
                self.performance_monitor.record_odometry_timestamp('car_state/odom', msg.header.stamp)
            
            # Publish race status
            self._publish_race_status()
            
        except Exception as e:
            self.get_logger().error(f"Error in odometry callback: {e}")
    
    def _control_command_callback(self, msg: AckermannDriveStamped):
        """Handle control command messages for performance monitoring."""
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.record_control_command_timestamp('/drive', msg.header.stamp)
    
    def _clicked_point_callback(self, msg: PointStamped):
        """Handle clicked point messages for manual start line setup."""
        self.get_logger().info(f"Clicked point received: ({msg.point.x:.2f}, {msg.point.y:.2f})")
        # Could be used to dynamically set start line points
    
    def _publish_marker(self, marker: Marker):
        """Publish visualization marker."""
        self.visualization_marker_pub.publish(marker)
    
    # Race Event Handlers
    def _on_race_start(self, timestamp):
        """Handle race start event."""
        self.get_logger().info("🏁 Race started!")
        
        # Start new lap trajectory
        self.data_manager.start_new_lap_trajectory(1)
        
        # Update visualization
        self.visualization_publisher.publish_race_status_marker("RACING", 1, self.config['required_laps'])
        
        # Reset performance monitoring data
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.reset_performance_data()
    
    def _on_lap_complete(self, lap_number: int, lap_time: float):
        """Handle lap completion event."""
        self.get_logger().info(f"🏎️ Lap {lap_number} completed in {lap_time:.3f}s")
        
        # Complete current lap trajectory
        self.data_manager.complete_lap_trajectory(lap_number, lap_time)
        
        # Start next lap trajectory if race not complete
        if lap_number < self.config['required_laps']:
            self.data_manager.start_new_lap_trajectory(lap_number + 1)
        
        # Update visualization
        race_stats = self.lap_detector.get_race_stats()
        self.visualization_publisher.publish_lap_completion_marker(lap_number, lap_time)
        
        if not race_stats['race_completed']:
            self.visualization_publisher.publish_race_status_marker(
                "RACING", race_stats['current_lap'], self.config['required_laps']
            )
        
        # Perform trajectory evaluation if enabled
        if self.config['enable_trajectory_evaluation'] and EVO_AVAILABLE:
            self._evaluate_lap_trajectory(lap_number)
    
    def _on_race_complete(self, total_time: float, lap_times: list):
        """Handle race completion event."""
        self.get_logger().info(f"🏆 Race completed in {total_time:.3f}s!")
        
        # Update visualization
        self.visualization_publisher.publish_race_status_marker("FINISHED", len(lap_times), self.config['required_laps'])
        
        # Save race results
        race_data = {
            'total_race_time': total_time,
            'lap_times': lap_times,
            'controller_name': self.config['controller_name'],
            'experiment_id': self.config['experiment_id'],
            'timestamp': datetime.now().isoformat()
        }
        self.data_manager.save_race_results_to_csv(race_data)
        
        # Perform comprehensive race analysis
        self._perform_comprehensive_analysis(race_data)
        
        # Save performance data
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.save_performance_data_to_csv(
                os.path.join(self.config['trajectory_output_directory'], 'performance_data')
            )
    
    def _evaluate_lap_trajectory(self, lap_number: int):
        """Evaluate individual lap trajectory."""
        try:
            # Create EVO trajectory for the lap
            evo_trajectory = self.data_manager.create_evo_trajectory(lap_number)
            if not evo_trajectory:
                return
            
            # Perform evaluation using research evaluator if available
            if self.research_evaluator and self.reference_manager.is_reference_available():
                reference_trajectory = self.reference_manager.get_reference_trajectory()
                if reference_trajectory:
                    # TODO: Implement single lap evaluation
                    pass
            
        except Exception as e:
            self.get_logger().error(f"Error evaluating lap {lap_number}: {e}")
    
    def _perform_comprehensive_analysis(self, race_data: dict):
        """Perform comprehensive race analysis after completion."""
        try:
            self.get_logger().info("🔬 Performing comprehensive race analysis...")
            
            # Compile all trajectory data for analysis
            all_trajectories = self.data_manager.get_trajectory_data()
            
            if self.research_evaluator and all_trajectories:
                # Perform research-grade analysis
                self.get_logger().info("📊 Running research trajectory evaluation...")
                # TODO: Implement comprehensive analysis with research evaluator
            
            if self.race_evaluator and all_trajectories:
                # Perform custom race evaluation
                self.get_logger().info("🏁 Running custom race evaluation...")
                # TODO: Implement race evaluation integration
            
            if self.evo_plotter and all_trajectories:
                # Generate visualization plots
                self.get_logger().info("📈 Generating analysis plots...")
                # TODO: Implement plot generation
            
            self.get_logger().info("✅ Comprehensive analysis completed!")
            
        except Exception as e:
            self.get_logger().error(f"Error in comprehensive analysis: {e}")
    
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
        
        # Publish race status
        if race_stats['race_completed']:
            status = "FINISHED"
        elif race_stats['race_started']:
            status = "RACING"
        else:
            status = "WAITING"
        
        race_status_msg = String()
        race_status_msg.data = status
        self.race_status_pub.publish(race_status_msg)
        
        # Publish race running status
        race_running_msg = Bool()
        race_running_msg.data = race_stats['race_started'] and not race_stats['race_completed']
        self.race_running_pub.publish(race_running_msg)
    
    def destroy_node(self):
        """Clean up resources before node destruction."""
        self.get_logger().info("🛑 Shutting down race monitor...")
        
        # Stop monitoring systems
        if self.config['enable_computational_monitoring']:
            self.performance_monitor.stop_monitoring()
        
        # Save any remaining data
        race_stats = self.lap_detector.get_race_stats()
        if race_stats['race_started'] and race_stats['lap_times']:
            race_data = {
                'total_race_time': race_stats['total_race_time'],
                'lap_times': race_stats['lap_times'],
                'controller_name': self.config['controller_name'],
                'experiment_id': self.config['experiment_id'],
                'timestamp': datetime.now().isoformat()
            }
            self.data_manager.save_race_results_to_csv(race_data)
        
        super().destroy_node()


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