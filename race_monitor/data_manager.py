#!/usr/bin/env python3

"""
Race Monitor Data Manager

Centralized data management system for race monitoring with comprehensive
file I/O operations, trajectory storage, and multi-format export capabilities.

This module provides the core data storage infrastructure for the race monitor,
handling everything from real-time trajectory recording to advanced metrics
export and research data compilation.

Key Features:
    - Real-time trajectory recording and storage
    - Multi-format export (CSV, TUM, JSON, Pickle, MAT, compressed)
    - Advanced metrics compilation (APE/RPE analysis)
    - Intelligent directory management and organization
    - Race evaluation data handling with fallback mechanisms
    - Research-ready data export with filtering capabilities

Output Formats:
    - Stable filenames without timestamps for consistent access
    - Compressed .rsum format for efficient storage
    - Consolidated JSON results for easy parsing
    - EVO-compatible trajectory formats for analysis

License: MIT
"""

import rclpy
import os
import csv
import json
import pickle
import numpy as np
import gzip
from datetime import datetime
from typing import Dict, List, Any, Optional
from .metadata_manager import MetadataManager

# MAT file support
try:
    from scipy.io import savemat
    MAT_AVAILABLE = True
except ImportError:
    MAT_AVAILABLE = False


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


# EVO imports for trajectory handling
try:
    from evo.core import trajectory
    from evo.tools import file_interface
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


class DataManager:
    """
    Manages data storage and file operations for race monitoring.

    Handles trajectory data storage, file export in multiple formats,
    and provides centralized data management for all race monitor components.
    """

    def __init__(self, logger):
        """
        Initialize the data manager.

        Args:
            logger: ROS2 logger instance
        """
        self.logger = logger

        # Basic configuration
        self.trajectory_output_directory = ""
        self.save_trajectories = True
        self.save_intermediate_results = True

        # Advanced file output configuration
        self.output_formats = ["csv", "json"]
        self.include_timestamps = True
        self.export_to_pandas = True
        self.save_detailed_statistics = True
        self.save_filtered_trajectories = True
        self.export_research_summary = True

        # Research and analysis configuration
        self.controller_name = 'custom_controller'
        self.experiment_id = 'exp_001'
        self.test_description = 'Controller performance evaluation'
        self.enable_advanced_metrics = True
        self.calculate_all_statistics = True

        # Trajectory analysis configuration
        self.evaluate_smoothness = True
        self.evaluate_consistency = True
        self.evaluate_efficiency = True
        self.evaluate_aggressiveness = True
        self.evaluate_stability = True

        # EVO metrics configuration
        self.pose_relations = ['translation_part', 'rotation_part', 'full_transformation']
        self.statistics_types = ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse']

        # Filtering configuration
        self.apply_trajectory_filtering = True
        self.filter_types = ['motion', 'distance', 'angle']
        self.filter_parameters = {
            'motion_threshold': 0.1,
            'distance_threshold': 0.05,
            'angle_threshold': 0.1
        }

        # Graph generation configuration
        self.auto_generate_graphs = True
        self.graph_output_directory = ''
        self.graph_formats = ['png', 'pdf']

        # Plot appearance configuration
        self.plot_figsize = [12.0, 8.0]
        self.plot_dpi = 300
        self.plot_style = 'seaborn'
        self.plot_color_scheme = 'viridis'

        # Graph types configuration
        self.generate_trajectory_plots = True
        self.generate_xyz_plots = True
        self.generate_rpy_plots = True
        self.generate_speed_plots = True
        self.generate_error_plots = True
        self.generate_metrics_plots = True

        # Data storage
        self.current_lap_trajectory = []
        self.completed_trajectories = {}  # lap_number -> trajectory_data
        self.evaluation_summaries = []
        self.race_results = []

        # File paths
        self.base_output_dir = ""
        self.trajectory_dir = ""
        self.results_dir = ""

    def configure(self, config: dict):
        """
        Configure data manager parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        # Core data storage configuration
        raw_trajectory_dir = config.get('trajectory_output_directory', "")

        # Use the trajectory_output_directory directly (absolute paths are used as-is)
        self.trajectory_output_directory = raw_trajectory_dir

        self.save_trajectories = config.get('save_trajectories', self.save_trajectories)
        self.save_intermediate_results = config.get('save_intermediate_results', self.save_intermediate_results)

        # Advanced file output configuration
        self.output_formats = config.get('output_formats', self.output_formats)
        self.include_timestamps = config.get('include_timestamps', self.include_timestamps)
        self.export_to_pandas = config.get('export_to_pandas', True)
        self.save_detailed_statistics = config.get('save_detailed_statistics', True)
        self.save_filtered_trajectories = config.get('save_filtered_trajectories', True)
        self.export_research_summary = config.get('export_research_summary', True)

        # Research and analysis configuration
        self.controller_name = config.get('controller_name', 'custom_controller')
        self.experiment_id = config.get('experiment_id', 'exp_001')
        self.test_description = config.get('test_description', 'Controller performance evaluation')
        self.enable_advanced_metrics = config.get('enable_advanced_metrics', True)
        self.calculate_all_statistics = config.get('calculate_all_statistics', True)

        # Trajectory analysis configuration
        self.evaluate_smoothness = config.get('evaluate_smoothness', True)
        self.evaluate_consistency = config.get('evaluate_consistency', True)
        self.evaluate_efficiency = config.get('evaluate_efficiency', True)
        self.evaluate_aggressiveness = config.get('evaluate_aggressiveness', True)
        self.evaluate_stability = config.get('evaluate_stability', True)

        # EVO metrics configuration
        self.pose_relations = config.get('pose_relations', ['translation_part', 'rotation_part', 'full_transformation'])
        self.statistics_types = config.get('statistics_types', ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'])

        # Filtering configuration
        self.apply_trajectory_filtering = config.get('apply_trajectory_filtering', True)
        self.filter_types = config.get('filter_types', ['motion', 'distance', 'angle'])
        self.filter_parameters = config.get('filter_parameters', {
            'motion_threshold': 0.1,
            'distance_threshold': 0.05,
            'angle_threshold': 0.1
        })

        # Graph generation configuration
        self.auto_generate_graphs = config.get('auto_generate_graphs', True)
        self.graph_output_directory = config.get('graph_output_directory', '')
        self.graph_formats = config.get('graph_formats', ['png', 'pdf'])

        # Plot appearance configuration
        self.plot_figsize = config.get('plot_figsize', [12.0, 8.0])
        self.plot_dpi = config.get('plot_dpi', 300)
        self.plot_style = config.get('plot_style', 'seaborn')
        self.plot_color_scheme = config.get('plot_color_scheme', 'viridis')

        # Graph types configuration
        self.generate_trajectory_plots = config.get('generate_trajectory_plots', True)
        self.generate_xyz_plots = config.get('generate_xyz_plots', True)
        self.generate_rpy_plots = config.get('generate_rpy_plots', True)
        self.generate_speed_plots = config.get('generate_speed_plots', True)
        self.generate_error_plots = config.get('generate_error_plots', True)
        self.generate_metrics_plots = config.get('generate_metrics_plots', True)

        # Set up directory structure
        if self.trajectory_output_directory:
            self._setup_directories()

        # Initialize metadata manager
        self.metadata_manager = MetadataManager(self.trajectory_output_directory, self.logger)

        self.logger.info(f"Data manager configured with {len(self.output_formats)} format(s)")
        if self.enable_advanced_metrics:
            self.logger.info("Advanced metrics and plotting enabled")

    def _setup_directories(self):
        """Create base directory structure for data storage."""
        try:
            self.base_output_dir = self.trajectory_output_directory

            # Only create base output directory initially
            # Specific experiment directories will be created by create_run_directory()
            os.makedirs(self.base_output_dir, exist_ok=True)
            self.trajectory_dir = None
            self.results_dir = None

        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")

    def start_new_lap_trajectory(self, lap_number: int):
        """Start recording a new lap trajectory."""
        self.current_lap_trajectory = []

    def add_trajectory_point(self, x: float, y: float, z: float = 0.0,
                             timestamp=None, additional_data: Dict = None):
        """
        Add a point to the current lap trajectory.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            timestamp: Point timestamp
            additional_data: Additional data to store with the point
        """
        point_data = {
            'x': float(x),
            'y': float(y),
            'z': float(z)
        }

        if timestamp and self.include_timestamps:
            point_data['timestamp'] = time_to_nanoseconds(timestamp) / 1e9

        if additional_data:
            point_data.update(additional_data)

        self.current_lap_trajectory.append(point_data)

    def complete_lap_trajectory(self, lap_number: int, lap_time: float) -> bool:
        """
        Complete the current lap trajectory and save it.

        Args:
            lap_number: Completed lap number
            lap_time: Lap time in seconds

        Returns:
            bool: True if successfully saved
        """
        if not self.current_lap_trajectory:
            self.logger.warn(f"No trajectory data for lap {lap_number}")
            return False

        # Store completed trajectory
        trajectory_data = {
            'lap_number': lap_number,
            'lap_time': lap_time,
            'points': self.current_lap_trajectory.copy(),
            'timestamp': datetime.now().isoformat(),
            'point_count': len(self.current_lap_trajectory)
        }

        self.completed_trajectories[lap_number] = trajectory_data

        if self.save_trajectories:
            success = self.save_trajectory_to_file(trajectory_data)
            return success

        return True

    def save_trajectory_to_file(self, trajectory_data: Dict) -> bool:
        """
        Save trajectory data to file in multiple formats.

        Args:
            trajectory_data: Trajectory data to save

        Returns:
            bool: True if successfully saved
        """
        # Ensure trajectory directory is set up
        if self.trajectory_dir is None:
            self.logger.warning("Trajectory directory not set up yet - cannot save trajectory")
            return False

        lap_number = trajectory_data['lap_number']
        points = trajectory_data['points']

        try:
            # Save in requested formats
            success = True

            if 'csv' in self.output_formats:
                success &= self._save_trajectory_csv(lap_number, points)

            if 'tum' in self.output_formats:
                success &= self._save_trajectory_tum(lap_number, points)

            if 'json' in self.output_formats:
                success &= self._save_trajectory_json(lap_number, trajectory_data)

            if 'pickle' in self.output_formats:
                success &= self._save_trajectory_pickle(lap_number, trajectory_data)

            if 'mat' in self.output_formats:
                success &= self._save_trajectory_mat(lap_number, trajectory_data)

            return success

        except Exception as e:
            self.logger.error(f"Error saving trajectory for lap {lap_number}: {e}")
            return False

    def _save_trajectory_csv(self, lap_number: int, points: List[Dict]) -> bool:
        """Save trajectory as CSV file."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.csv"
            filepath = self._get_organized_file_path(self.trajectory_dir, filename)

            with open(filepath, 'w', newline='') as csvfile:
                if points:
                    fieldnames = list(points[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(points)

            return True

        except Exception as e:
            self.logger.error(f"Error saving CSV trajectory: {e}")
            return False

    def _save_trajectory_tum(self, lap_number: int, points: List[Dict]) -> bool:
        """Save trajectory in TUM format."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.tum"
            filepath = self._get_organized_file_path(self.trajectory_dir, filename)

            with open(filepath, 'w') as f:
                for point in points:
                    timestamp = point.get('timestamp', 0.0)
                    x = point['x']
                    y = point['y']
                    z = point['z']
                    # Use identity quaternion if no orientation data
                    qx = point.get('qx', 0.0)
                    qy = point.get('qy', 0.0)
                    qz = point.get('qz', 0.0)
                    qw = point.get('qw', 1.0)

                    f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

            return True

        except Exception as e:
            self.logger.error(f"Error saving TUM trajectory: {e}")
            return False

    def _save_trajectory_json(self, lap_number: int, trajectory_data: Dict) -> bool:
        """Save trajectory as JSON file."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.json"
            filepath = self._get_organized_file_path(self.trajectory_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(trajectory_data, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error saving JSON trajectory: {e}")
            return False

    def _save_trajectory_pickle(self, lap_number: int, trajectory_data: Dict) -> bool:
        """Save trajectory as Pickle file."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.pkl"
            filepath = self._get_organized_file_path(self.trajectory_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(trajectory_data, f)

            return True

        except Exception as e:
            self.logger.error(f"Error saving Pickle trajectory: {e}")
            return False

    def _save_trajectory_mat(self, lap_number: int, trajectory_data: Dict) -> bool:
        """Save trajectory as MATLAB MAT file."""
        if not MAT_AVAILABLE:
            self.logger.warning("scipy.io not available, cannot save MAT files")
            return False

        try:
            filename = f"lap_{lap_number:03d}_trajectory.mat"
            filepath = self._get_organized_file_path(self.trajectory_dir, filename)

            # Prepare data for MAT format
            points = trajectory_data['points']
            mat_data = {
                'lap_number': lap_number,
                'lap_time': trajectory_data.get('lap_time', 0.0),
                'point_count': len(points),
                'timestamp': trajectory_data.get('timestamp', ''),
                'x': np.array([p['x'] for p in points]),
                'y': np.array([p['y'] for p in points]),
                'z': np.array([p['z'] for p in points])
            }

            # Add timestamps if available
            if points and 'timestamp' in points[0]:
                mat_data['timestamps'] = np.array([p['timestamp'] for p in points])

            # Add orientation data if available
            if points and 'qx' in points[0]:
                mat_data['qx'] = np.array([p['qx'] for p in points])
                mat_data['qy'] = np.array([p['qy'] for p in points])
                mat_data['qz'] = np.array([p['qz'] for p in points])
                mat_data['qw'] = np.array([p['qw'] for p in points])

            # Add any additional data fields
            additional_fields = {}
            for point in points:
                for key, value in point.items():
                    if key not in ['x', 'y', 'z', 'timestamp', 'qx', 'qy', 'qz', 'qw']:
                        if key not in additional_fields:
                            additional_fields[key] = []
                        additional_fields[key].append(value)

            # Convert additional fields to numpy arrays
            for key, values in additional_fields.items():
                mat_data[key] = np.array(values)

            # Save MAT file
            savemat(filepath, mat_data)

            return True

        except Exception as e:
            self.logger.error(f"Error saving MAT trajectory: {e}")
            return False

    def save_race_results_to_csv(self, race_data: Dict, filename_override: str = None) -> bool:
        """
        Save race results to CSV file.

        Args:
            race_data: Race data including lap times, statistics, etc.
            filename_override: Optional custom filename

        Returns:
            bool: True if successfully saved
        """
        try:
            # By default we don't produce timestamped CSV race results anymore (they were confusing/misleading).
            # If a filename_override is provided we will honour it and write a CSV, otherwise produce a
            # consolidated race_results.json via save_consolidated_race_results and skip creating the CSV.
            if filename_override:
                filename = filename_override
                filepath = self._get_organized_file_path(self.results_dir, filename)

                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    # Header
                    writer.writerow(['Race Results', datetime.now().isoformat()])
                    writer.writerow([])

                    # Basic race info
                    writer.writerow(['Total Race Time (s)', race_data.get('total_race_time', 0.0)])
                    writer.writerow(['Total Laps', len(race_data.get('lap_times', []))])
                    writer.writerow(['Controller', race_data.get('controller_name', 'unknown')])
                    writer.writerow(['Experiment ID', race_data.get('experiment_id', 'unknown')])
                    writer.writerow(['Race Ending Mode', race_data.get('race_ending_mode', 'unknown')])
                    writer.writerow(['Race Ending Reason', race_data.get('race_ending_reason', 'unknown')])
                    writer.writerow(['Crashed', race_data.get('crashed', False)])
                    writer.writerow(['Forced Shutdown', race_data.get('forced_shutdown', False)])
                    writer.writerow(['Intermediate Save', race_data.get('intermediate_save', False)])
                    writer.writerow([])

                    # Lap times
                    writer.writerow(['Lap Number', 'Lap Time (s)'])
                    for i, lap_time in enumerate(race_data.get('lap_times', []), 1):
                        writer.writerow([i, f"{lap_time:.4f}"])

                    writer.writerow([])

                    # Statistics
                    lap_times = race_data.get('lap_times', [])
                    if lap_times:
                        writer.writerow(['best_lap_s', f"{min(lap_times):.4f}"])
                        writer.writerow(['worst_lap_s', f"{max(lap_times):.4f}"])
                        writer.writerow(['average_lap_s', f"{np.mean(lap_times):.4f}"])

                self.logger.info(f"Saved race results CSV to: {filepath}")
                return True

            try:
                self.save_consolidated_race_results(race_data)
                return True
            except Exception as e:
                self.logger.error(f"Failed to save consolidated results as fallback: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error saving race results: {e}")
            return False

    def save_evaluation_summary(self, evaluation_data: Dict) -> bool:
        """
        Save evaluation summary data.

        Args:
            evaluation_data: Evaluation summary data

        Returns:
            bool: True if successfully saved
        """
        try:
            self.evaluation_summaries.append(evaluation_data)
            return True

        except Exception as e:
            self.logger.error(f"Error saving evaluation summary: {e}")
            return False

    def get_trajectory_data(self, lap_number: int = None) -> Optional[Dict]:
        """
        Get trajectory data for a specific lap or all laps.

        Args:
            lap_number: Specific lap number, or None for all laps

        Returns:
            Trajectory data or None if not found
        """
        if lap_number is not None:
            return self.completed_trajectories.get(lap_number)
        else:
            return self.completed_trajectories

    def get_current_trajectory_points(self) -> List[Dict]:
        """Get current lap trajectory points."""
        return self.current_lap_trajectory.copy()

    def clear_trajectory_data(self):
        """Clear all stored trajectory data."""
        self.current_lap_trajectory.clear()
        self.completed_trajectories.clear()
        self.evaluation_summaries.clear()
        self.race_results.clear()

    def create_evo_trajectory(self, lap_number: int):
        """
        Create EVO trajectory object from stored trajectory data.

        Args:
            lap_number: Lap number to create trajectory for

        Returns:
            EVO trajectory object or None
        """
        if not EVO_AVAILABLE:
            return None

        trajectory_data = self.completed_trajectories.get(lap_number)
        if not trajectory_data:
            return None

        try:
            points = trajectory_data['points']
            timestamps = np.array([p.get('timestamp', i) for i, p in enumerate(points)])
            positions = np.array([[p['x'], p['y'], p['z']] for p in points])

            if all('qx' in p for p in points):
                # Has orientation data
                orientations = np.array([[p['qx'], p['qy'], p['qz'], p['qw']] for p in points])
                return trajectory.PoseTrajectory3D(positions, orientations, timestamps)
            else:
                # Position only
                return trajectory.PosePath3D(positions, timestamps)

        except Exception as e:
            self.logger.error(f"Error creating EVO trajectory: {e}")
            return None

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        return {
            "completed_laps": len(self.completed_trajectories),
            "current_lap_points": len(self.current_lap_trajectory),
            "evaluation_summaries": len(self.evaluation_summaries),
            "output_directory": self.base_output_dir,
            "output_formats": self.output_formats,
            "total_trajectory_points": sum(
                len(traj['points']) for traj in self.completed_trajectories.values()
            )
        }

    def get_metrics_directory(self) -> str:
        """Get the metrics directory path."""
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "metrics")

        metrics_dir = os.path.join(self.base_output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        return metrics_dir

    def get_filtered_directory(self) -> str:
        """Get the filtered trajectories directory path."""
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "filtered")

        filtered_dir = os.path.join(self.base_output_dir, "filtered")
        os.makedirs(filtered_dir, exist_ok=True)
        return filtered_dir

    def get_graphs_directory(self) -> str:
        """Get the graphs directory path."""
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "graphs")

        graphs_dir = os.path.join(self.base_output_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        return graphs_dir

    def _get_organized_file_path(self, base_dir: str, filename: str) -> str:
        """
        Get organized file path with extension-specific subdirectory.

        Exceptions:
        - experiment_metadata.txt files are saved directly in base directory
        - .md files are saved directly in base directory

        Args:
            base_dir: Base directory path
            filename: Filename with extension

        Returns:
            str: Full path with extension subdirectory (or base directory for exceptions)
        """
        # Check for exceptions that should NOT be organized
        if filename == 'experiment_metadata.txt' or filename.lower().endswith('.md'):
            return os.path.join(base_dir, filename)

        # Extract extension from filename
        _, ext = os.path.splitext(filename)
        ext = ext.lstrip('.').lower()  # Remove dot and make lowercase

        if ext:
            # Create extension-specific subdirectory
            ext_dir = os.path.join(base_dir, ext)
            os.makedirs(ext_dir, exist_ok=True)
            return os.path.join(ext_dir, filename)
        else:
            # No extension, save directly in base directory
            return os.path.join(base_dir, filename)

    def get_current_trajectory_directory(self) -> str:
        """Get the current trajectory base directory path."""
        return self.base_output_dir

    def _is_in_experiment_directory(self) -> bool:
        """Check if base_output_dir points to a controller/experiment directory."""
        path_parts = self.base_output_dir.split(os.sep)
        has_experiment = any('exp_' in part and len(part) > 15 for part in path_parts)
        has_controller = hasattr(self, 'controller_name') and self.controller_name and self.controller_name.strip()
        is_base_level = self.base_output_dir.endswith('evaluation_results')

        return has_experiment and has_controller and not is_base_level

    def get_comparisons_directory(self) -> str:
        """Get the comparisons directory path (at controller level, not experiment level)."""
        # Comparisons should be at the controller level, not experiment level
        # To get controller level, we need to go up from the experiment directory
        controller_name = self.controller_name if hasattr(self, 'controller_name') else 'unknown_controller'

        # If we're in an experiment directory, get its parent's parent (controller dir)
        if os.path.basename(os.path.dirname(self.base_output_dir)) == controller_name:
            # base_output_dir is an experiment dir inside controller dir
            controller_dir = os.path.dirname(self.base_output_dir)
        else:
            # Fallback: try to find controller dir relative to original base
            original_base = self.trajectory_output_directory if self.trajectory_output_directory else os.path.dirname(
                self.base_output_dir)
            controller_dir = os.path.join(original_base, controller_name)

        comparisons_dir = os.path.join(controller_dir, "comparisons")
        os.makedirs(comparisons_dir, exist_ok=True)
        return comparisons_dir

    def save_race_summary(self, race_summary: Dict) -> bool:
        """
        Save comprehensive race summary to multiple formats.

        Args:
            race_summary: Comprehensive race summary data

        Returns:
            bool: True if successfully saved
        """
        try:
            # Extract data from race_summary
            race_metadata = race_summary.get('race_metadata', {})
            lap_statistics = race_summary.get('lap_statistics', {})
            trajectory_statistics = race_summary.get('trajectory_statistics', {})
            performance_metrics = race_summary.get('performance_metrics', {})
            advanced_metrics = race_summary.get('advanced_metrics', {})

            # Create new structured race_summary.json format
            structured_summary = {
                "metadata": {
                    "timestamp": race_metadata.get('timestamp', ''),
                    "controller_name": race_metadata.get('controller_name', ''),
                    "experiment_id": race_metadata.get('experiment_id', '')
                },
                "lap_statistics": lap_statistics,
                "performance_metrics": {
                    "total_distance": trajectory_statistics.get('total_distance', 0),
                    "average_speed": performance_metrics.get('average_speed', 0),
                    "best_lap_speed": performance_metrics.get('best_lap_speed', 0),
                    "distance_per_second": performance_metrics.get('distance_per_second', 0)
                },
                "trajectory_statistics": trajectory_statistics,
                "trajectory_quality": {
                    "path_length_mean": advanced_metrics.get('path_length_mean', 0),
                    "path_length_std": advanced_metrics.get('path_length_std', 0),
                    "ape_rmse_mean": advanced_metrics.get('ape_translation_part_rmse_mean', 0),
                    "rpe_rmse_mean": advanced_metrics.get('rpe_translation_part_rmse_mean', 0),
                    "overall_ape_mean": advanced_metrics.get('overall_ape_mean', 0),
                    "overall_rpe_mean": advanced_metrics.get('overall_rpe_mean', 0)
                },
                "vehicle_dynamics": {
                    "velocity_mean": advanced_metrics.get('velocity_mean_mean', 0),
                    "velocity_std": advanced_metrics.get('velocity_std_mean', 0),
                    "velocity_max_mean": advanced_metrics.get('velocity_max_mean', 0),
                    "acceleration_mean": advanced_metrics.get('acceleration_mean_mean', 0),
                    "acceleration_std": advanced_metrics.get('acceleration_std_mean', 0),
                    "acceleration_max_mean": advanced_metrics.get('acceleration_max_mean', 0),
                    "jerk_mean": advanced_metrics.get('jerk_mean', 0),
                    "angular_velocity_mean": advanced_metrics.get('angular_velocity_mean_mean', 0)
                },
                "path_metrics": {
                    "mean_curvature": advanced_metrics.get('mean_curvature_mean', 0),
                    "max_curvature_mean": advanced_metrics.get('max_curvature_mean', 0),
                    "path_efficiency_mean": advanced_metrics.get('path_efficiency_mean', 0)
                },
                "summary_stats": {
                    "total_trajectory_points": trajectory_statistics.get('total_trajectory_points', 0),
                    "sampling_rate_mean": advanced_metrics.get('sampling_rate_mean_mean', 10.0),
                    "overall_consistency_cv": advanced_metrics.get('overall_consistency_cv', 0)
                }
            }

            # Save structured race_summary.json
            json_filepath = self._get_organized_file_path(self.results_dir, "race_summary.json")
            with open(json_filepath, 'w') as f:
                json.dump(structured_summary, f, indent=2, default=str)

            # Also write a gzipped compact binary representation (custom .rsum) - compatible
            # with our tools: contains gzipped JSON bytes.
            try:
                rsum_path = self._get_organized_file_path(self.results_dir, "race_summary.rsum")
                with gzip.open(rsum_path, 'wt', encoding='utf-8') as gz:
                    json.dump(race_summary, gz, separators=(',', ':'), default=str)
            except Exception:
                pass

            # For backward compatibility keep a detailed CSV if explicitly requested via config
            if 'csv' in self.output_formats:
                csv_filepath = self._get_organized_file_path(self.results_dir, "race_summary.csv")
                try:
                    with open(csv_filepath, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        # Write metadata section
                        writer.writerow(['=== RACE SUMMARY ==='])
                        writer.writerow(['Generated', datetime.now().isoformat()])
                        writer.writerow([])

                        # Write race metadata
                        writer.writerow(['=== RACE METADATA ==='])
                        metadata = race_summary.get('race_metadata', {})
                        for key, value in metadata.items():
                            writer.writerow([key.replace('_', ' ').title(), value])
                        writer.writerow([])

                        # Write lap statistics
                        writer.writerow(['=== LAP STATISTICS ==='])
                        lap_stats = race_summary.get('lap_statistics', {})
                        for key, value in lap_stats.items():
                            if key != 'lap_times':
                                writer.writerow([key.replace('_', ' ').title(),
                                                 f"{value:.4f}" if isinstance(value, float) else value])

                        writer.writerow([])
                        writer.writerow(['=== INDIVIDUAL LAP TIMES ==='])
                        writer.writerow(['Lap Number', 'Lap Time (s)'])
                        for i, lap_time in enumerate(lap_stats.get('lap_times', []), 1):
                            writer.writerow([i, f"{lap_time:.4f}"])

                        # Write trajectory statistics
                        writer.writerow([])
                        writer.writerow(['=== TRAJECTORY STATISTICS ==='])
                        traj_stats = race_summary.get('trajectory_statistics', {})
                        for key, value in traj_stats.items():
                            writer.writerow([key.replace('_', ' ').title(),
                                            f"{value:.4f}" if isinstance(value, float) else value])

                        # Write performance metrics
                        writer.writerow([])
                        writer.writerow(['=== PERFORMANCE METRICS ==='])
                        perf_metrics = race_summary.get('performance_metrics', {})
                        for key, value in perf_metrics.items():
                            writer.writerow([key.replace('_', ' ').title(),
                                            f"{value:.4f}" if isinstance(value, float) else value])

                        # Write advanced metrics (including APE/RPE metrics)
                        advanced_metrics = race_summary.get('advanced_metrics', {})
                        if advanced_metrics:
                            writer.writerow([])
                            writer.writerow(['=== ADVANCED METRICS ==='])

                            # Group APE metrics first
                            ape_metrics = {k: v for k, v in advanced_metrics.items() if 'ape_' in k}
                            if ape_metrics:
                                writer.writerow(['--- Absolute Pose Error (APE) Metrics ---'])
                                for key, value in sorted(ape_metrics.items()):
                                    formatted_key = key.replace('_', ' ').replace('ape ', 'APE ').title()
                                    writer.writerow(
                                        [formatted_key, f"{value:.4f}" if isinstance(value, float) else value])

                            # Group RPE metrics second
                            rpe_metrics = {k: v for k, v in advanced_metrics.items() if 'rpe_' in k}
                            if rpe_metrics:
                                writer.writerow(['--- Relative Pose Error (RPE) Metrics ---'])
                                for key, value in sorted(rpe_metrics.items()):
                                    formatted_key = key.replace('_', ' ').replace('rpe ', 'RPE ').title()
                                    writer.writerow(
                                        [formatted_key, f"{value:.4f}" if isinstance(value, float) else value])

                            # Add other important advanced metrics
                            other_important_metrics = {
                                k: v for k,
                                v in advanced_metrics.items() if any(
                                    keyword in k for keyword in [
                                        'overall_',
                                        'consistency',
                                        'speed',
                                        'path_length',
                                        'duration']) and 'ape_' not in k and 'rpe_' not in k}
                            if other_important_metrics:
                                writer.writerow(['--- Other Advanced Metrics ---'])
                                for key, value in sorted(other_important_metrics.items()):
                                    formatted_key = key.replace('_', ' ').title()
                                    writer.writerow(
                                        [formatted_key, f"{value:.4f}" if isinstance(value, float) else value])

                    self.logger.info(f"Saved detailed CSV summary to: {csv_filepath}")
                except Exception as e:
                    pass

            return True

        except Exception as e:
            self.logger.error(f"Error saving race summary: {e}")
            return False
            return False

    def save_consolidated_race_results(self, race_summary: Dict, race_evaluation: Dict = None) -> bool:
        """
        Save a single consolidated race_results.json file with all important information.

        Args:
            race_summary: Comprehensive race summary data
            race_evaluation: Race evaluation data (optional)

        Returns:
            bool: True if successfully saved
        """
        try:
            # Extract data from race_summary
            race_metadata = race_summary.get('race_metadata', {})
            lap_statistics = race_summary.get('lap_statistics', {})
            trajectory_statistics = race_summary.get('trajectory_statistics', {})
            performance_metrics = race_summary.get('performance_metrics', {})

            # Create flatter race_results.json structure
            consolidated_results = {
                "timestamp": race_metadata.get('timestamp', ''),
                "controller_name": race_metadata.get('controller_name', ''),
                "experiment_id": race_metadata.get('experiment_id', ''),
                "total_race_time": race_metadata.get('total_race_time', 0),
                "laps_completed": race_metadata.get('laps_completed', 0),
                "race_ending_reason": race_metadata.get('race_ending_reason', ''),
                "crashed": race_metadata.get('crashed', False),
                "lap_times": lap_statistics.get('lap_times', []),
                "total_distance": trajectory_statistics.get('total_distance', 0),
                "sampling_rate": 10  # Default sampling rate
            }

            # Save as race_results.json
            results_filepath = self._get_organized_file_path(self.results_dir, "race_results.json")
            with open(results_filepath, 'w') as f:
                json.dump(consolidated_results, f, indent=2, default=str)

            # Also save as CSV
            self.save_race_results_csv(consolidated_results)

            return True

        except Exception as e:
            self.logger.error(f"Error saving consolidated race results: {e}")
            return False

    def save_race_evaluation(self, evaluation_data: Dict) -> bool:
        """
        Save race evaluation results.

        Args:
            evaluation_data: Race evaluation results

        Returns:
            bool: True if successfully saved
        """
        try:
            # Use stable filename without timestamp
            filename = "race_evaluation.json"
            filepath = self._get_organized_file_path(self.results_dir, filename)

            # If the evaluation_data is empty or contains only zeros, try to populate from
            # consolidated results or a recent race_summary to avoid zeroed outputs.
            def _is_empty_or_all_zero(d: Dict) -> bool:
                if not d:
                    return True
                # check nested values recursively for any non-zero / non-empty

                def any_nonzero(obj):
                    if obj is None:
                        return False
                    if isinstance(obj, (int, float)):
                        return not (obj == 0 or (isinstance(obj, float) and np.isnan(obj)))
                    if isinstance(obj, str):
                        return obj.strip() != ''
                    if isinstance(obj, dict):
                        return any(any_nonzero(v) for v in obj.values())
                    if isinstance(obj, (list, tuple)):
                        return any(any_nonzero(v) for v in obj)
                    return True

                return not any_nonzero(d)

            if _is_empty_or_all_zero(evaluation_data):
                try:
                    consolidated_path = os.path.join(self.results_dir, 'race_results.json')
                    if os.path.exists(consolidated_path):
                        with open(consolidated_path, 'r') as f:
                            loaded = json.load(f)
                        evaluation_data = loaded.get('race_evaluation', loaded)
                except Exception:
                    try:
                        summary_path = os.path.join(self.results_dir, 'race_summary.json')
                        if os.path.exists(summary_path):
                            with open(summary_path, 'r') as f:
                                loaded = json.load(f)
                            evaluation_data = loaded
                    except Exception:
                        pass

            with open(filepath, 'w') as f:
                json.dump(evaluation_data, f, indent=2, default=str)

            # Also save as CSV
            self.save_race_evaluation_csv(evaluation_data)

            return True

        except Exception as e:
            self.logger.error(f"Error saving race evaluation: {e}")
            return False

    def save_race_evaluation_csv(self, evaluation_data: Dict) -> bool:
        """
        Save race evaluation results to CSV format.

        Args:
            evaluation_data: Race evaluation results

        Returns:
            bool: True if successfully saved
        """
        try:
            csv_filepath = self._get_organized_file_path(self.results_dir, "race_evaluation.csv")

            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Header
                writer.writerow(['=== RACE EVALUATION ==='])
                writer.writerow(['Generated', datetime.now().isoformat()])
                writer.writerow([])

                race_eval = evaluation_data.get('race_evaluation', evaluation_data)

                # Metadata
                writer.writerow(['=== METADATA ==='])
                metadata = race_eval.get('metadata', {})
                for key, value in metadata.items():
                    writer.writerow([key.replace('_', ' ').title(), value])
                writer.writerow([])

                # Performance Summary
                writer.writerow(['=== PERFORMANCE SUMMARY ==='])
                perf_summary = race_eval.get('performance_summary', {})
                writer.writerow(['Overall Grade', perf_summary.get('overall_grade', 'N/A')])
                writer.writerow(['Numerical Score', perf_summary.get('numerical_score', 0)])

                # Lap Times
                lap_times = perf_summary.get('lap_times', {})
                writer.writerow(['Best Lap Time', lap_times.get('best', 0)])
                writer.writerow(['Average Lap Time', lap_times.get('average', 0)])
                writer.writerow(['Worst Lap Time', lap_times.get('worst', 0)])
                writer.writerow(['Lap Consistency CV', lap_times.get('consistency_cv', 0)])
                writer.writerow(['Total Laps', lap_times.get('total_laps', 0)])
                writer.writerow(['Lap Times Grade', lap_times.get('grade', 'N/A')])
                writer.writerow([])

                # Speed Analysis
                writer.writerow(['=== SPEED ANALYSIS ==='])
                speed_analysis = perf_summary.get('speed_analysis', {})
                for key, value in speed_analysis.items():
                    writer.writerow([key.replace('_', ' ').title(), value])
                writer.writerow([])

                # Category Grades
                writer.writerow(['=== CATEGORY GRADES ==='])
                category_grades = perf_summary.get('category_grades', {})
                for key, value in category_grades.items():
                    writer.writerow([key.replace('_', ' ').title(), value])
                writer.writerow([])

                # Trajectory Evaluation
                writer.writerow(['=== TRAJECTORY EVALUATION ==='])
                traj_eval = race_eval.get('trajectory_evaluation', {})

                # APE Analysis
                ape_analysis = traj_eval.get('ape_analysis', {})
                if ape_analysis:
                    writer.writerow(['--- APE Analysis ---'])
                    for key, value in ape_analysis.items():
                        writer.writerow([f'APE {key.title()}', f"{value:.4f}" if isinstance(value, float) else value])

                # RPE Analysis
                rpe_analysis = traj_eval.get('rpe_analysis', {})
                if rpe_analysis:
                    writer.writerow(['--- RPE Analysis ---'])
                    for key, value in rpe_analysis.items():
                        writer.writerow([f'RPE {key.title()}', f"{value:.4f}" if isinstance(value, float) else value])
                writer.writerow([])

                # Recommendations
                recommendations = race_eval.get('recommendations', [])
                if recommendations:
                    writer.writerow(['=== RECOMMENDATIONS ==='])
                    for i, rec in enumerate(recommendations, 1):
                        writer.writerow([f'Recommendation {i}', rec])

            return True

        except Exception as e:
            self.logger.error(f"Error saving race evaluation CSV: {e}")
            return False

    def save_race_results_csv(self, race_results: Dict) -> bool:
        """
        Save race results to CSV format.

        Args:
            race_results: Race results data

        Returns:
            bool: True if successfully saved
        """
        try:
            csv_filepath = self._get_organized_file_path(self.results_dir, "race_results.csv")

            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Header
                writer.writerow(['=== RACE RESULTS ==='])
                writer.writerow(['Generated', datetime.now().isoformat()])
                writer.writerow([])

                # Basic Info
                writer.writerow(['=== BASIC INFORMATION ==='])
                writer.writerow(['Timestamp', race_results.get('timestamp', '')])
                writer.writerow(['Controller Name', race_results.get('controller_name', '')])
                writer.writerow(['Experiment ID', race_results.get('experiment_id', '')])
                writer.writerow(['Total Race Time (s)', f"{race_results.get('total_race_time', 0):.3f}"])
                writer.writerow(['Laps Completed', race_results.get('laps_completed', 0)])
                writer.writerow(['Race Ending Reason', race_results.get('race_ending_reason', '')])
                writer.writerow(['Crashed', 'Yes' if race_results.get('crashed', False) else 'No'])
                writer.writerow(['Total Distance (m)', f"{race_results.get('total_distance', 0):.3f}"])
                writer.writerow(['Sampling Rate (Hz)', race_results.get('sampling_rate', 10)])
                writer.writerow([])

                # Lap Times
                lap_times = race_results.get('lap_times', [])
                if lap_times:
                    writer.writerow(['=== LAP TIMES ==='])
                    writer.writerow(['Lap Number', 'Lap Time (s)'])
                    for i, lap_time in enumerate(lap_times, 1):
                        writer.writerow([i, f"{lap_time:.3f}"])

            return True

        except Exception as e:
            self.logger.error(f"Error saving race results CSV: {e}")
            return False

    def create_run_directory(self, controller_name: str, experiment_id: str) -> str:
        """
        Create a dedicated directory for this run.

        Args:
            controller_name: Name of the controller
            experiment_id: Experiment identifier

        Returns:
            str: Path to the created experiment directory
        """
        try:
            controller_dir = os.path.join(self.base_output_dir, controller_name)
            os.makedirs(controller_dir, exist_ok=True)

            experiment_dir_name = experiment_id
            experiment_dir = os.path.join(controller_dir, experiment_dir_name)

            # Create subdirectories for race monitor components
            os.makedirs(os.path.join(experiment_dir, "trajectories"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "graphs"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "metrics"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "filtered"), exist_ok=True)

            self.trajectory_dir = os.path.join(experiment_dir, "trajectories")
            self.results_dir = os.path.join(experiment_dir, "results")
            self.base_output_dir = experiment_dir

            # Create and save experiment metadata
            if hasattr(self, 'metadata_manager'):
                custom_data = {
                    'experiment_setup': {
                        'output_formats': self.output_formats,
                        'advanced_metrics_enabled': self.enable_advanced_metrics,
                        'trajectory_filtering_enabled': self.apply_trajectory_filtering,
                        'auto_graph_generation': self.auto_generate_graphs
                    }
                }
                self.metadata_manager.create_experiment_metadata(
                    experiment_id=experiment_id,
                    controller_name=controller_name,
                    custom_data=custom_data
                )
                # Update metadata manager output directory to the experiment directory
                self.metadata_manager.output_directory = experiment_dir

                # Save text metadata file
                self.metadata_manager.save_metadata_file('experiment_metadata.txt')

                # Log experiment summary
                self.logger.info(f"Experiment setup: {self.metadata_manager.get_experiment_summary()}")

            return experiment_dir

        except Exception as e:
            self.logger.error(f"Error creating run directory: {e}")
            return self.base_output_dir

    def save_ape_rpe_metrics_files(self, advanced_metrics: Dict) -> bool:
        """
        Save APE and RPE metrics as separate files in the metrics directory.

        Args:
            advanced_metrics: Dictionary containing all advanced metrics

        Returns:
            bool: True if successfully saved
        """
        try:
            if not advanced_metrics:
                self.logger.warn("No advanced metrics provided for APE/RPE file generation")
                return False

            metrics_dir = self.get_metrics_directory()

            # Extract APE metrics
            ape_metrics = {k: v for k, v in advanced_metrics.items() if 'ape_' in k}
            if ape_metrics:
                ape_filepath = self._get_organized_file_path(metrics_dir, 'ape_metrics_summary.json')
                with open(ape_filepath, 'w') as f:
                    json.dump(ape_metrics, f, indent=2, default=str)
                self.logger.info(f"Saved APE metrics summary to: {ape_filepath}")

            # Extract RPE metrics
            rpe_metrics = {k: v for k, v in advanced_metrics.items() if 'rpe_' in k}
            if rpe_metrics:
                rpe_filepath = self._get_organized_file_path(metrics_dir, 'rpe_metrics_summary.json')
                with open(rpe_filepath, 'w') as f:
                    json.dump(rpe_metrics, f, indent=2, default=str)
                self.logger.info(f"Saved RPE metrics summary to: {rpe_filepath}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving APE/RPE metrics files: {e}")
            return False

    def generate_ape_rpe_plots(self, advanced_metrics: Dict) -> bool:
        """
        Generate plots for APE and RPE metrics.

        Args:
            advanced_metrics: Dictionary containing all advanced metrics

        Returns:
            bool: True if successfully generated
        """
        try:
            # Check if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                import numpy as np
            except ImportError:
                self.logger.warn("Matplotlib not available - skipping APE/RPE plot generation")
                return False

            if not advanced_metrics:
                self.logger.warn("No advanced metrics provided for APE/RPE plot generation")
                return False

            plots_dir = self.get_plots_directory()

            # Extract APE metrics for plotting
            ape_metrics = {k: v for k, v in advanced_metrics.items() if 'ape_' in k}
            rpe_metrics = {k: v for k, v in advanced_metrics.items() if 'rpe_' in k}

            if ape_metrics:
                # Create APE metrics plot
                plt.figure(figsize=(12, 8))

                # Group by metric type (mean, std, min, max, median, cv)
                ape_types = {}
                for key, value in ape_metrics.items():
                    # Extract the last part after the final underscore (metric type)
                    parts = key.split('_')
                    if len(parts) >= 2:
                        metric_type = parts[-1]  # mean, std, min, max, etc.
                        metric_name = '_'.join(parts[:-1])  # everything before the type

                        if metric_type not in ape_types:
                            ape_types[metric_type] = {}
                        ape_types[metric_type][metric_name] = value

                # Create subplots for different APE metric types
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()

                for idx, (metric_type, metrics) in enumerate(ape_types.items()):
                    if idx >= 6:  # Limit to 6 subplots
                        break

                    ax = axes[idx]
                    names = list(metrics.keys())
                    values = list(metrics.values())

                    # Create bar plot
                    bars = ax.bar(range(len(names)), values)
                    ax.set_title(f'APE {metric_type.upper()} Metrics', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Metric Type')
                    ax.set_ylabel(f'{metric_type.capitalize()} Value')

                    # Rotate x-axis labels for better readability
                    ax.set_xticks(range(len(names)))
                    ax.set_xticklabels([name.replace('ape_', '').replace('_', ' ').title() for name in names],
                                       rotation=45, ha='right')

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

                    ax.grid(True, alpha=0.3)

                # Hide unused subplots
                for idx in range(len(ape_types), 6):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                ape_plot_path = os.path.join(plots_dir, f'ape_metrics_analysis.png')
                plt.savefig(ape_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved APE metrics plot to: {ape_plot_path}")

            if rpe_metrics:
                # Create RPE metrics plot
                plt.figure(figsize=(12, 8))

                # Group by metric type (mean, std, min, max, median, cv)
                rpe_types = {}
                for key, value in rpe_metrics.items():
                    # Extract the last part after the final underscore (metric type)
                    parts = key.split('_')
                    if len(parts) >= 2:
                        metric_type = parts[-1]  # mean, std, min, max, etc.
                        metric_name = '_'.join(parts[:-1])  # everything before the type

                        if metric_type not in rpe_types:
                            rpe_types[metric_type] = {}
                        rpe_types[metric_type][metric_name] = value

                # Create subplots for different RPE metric types
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()

                for idx, (metric_type, metrics) in enumerate(rpe_types.items()):
                    if idx >= 6:  # Limit to 6 subplots
                        break

                    ax = axes[idx]
                    names = list(metrics.keys())
                    values = list(metrics.values())

                    # Create bar plot
                    bars = ax.bar(range(len(names)), values, color='orange')
                    ax.set_title(f'RPE {metric_type.upper()} Metrics', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Metric Type')
                    ax.set_ylabel(f'{metric_type.capitalize()} Value')

                    # Rotate x-axis labels for better readability
                    ax.set_xticks(range(len(names)))
                    ax.set_xticklabels([name.replace('rpe_', '').replace('_', ' ').title() for name in names],
                                       rotation=45, ha='right')

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

                    ax.grid(True, alpha=0.3)

                # Hide unused subplots
                for idx in range(len(rpe_types), 6):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                rpe_plot_path = os.path.join(plots_dir, f'rpe_metrics_analysis.png')
                plt.savefig(rpe_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved RPE metrics plot to: {rpe_plot_path}")

            # Create a combined APE vs RPE comparison plot
            if ape_metrics and rpe_metrics:
                plt.figure(figsize=(14, 8))

                # Extract mean values for comparison
                ape_means = {k: v for k, v in ape_metrics.items() if k.endswith('_mean')}
                rpe_means = {k: v for k, v in rpe_metrics.items() if k.endswith('_mean')}

                if ape_means and rpe_means:
                    # Create side-by-side comparison
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                    # APE means
                    ape_names = [name.replace('ape_', '').replace('_mean', '').replace('_', ' ').title()
                                 for name in ape_means.keys()]
                    ape_values = list(ape_means.values())

                    bars1 = ax1.bar(range(len(ape_names)), ape_values, color='skyblue')
                    ax1.set_title('APE Mean Values Comparison', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Metric Type')
                    ax1.set_ylabel('APE Mean Value (meters)')
                    ax1.set_xticks(range(len(ape_names)))
                    ax1.set_xticklabels(ape_names, rotation=45, ha='right')

                    for bar, value in zip(bars1, ape_values):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    ax1.grid(True, alpha=0.3)

                    # RPE means
                    rpe_names = [name.replace('rpe_', '').replace('_mean', '').replace('_', ' ').title()
                                 for name in rpe_means.keys()]
                    rpe_values = list(rpe_means.values())

                    bars2 = ax2.bar(range(len(rpe_names)), rpe_values, color='lightcoral')
                    ax2.set_title('RPE Mean Values Comparison', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Metric Type')
                    ax2.set_ylabel('RPE Mean Value (meters)')
                    ax2.set_xticks(range(len(rpe_names)))
                    ax2.set_xticklabels(rpe_names, rotation=45, ha='right')

                    for bar, value in zip(bars2, rpe_values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    comparison_plot_path = os.path.join(plots_dir, f'ape_vs_rpe_comparison.png')
                    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.logger.info(f"Saved APE vs RPE comparison plot to: {comparison_plot_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error generating APE/RPE plots: {e}")
            return False
