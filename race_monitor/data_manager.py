#!/usr/bin/env python3

"""
Data Manager

Handles data storage, file I/O operations, and trajectory management
for the race monitor system. Provides centralized data management
with support for multiple file formats and data export capabilities.

Features:
    - Trajectory data storage and management
    - Multi-format file export (CSV, TUM, JSON, Pickle, MAT)
    - Research data compilation and analysis
    - File organization and directory management
    - Data validation and error handling

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

import rclpy
import os
import csv
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

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

        self.logger.info(f"Data manager configured: output_dir={self.trajectory_output_directory}, "
                         f"formats={self.output_formats}, controller={self.controller_name}, "
                         f"experiment={self.experiment_id}")
        self.logger.info(f"Advanced features: pandas_export={self.export_to_pandas}, "
                         f"graphs={self.auto_generate_graphs}, filtering={self.apply_trajectory_filtering}")

    def _setup_directories(self):
        """Create base directory structure for data storage."""
        try:
            self.base_output_dir = self.trajectory_output_directory

            # Only create base output directory initially
            # Specific experiment directories will be created by create_run_directory()
            os.makedirs(self.base_output_dir, exist_ok=True)

            # Initialize trajectory and results dirs as None - they will be set when run directory is created
            self.trajectory_dir = None
            self.results_dir = None

            self.logger.info(f"Base data directory created: {self.base_output_dir}")

        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")

    def start_new_lap_trajectory(self, lap_number: int):
        """
        Start recording a new lap trajectory.

        Args:
            lap_number: Lap number to start recording
        """
        self.current_lap_trajectory = []
        self.logger.debug(f"Started recording trajectory for lap {lap_number}")

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

        # Save to file if enabled
        if self.save_trajectories:
            success = self.save_trajectory_to_file(trajectory_data)
            if success:
                self.logger.info(f"Saved trajectory for lap {lap_number} ({len(self.current_lap_trajectory)} points)")
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
            filepath = os.path.join(self.trajectory_dir, filename)

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
            filepath = os.path.join(self.trajectory_dir, filename)

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
            filepath = os.path.join(self.trajectory_dir, filename)

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
            filepath = os.path.join(self.trajectory_dir, filename)

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
            filepath = os.path.join(self.trajectory_dir, filename)

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
            if filename_override:
                filename = filename_override
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"race_results_{timestamp}.csv"
            filepath = os.path.join(self.results_dir, filename)

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

            self.logger.info(f"Saved race results to: {filepath}")
            return True

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
            filename = "evaluation_summary.csv"
            # Save evaluation summary in the results directory
            filepath = os.path.join(self.results_dir, filename)

            # Store evaluation data
            self.evaluation_summaries.append(evaluation_data)

            # Write to CSV
            with open(filepath, 'w', newline='') as csvfile:
                if self.evaluation_summaries:
                    fieldnames = list(self.evaluation_summaries[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.evaluation_summaries)

            self.logger.info(f"Saved evaluation summary to: {filepath}")
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
        self.logger.info("Cleared all trajectory data")

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
        # Only create directories if we're in a controller/experiment directory
        if not self._is_in_experiment_directory():
            self.logger.info(f"get_metrics_directory: SKIPPING creation at base level: {self.base_output_dir}")
            return os.path.join(self.base_output_dir, "metrics")  # Return path but don't create

        metrics_dir = os.path.join(self.base_output_dir, "metrics")
        self.logger.info(f"get_metrics_directory: CREATING in experiment dir: {metrics_dir}")
        os.makedirs(metrics_dir, exist_ok=True)
        return metrics_dir

    def get_filtered_directory(self) -> str:
        """Get the filtered trajectories directory path."""
        # Only create directories if we're in a controller/experiment directory
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "filtered")  # Return path but don't create

        filtered_dir = os.path.join(self.base_output_dir, "filtered")
        self.logger.info(f"get_filtered_directory called: {filtered_dir} (base_output_dir: {self.base_output_dir})")
        os.makedirs(filtered_dir, exist_ok=True)
        return filtered_dir

    def get_exports_directory(self) -> str:
        """Get the exports directory path."""
        # Only create directories if we're in a controller/experiment directory
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "exports")  # Return path but don't create

        exports_dir = os.path.join(self.base_output_dir, "exports")
        self.logger.info(f"get_exports_directory called: {exports_dir} (base_output_dir: {self.base_output_dir})")
        os.makedirs(exports_dir, exist_ok=True)
        return exports_dir

    def get_plots_directory(self) -> str:
        """Get the plots directory path."""
        # Only create directories if we're in a controller/experiment directory
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "plots")  # Return path but don't create

        plots_dir = os.path.join(self.base_output_dir, "plots")
        self.logger.info(f"get_plots_directory called: {plots_dir} (base_output_dir: {self.base_output_dir})")
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir

    def get_statistics_directory(self) -> str:
        """Get the statistics directory path."""
        # Only create directories if we're in a controller/experiment directory
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "statistics")  # Return path but don't create

        stats_dir = os.path.join(self.base_output_dir, "statistics")
        self.logger.info(f"get_statistics_directory called: {stats_dir} (base_output_dir: {self.base_output_dir})")
        os.makedirs(stats_dir, exist_ok=True)
        return stats_dir

    def get_graphs_directory(self) -> str:
        """Get the graphs directory path."""
        # Only create directories if we're in a controller/experiment directory
        if not self._is_in_experiment_directory():
            return os.path.join(self.base_output_dir, "graphs")  # Return path but don't create

        graphs_dir = os.path.join(self.base_output_dir, "graphs")
        self.logger.info(f"get_graphs_directory called: {graphs_dir} (base_output_dir: {self.base_output_dir})")
        os.makedirs(graphs_dir, exist_ok=True)
        return graphs_dir

    def get_current_trajectory_directory(self) -> str:
        """Get the current trajectory base directory path."""
        return self.base_output_dir

    def _is_in_experiment_directory(self) -> bool:
        """Check if base_output_dir points to a controller/experiment directory."""
        # Check if the path contains a controller name and experiment pattern
        path_parts = self.base_output_dir.split(os.sep)

        # Look for experiment pattern (exp_xxx_timestamp)
        has_experiment = any('exp_' in part and len(part) > 15 for part in path_parts)

        # Also check if we have a controller name set and it's not empty
        has_controller = hasattr(self, 'controller_name') and self.controller_name and self.controller_name.strip()

        # Additional check: make sure we're not at the base level
        is_base_level = self.base_output_dir.endswith('evaluation_results')

        result = has_experiment and has_controller and not is_base_level
        return result

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            controller_name = race_summary.get('race_metadata', {}).get('controller_name', 'unknown')
            experiment_id = race_summary.get('race_metadata', {}).get('experiment_id', 'exp')

            # Create filename with run identification
            base_filename = f"race_summary_{controller_name}_{experiment_id}_{timestamp}"

            # Save as JSON
            json_filepath = os.path.join(self.results_dir, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(race_summary, f, indent=2, default=str)

            # Save as detailed CSV
            csv_filepath = os.path.join(self.results_dir, f"{base_filename}.csv")
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
                            writer.writerow([formatted_key, f"{value:.4f}" if isinstance(value, float) else value])

                    # Group RPE metrics second
                    rpe_metrics = {k: v for k, v in advanced_metrics.items() if 'rpe_' in k}
                    if rpe_metrics:
                        writer.writerow(['--- Relative Pose Error (RPE) Metrics ---'])
                        for key, value in sorted(rpe_metrics.items()):
                            formatted_key = key.replace('_', ' ').replace('rpe ', 'RPE ').title()
                            writer.writerow([formatted_key, f"{value:.4f}" if isinstance(value, float) else value])

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
                            writer.writerow([formatted_key, f"{value:.4f}" if isinstance(value, float) else value])

            self.logger.info(f"Saved comprehensive race summary to: {json_filepath}")
            self.logger.info(f"Saved detailed CSV summary to: {csv_filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving race summary: {e}")
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
            # Create consolidated results structure
            consolidated_results = {
                "race_info": race_summary.get('race_metadata', {}),
                "lap_analysis": {
                    "statistics": race_summary.get('lap_statistics', {}),
                    "trajectory_stats": race_summary.get('trajectory_statistics', {}),
                    "performance_metrics": race_summary.get('performance_metrics', {})
                },
                "advanced_metrics": race_summary.get('advanced_metrics', {}),
                "race_evaluation": race_evaluation or {}
            }

            # Save as race_results.json
            results_filepath = os.path.join(self.results_dir, "race_results.json")
            with open(results_filepath, 'w') as f:
                json.dump(consolidated_results, f, indent=2, default=str)

            self.logger.info(f"Saved consolidated race results to: {results_filepath}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"race_evaluation_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(evaluation_data, f, indent=2, default=str)

            self.logger.info(f"Saved race evaluation to: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving race evaluation: {e}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.logger.info(f"Creating run directory - controller: {controller_name}, experiment: {experiment_id}")
            self.logger.info(f"Current base_output_dir before update: {self.base_output_dir}")

            # Create controller-based hierarchy structure
            # Create controller directory first
            controller_dir = os.path.join(self.base_output_dir, controller_name)
            os.makedirs(controller_dir, exist_ok=True)

            # Create experiment directory inside controller directory
            experiment_dir_name = f"{experiment_id}_{timestamp}"
            experiment_dir = os.path.join(controller_dir, experiment_dir_name)

            self.logger.info(f"Creating experiment directory: {experiment_dir}")

            # Create subdirectories for all race monitor components under the experiment directory
            os.makedirs(os.path.join(experiment_dir, "trajectories"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "graphs"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "metrics"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "filtered"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "exports"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
            os.makedirs(os.path.join(experiment_dir, "statistics"), exist_ok=True)

            # Update paths to use experiment directory
            self.trajectory_dir = os.path.join(experiment_dir, "trajectories")
            self.results_dir = os.path.join(experiment_dir, "results")

            # Update base_output_dir to point to the experiment directory
            # This ensures all other components save to the controller-specific structure
            self.base_output_dir = experiment_dir

            self.logger.info(f"Created experiment directory: {experiment_dir}")
            self.logger.info(f"Updated base output directory to: {self.base_output_dir}")
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
                ape_filepath = os.path.join(metrics_dir, 'ape_metrics_summary.json')
                with open(ape_filepath, 'w') as f:
                    json.dump(ape_metrics, f, indent=2, default=str)
                self.logger.info(f"Saved APE metrics summary to: {ape_filepath}")

                # Also save as CSV for easy analysis
                ape_csv_filepath = os.path.join(metrics_dir, 'ape_metrics_summary.csv')
                with open(ape_csv_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['APE Metric', 'Value'])
                    for key, value in sorted(ape_metrics.items()):
                        formatted_key = key.replace('_', ' ').replace('ape ', 'APE ').title()
                        writer.writerow([formatted_key, f"{value:.6f}" if isinstance(value, float) else value])
                self.logger.info(f"Saved APE metrics CSV to: {ape_csv_filepath}")

            # Extract RPE metrics
            rpe_metrics = {k: v for k, v in advanced_metrics.items() if 'rpe_' in k}
            if rpe_metrics:
                rpe_filepath = os.path.join(metrics_dir, 'rpe_metrics_summary.json')
                with open(rpe_filepath, 'w') as f:
                    json.dump(rpe_metrics, f, indent=2, default=str)
                self.logger.info(f"Saved RPE metrics summary to: {rpe_filepath}")

                # Also save as CSV for easy analysis
                rpe_csv_filepath = os.path.join(metrics_dir, 'rpe_metrics_summary.csv')
                with open(rpe_csv_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['RPE Metric', 'Value'])
                    for key, value in sorted(rpe_metrics.items()):
                        formatted_key = key.replace('_', ' ').replace('rpe ', 'RPE ').title()
                        writer.writerow([formatted_key, f"{value:.6f}" if isinstance(value, float) else value])
                self.logger.info(f"Saved RPE metrics CSV to: {rpe_csv_filepath}")

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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ape_plot_path = os.path.join(plots_dir, f'ape_metrics_analysis_{timestamp}.png')
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rpe_plot_path = os.path.join(plots_dir, f'rpe_metrics_analysis_{timestamp}.png')
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    comparison_plot_path = os.path.join(plots_dir, f'ape_vs_rpe_comparison_{timestamp}.png')
                    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.logger.info(f"Saved APE vs RPE comparison plot to: {comparison_plot_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error generating APE/RPE plots: {e}")
            return False
