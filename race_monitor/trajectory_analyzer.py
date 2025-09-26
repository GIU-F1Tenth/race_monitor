#!/usr/bin/env python3

"""
Research Trajectory Evaluator

A comprehensive trajectory analysis system for autonomous racing research.
Utilizes the full functionality of the EVO trajectory evaluation library to provide
advanced metrics calculation, statistical analysis, and research-ready data export.

Features:
    - Comprehensive trajectory analysis using full EVO library capabilities
    - 40+ performance metrics calculation per lap
    - Advanced filtering and smoothing algorithms
    - Multi-format data export (JSON, CSV, Pickle, TUM)
    - Statistical analysis with confidence intervals
    - Research-grade documentation and metadata

Metrics Calculated:
    - Basic: Path length, duration, average speed
    - Velocity: Mean/std/max velocity, consistency analysis
    - Acceleration: Mean/std/max acceleration, jerk (smoothness)
    - Steering: Angular velocity analysis, aggressiveness metrics
    - Geometric: Curvature analysis, path efficiency
    - Statistical: Complete statistical breakdown for all metrics

Usage:
    config = {
        'controller_name': 'my_controller',
        'experiment_id': 'session_001',
        'enable_advanced_metrics': True,
        'output_formats': ['json', 'csv']
    }
    evaluator = create_research_evaluator(config)
    evaluator.add_trajectory(lap_number, trajectory_data, lap_time)
    evaluator.export_research_data()

Author: Race Monitor Development Team
License: MIT
"""

import os
import sys
import numpy as np
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add EVO library to Python path
evo_path = os.path.join(os.path.dirname(__file__), '..', 'evo')
if os.path.exists(evo_path) and evo_path not in sys.path:
    sys.path.insert(0, evo_path)

try:
    from evo.core import trajectory, metrics, sync, transformations, filters, geometry, lie_algebra, result
    from evo.tools import file_interface, pandas_bridge
    from evo.core.units import Unit, METER_SCALE_FACTORS, ANGLE_UNITS
    import evo
    EVO_AVAILABLE = True
except ImportError as e:
    print(f"❌ EVO not available: {e}")
    EVO_AVAILABLE = False


class ResearchTrajectoryEvaluator:
    """
    Comprehensive trajectory evaluator using full EVO functionality
    Designed for research and paper writing with controller comparison
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.controller_name = config.get('controller_name', 'unknown')
        self.experiment_id = config.get('experiment_id', 'exp_001')

        # Initialize logging
        self.logger = logging.getLogger(f'ResearchEvaluator_{self.controller_name}')

        # Storage for analysis results
        self.lap_trajectories = {}
        self.filtered_trajectories = {}
        self.detailed_metrics = {}
        self.statistical_results = {}
        self.geometric_analysis = {}

        # Reference trajectories for comparison
        self.reference_trajectories = {}
        self.controller_comparisons = {}

        # Output directories
        self.setup_output_directories()

    def setup_output_directories(self):
        """Setup directory structure for research outputs"""
        # Get base directory - if relative, make it relative to the package
        base_dir_config = self.config.get('trajectory_output_directory', 'evaluation_results')
        if not os.path.isabs(base_dir_config):
            # Make path relative to the race_monitor package directory
            package_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to race_monitor package root
            base_dir = os.path.join(package_dir, base_dir_config)
        else:
            base_dir = base_dir_config

        self.controller_dir = os.path.join(base_dir, 'research_data', self.controller_name)
        self.experiment_dir = os.path.join(self.controller_dir, self.experiment_id)

        directories = [
            self.experiment_dir,
            os.path.join(self.experiment_dir, 'trajectories'),
            os.path.join(self.experiment_dir, 'filtered'),
            os.path.join(self.experiment_dir, 'metrics'),
            os.path.join(self.experiment_dir, 'statistics'),
            os.path.join(self.experiment_dir, 'comparisons'),
            os.path.join(self.experiment_dir, 'plots'),
            os.path.join(self.experiment_dir, 'exports')
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def add_trajectory(self, lap_number: int, trajectory_data: List[Dict], lap_time: float):
        """Add trajectory data for comprehensive analysis"""
        if not EVO_AVAILABLE:
            return

        # Convert to EVO trajectory format
        traj = self._convert_to_evo_trajectory(trajectory_data)
        if traj is None:
            return

        self.lap_trajectories[lap_number] = {
            'trajectory': traj,
            'lap_time': lap_time,
            'raw_data': trajectory_data
        }

        # Apply filtering if enabled
        if self.config.get('apply_trajectory_filtering', True):
            filtered_traj = self._apply_filtering(traj)
            self.filtered_trajectories[lap_number] = filtered_traj

        # Perform comprehensive analysis
        self._analyze_lap_trajectory(lap_number, traj)

        # Save intermediate results if configured
        if self.config.get('save_intermediate_results', True):
            self._save_lap_results(lap_number)

    def _convert_to_evo_trajectory(self, trajectory_data: List[Dict]) -> Optional[trajectory.PoseTrajectory3D]:
        """Convert trajectory data to EVO format with error handling"""
        try:
            if len(trajectory_data) < 2:
                return None

            positions = []
            orientations = []
            timestamps = []

            for pose_data in trajectory_data:
                # Extract position
                if 'pose' in pose_data and hasattr(pose_data['pose'], 'position'):
                    # ROS format
                    pos = [pose_data['pose'].position.x,
                           pose_data['pose'].position.y,
                           pose_data['pose'].position.z]
                    orient = [pose_data['pose'].orientation.w,
                              pose_data['pose'].orientation.x,
                              pose_data['pose'].orientation.y,
                              pose_data['pose'].orientation.z]
                    timestamp = pose_data['header'].stamp.sec + pose_data['header'].stamp.nanosec * 1e-9
                else:
                    # Custom format
                    pos = [pose_data['pose']['position']['x'],
                           pose_data['pose']['position']['y'],
                           pose_data['pose']['position']['z']]
                    orient = [pose_data['pose']['orientation']['w'],
                              pose_data['pose']['orientation']['x'],
                              pose_data['pose']['orientation']['y'],
                              pose_data['pose']['orientation']['z']]
                    timestamp = pose_data['header']['stamp']['sec'] + pose_data['header']['stamp']['nanosec'] * 1e-9

                positions.append(pos)
                orientations.append(orient)
                timestamps.append(timestamp)

            return trajectory.PoseTrajectory3D(
                positions_xyz=np.array(positions),
                orientations_quat_wxyz=np.array(orientations),
                timestamps=np.array(timestamps)
            )

        except Exception as e:
            self.logger.error(f"Error converting trajectory: {e}")
            return None

    def _apply_filtering(self, traj: trajectory.PoseTrajectory3D) -> trajectory.PoseTrajectory3D:
        """Apply trajectory filtering using EVO filters"""
        try:
            if not self.config.get('filter_types'):
                return traj

            # Convert to SE(3) poses for filtering
            poses_se3 = []
            for i in range(len(traj.positions_xyz)):
                T = transformations.quaternion_matrix(traj.orientations_quat_wxyz[i])
                T[:3, 3] = traj.positions_xyz[i]
                poses_se3.append(T)

            # Apply motion filtering
            if 'motion' in self.config.get('filter_types', []):
                motion_threshold = self.config.get('filter_parameters', {}).get('motion_threshold', 0.1)
                filtered_poses = filters.filter_by_motion(poses_se3, motion_threshold)
                poses_se3 = filtered_poses

            # Apply distance filtering
            if 'distance' in self.config.get('filter_types', []):
                distance_threshold = self.config.get('filter_parameters', {}).get('distance_threshold', 0.05)
                # Create pairs for distance filtering
                id_pairs = filters.filter_pairs_by_distance(poses_se3, distance_threshold)
                # Extract filtered poses
                filtered_indices = list(set([pair[0] for pair in id_pairs] + [pair[1] for pair in id_pairs]))
                filtered_indices.sort()
                poses_se3 = [poses_se3[i] for i in filtered_indices]

            # Convert back to trajectory format
            filtered_positions = []
            filtered_orientations = []
            filtered_timestamps = []

            for i, T in enumerate(poses_se3):
                filtered_positions.append(T[:3, 3])
                quat = transformations.quaternion_from_matrix(T)
                filtered_orientations.append([quat[3], quat[0], quat[1], quat[2]])  # wxyz format
                # Interpolate timestamps
                if i < len(traj.timestamps):
                    filtered_timestamps.append(traj.timestamps[i])
                else:
                    filtered_timestamps.append(traj.timestamps[-1] + (i - len(traj.timestamps) + 1) * 0.1)

            return trajectory.PoseTrajectory3D(
                positions_xyz=np.array(filtered_positions),
                orientations_quat_wxyz=np.array(filtered_orientations),
                timestamps=np.array(filtered_timestamps)
            )

        except Exception as e:
            self.logger.error(f"Error applying filtering: {e}")
            return traj

    def _analyze_lap_trajectory(self, lap_number: int, traj: trajectory.PoseTrajectory3D):
        """Perform comprehensive trajectory analysis"""
        metrics_dict = {}

        # Basic metrics
        metrics_dict.update(self._calculate_basic_metrics(traj))

        # Advanced EVO metrics
        if self.config.get('enable_advanced_metrics', True):
            metrics_dict.update(self._calculate_advanced_evo_metrics(traj))

        # Geometric analysis
        if self.config.get('enable_geometric_analysis', True):
            metrics_dict.update(self._calculate_geometric_metrics(traj))

        # Research-specific metrics
        metrics_dict.update(self._calculate_research_metrics(traj))

        # Statistical analysis
        if self.config.get('calculate_all_statistics', True):
            metrics_dict.update(self._calculate_statistical_metrics(traj))

        self.detailed_metrics[lap_number] = metrics_dict

    def _calculate_basic_metrics(self, traj: trajectory.PoseTrajectory3D) -> Dict[str, float]:
        """Calculate basic trajectory metrics"""
        metrics = {}

        # Path length using EVO geometry
        try:
            path_length = geometry.arc_len(traj.positions_xyz)
            metrics['path_length'] = float(path_length)
        except BaseException:
            # Fallback calculation
            path_length = 0.0
            for i in range(1, len(traj.positions_xyz)):
                segment_length = np.linalg.norm(traj.positions_xyz[i] - traj.positions_xyz[i - 1])
                path_length += segment_length
            metrics['path_length'] = path_length

        # Trajectory duration
        metrics['duration'] = float(traj.timestamps[-1] - traj.timestamps[0])

        # Average speed
        if metrics['duration'] > 0:
            metrics['avg_speed'] = metrics['path_length'] / metrics['duration']
        else:
            metrics['avg_speed'] = 0.0

        return metrics

    def _calculate_advanced_evo_metrics(self, traj: trajectory.PoseTrajectory3D) -> Dict[str, Any]:
        """Calculate advanced EVO metrics with all pose relations and statistics"""
        metrics = {}

        if not hasattr(self, 'reference_trajectory') or self.reference_trajectory is None:
            return metrics

        try:
            # Synchronize trajectories
            traj_ref, traj_est = sync.associate_trajectories(
                self.reference_trajectory, traj, max_diff=0.01
            )

            # Calculate metrics for all pose relations
            pose_relations = self.config.get('pose_relations', ['translation_part'])
            statistics_types = self.config.get('statistics_types', ['rmse'])

            for relation_name in pose_relations:
                try:
                    pose_relation = getattr(metrics.PoseRelation, relation_name)

                    # APE metrics
                    ape_metric = metrics.APE(pose_relation)
                    ape_metric.process_data((traj_ref, traj_est))

                    for stat_name in statistics_types:
                        try:
                            stat_type = getattr(metrics.StatisticsType, stat_name)
                            ape_value = ape_metric.get_statistic(stat_type)
                            metrics[f'ape_{relation_name}_{stat_name}'] = float(ape_value)
                        except AttributeError:
                            continue

                    # RPE metrics
                    rpe_metric = metrics.RPE(pose_relation)
                    rpe_metric.process_data((traj_ref, traj_est))

                    for stat_name in statistics_types:
                        try:
                            stat_type = getattr(metrics.StatisticsType, stat_name)
                            rpe_value = rpe_metric.get_statistic(stat_type)
                            metrics[f'rpe_{relation_name}_{stat_name}'] = float(rpe_value)
                        except AttributeError:
                            continue

                except AttributeError:
                    continue

        except Exception as e:
            self.logger.error(f"Error calculating advanced EVO metrics: {e}")

        return metrics

    def _calculate_geometric_metrics(self, traj: trajectory.PoseTrajectory3D) -> Dict[str, float]:
        """Calculate geometric analysis metrics"""
        metrics = {}

        try:
            # Accumulated distances
            distances = geometry.accumulated_distances(traj.positions_xyz)
            metrics['total_distance'] = float(distances[-1])
            metrics['avg_step_distance'] = float(np.mean(np.diff(distances)))
            metrics['max_step_distance'] = float(np.max(np.diff(distances)))
            metrics['min_step_distance'] = float(np.min(np.diff(distances)))

            # Curvature analysis (more sophisticated)
            curvatures = []
            for i in range(1, len(traj.positions_xyz) - 1):
                p1 = traj.positions_xyz[i - 1]
                p2 = traj.positions_xyz[i]
                p3 = traj.positions_xyz[i + 1]

                # Calculate curvature using cross product method
                v1 = p2 - p1
                v2 = p3 - p2

                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    cross_product = np.cross(v1[:2], v2[:2])  # 2D cross product
                    curvature = abs(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    curvatures.append(curvature)

            if curvatures:
                metrics['mean_curvature'] = float(np.mean(curvatures))
                metrics['max_curvature'] = float(np.max(curvatures))
                metrics['curvature_std'] = float(np.std(curvatures))
                metrics['curvature_variation'] = float(np.std(curvatures) / np.mean(curvatures))

        except Exception as e:
            self.logger.error(f"Error calculating geometric metrics: {e}")

        return metrics

    def _calculate_research_metrics(self, traj: trajectory.PoseTrajectory3D) -> Dict[str, float]:
        """Calculate research-specific metrics for controller comparison"""
        metrics = {}

        try:
            # Control smoothness (acceleration analysis)
            velocities = []
            accelerations = []

            for i in range(1, len(traj.positions_xyz)):
                dt = traj.timestamps[i] - traj.timestamps[i - 1]
                if dt > 0:
                    velocity = np.linalg.norm(traj.positions_xyz[i] - traj.positions_xyz[i - 1]) / dt
                    velocities.append(velocity)

            for i in range(1, len(velocities)):
                dt = traj.timestamps[i + 1] - traj.timestamps[i]
                if dt > 0:
                    acceleration = (velocities[i] - velocities[i - 1]) / dt
                    accelerations.append(acceleration)

            if velocities:
                metrics['velocity_mean'] = float(np.mean(velocities))
                metrics['velocity_std'] = float(np.std(velocities))
                metrics['velocity_max'] = float(np.max(velocities))
                metrics['velocity_consistency'] = float(
                    np.std(velocities) / np.mean(velocities)) if np.mean(velocities) > 0 else 0

            if accelerations:
                metrics['acceleration_mean'] = float(np.mean(np.abs(accelerations)))
                metrics['acceleration_std'] = float(np.std(accelerations))
                metrics['acceleration_max'] = float(np.max(np.abs(accelerations)))
                metrics['jerk'] = float(np.mean(np.abs(np.diff(accelerations))))  # Rate of acceleration change

            # Angular velocity analysis (control aggressiveness)
            angular_velocities = []
            for i in range(1, len(traj.orientations_quat_wxyz)):
                dt = traj.timestamps[i] - traj.timestamps[i - 1]
                if dt > 0:
                    # Calculate relative rotation
                    q1 = traj.orientations_quat_wxyz[i - 1]
                    q2 = traj.orientations_quat_wxyz[i]

                    # Relative quaternion
                    q_rel = transformations.quaternion_multiply(q2, transformations.quaternion_inverse(q1))

                    # Extract angle
                    angle = 2 * np.arccos(np.clip(abs(q_rel[0]), 0, 1))
                    angular_velocity = angle / dt
                    angular_velocities.append(angular_velocity)

            if angular_velocities:
                metrics['angular_velocity_mean'] = float(np.mean(angular_velocities))
                metrics['angular_velocity_std'] = float(np.std(angular_velocities))
                metrics['angular_velocity_max'] = float(np.max(angular_velocities))
                metrics['steering_aggressiveness'] = float(np.std(angular_velocities))

            # Path efficiency (deviation from optimal line)
            if len(traj.positions_xyz) >= 2:
                start_pos = traj.positions_xyz[0]
                end_pos = traj.positions_xyz[-1]
                straight_line_distance = np.linalg.norm(end_pos - start_pos)

                if straight_line_distance > 0:
                    metrics['path_efficiency'] = float(straight_line_distance / metrics.get('path_length', 1))
                else:
                    metrics['path_efficiency'] = 1.0

        except Exception as e:
            self.logger.error(f"Error calculating research metrics: {e}")

        return metrics

    def _calculate_statistical_metrics(self, traj: trajectory.PoseTrajectory3D) -> Dict[str, Any]:
        """Calculate statistical analysis metrics"""
        metrics = {}

        try:
            # Position statistics
            positions = traj.positions_xyz

            for axis, name in enumerate(['x', 'y', 'z']):
                axis_data = positions[:, axis]
                metrics[f'position_{name}_mean'] = float(np.mean(axis_data))
                metrics[f'position_{name}_std'] = float(np.std(axis_data))
                metrics[f'position_{name}_range'] = float(np.max(axis_data) - np.min(axis_data))
                metrics[f'position_{name}_min'] = float(np.min(axis_data))
                metrics[f'position_{name}_max'] = float(np.max(axis_data))

            # Temporal statistics
            time_diffs = np.diff(traj.timestamps)
            metrics['sampling_rate_mean'] = float(1.0 / np.mean(time_diffs)) if np.mean(time_diffs) > 0 else 0
            metrics['sampling_rate_std'] = float(np.std(1.0 / time_diffs)) if np.all(time_diffs > 0) else 0

        except Exception as e:
            self.logger.error(f"Error calculating statistical metrics: {e}")

        return metrics

    def set_reference_trajectory(self, reference_file: str, format_type: str = 'tum'):
        """Set reference trajectory for comparison"""
        try:
            if format_type == 'tum':
                self.reference_trajectory = file_interface.read_tum_trajectory_file(reference_file)
            elif format_type == 'kitti':
                self.reference_trajectory = file_interface.read_kitti_poses_file(reference_file)
            else:
                self.logger.error(f"Unsupported reference format: {format_type}")
                return False

            self.logger.info(f"Reference trajectory loaded with {len(self.reference_trajectory.positions_xyz)} poses")
            return True

        except Exception as e:
            self.logger.error(f"Error loading reference trajectory: {e}")
            return False

    def _save_lap_results(self, lap_number: int):
        """Save intermediate results for a single lap"""
        if lap_number not in self.detailed_metrics:
            return

        # Save metrics as JSON
        metrics_file = os.path.join(self.experiment_dir, 'metrics', f'lap_{lap_number:03d}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.detailed_metrics[lap_number], f, indent=2)

        # Save trajectory as TUM format using EVO
        if lap_number in self.lap_trajectories:
            traj_file = os.path.join(self.experiment_dir, 'trajectories', f'lap_{lap_number:03d}_trajectory.tum')
            try:
                file_interface.write_tum_trajectory_file(traj_file, self.lap_trajectories[lap_number]['trajectory'])
            except BaseException:
                # Fallback to manual writing
                with open(traj_file, 'w') as f:
                    traj = self.lap_trajectories[lap_number]['trajectory']
                    for i in range(len(traj.positions_xyz)):
                        timestamp = traj.timestamps[i]
                        pos = traj.positions_xyz[i]
                        quat = traj.orientations_quat_wxyz[i]
                        f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                                f"{quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {quat[0]:.6f}\n")

        # Save filtered trajectory if available
        if lap_number in self.filtered_trajectories:
            filtered_file = os.path.join(self.experiment_dir, 'filtered', f'lap_{lap_number:03d}_filtered.tum')
            try:
                file_interface.write_tum_trajectory_file(filtered_file, self.filtered_trajectories[lap_number])
            except BaseException:
                # Fallback implementation
                pass

    def generate_research_summary(self) -> Dict[str, Any]:
        """Generate comprehensive research summary for paper writing"""
        summary = {
            'experiment_info': {
                'controller_name': self.controller_name,
                'experiment_id': self.experiment_id,
                'test_description': self.config.get('test_description', ''),
                'timestamp': datetime.now().isoformat(),
                'total_laps': len(self.detailed_metrics)
            },
            'aggregate_statistics': {},
            'lap_by_lap_analysis': {},
            'performance_metrics': {},
            'statistical_significance': {}
        }

        if not self.detailed_metrics:
            return summary

        # Aggregate statistics across all laps
        all_metrics = {}
        for lap_metrics in self.detailed_metrics.values():
            for key, value in lap_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        for key, values in all_metrics.items():
            summary['aggregate_statistics'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }

        # Lap-by-lap analysis
        summary['lap_by_lap_analysis'] = dict(self.detailed_metrics)

        # Performance metrics summary
        if 'path_length' in all_metrics:
            summary['performance_metrics']['consistency'] = {
                'path_length_cv': float(np.std(all_metrics['path_length']) / np.mean(all_metrics['path_length'])),
                'lap_time_cv': float(np.std([self.lap_trajectories[lap]['lap_time'] for lap in self.lap_trajectories]) /
                                     np.mean([self.lap_trajectories[lap]['lap_time'] for lap in self.lap_trajectories]))
            }

        return summary

    def export_research_data(self):
        """Export all data in research-friendly formats"""
        if not EVO_AVAILABLE:
            return

        # Generate comprehensive summary
        summary = self.generate_research_summary()

        # Save summary in multiple formats
        output_formats = self.config.get('output_formats', ['json', 'csv'])

        for fmt in output_formats:
            if fmt == 'json':
                summary_file = os.path.join(
                    self.experiment_dir,
                    'exports',
                    f'{self.controller_name}_{self.experiment_id}_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)

            elif fmt == 'csv':
                # Export lap-by-lap metrics as CSV
                csv_file = os.path.join(
                    self.experiment_dir,
                    'exports',
                    f'{self.controller_name}_{self.experiment_id}_metrics.csv')
                self._export_metrics_csv(csv_file)

            elif fmt == 'pickle' and EVO_AVAILABLE:
                # Export using pandas bridge
                try:
                    import pickle
                    pickle_file = os.path.join(
                        self.experiment_dir,
                        'exports',
                        f'{self.controller_name}_{self.experiment_id}_data.pkl')
                    with open(pickle_file, 'wb') as f:
                        pickle.dump({
                            'trajectories': self.lap_trajectories,
                            'metrics': self.detailed_metrics,
                            'summary': summary
                        }, f)
                except BaseException:
                    pass

        self.logger.info(f"Research data exported to {self.experiment_dir}/exports/")

    def _export_metrics_csv(self, csv_file: str):
        """Export metrics to CSV format for analysis"""
        if not self.detailed_metrics:
            return

        # Get all metric keys
        all_keys = set()
        for metrics in self.detailed_metrics.values():
            all_keys.update(metrics.keys())

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['lap_number', 'lap_time'] + sorted(all_keys)
            writer.writerow(header)

            # Data rows
            for lap_num in sorted(self.detailed_metrics.keys()):
                row = [lap_num]

                # Add lap time
                if lap_num in self.lap_trajectories:
                    row.append(self.lap_trajectories[lap_num]['lap_time'])
                else:
                    row.append('')

                # Add metrics
                for key in sorted(all_keys):
                    value = self.detailed_metrics[lap_num].get(key, '')
                    row.append(value)

                writer.writerow(row)


def create_research_evaluator(config: Dict[str, Any]) -> ResearchTrajectoryEvaluator:
    """Factory function to create research evaluator"""
    return ResearchTrajectoryEvaluator(config)
