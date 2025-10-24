#!/usr/bin/env python3

"""
EVO Plotter Module

Integrates EVO's plotting capabilities for automated race analysis graph generation.
Provides comprehensive visualization of trajectory data, performance metrics, and
comparative analysis across multiple laps.

License: MIT
"""

import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# EVO library setup for plotting
evo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'evo')
if os.path.exists(evo_path) and evo_path not in sys.path:
    sys.path.insert(0, evo_path)

# Import EVO modules
try:
    import evo
    print(f"EVO version: {evo.__version__}")
    from evo.tools import plot
    from evo.core import trajectory, metrics, sync
    from evo.tools import file_interface
    from evo.core.units import Unit
    EVO_AVAILABLE = True
    print("EVO modules imported successfully")
except ImportError as e:
    EVO_AVAILABLE = False
    print(f"Warning: EVO not available. Plotting features will be disabled. Error: {e}")
    print(f"Python path: {sys.path}")
    print("This is likely due to matplotlib version incompatibility between system and user packages.")
    print("Try: pip uninstall matplotlib && sudo apt install python3-matplotlib")


class EVOPlotter:
    """Enhanced EVO plotting integration for race monitor"""

    def __init__(self, config):
        """Initialize the EVO plotter with configuration"""
        self.config = config
        self.EVO_AVAILABLE = EVO_AVAILABLE  # Make EVO_AVAILABLE accessible as instance attribute
        self.plot_collection = None
        self.reference_trajectory = None
        self.lap_trajectories = {}
        self.lap_metrics = {}

        # Create graph output directory
        if self.config.get('auto_generate_graphs', False):
            graph_dir_config = self.config.get('graph_output_directory', 'evaluation_results/graphs')
            if not os.path.isabs(graph_dir_config):
                # Make path relative to the race_monitor package directory
                package_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to race_monitor package root
                graph_dir = os.path.join(package_dir, graph_dir_config)
            else:
                graph_dir = graph_dir_config
            os.makedirs(graph_dir, exist_ok=True)

    def load_reference_trajectory(self, filepath, format_type='csv'):
        """Load reference trajectory from file"""
        if not EVO_AVAILABLE:
            return False

        try:
            if format_type == 'csv':
                # Load CSV format (x, y, v, theta)
                data = np.loadtxt(filepath, delimiter=',', skiprows=1)
                x = data[:, 0]
                y = data[:, 1]
                v = data[:, 2]
                theta = data[:, 3]

                # Convert to EVO trajectory format
                # Create timestamps (assuming 10Hz sampling)
                timestamps = np.arange(len(x)) * 0.1

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

            elif format_type == 'tum':
                self.reference_trajectory = file_interface.read_tum_trajectory_file(filepath)
            elif format_type == 'kitti':
                self.reference_trajectory = file_interface.read_kitti_poses_file(filepath)
            else:
                print(f"Unsupported reference trajectory format: {format_type}")
                return False

            # Check the actual attribute name for EVO trajectory
            if hasattr(self.reference_trajectory, 'poses'):
                num_poses = len(self.reference_trajectory.poses)
            elif hasattr(self.reference_trajectory, 'positions_xyz'):
                num_poses = len(self.reference_trajectory.positions_xyz)
            else:
                num_poses = 'unknown'
            print(f"Loaded reference trajectory with {num_poses} poses")
            return True

        except Exception as e:
            print(f"❌ Failed to load reference trajectory from data: {e}")

    def _create_reference_from_laps(self):
        """Create a reference trajectory from the average of all lap trajectories"""
        if not self.lap_trajectories:
            return

        try:
            print(f"🎯 Creating reference trajectory from {len(self.lap_trajectories)} laps...")

            # Get all trajectory positions
            all_positions = []
            for lap_num, traj in self.lap_trajectories.items():
                positions = traj.positions_xyz
                all_positions.append(positions)

            # Find minimum length to avoid index issues
            min_length = min(len(pos) for pos in all_positions)
            print(f"📏 Using first {min_length} points for reference")

            # Truncate all trajectories to same length
            truncated_positions = [pos[:min_length] for pos in all_positions]

            # Calculate average trajectory (centroid)
            avg_positions = np.mean(truncated_positions, axis=0)

            # Smooth the reference trajectory if scipy is available
            try:
                from scipy import signal
                window_length = min(51, len(avg_positions) // 10)
                if window_length % 2 == 0:
                    window_length += 1

                if window_length >= 3:
                    avg_positions[:, 0] = signal.savgol_filter(avg_positions[:, 0], window_length, 3)
                    avg_positions[:, 1] = signal.savgol_filter(avg_positions[:, 1], window_length, 3)
                    avg_positions[:, 2] = signal.savgol_filter(avg_positions[:, 2], window_length, 3)
                    print(f"✅ Applied smoothing to reference trajectory")
            except ImportError:
                print(f"⚠️ Scipy not available, using unsmoothed reference")

            # Create timestamps
            timestamps = np.arange(len(avg_positions)) * 0.1  # Assume 10Hz

            # Create default orientations (looking forward)
            qw = np.ones(len(avg_positions))
            qx = np.zeros(len(avg_positions))
            qy = np.zeros(len(avg_positions))
            qz = np.zeros(len(avg_positions))

            # Create EVO trajectory
            self.reference_trajectory = trajectory.PoseTrajectory3D(
                positions_xyz=avg_positions,
                orientations_quat_wxyz=np.column_stack([qw, qx, qy, qz]),
                timestamps=timestamps
            )

            print(f"✅ Created reference trajectory with {len(avg_positions)} points")

        except Exception as e:
            print(f"❌ Failed to create reference trajectory from laps: {e}")

    def _find_best_lap(self):
        """Find the best lap based on lowest average error to reference trajectory"""
        if not self.lap_trajectories or not self.reference_trajectory:
            return None

        best_lap = None
        lowest_error = float('inf')

        try:
            for lap_num, traj in self.lap_trajectories.items():
                try:
                    # Synchronize trajectories
                    traj_ref, traj_est = sync.associate_trajectories(
                        self.reference_trajectory, traj, max_diff=0.01)

                    # Calculate average error
                    pose_relation = metrics.PoseRelation.translation_part
                    ape_metric = metrics.APE(pose_relation)
                    ape_metric.process_data((traj_ref, traj_est))
                    avg_error = np.mean(ape_metric.error)

                    print(f"🏁 Lap {lap_num}: Average error = {avg_error:.4f}m")

                    if avg_error < lowest_error:
                        lowest_error = avg_error
                        best_lap = lap_num

                except Exception as e:
                    print(f"Warning: Could not evaluate lap {lap_num}: {e}")
                    continue

            if best_lap is not None:
                print(f"🏆 Best lap identified: Lap {best_lap} (avg error: {lowest_error:.4f}m)")

            return best_lap

        except Exception as e:
            print(f"❌ Failed to find best lap: {e}")
            return None

    def generate_plots(self):
        return False

    def add_lap_trajectory(self, lap_number, trajectory_data):
        """Add a completed lap trajectory for plotting"""
        print(f"=== ADDING LAP TRAJECTORY DEBUG ===")
        print(f"lap_number: {lap_number}")
        print(f"trajectory_data type: {type(trajectory_data)}")
        print(f"trajectory_data length: {len(trajectory_data) if hasattr(trajectory_data, '__len__') else 'no length'}")
        print(f"EVO_AVAILABLE: {EVO_AVAILABLE}")

        if not EVO_AVAILABLE:
            print("❌ EVO not available, returning early")
            return

        try:
            # Convert trajectory data to EVO format
            poses = trajectory_data
            if len(poses) < 2:
                print(f"❌ Not enough poses: {len(poses)}")
                return

            # Handle both ROS2 messages and F1Tenth data
            positions = []
            orientations = []
            timestamps = []

            for i, pose in enumerate(poses):
                # Extract position
                if hasattr(pose['pose'], 'position'):
                    x = pose['pose'].position.x
                    y = pose['pose'].position.y
                    z = getattr(pose['pose'].position, 'z', 0.0)
                else:
                    # Roboracer format
                    x = pose['pose'].x
                    y = pose['pose'].y
                    z = 0.0

                positions.append([x, y, z])

                # Extract orientation
                if hasattr(pose['pose'], 'orientation'):
                    # ROS2 message format
                    orientations.append([
                        pose['pose'].orientation.w,
                        pose['pose'].orientation.x,
                        pose['pose'].orientation.y,
                        pose['pose'].orientation.z
                    ])
                else:
                    # Roboracer format - convert theta to quaternion
                    theta = pose['pose'].theta
                    orientations.append([
                        np.cos(theta / 2),  # w
                        0.0,                # x
                        0.0,                # y
                        np.sin(theta / 2)   # z
                    ])

                # Handle timestamps
                if pose['header'] is not None and hasattr(pose['header'], 'stamp'):
                    # ROS2 message with header
                    timestamp = pose['header'].stamp.sec + pose['header'].stamp.nanosec * 1e-9
                else:
                    # Roboracer format - create synthetic timestamps
                    timestamp = i * 0.1  # Assume 10Hz sampling

                timestamps.append(timestamp)

            # Create trajectory object
            traj = trajectory.PoseTrajectory3D(
                positions_xyz=np.array(positions),
                orientations_quat_wxyz=np.array(orientations),
                timestamps=np.array(timestamps)
            )

            self.lap_trajectories[lap_number] = traj
            print(f"✅ Added lap {lap_number} trajectory for plotting with {len(poses)} poses")
            print(
                f"Trajectory data: positions shape {traj.positions_xyz.shape}, orientations shape {traj.orientations_quat_wxyz.shape}")
            print(f"Total trajectories stored: {len(self.lap_trajectories)}")
            print(f"Trajectory keys: {list(self.lap_trajectories.keys())}")

        except Exception as e:
            print(f"Error adding lap {lap_number} trajectory: {e}")

    def add_lap_trajectory_evo(self, lap_number: int, evo_trajectory):
        """Add EVO trajectory object directly for plotting"""
        if not EVO_AVAILABLE:
            return

        try:
            self.lap_trajectories[lap_number] = evo_trajectory
            print(f"✅ Added EVO lap {lap_number} trajectory for plotting with {len(evo_trajectory.timestamps)} poses")
            print(f"Total trajectories stored: {len(self.lap_trajectories)}")
            print(f"Trajectory keys: {list(self.lap_trajectories.keys())}")
        except Exception as e:
            print(f"Error adding EVO lap {lap_number} trajectory: {e}")

    def load_reference_trajectory_from_data(self, traj_data):
        """Load reference trajectory from EVO trajectory object directly"""
        if not EVO_AVAILABLE:
            return False

        try:
            self.reference_trajectory = traj_data
            return True
        except Exception as e:
            print(f"Could not load reference trajectory from data: {e}")
            return False

    def add_lap_metrics(self, lap_number, metrics_data):
        """Add lap metrics for plotting"""
        self.lap_metrics[lap_number] = metrics_data

    def generate_all_plots(self):
        """Generate all configured plots using EVO"""
        print(f"=== EVO PLOT GENERATION DEBUG ===")
        print(f"EVO_AVAILABLE: {EVO_AVAILABLE}")
        print(f"auto_generate_graphs: {self.config.get('auto_generate_graphs', False)}")
        print(f"lap_trajectories count: {len(self.lap_trajectories)}")
        print(f"lap_trajectories keys: {list(self.lap_trajectories.keys())}")
        print(f"lap_metrics count: {len(self.lap_metrics)}")
        print(f"lap_metrics keys: {list(self.lap_metrics.keys())}")

        if not EVO_AVAILABLE or not self.config.get('auto_generate_graphs', False):
            print(f"❌ EVO not available or auto_generate_graphs disabled")
            return False

        if not self.lap_trajectories:
            print(f"❌ No lap trajectories available for plotting")
            return False

        print(f"✅ Starting plot generation...")
        # Warn if no reference trajectory
        if not self.reference_trajectory:
            import warnings
            warnings.warn("No reference trajectory provided. Comparison plots will be skipped.", UserWarning)
            print("⚠️  No reference trajectory - comparison plots will be skipped")

        try:
            # Initialize plot collection
            print(f"Creating plot collection...")
            self.plot_collection = plot.PlotCollection("Race Monitor - EVO Analysis")

            # Auto-generate reference trajectory if not available
            if not self.reference_trajectory and self.lap_trajectories:
                print(f"🎯 Auto-generating reference trajectory from lap data...")
                self._create_reference_from_laps()

            # Generate different types of plots
            print(f"Generating trajectory plots...")
            if self.config.get('generate_trajectory_plots', True):
                self._generate_trajectory_plots()

            print(f"Generating XYZ plots...")
            if self.config.get('generate_xyz_plots', True):
                self._generate_xyz_plots()

            print(f"Generating RPY plots...")
            if self.config.get('generate_rpy_plots', True):
                self._generate_rpy_plots()

            print(f"Generating speed plots...")
            if self.config.get('generate_speed_plots', True):
                self._generate_speed_plots()

            print(f"Generating error plots...")
            if self.config.get('generate_error_plots', True) and self.reference_trajectory:
                self._generate_error_plots()

            print(f"Generating metrics plots...")
            if self.config.get('generate_metrics_plots', True):
                self._generate_metrics_plots()

            # Generate new advanced plot types
            print(f"Generating error-mapped trajectory plots...")
            if self.config.get('generate_error_mapped_plots', True) and self.reference_trajectory:
                self._generate_error_mapped_trajectory()

            print(f"Generating violin/box plots...")
            if self.config.get('generate_violin_plots', True) and self.reference_trajectory:
                self._generate_violin_plots()

            print(f"Generating 3D trajectory plots with vectors...")
            if self.config.get('generate_3d_vector_plots', True):
                self._generate_3d_trajectory_with_vectors()

            # Save all plots
            print(f"Saving all plots...")
            success = self._save_all_plots()

            if success:
                print("✅ Generated all EVO plots successfully!")
                return True
            else:
                print("⚠️ Some plots may have failed to save")
                return False

        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            return False

    def _generate_trajectory_plots(self):
        """Generate 2D trajectory plots"""
        if not self.lap_trajectories:
            return

        # Create trajectory plot using standard matplotlib
        fig_traj = plt.figure(figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))
        ax_traj = fig_traj.add_subplot(111)

        # Plot reference trajectory if available
        if self.reference_trajectory:
            ref_positions = self.reference_trajectory.positions_xyz
            ax_traj.plot(ref_positions[:, 0], ref_positions[:, 1],
                         '--', color='black', label='Reference', alpha=0.7, linewidth=2)
            # Add only start marker
            ax_traj.scatter(ref_positions[0, 0], ref_positions[0, 1],
                            color='darkgreen', s=100, marker='o', label='Start', zorder=5)

        # Plot all lap trajectories
        colors = cm.get_cmap(self.config.get('plot_color_scheme', 'viridis'))
        for i, (lap_num, traj) in enumerate(self.lap_trajectories.items()):
            color = colors(i / len(self.lap_trajectories))
            positions = traj.positions_xyz
            ax_traj.plot(positions[:, 0], positions[:, 1],
                         '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            # Add only start marker for first lap
            if i == 0 and not self.reference_trajectory:
                # Only show start if no reference trajectory
                ax_traj.scatter(positions[0, 0], positions[0, 1],
                                color='darkgreen', s=100, marker='o', label='Start', zorder=5)
            elif i == 0:
                # Don't add duplicate start label
                ax_traj.scatter(positions[0, 0], positions[0, 1],
                                color=color, s=80, marker='o', alpha=0.9, zorder=4)
            else:
                ax_traj.scatter(positions[0, 0], positions[0, 1],
                                color=color, s=80, marker='o', alpha=0.9, zorder=4)

        ax_traj.set_title('Trajectory Comparison')
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.legend()
        ax_traj.grid(True)
        ax_traj.axis('equal')

        self.plot_collection.add_figure("trajectories", fig_traj)

    def _generate_xyz_plots(self):
        """Generate X, Y, Z position plots (without reference trajectory)"""
        if not self.lap_trajectories:
            return

        fig_xyz, axarr_xyz = plt.subplots(3, 1, sharex=True,
                                          figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])),
                                          constrained_layout=False)

        # Plot all lap trajectories (no reference)
        colors = cm.get_cmap(self.config.get('plot_color_scheme', 'viridis'))
        for i, (lap_num, traj) in enumerate(self.lap_trajectories.items()):
            color = colors(i / len(self.lap_trajectories))
            positions = traj.positions_xyz
            timestamps = traj.timestamps
            axarr_xyz[0].plot(timestamps, positions[:, 0],
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            axarr_xyz[1].plot(timestamps, positions[:, 1],
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            axarr_xyz[2].plot(timestamps, positions[:, 2],
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)

        axarr_xyz[0].set_title('X Position')
        axarr_xyz[0].set_ylabel('X (m)')
        axarr_xyz[1].set_title('Y Position')
        axarr_xyz[1].set_ylabel('Y (m)')
        axarr_xyz[2].set_title('Z Position')
        axarr_xyz[2].set_ylabel('Z (m)')
        axarr_xyz[2].set_xlabel('Time (s)')

        for ax in axarr_xyz:
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        self.plot_collection.add_figure("xyz", fig_xyz)

    def _generate_rpy_plots(self):
        """Generate Roll, Pitch, Yaw plots (without reference trajectory)"""
        if not self.lap_trajectories:
            return

        fig_rpy, axarr_rpy = plt.subplots(3, 1, sharex=True,
                                          figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])),
                                          constrained_layout=False)

        # Plot all lap trajectories (no reference)
        colors = cm.get_cmap(self.config.get('plot_color_scheme', 'viridis'))
        for i, (lap_num, traj) in enumerate(self.lap_trajectories.items()):
            color = colors(i / len(self.lap_trajectories))
            orientations = traj.orientations_quat_wxyz
            timestamps = traj.timestamps
            # Convert quaternions to Euler angles for plotting
            rpy = self._quat_to_euler(orientations)
            axarr_rpy[0].plot(timestamps, rpy[:, 0],
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            axarr_rpy[1].plot(timestamps, rpy[:, 1],
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            axarr_rpy[2].plot(timestamps, rpy[:, 2],
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)

        axarr_rpy[0].set_title('Roll')
        axarr_rpy[0].set_ylabel('Roll (rad)')
        axarr_rpy[1].set_title('Pitch')
        axarr_rpy[1].set_ylabel('Pitch (rad)')
        axarr_rpy[2].set_title('Yaw')
        axarr_rpy[2].set_ylabel('Yaw (rad)')
        axarr_rpy[2].set_xlabel('Time (s)')

        for ax in axarr_rpy:
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        self.plot_collection.add_figure("rpy", fig_rpy)

    def _quat_to_euler(self, quaternions):
        """Convert quaternions (w,x,y,z) to Euler angles (roll,pitch,yaw)"""
        roll = np.arctan2(2 * (quaternions[:, 0] * quaternions[:, 1] + quaternions[:, 2] * quaternions[:, 3]),
                          1 - 2 * (quaternions[:, 1]**2 + quaternions[:, 2]**2))
        pitch = np.arcsin(2 * (quaternions[:, 0] * quaternions[:, 2] - quaternions[:, 3] * quaternions[:, 1]))
        yaw = np.arctan2(2 * (quaternions[:, 0] * quaternions[:, 3] + quaternions[:, 1] * quaternions[:, 2]),
                         1 - 2 * (quaternions[:, 2]**2 + quaternions[:, 3]**2))
        return np.column_stack([roll, pitch, yaw])

    def _generate_speed_plots(self):
        """Generate speed analysis plots (without reference trajectory)"""
        if not self.lap_trajectories:
            return

        fig_speed = plt.figure(figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))
        ax_speed = fig_speed.gca()

        # Plot all lap trajectories (no reference)
        colors = cm.get_cmap(self.config.get('plot_color_scheme', 'viridis'))
        for i, (lap_num, traj) in enumerate(self.lap_trajectories.items()):
            try:
                color = colors(i / len(self.lap_trajectories))
                speeds = self._calculate_speeds(traj)
                timestamps = traj.timestamps[1:]  # One less due to speed calculation
                ax_speed.plot(timestamps, speeds,
                              '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            except Exception as e:
                print(f"Could not plot speeds for lap {lap_num}: {e}")

        ax_speed.set_title('Speed Analysis')
        ax_speed.set_xlabel('Time (s)')
        ax_speed.set_ylabel('Speed (m/s)')
        ax_speed.legend()
        ax_speed.grid(True)

        self.plot_collection.add_figure("speeds", fig_speed)

    def _calculate_speeds(self, trajectory):
        """Calculate speeds from trajectory positions"""
        positions = trajectory.positions_xyz
        timestamps = trajectory.timestamps

        speeds = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 0:
                dp = np.linalg.norm(positions[i] - positions[i - 1])
                speed = dp / dt
                speeds.append(speed)
            else:
                speeds.append(0.0)

        return np.array(speeds)

    def _generate_error_plots(self):
        """Generate APE/RPE error analysis plots"""
        if not self.reference_trajectory or not self.lap_trajectories:
            return

        # Calculate APE and RPE for each lap
        ape_scores = {}
        rpe_scores = {}

        for lap_num, traj in self.lap_trajectories.items():
            try:
                # Synchronize trajectories
                traj_ref, traj_est = sync.associate_trajectories(
                    self.reference_trajectory, traj, max_diff=0.01)

                # Calculate APE
                pose_relation = metrics.PoseRelation.translation_part
                ape_metric = metrics.APE(pose_relation)
                ape_metric.process_data((traj_ref, traj_est))
                ape_scores[lap_num] = ape_metric.get_statistic(metrics.StatisticsType.rmse)

                # Calculate RPE
                rpe_metric = metrics.RPE(pose_relation)
                rpe_metric.process_data((traj_ref, traj_est))
                rpe_scores[lap_num] = rpe_metric.get_statistic(metrics.StatisticsType.rmse)

            except Exception as e:
                print(f"Could not calculate errors for lap {lap_num}: {e}")

        if not ape_scores:
            return

        # Create error plots
        fig_error, (ax_ape, ax_rpe) = plt.subplots(2, 1, figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])),
                                                   constrained_layout=False)

        # APE plot
        lap_numbers = list(ape_scores.keys())
        ape_values = list(ape_scores.values())
        ax_ape.bar(lap_numbers, ape_values, color='red', alpha=0.7)
        ax_ape.set_title('Absolute Pose Error (APE) - RMSE')
        ax_ape.set_xlabel('Lap Number')
        ax_ape.set_ylabel('APE RMSE (m)')
        ax_ape.grid(True)

        # RPE plot
        rpe_values = [rpe_scores.get(lap, 0) for lap in lap_numbers]
        ax_rpe.bar(lap_numbers, rpe_values, color='blue', alpha=0.7)
        ax_rpe.set_title('Relative Pose Error (RPE) - RMSE')
        ax_rpe.set_xlabel('Lap Number')
        ax_rpe.set_ylabel('RPE RMSE (m)')
        ax_rpe.grid(True)

        plt.tight_layout()
        self.plot_collection.add_figure("errors", fig_error)

    def _generate_metrics_plots(self):
        """Generate metrics over time plots"""
        if not self.lap_metrics:
            return

        # Extract metrics data
        lap_numbers = sorted(self.lap_metrics.keys())
        smoothness_values = [self.lap_metrics[lap].get('smoothness', 0) for lap in lap_numbers]
        consistency_values = [self.lap_metrics[lap].get('consistency', 0) for lap in lap_numbers]
        path_lengths = [self.lap_metrics[lap].get('path_length', 0) for lap in lap_numbers]

        # Create metrics plots
        fig_metrics, ((ax_smooth, ax_consist), (ax_length, ax_combined)) = plt.subplots(
            2, 2, figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])),
            constrained_layout=False)

        # Smoothness plot
        ax_smooth.plot(lap_numbers, smoothness_values, 'bo-', linewidth=2, markersize=8)
        ax_smooth.set_title('Trajectory Smoothness')
        ax_smooth.set_xlabel('Lap Number')
        ax_smooth.set_ylabel('Smoothness Score')
        ax_smooth.grid(True)

        # Consistency plot
        ax_consist.plot(lap_numbers, consistency_values, 'ro-', linewidth=2, markersize=8)
        ax_consist.set_title('Trajectory Consistency')
        ax_consist.set_xlabel('Lap Number')
        ax_consist.set_ylabel('Consistency Score')
        ax_consist.grid(True)

        # Path length plot
        ax_length.plot(lap_numbers, path_lengths, 'go-', linewidth=2, markersize=8)
        ax_length.set_title('Path Length')
        ax_length.set_xlabel('Lap Number')
        ax_length.set_ylabel('Length (m)')
        ax_length.grid(True)

        # Combined metrics plot
        ax_combined.plot(lap_numbers, smoothness_values, 'bo-', label='Smoothness', linewidth=2)
        ax_combined.plot(lap_numbers, consistency_values, 'ro-', label='Consistency', linewidth=2)
        ax_combined.set_title('Combined Metrics')
        ax_combined.set_xlabel('Lap Number')
        ax_combined.set_ylabel('Score')
        ax_combined.legend()
        ax_combined.grid(True)

        plt.tight_layout()
        self.plot_collection.add_figure("metrics", fig_metrics)

    def _generate_error_mapped_trajectory(self):
        """Generate trajectory plot with error mapped as colors for the best lap only"""
        if not self.reference_trajectory or not self.lap_trajectories:
            return

        try:
            print("🎨 Generating error-mapped trajectory plot for best lap...")

            # Find the best lap
            best_lap = self._find_best_lap()
            if best_lap is None:
                print("❌ Could not determine best lap")
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 10))

            # Plot reference trajectory with better visibility
            ref_xyz = self.reference_trajectory.positions_xyz
            ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], '--', color='black',
                    linewidth=4, alpha=0.9, label='Reference Trajectory', zorder=1)

            # Process only the best lap
            traj = self.lap_trajectories[best_lap]
            try:
                # Synchronize trajectories
                traj_ref, traj_est = sync.associate_trajectories(
                    self.reference_trajectory, traj, max_diff=0.01)

                # Calculate point-wise errors
                pose_relation = metrics.PoseRelation.translation_part
                ape_metric = metrics.APE(pose_relation)
                ape_metric.process_data((traj_ref, traj_est))
                errors = ape_metric.error

                # Get trajectory positions
                est_xyz = traj_est.positions_xyz

                # Create the main error-mapped visualization
                scatter = ax.scatter(est_xyz[:, 0], est_xyz[:, 1], c=errors, cmap='viridis',
                                     s=30, alpha=0.8, edgecolors='none', zorder=3)

            except Exception as e:
                print(f"Warning: Could not process best lap {best_lap}: {e}")
                return

            # Add colorbar at the bottom to avoid overlap with legend
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, location='bottom', pad=0.1)
            cbar.set_label('Error (m)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

            # Formatting
            ax.set_xlabel('x (m)', fontsize=12)
            ax.set_ylabel('y (m)', fontsize=12)
            ax.set_title(f'Error Mapped onto Trajectory (Best Lap {best_lap})', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)  # Moved to upper left to avoid colorbar
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # Improve layout
            plt.tight_layout()
            self.plot_collection.add_figure("best_lap_error_mapped_trajectory", fig)

            print(f"✅ Generated error-mapped trajectory for best lap {best_lap}")

        except Exception as e:
            print(f"❌ Could not generate error-mapped trajectory: {e}")

    def _generate_violin_plots(self):
        """Generate combined violin/box plots for error comparison across all laps"""
        if not self.reference_trajectory or not self.lap_trajectories:
            return

        try:
            import seaborn as sns

            print("📊 Generating combined statistical comparison plot...")

            # Collect APE data for all laps
            data_for_plot = []

            for lap_num, traj in self.lap_trajectories.items():
                try:
                    # Synchronize trajectories
                    traj_ref, traj_est = sync.associate_trajectories(
                        self.reference_trajectory, traj, max_diff=0.01)

                    # Calculate APE
                    pose_relation = metrics.PoseRelation.translation_part
                    ape_metric = metrics.APE(pose_relation)
                    ape_metric.process_data((traj_ref, traj_est))

                    # Add errors to data
                    for error in ape_metric.error:
                        data_for_plot.append({'Lap': f'Lap {lap_num}', 'Error (m)': error})

                except Exception as e:
                    print(f"Warning: Error processing lap {lap_num} for violin plot: {e}")
                    continue

            if not data_for_plot:
                print("❌ No data available for statistical comparison")
                return

            # Create DataFrame for plotting
            df = pd.DataFrame(data_for_plot)

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Generate violin plot
            sns.violinplot(data=df, x='Lap', y='Error (m)', ax=ax, hue='Lap', palette='Set2', legend=False)

            # Add box plot overlay for better statistics visibility
            sns.boxplot(data=df, x='Lap', y='Error (m)', ax=ax,
                        width=0.3, hue='Lap', palette='Set2', legend=False)

            # Formatting
            ax.set_title('Trajectory Error Distribution Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Error (m)', fontsize=12)
            ax.set_xlabel('Lap', fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            self.plot_collection.add_figure("Trajectory_Error_Distribution", fig)

            print(f"✅ Generated statistical comparison for {len(df)} data points")

        except ImportError:
            print("⚠️ Seaborn not available, generating simple box plot instead...")
            self._generate_simple_box_plot()
        except Exception as e:
            print(f"❌ Could not generate statistical comparison plot: {e}")

    def _generate_simple_box_plot(self):
        """Generate simple box plot as fallback when seaborn is not available"""
        try:
            # Collect APE data for all laps
            lap_errors = {}

            for lap_num, traj in self.lap_trajectories.items():
                try:
                    # Synchronize trajectories
                    traj_ref, traj_est = sync.associate_trajectories(
                        self.reference_trajectory, traj, max_diff=0.01)

                    # Calculate APE
                    pose_relation = metrics.PoseRelation.translation_part
                    ape_metric = metrics.APE(pose_relation)
                    ape_metric.process_data((traj_ref, traj_est))

                    lap_errors[f'Lap {lap_num}'] = ape_metric.error

                except Exception as e:
                    print(f"Warning: Error processing lap {lap_num} for box plot: {e}")
                    continue

            if not lap_errors:
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create box plot
            box_data = list(lap_errors.values())
            labels = list(lap_errors.keys())

            ax.boxplot(box_data, labels=labels)

            # Formatting
            ax.set_title('Trajectory Error Distribution Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Error (m)', fontsize=12)
            ax.set_xlabel('Lap', fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            self.plot_collection.add_figure("simple_statistical_comparison", fig)

        except Exception as e:
            print(f"❌ Could not generate simple box plot: {e}")

    def _generate_3d_trajectory_with_vectors(self):
        """Generate 3D trajectory visualization with direction vectors for the best lap only"""
        if not self.lap_trajectories:
            return

        try:
            print("🔵 Generating 3D trajectory plot for best lap...")

            # Find the best lap
            best_lap = self._find_best_lap()
            if best_lap is None:
                print("❌ Could not determine best lap for 3D plot")
                return

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot reference trajectory if available
            if self.reference_trajectory:
                ref_xyz = self.reference_trajectory.positions_xyz
                ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2],
                        '--', color='black', linewidth=3, alpha=0.8, label='Reference')

            # Plot only the best lap trajectory
            traj = self.lap_trajectories[best_lap]
            try:
                est_xyz = traj.positions_xyz

                ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2],
                        color='blue', linewidth=3, alpha=0.9, label=f'Best Lap {best_lap}')

                # Add directional vectors at regular intervals
                step = max(1, len(est_xyz) // 20)  # Show ~20 vectors
                for j in range(0, len(est_xyz), step):
                    if j + 1 < len(est_xyz):
                        # Calculate direction vector
                        direction = est_xyz[j + 1] - est_xyz[j]
                        direction = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize

                        # Scale the vector for visibility
                        scale = 0.4
                        direction *= scale

                        # Draw arrow
                        ax.quiver(est_xyz[j, 0], est_xyz[j, 1], est_xyz[j, 2],
                                  direction[0], direction[1], direction[2],
                                  color='red', alpha=0.8, arrow_length_ratio=0.1)

            except Exception as e:
                print(f"Warning: Could not process best lap {best_lap} for 3D plot: {e}")
                return

            # Formatting
            ax.set_xlabel('x (m)', fontsize=12)
            ax.set_ylabel('y (m)', fontsize=12)
            ax.set_zlabel('z (m)', fontsize=12)
            ax.legend(fontsize=10)
            ax.set_title(f'3D Trajectory with Direction Vectors (Best Lap {best_lap})', fontsize=14, fontweight='bold')

            # Set equal aspect ratio using best lap and reference trajectory
            trajectories = [traj.positions_xyz]
            if self.reference_trajectory:
                trajectories.append(self.reference_trajectory.positions_xyz)

            if trajectories:
                all_x = np.concatenate([traj[:, 0] for traj in trajectories])
                all_y = np.concatenate([traj[:, 1] for traj in trajectories])
                all_z = np.concatenate([traj[:, 2] for traj in trajectories])

                max_range = np.array([all_x.max() - all_x.min(),
                                     all_y.max() - all_y.min(),
                                     all_z.max() - all_z.min()]).max() / 2.0

                mid_x = (all_x.max() + all_x.min()) * 0.5
                mid_y = (all_y.max() + all_y.min()) * 0.5
                mid_z = (all_z.max() + all_z.min()) * 0.5

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.tight_layout()
            self.plot_collection.add_figure("best_lap_3d_trajectory_vectors", fig)

            print(f"✅ Generated 3D trajectory plot for best lap {best_lap}")

        except Exception as e:
            print(f"❌ Could not generate 3D trajectory plot: {e}")

    def _generate_box_plots(self):
        """Fallback box plots if seaborn is not available"""
        if not self.reference_trajectory or not self.lap_trajectories:
            return

        ape_data_by_lap = {}

        for lap_num, traj in self.lap_trajectories.items():
            try:
                # Synchronize trajectories
                traj_ref, traj_est = sync.associate_trajectories(
                    self.reference_trajectory, traj, max_diff=0.01)

                # Calculate APE
                pose_relation = metrics.PoseRelation.translation_part
                ape_metric = metrics.APE(pose_relation)
                ape_metric.process_data((traj_ref, traj_est))

                ape_data_by_lap[f'LAP_{lap_num:02d}'] = ape_metric.error

            except Exception as e:
                print(f"Error processing lap {lap_num} for box plot: {e}")

        if not ape_data_by_lap:
            return

        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))

        data_values = list(ape_data_by_lap.values())
        labels = list(ape_data_by_lap.keys())

        box_plot = ax.boxplot(data_values, labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['steelblue', 'green'] * (len(box_plot['boxes']) // 2 + 1)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('APE (m)')
        ax.set_title('Error Distribution Comparison')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        self.plot_collection.add_figure("box_comparison", fig)

    def _save_all_plots(self):
        """Save all generated plots"""
        if not self.plot_collection:
            print("No plot collection to save")
            return

        print(f"Plot collection has {len(self.plot_collection.figures)} figures")

        graph_dir_config = self.config.get('graph_output_directory', 'evaluation_results/graphs')
        if not os.path.isabs(graph_dir_config):
            # Make path relative to the race_monitor package directory
            package_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to race_monitor package root
            graph_dir = os.path.join(package_dir, graph_dir_config)
        else:
            graph_dir = graph_dir_config

        graph_formats = self.config.get('graph_formats', ['png'])

        print(f"Saving plots to {graph_dir} in formats: {graph_formats}")

        # Ensure directory exists
        os.makedirs(graph_dir, exist_ok=True)

        dpi = self.config.get('plot_dpi', 300)

        # Save each figure individually using matplotlib
        for plot_name, figure in self.plot_collection.figures.items():
            try:
                for fmt in graph_formats:
                    if fmt.lower() in ['png', 'pdf', 'svg', 'eps']:
                        # Create extension-specific subdirectory
                        format_dir = os.path.join(graph_dir, fmt.lower())
                        os.makedirs(format_dir, exist_ok=True)

                        output_path = os.path.join(format_dir, f"{plot_name}.{fmt.lower()}")
                        print(f"Saving {plot_name} as {fmt.upper()} to {output_path}")
                        figure.savefig(output_path, format=fmt.lower(), dpi=dpi, bbox_inches='tight')

                print(f"✅ Successfully saved {plot_name} plots")

            except Exception as e:
                print(f"❌ Error saving {plot_name} plot: {e}")

        # Note: EVO native export disabled to avoid duplicate plots
        # All plots are saved individually using matplotlib

        print(f"✅ All plots saved to {graph_dir}")
        return True

    def update_reference_trajectory_from_horizon_mapper(self, trajectory_data):
        """Update reference trajectory from horizon mapper VehicleStateArray data"""
        if not EVO_AVAILABLE:
            return False

        try:
            if not trajectory_data:
                return False

            # Extract data from horizon mapper format
            x = [point['x'] for point in trajectory_data]
            y = [point['y'] for point in trajectory_data]
            theta = [point['theta'] for point in trajectory_data]
            v = [point['v'] for point in trajectory_data]

            # Convert to EVO trajectory format
            # Create timestamps (assuming 10Hz sampling)
            timestamps = np.arange(len(x)) * 0.1

            # Convert to quaternions (assuming 2D motion)
            z = np.zeros_like(x)
            qx = np.zeros_like(x)
            qy = np.zeros_like(x)
            qz = np.sin(np.array(theta) / 2)
            qw = np.cos(np.array(theta) / 2)

            # Create EVO trajectory
            self.reference_trajectory = trajectory.PoseTrajectory3D(
                positions_xyz=np.column_stack([x, y, z]),
                orientations_quat_wxyz=np.column_stack([qw, qx, qy, qz]),
                timestamps=timestamps
            )

            print(f"Updated reference trajectory from horizon mapper with {len(x)} points")
            return True

        except Exception as e:
            print(f"Error updating reference trajectory from horizon mapper: {e}")
            return False
