#!/usr/bin/env python3

"""
EVO Plotter Module for Race Monitor
Integrates EVO's plotting capabilities for automatic graph generation
"""

import json
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# Please ensure the 'evo' library is installed and available in your PYTHONPATH.
# Add EVO library to Python path
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

            print(f"Loaded reference trajectory with {len(self.reference_trajectory.poses)} poses")
            return True

        except Exception as e:
            print(f"Error loading reference trajectory: {e}")
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
                    # F1Tenth format
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
                    # F1Tenth format - convert theta to quaternion
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
                    # F1Tenth format - create synthetic timestamps
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
            return

        print(f"✅ Starting plot generation...")

        try:
            # Initialize plot collection
            print(f"Creating plot collection...")
            self.plot_collection = plot.PlotCollection("Race Monitor - EVO Analysis")

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

            # Save all plots
            print(f"Saving all plots...")
            self._save_all_plots()

            print("✅ Generated all EVO plots successfully!")

        except Exception as e:
            print(f"Error generating plots: {e}")

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
            # Add start/end markers
            ax_traj.scatter(ref_positions[0, 0], ref_positions[0, 1],
                            color='darkgreen', s=100, marker='o', label='Ref Start', zorder=5)
            ax_traj.scatter(ref_positions[-1, 0], ref_positions[-1, 1],
                            color='darkred', s=100, marker='s', label='Ref End', zorder=5)

        # Plot all lap trajectories
        colors = cm.get_cmap(self.config.get('plot_color_scheme', 'viridis'))
        for i, (lap_num, traj) in enumerate(self.lap_trajectories.items()):
            color = colors(i / len(self.lap_trajectories))
            positions = traj.positions_xyz
            ax_traj.plot(positions[:, 0], positions[:, 1],
                         '-', color=color, label=f'Lap {lap_num}', alpha=0.8, linewidth=2)
            # Add start/end markers
            ax_traj.scatter(positions[0, 0], positions[0, 1],
                            color=color, s=80, marker='o', alpha=0.9, zorder=4)
            ax_traj.scatter(positions[-1, 0], positions[-1, 1],
                            color=color, s=80, marker='s', alpha=0.9, zorder=4)

        ax_traj.set_title('Trajectory Comparison')
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.legend()
        ax_traj.grid(True)
        ax_traj.axis('equal')

        self.plot_collection.add_figure("trajectories", fig_traj)

    def _generate_xyz_plots(self):
        """Generate X, Y, Z position plots"""
        if not self.lap_trajectories:
            return

        fig_xyz, axarr_xyz = plt.subplots(3, 1, sharex=True,
                                          figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))

        # Plot reference trajectory if available
        if self.reference_trajectory:
            ref_positions = self.reference_trajectory.positions_xyz
            ref_timestamps = self.reference_trajectory.timestamps
            axarr_xyz[0].plot(ref_timestamps, ref_positions[:, 0],
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)
            axarr_xyz[1].plot(ref_timestamps, ref_positions[:, 1],
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)
            axarr_xyz[2].plot(ref_timestamps, ref_positions[:, 2],
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)

        # Plot all lap trajectories
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
        """Generate Roll, Pitch, Yaw plots"""
        if not self.lap_trajectories:
            return

        fig_rpy, axarr_rpy = plt.subplots(3, 1, sharex=True,
                                          figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))

        # Plot reference trajectory if available
        if self.reference_trajectory:
            ref_orientations = self.reference_trajectory.orientations_quat_wxyz
            ref_timestamps = self.reference_trajectory.timestamps
            # Convert quaternions to Euler angles for plotting
            ref_rpy = self._quat_to_euler(ref_orientations)
            axarr_rpy[0].plot(ref_timestamps, ref_rpy[:, 0],
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)
            axarr_rpy[1].plot(ref_timestamps, ref_rpy[:, 1],
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)
            axarr_rpy[2].plot(ref_timestamps, ref_rpy[:, 2],
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)

        # Plot all lap trajectories
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
        """Generate speed analysis plots"""
        if not self.lap_trajectories:
            return

        fig_speed = plt.figure(figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))
        ax_speed = fig_speed.gca()

        # Plot reference trajectory if available
        if self.reference_trajectory:
            try:
                ref_speeds = self._calculate_speeds(self.reference_trajectory)
                ref_timestamps = self.reference_trajectory.timestamps[1:]  # One less due to speed calculation
                ax_speed.plot(ref_timestamps, ref_speeds,
                              '--', color='black', label='Reference', alpha=0.7, linewidth=2)
            except Exception as e:
                print(f"Could not plot reference speeds: {e}")

        # Plot all lap trajectories
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
        fig_error, (ax_ape, ax_rpe) = plt.subplots(2, 1, figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))

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
            2, 2, figsize=tuple(self.config.get('plot_figsize', [12.0, 8.0])))

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for fmt in graph_formats:
            try:
                if fmt == 'png':
                    output_path = os.path.join(graph_dir, f"race_analysis_{timestamp}.png")
                    print(f"Saving PNG to {output_path}")
                    self.plot_collection.export(output_path, confirm_overwrite=True)
                elif fmt == 'pdf':
                    output_path = os.path.join(graph_dir, f"race_analysis_{timestamp}.pdf")
                    print(f"Saving PDF to {output_path}")
                    self.plot_collection.export(output_path, confirm_overwrite=True)
                elif fmt == 'svg':
                    output_path = os.path.join(graph_dir, f"race_analysis_{timestamp}.svg")
                    print(f"Saving SVG to {output_path}")
                    self.plot_collection.export(output_path, confirm_overwrite=True)
                elif fmt == 'html':
                    # Save as serialized plot for later viewing
                    output_path = os.path.join(graph_dir, f"race_analysis_{timestamp}.evo")
                    print(f"Saving EVO to {output_path}")
                    self.plot_collection.serialize(output_path, confirm_overwrite=True)
            except Exception as e:
                print(f"Error saving {fmt} format: {e}")

        print(f"Saved plots to {graph_dir}")

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
