#!/usr/bin/env python3

"""
Reference Trajectory Manager

Handles loading, managing, and providing access to reference trajectories
for trajectory evaluation and racing line visualization.

Features:
    - Multiple reference trajectory formats (CSV, TUM, KITTI)
    - Real-time reference trajectory from topics
    - Horizon mapper integration
    - Reference path management
    - Trajectory format conversion

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

import os
import csv
import numpy as np
from typing import Optional, List, Dict, Any
import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# EVO imports
try:
    from evo.core import trajectory
    from evo.tools import file_interface
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


class ReferenceTrajectoryManager:
    """
    Manages reference trajectories for racing analysis and evaluation.

    Supports multiple sources:
    - Static file-based trajectories (CSV, TUM, KITTI formats)
    - Real-time reference trajectories from ROS topics
    - Horizon mapper integration for dynamic reference paths
    """

    def __init__(self, logger):
        """
        Initialize the reference trajectory manager.

        Args:
            logger: ROS2 logger instance
        """
        self.logger = logger

        # Configuration
        self.reference_trajectory_file = ""
        self.reference_trajectory_format = "csv"
        self.enable_horizon_mapper_reference = False
        self.horizon_mapper_reference_topic = '/horizon_mapper/reference_trajectory'
        self.use_complete_reference_path = True
        self.horizon_mapper_path_topic = '/horizon_mapper/reference_path'

        # Reference data storage
        self.reference_trajectory = None
        self.reference_path_data = []
        self.horizon_reference_data = []

        # State
        self.reference_loaded = False
        self.reference_updated = False

    def configure(self, config: dict):
        """
        Configure reference trajectory manager parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        # File-based reference trajectory configuration
        self.reference_trajectory_file = config.get('reference_trajectory_file', "")
        self.reference_trajectory_format = config.get('reference_trajectory_format', "csv")

        # Horizon mapper interface configuration
        self.enable_horizon_mapper_reference = config.get('enable_horizon_mapper_reference', False)
        self.horizon_mapper_reference_topic = config.get(
            'horizon_mapper_reference_topic',
            '/horizon_mapper/reference_trajectory')
        self.use_complete_reference_path = config.get('use_complete_reference_path', True)
        self.horizon_mapper_path_topic = config.get('horizon_mapper_path_topic', '/horizon_mapper/reference_path')

        self.logger.info(f"Reference trajectory manager configured:")
        self.logger.info(f"  File: {self.reference_trajectory_file}")
        self.logger.info(f"  Format: {self.reference_trajectory_format}")
        self.logger.info(f"  Horizon mapper enabled: {self.enable_horizon_mapper_reference}")
        if self.enable_horizon_mapper_reference:
            self.logger.info(f"  Reference topic: {self.horizon_mapper_reference_topic}")
            self.logger.info(f"  Path topic: {self.horizon_mapper_path_topic}")
            self.logger.info(f"  Use complete path: {self.use_complete_reference_path}")

        # Try to load reference trajectory from file if specified
        if self.reference_trajectory_file:
            self.load_reference_trajectory()

    def load_reference_trajectory(self) -> bool:
        """
        Load reference trajectory from file.

        Returns:
            bool: True if successfully loaded
        """
        if not self.reference_trajectory_file:
            self.logger.warn("No reference trajectory file specified")
            return False

        if not os.path.exists(self.reference_trajectory_file):
            self.logger.warn(f"Reference trajectory file not found: {self.reference_trajectory_file}")
            return False

        try:
            if self.reference_trajectory_format.lower() == "csv":
                success = self._load_csv_reference()
            elif self.reference_trajectory_format.lower() == "tum":
                success = self._load_tum_reference()
            elif self.reference_trajectory_format.lower() == "kitti":
                success = self._load_kitti_reference()
            else:
                self.logger.error(f"Unsupported reference trajectory format: {self.reference_trajectory_format}")
                return False

            if success:
                self.reference_loaded = True
                self.logger.info(f"Successfully loaded reference trajectory from {self.reference_trajectory_file}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error loading reference trajectory: {e}")
            return False

    def _load_csv_reference(self) -> bool:
        """Load reference trajectory from CSV file."""
        try:
            with open(self.reference_trajectory_file, 'r') as file:
                reader = csv.reader(file)
                header = next(reader, None)

                timestamps = []
                positions = []
                has_orientation = False

                for row in reader:
                    if len(row) >= 3:
                        # Assume format: timestamp, x, y, [z], [qx, qy, qz, qw]
                        t = float(row[0])
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3]) if len(row) > 3 else 0.0

                        timestamps.append(t)
                        positions.append([x, y, z])

                        # Check if this row has orientation data
                        if len(row) >= 7:
                            has_orientation = True

                if EVO_AVAILABLE and len(positions) > 0:
                    # Create EVO trajectory
                    timestamps = np.array(timestamps)
                    positions = np.array(positions)

                    if has_orientation:  # Has orientation data
                        orientations = []
                        with open(self.reference_trajectory_file, 'r') as file:
                            reader = csv.reader(file)
                            next(reader)  # Skip header
                            for row in reader:
                                if len(row) >= 7:
                                    qx, qy, qz, qw = map(float, row[3:7])
                                    orientations.append([qw, qx, qy, qz])  # EVO expects w,x,y,z format

                        if len(orientations) > 0:
                            orientations = np.array(orientations)
                            self.reference_trajectory = trajectory.PoseTrajectory3D(
                                positions, orientations, timestamps
                            )
                        else:
                            # If orientation loading failed, create path without orientation
                            self.reference_trajectory = trajectory.PosePath3D(positions, timestamps)
                    else:
                        self.reference_trajectory = trajectory.PosePath3D(positions, timestamps)

                self.logger.info(f"Loaded {len(positions)} reference points from CSV")
                return True

        except Exception as e:
            self.logger.error(f"Error loading CSV reference: {e}")
            return False

    def _load_tum_reference(self) -> bool:
        """Load reference trajectory from TUM format file."""
        if not EVO_AVAILABLE:
            self.logger.error("EVO library not available for TUM format loading")
            return False

        try:
            self.reference_trajectory = file_interface.read_tum_trajectory_file(
                self.reference_trajectory_file
            )
            self.logger.info(f"Loaded TUM reference trajectory with {len(self.reference_trajectory.poses_se3)} poses")
            return True
        except Exception as e:
            self.logger.error(f"Error loading TUM reference: {e}")
            return False

    def _load_kitti_reference(self) -> bool:
        """Load reference trajectory from KITTI format file."""
        if not EVO_AVAILABLE:
            self.logger.error("EVO library not available for KITTI format loading")
            return False

        try:
            self.reference_trajectory = file_interface.read_kitti_poses_file(
                self.reference_trajectory_file
            )
            self.logger.info(f"Loaded KITTI reference trajectory with {len(self.reference_trajectory.poses_se3)} poses")
            return True
        except Exception as e:
            self.logger.error(f"Error loading KITTI reference: {e}")
            return False

    def update_reference_trajectory(self, msg):
        """
        Update reference trajectory from ROS topic.

        Args:
            msg: PoseArray or similar message containing reference trajectory
        """
        try:
            # Store the reference trajectory data
            self.horizon_reference_data = []

            for pose in msg.poses:
                point_data = {
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z,
                    'qx': pose.orientation.x,
                    'qy': pose.orientation.y,
                    'qz': pose.orientation.z,
                    'qw': pose.orientation.w
                }
                self.horizon_reference_data.append(point_data)

            self.reference_updated = True
            self.logger.debug(f"Updated reference trajectory with {len(self.horizon_reference_data)} points")

        except Exception as e:
            self.logger.error(f"Error updating reference trajectory: {e}")

    def update_reference_path(self, msg: Path):
        """
        Update reference path from ROS Path message.

        Args:
            msg: nav_msgs/Path message
        """
        try:
            self.reference_path_data = []

            for pose_stamped in msg.poses:
                pose = pose_stamped.pose
                point_data = {
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z,
                    'qx': pose.orientation.x,
                    'qy': pose.orientation.y,
                    'qz': pose.orientation.z,
                    'qw': pose.orientation.w
                }
                self.reference_path_data.append(point_data)

            self.reference_updated = True
            self.logger.debug(f"Updated reference path with {len(self.reference_path_data)} points")

        except Exception as e:
            self.logger.error(f"Error updating reference path: {e}")

    def get_reference_trajectory(self):
        """
        Get the current reference trajectory.

        Returns:
            EVO trajectory object or None if not available
        """
        return self.reference_trajectory

    def get_reference_points(self) -> List[Dict[str, float]]:
        """
        Get reference trajectory points as a list of dictionaries.

        Returns:
            List of reference points with x, y, z coordinates
        """
        if self.use_complete_reference_path and self.reference_path_data:
            return self.reference_path_data
        elif self.horizon_reference_data:
            return self.horizon_reference_data
        elif self.reference_trajectory is not None:
            # Convert EVO trajectory to point list
            points = []
            positions = self.reference_trajectory.positions_xyz

            # Ensure positions is properly shaped
            if len(positions.shape) == 1:
                # If positions is 1D, it might be a single point
                if len(positions) >= 3:
                    points.append({'x': positions[0], 'y': positions[1], 'z': positions[2]})
                return points

            if hasattr(self.reference_trajectory, 'orientations_quat_wxyz'):
                orientations = self.reference_trajectory.orientations_quat_wxyz
                for i, pos in enumerate(positions):
                    if len(pos) >= 3 and i < len(orientations):
                        quat = orientations[i]
                        # Ensure quat is iterable and has at least 4 elements
                        if hasattr(quat, '__len__') and len(quat) >= 4:
                            points.append({
                                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                                'qw': float(quat[0]), 'qx': float(quat[1]), 'qy': float(quat[2]), 'qz': float(quat[3])
                            })
                        else:
                            # If quaternion is not properly formatted, skip orientation
                            points.append({'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})
                    elif len(pos) >= 3:
                        # No orientation available for this position
                        points.append({'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})
            else:
                for pos in positions:
                    if len(pos) >= 3:
                        points.append({'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})
            return points
        else:
            return []

    def save_horizon_reference_trajectory(self, output_dir: str):
        """
        Save horizon reference trajectory data to file.

        Args:
            output_dir: Directory to save the file
        """
        if not self.horizon_reference_data:
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "horizon_reference_trajectory.txt")

            with open(filepath, 'w') as f:
                f.write("# Horizon Reference Trajectory Data\n")
                f.write("# Format: x y z qx qy qz qw\n")

                for point in self.horizon_reference_data:
                    f.write(f"{point['x']:.6f} {point['y']:.6f} {point['z']:.6f} "
                            f"{point['qx']:.6f} {point['qy']:.6f} {point['qz']:.6f} {point['qw']:.6f}\n")

            self.logger.info(f"Saved horizon reference trajectory to: {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving horizon reference trajectory: {e}")

    def is_reference_available(self) -> bool:
        """Check if any reference trajectory is available."""
        return (self.reference_loaded or
                bool(self.horizon_reference_data) or
                bool(self.reference_path_data))

    def get_reference_source(self) -> str:
        """Get the current reference trajectory source."""
        if self.use_complete_reference_path and self.reference_path_data:
            return "reference_path"
        elif self.horizon_reference_data:
            return "horizon_reference"
        elif self.reference_loaded:
            return "file"
        else:
            return "none"

    def get_reference_stats(self) -> Dict[str, Any]:
        """Get statistics about the current reference trajectory."""
        points = self.get_reference_points()

        if not points:
            return {"available": False}

        # Calculate basic statistics
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]

        return {
            "available": True,
            "source": self.get_reference_source(),
            "point_count": len(points),
            "x_range": [min(x_coords), max(x_coords)],
            "y_range": [min(y_coords), max(y_coords)],
            "has_orientation": 'qx' in points[0] if points else False
        }
