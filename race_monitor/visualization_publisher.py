#!/usr/bin/env python3

"""
Visualization Publisher

Handles RViz visualization and marker publishing for the race monitor system.
Provides visual feedback including start/finish line markers, race line
visualization, and racing status indicators.

Features:
    - Start/finish line visualization
    - Racing line and trajectory markers
    - Real-time race status displays
    - Reference trajectory visualization
    - Configurable marker styles and colors

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

import numpy as np
from typing import List, Dict, Optional, Any
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import rclpy
from race_monitor.logger_utils import RaceMonitorLogger, LogLevel

# EVO imports
try:
    from evo.core import trajectory
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


class VisualizationPublisher:
    """
    Manages RViz visualization and marker publishing for race monitoring.

    Provides comprehensive visualization including race lines, start/finish
    markers, trajectory visualization, and status indicators.
    """

    def __init__(self, logger, publisher_callback):
        """
        Initialize the visualization publisher.

        Args:
            logger: ROS2 logger instance
            publisher_callback: Callback function to publish markers
        """
        self.logger = logger
        self.publish_marker = publisher_callback

        # Configuration
        self.frame_id = "map"
        self.start_line_p1 = [0.0, -1.0]
        self.start_line_p2 = [0.0, 1.0]

        # Marker IDs
        self.marker_id_counter = 0
        self.start_line_marker_id = 1000
        self.raceline_marker_id = 2000
        self.trajectory_marker_id = 3000

        # Colors
        self.colors = {
            'start_line': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),     # Red
            'raceline': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),       # Green
            'trajectory': ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.6),     # Blue
            'reference': ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7),      # Yellow
            'current_lap': ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)     # Orange
        }

    def configure(self, config: dict):
        """
        Configure visualization publisher parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.frame_id = config.get('frame_id', self.frame_id)
        self.start_line_p1 = config.get('start_line_p1', self.start_line_p1)
        self.start_line_p2 = config.get('start_line_p2', self.start_line_p2)

    def publish_start_line_marker(self):
        """Publish enhanced start/finish line marker for RViz visualization with endpoint markers."""
        try:
            # Check if line has changed to avoid unnecessary republishing
            current_line = (tuple(self.start_line_p1), tuple(self.start_line_p2))

            if not hasattr(self, '_last_published_line'):
                self._last_published_line = None

            # Only publish if line has changed or it's been a while (for new subscribers)
            if not hasattr(self, '_last_line_publish_time'):
                self._last_line_publish_time = 0

            import time
            current_time = time.time()
            should_republish = (current_time - self._last_line_publish_time > 5.0)  # Every 5 seconds

            if self._last_published_line == current_line and not should_republish:
                return

            self._last_published_line = current_line
            self._last_line_publish_time = current_time

            # Delete existing markers first
            clear_marker = Marker()
            clear_marker.header.frame_id = self.frame_id
            clear_marker.header.stamp = rclpy.time.Time().to_msg()
            clear_marker.ns = "race_monitor_start_line"
            clear_marker.action = Marker.DELETEALL
            self.publish_marker(clear_marker)

            # Create main line marker (sleek green line)
            line_marker = Marker()
            line_marker.header.frame_id = self.frame_id
            line_marker.header.stamp = rclpy.time.Time().to_msg()
            line_marker.ns = "race_monitor_start_line"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.15  # Reduced thickness (15cm) for cleaner look

            # Bright green color for start/finish line with reduced opacity
            line_marker.color.r = 0.0
            line_marker.color.g = 0.8
            line_marker.color.b = 0.2
            line_marker.color.a = 0.6  # More transparent for subtle appearance

            # Add two points (endpoints) to the line strip
            pt1 = Point()
            pt1.x = float(self.start_line_p1[0])
            pt1.y = float(self.start_line_p1[1])
            pt1.z = 0.02  # Lower to ground for more realistic appearance

            pt2 = Point()
            pt2.x = float(self.start_line_p2[0])
            pt2.y = float(self.start_line_p2[1])
            pt2.z = 0.02  # Lower to ground for more realistic appearance

            line_marker.points = [pt1, pt2]
            self.publish_marker(line_marker)

            # Add subtle endpoint markers for better visibility
            for i, point in enumerate([self.start_line_p1, self.start_line_p2]):
                endpoint_marker = Marker()
                endpoint_marker.header.frame_id = self.frame_id
                endpoint_marker.header.stamp = rclpy.time.Time().to_msg()
                endpoint_marker.ns = "race_monitor_start_line"
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

                self.publish_marker(endpoint_marker)

            # Calculate line length for logging
            import numpy as np
            line_length = np.linalg.norm(np.array(self.start_line_p2) - np.array(self.start_line_p1))
            self.logger.verbose(f"Published enhanced start line (length: {line_length:.2f}m) in frame '{self.frame_id}'")

        except Exception as e:
            self.logger.error(f"Error publishing start line marker", exception=e)

    def publish_start_line_markers(self, p1: List[float], p2: List[float], frame_id: str = None):
        """
        Publish start/finish line markers with specific points (for clicked point updates).

        Args:
            p1: First point coordinates [x, y]
            p2: Second point coordinates [x, y]
            frame_id: Frame ID for the markers (defaults to self.frame_id)
        """
        try:
            if frame_id is None:
                frame_id = self.frame_id

            # Update internal points
            self.start_line_p1 = p1
            self.start_line_p2 = p2

            # Force republish by clearing cache
            if hasattr(self, '_last_published_line'):
                self._last_published_line = None

            # Publish the enhanced start line
            self.publish_start_line_marker()

        except Exception as e:
            self.logger.error(f"Error publishing start line markers with specific points", exception=e)

    def publish_raceline_markers(self, reference_points: List[Dict[str, float]] = None,
                                 evo_trajectory=None, csv_data: List[List[str]] = None):
        """
        Publish raceline markers for visualization.

        Args:
            reference_points: List of reference trajectory points
            evo_trajectory: EVO trajectory object
            csv_data: Raw CSV data for raceline
        """
        try:
            if reference_points:
                self._publish_raceline_from_points(reference_points)
            elif evo_trajectory and EVO_AVAILABLE:
                self._publish_raceline_from_evo_trajectory(evo_trajectory)
            elif csv_data:
                self._publish_raceline_from_csv_data(csv_data)
            else:
                self.logger.verbose("No raceline data available for visualization")

        except Exception as e:
            self.logger.error(f"Error publishing raceline markers", exception=e)

    def _publish_raceline_from_points(self, reference_points: List[Dict[str, float]]):
        """Publish raceline from reference points with enhanced visualization."""
        if not reference_points:
            return

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.ns = "race_monitor"
        marker.id = self.raceline_marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line properties - make it green and thicker
        marker.scale.x = 0.1  # Thicker line
        marker.color = self.colors['raceline']  # Green color

        # Add points
        points = []
        for point_data in reference_points:
            point = Point()
            point.x = float(point_data['x'])
            point.y = float(point_data['y'])
            point.z = float(point_data.get('z', 0.0))
            marker.points.append(point)
            points.append([point.x, point.y, point.z])

        self.publish_marker(marker)
        self.logger.verbose(f"Published raceline with {len(reference_points)} points")

        # Add start and end markers (balls)
        if len(points) >= 2:
            self._publish_start_end_markers(points[0], points[-1])

    def _publish_raceline_from_csv_data(self, data: List[List[str]]):
        """Publish raceline from CSV data with enhanced visualization."""
        if not data or len(data) < 2:
            return

        # First publish the raceline
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.ns = "race_monitor"
        marker.id = self.raceline_marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line properties - make it green and thicker
        marker.scale.x = 0.1  # Thicker line
        marker.color = self.colors['raceline']  # Green color

        # Skip header row if it exists
        start_idx = 1 if not self._is_numeric(data[0][1]) else 0

        # Collect points
        points = []
        for row in data[start_idx:]:
            if len(row) >= 3:
                try:
                    point = Point()
                    point.x = float(row[1])  # Assume format: timestamp, x, y, [z]
                    point.y = float(row[2])
                    point.z = float(row[3]) if len(row) > 3 and self._is_numeric(row[3]) else 0.0
                    marker.points.append(point)
                    points.append([point.x, point.y, point.z])
                except (ValueError, IndexError):
                    continue

        if marker.points:
            self.publish_marker(marker)
            self.logger.verbose(f"Published raceline from CSV with {len(marker.points)} points")
            # Add start and end markers (balls)
            if len(points) >= 2:
                self._publish_start_end_markers(points[0], points[-1])

    def _publish_raceline_from_evo_trajectory(self, evo_trajectory):
        """Publish raceline from EVO trajectory object."""
        if not EVO_AVAILABLE or evo_trajectory is None:
            return

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.ns = "race_monitor"
        marker.id = self.raceline_marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line properties
        marker.scale.x = 0.05
        marker.color = self.colors['reference']

        # Extract positions from EVO trajectory
        positions = evo_trajectory.positions_xyz

        for pos in positions:
            point = Point()
            point.x = float(pos[0])
            point.y = float(pos[1])
            point.z = float(pos[2])
            marker.points.append(point)

        self.publish_marker(marker)
        self.logger.verbose(f"Published EVO raceline with {len(positions)} points")

    def publish_trajectory_markers(self, trajectory_data: List[Dict[str, float]],
                                   marker_type: str = "current_lap"):
        """
        Publish trajectory markers for a specific lap or trajectory.

        Args:
            trajectory_data: List of trajectory points
            marker_type: Type of marker ('current_lap', 'trajectory', 'reference')
        """
        if not trajectory_data:
            return

        try:
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rclpy.time.Time().to_msg()
            marker.ns = "race_monitor"
            marker.id = self.trajectory_marker_id + hash(marker_type) % 1000
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Line properties
            marker.scale.x = 0.03
            marker.color = self.colors.get(marker_type, self.colors['trajectory'])

            # Add trajectory points
            for point_data in trajectory_data:
                point = Point()
                point.x = float(point_data.get('x', 0.0))
                point.y = float(point_data.get('y', 0.0))
                point.z = float(point_data.get('z', 0.0))
                marker.points.append(point)

            self.publish_marker(marker)
            self.logger.verbose(f"Published {marker_type} trajectory with {len(trajectory_data)} points")

        except Exception as e:
            self.logger.error(f"Error publishing trajectory markers", exception=e)

    def publish_race_status_marker(self, race_status: str, current_lap: int,
                                   total_laps: int, position: List[float] = None):
        """
        Publish race status text marker.
        DISABLED: Text markers are disabled to clean up visualization.
        """
        # Disabled text markers as requested
        pass

    def clear_markers(self, marker_namespace: str = "race_monitor"):
        """
        Clear all markers in the specified namespace.

        Args:
            marker_namespace: Namespace of markers to clear
        """
        try:
            # Delete all markers by publishing a delete marker
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rclpy.time.Time().to_msg()
            marker.ns = marker_namespace
            marker.action = Marker.DELETEALL

            self.publish_marker(marker)
            self.logger.verbose(f"Cleared all markers in namespace: {marker_namespace}")

        except Exception as e:
            self.logger.error(f"Error clearing markers", exception=e)

    def publish_lap_completion_marker(self, lap_number: int, lap_time: float,
                                      position: List[float] = None):
        """
        Publish temporary marker showing lap completion.
        DISABLED: Text markers are disabled to clean up visualization.
        """
        # Disabled text markers as requested
        pass

    def _publish_start_end_markers(self, start_point: List[float], end_point: List[float]):
        """Publish start and end markers (balls) for the raceline."""
        try:
            # Start marker (green ball)
            start_marker = Marker()
            start_marker.header.frame_id = self.frame_id
            start_marker.header.stamp = rclpy.time.Time().to_msg()
            start_marker.ns = "race_monitor"
            start_marker.id = self.raceline_marker_id + 1
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD

            # Position
            start_marker.pose.position.x = float(start_point[0])
            start_marker.pose.position.y = float(start_point[1])
            start_marker.pose.position.z = float(start_point[2]) + 0.1
            start_marker.pose.orientation.w = 1.0

            # Size and color
            start_marker.scale.x = 0.3
            start_marker.scale.y = 0.3
            start_marker.scale.z = 0.3
            start_marker.color.r = 0.0
            start_marker.color.g = 1.0  # Green
            start_marker.color.b = 0.0
            start_marker.color.a = 1.0

            self.publish_marker(start_marker)

            # End marker (red ball)
            end_marker = Marker()
            end_marker.header.frame_id = self.frame_id
            end_marker.header.stamp = rclpy.time.Time().to_msg()
            end_marker.ns = "race_monitor"
            end_marker.id = self.raceline_marker_id + 2
            end_marker.type = Marker.SPHERE
            end_marker.action = Marker.ADD

            # Position
            end_marker.pose.position.x = float(end_point[0])
            end_marker.pose.position.y = float(end_point[1])
            end_marker.pose.position.z = float(end_point[2]) + 0.1
            end_marker.pose.orientation.w = 1.0

            # Size and color
            end_marker.scale.x = 0.3
            end_marker.scale.y = 0.3
            end_marker.scale.z = 0.3
            end_marker.color.r = 1.0  # Red
            end_marker.color.g = 0.0
            end_marker.color.b = 0.0
            end_marker.color.a = 1.0

            self.publish_marker(end_marker)

            self.logger.verbose("Published start and end markers for raceline")

        except Exception as e:
            self.logger.error(f"Error publishing start/end markers", exception=e)

    def _is_numeric(self, value: str) -> bool:
        """Check if a string value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def get_next_marker_id(self) -> int:
        """Get the next available marker ID."""
        self.marker_id_counter += 1
        return self.marker_id_counter
