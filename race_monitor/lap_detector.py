#!/usr/bin/env python3

"""
Lap Detection Module

Handles lap detection logic, timing, and race state management for the race monitor system.
Provides clean separation of lap detection functionality from other components.

Features:
    - Start/finish line crossing detection
    - Lap timing and counting
    - Race state management
    - Debounce logic to prevent false triggers

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

import numpy as np
import math
from datetime import datetime
from typing import Tuple, Optional, Callable
import rclpy
from rclpy.node import Node


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


class LapDetector:
    """
    Handles lap detection and timing logic for autonomous racing.

    This class provides lap crossing detection using a start/finish line
    defined by two points, with configurable debounce time to prevent
    false triggers.
    """

    def __init__(self, logger, clock):
        """
        Initialize the lap detector.

        Args:
            logger: ROS2 logger instance
            clock: ROS2 clock instance
        """
        self.logger = logger
        self.clock = clock

        # Lap detection configuration
        self.start_line_p1 = [8.0, -10.0]
        self.start_line_p2 = [8.5, -10.0]
        self.debounce_time = 2.0  # seconds
        self.required_laps = 20

        # Race ending configuration
        self.race_ending_mode = "lap_complete"  # "lap_complete", "crash", "manual"

        # Crash detection configuration
        self.crash_detection_enabled = True
        self.max_stationary_time = 5.0
        self.min_velocity_threshold = 0.1
        self.max_odometry_timeout = 3.0
        self.enable_collision_detection = True
        self.collision_velocity_threshold = 2.0
        self.collision_detection_window = 0.5

        # Manual mode configuration
        self.max_race_duration = 0  # 0 = no limit

        # Race state
        self.race_started = False
        self.race_completed = False
        self.race_ended_by_crash = False
        self.race_ended_manually = False
        self.current_lap = 0
        self.lap_times = []
        self.race_start_time = None
        self.last_lap_time = None
        self.total_race_time = 0.0

        # Lap detection state
        self.last_crossing_time = None
        self.previous_side = None
        self.current_position = None
        self.previous_position = None
        self.current_heading = 0.0  # Vehicle heading in radians

        # Lap detection configuration
        self.expected_direction = "any"  # "any", "positive", "negative"
        self.validate_heading_direction = False
        self.log_level = "normal"  # "minimal", "normal", "debug"

        # Crash detection state
        self.last_odometry_time = None
        self.last_velocity = 0.0
        self.stationary_start_time = None
        self.velocity_history = []  # For collision detection
        self.last_position_update = None

        # Callbacks for events
        self.on_race_start = None
        self.on_lap_complete = None
        self.on_race_complete = None
        self.on_race_crash = None

    def configure(self, config: dict):
        """
        Configure lap detector parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.start_line_p1 = config.get('start_line_p1', self.start_line_p1)
        self.start_line_p2 = config.get('start_line_p2', self.start_line_p2)
        self.debounce_time = config.get('debounce_time', self.debounce_time)
        self.required_laps = config.get('required_laps', self.required_laps)

        # Lap detection configuration
        lap_detection_config = config.get('lap_detection', {})
        self.expected_direction = lap_detection_config.get('expected_direction', self.expected_direction)
        self.validate_heading_direction = lap_detection_config.get(
            'validate_heading_direction', self.validate_heading_direction)
        self.log_level = lap_detection_config.get('log_level', self.log_level)

        # Race ending configuration
        self.race_ending_mode = config.get('race_ending_mode', self.race_ending_mode)

        # Crash detection configuration
        crash_config = config.get('crash_detection', {})
        self.crash_detection_enabled = crash_config.get('enable_crash_detection', self.crash_detection_enabled)
        self.max_stationary_time = crash_config.get('max_stationary_time', self.max_stationary_time)
        self.min_velocity_threshold = crash_config.get('min_velocity_threshold', self.min_velocity_threshold)
        self.max_odometry_timeout = crash_config.get('max_odometry_timeout', self.max_odometry_timeout)
        self.enable_collision_detection = crash_config.get(
            'enable_collision_detection', self.enable_collision_detection)
        self.collision_velocity_threshold = crash_config.get(
            'collision_velocity_threshold', self.collision_velocity_threshold)
        self.collision_detection_window = crash_config.get(
            'collision_detection_window', self.collision_detection_window)

        # Manual mode configuration
        manual_config = config.get('manual_mode', {})
        self.max_race_duration = manual_config.get('max_race_duration', self.max_race_duration)

        self.logger.info(f"Lap detector configuration:")
        self.logger.info(f"  Start line: P1={self.start_line_p1}, P2={self.start_line_p2}")
        self.logger.info(f"  Required laps: {self.required_laps}")
        self.logger.info(f"  Debounce time: {self.debounce_time}s")
        self.logger.info(f"  Race ending mode: {self.race_ending_mode}")
        self.logger.info(f"  Expected direction: {self.expected_direction}")
        self.logger.info(f"  Validate heading: {self.validate_heading_direction}")
        self.logger.info(f"  Log level: {self.log_level}")
        if self.race_ending_mode == "crash":
            self.logger.info(f"  Crash detection enabled: {self.crash_detection_enabled}")
            self.logger.info(f"  Max stationary time: {self.max_stationary_time}s")
        elif self.race_ending_mode == "manual":
            self.logger.info(
                f"  Max race duration: {self.max_race_duration}s ({'no limit' if self.max_race_duration == 0 else 'limit'})")

    def set_callbacks(self, on_race_start: Callable = None,
                      on_lap_complete: Callable = None,
                      on_race_complete: Callable = None,
                      on_race_crash: Callable = None):
        """
        Set callback functions for race events.

        Args:
            on_race_start: Called when race starts (lap 1 begins)
            on_lap_complete: Called when a lap is completed (lap_number, lap_time)
            on_race_complete: Called when race is finished (total_time, lap_times)
            on_race_crash: Called when race ends due to crash (crash_reason, total_time, lap_times)
        """
        self.on_race_start = on_race_start
        self.on_lap_complete = on_lap_complete
        self.on_race_complete = on_race_complete
        self.on_race_crash = on_race_crash

    def update_position(self, x: float, y: float, timestamp=None, velocity: float = None, heading: float = None):
        """
        Update current position and check for lap crossing and crash conditions.

        Args:
            x: X coordinate
            y: Y coordinate
            timestamp: Optional timestamp, uses current time if None
            velocity: Optional velocity for crash detection
            heading: Vehicle heading in radians for direction checking
        """
        # Store previous position before updating current
        self.previous_position = self.current_position
        self.current_position = [x, y]

        # Store heading for direction checking
        if heading is not None:
            self.current_heading = heading

        if timestamp is None:
            timestamp = self.clock.now()        # Update crash detection parameters
        self.last_odometry_time = timestamp
        if velocity is not None:
            self.last_velocity = velocity
            self._update_velocity_history(velocity, timestamp)

        # Update position tracking
        self.last_position_update = timestamp

        # Check for race ending conditions if race is active
        if self.is_race_active():
            # Check for crash conditions first (higher priority)
            if self.race_ending_mode in ["crash", "manual"] and self._check_crash_conditions(timestamp):
                return  # Race ended due to crash

            # Check for maximum race duration (safety feature for manual mode)
            if self.race_ending_mode == "manual" and self.max_race_duration > 0:
                race_duration = (time_to_nanoseconds(timestamp) - time_to_nanoseconds(self.race_start_time)) / 1e9
                if race_duration >= self.max_race_duration:
                    self._end_race_by_timeout(timestamp)
                    return

        # Check for lap crossing (normal race progression)
        if self._detect_lap_crossing(timestamp):
            self._handle_lap_crossing(timestamp)

    def _detect_lap_crossing(self, current_time) -> bool:
        """
        Detect if the vehicle has crossed the start/finish line in the correct direction.

        Args:
            current_time: Current timestamp

        Returns:
            bool: True if a valid lap crossing is detected
        """
        if self.current_position is None or self.previous_position is None:
            return False

        # Skip if we don't have a valid position history
        if np.allclose(self.current_position, self.previous_position, atol=1e-6):
            return False

        # Check if start line is degenerate
        if np.allclose(self.start_line_p1, self.start_line_p2, atol=1e-6):
            self.logger.warning(f"Start line is degenerate: P1={self.start_line_p1}, P2={self.start_line_p2}")
            return False

        # Check debounce time
        if (self.last_crossing_time is not None and (time_to_nanoseconds(current_time) - \
            time_to_nanoseconds(self.last_crossing_time)) / 1e9 < self.debounce_time):
            return False

        # Check if the movement crosses the start/finish line
        intersection = self._line_intersection(self.previous_position, self.current_position,
                                               self.start_line_p1, self.start_line_p2)

        if intersection:
            # Check if crossing in the correct direction using heading
            direction_ok = self._check_crossing_direction()
            if direction_ok:
                if self.log_level == "debug":
                    self.logger.debug("Valid line crossing detected")
                return True
            else:
                if self.log_level in ["normal", "debug"]:
                    self.logger.info("Invalid crossing direction - ignoring")

        return False

    def _line_intersection(self, p1, p2, q1, q2):
        """
        Check if line segment p1-p2 intersects with line segment q1-q2.

        Uses the CCW (counter-clockwise) method for robust intersection detection.

        Args:
            p1, p2: Movement line (previous position to current position)
            q1, q2: Start/finish line points

        Returns:
            bool: True if segments intersect
        """
        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A = np.array(p1, dtype=float)
        B = np.array(p2, dtype=float)
        C = np.array(q1, dtype=float)
        D = np.array(q2, dtype=float)

        result = (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

        if self.log_level == "debug":
            self.logger.debug(f"Line intersection: path {p1}->{p2} vs line {q1}->{q2} = {result}")

        return result

    def _check_crossing_direction(self):
        """
        Check if the vehicle is crossing the line in the expected direction.

        Returns:
            bool: True if crossing is valid based on configuration
        """
        # If no direction validation required, accept all crossings
        if not self.validate_heading_direction or self.expected_direction == "any":
            if self.log_level == "debug":
                self.logger.debug("Direction validation disabled - accepting crossing")
            return True

        # Need heading data for direction validation
        if self.current_heading is None:
            if self.log_level in ["normal", "debug"]:
                self.logger.warning("No heading data available for direction validation")
            return self.expected_direction == "any"

        # Calculate line vector (from p1 to p2)
        line_vector = np.array([
            self.start_line_p2[0] - self.start_line_p1[0],
            self.start_line_p2[1] - self.start_line_p1[1]
        ], dtype=float)

        # Check for degenerate line
        if np.linalg.norm(line_vector) == 0.0:
            if self.log_level in ["normal", "debug"]:
                self.logger.warning("Degenerate start line detected")
            return False

        # Calculate line normal (perpendicular to line)
        line_normal = np.array([-line_vector[1], line_vector[0]], dtype=float)
        line_normal = line_normal / np.linalg.norm(line_normal)

        # Vehicle direction vector from heading
        vehicle_direction = np.array([
            np.cos(self.current_heading),
            np.sin(self.current_heading)
        ], dtype=float)

        # Calculate cross product to determine direction
        cross_product = np.dot(vehicle_direction, line_normal)

        # Determine if crossing is in expected direction
        crossing_positive = cross_product > 0

        if self.log_level == "debug":
            self.logger.debug(f"Direction analysis: heading={self.current_heading:.3f}rad, "
                              f"cross_product={cross_product:.3f}, positive={crossing_positive}")

        if self.expected_direction == "positive":
            return crossing_positive
        elif self.expected_direction == "negative":
            return not crossing_positive
        else:  # "any"
            return True

    def quaternion_to_yaw(self, q):
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

    def _get_side_of_line(self, position: list) -> int:
        """
        Determine which side of the start/finish line the position is on.

        Args:
            position: [x, y] coordinates

        Returns:
            int: 1 or -1 indicating which side of the line
        """
        x, y = position
        x1, y1 = self.start_line_p1
        x2, y2 = self.start_line_p2

        # Calculate cross product to determine side
        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        return 1 if cross_product > 0 else -1

    def _handle_lap_crossing(self, timestamp):
        """
        Handle a detected lap crossing event.

        Args:
            timestamp: Time of the crossing
        """
        self.last_crossing_time = timestamp

        if not self.race_started:
            # First crossing - start the race
            self._start_race(timestamp)
        else:
            # Subsequent crossings - complete laps
            self._complete_lap(timestamp)

    def _start_race(self, timestamp):
        """
        Start a new race.

        Args:
            timestamp: Race start time
        """
        self.race_started = True
        self.race_completed = False
        self.current_lap = 1
        self.race_start_time = timestamp
        self.lap_times = []
        self.last_lap_time = timestamp

        if self.log_level == "minimal":
            self.logger.info(f"Race started - Lap {self.current_lap}")
        else:
            self.logger.info(f"Race started! Beginning lap {self.current_lap}")

        if self.on_race_start:
            self.on_race_start(timestamp)

    def _complete_lap(self, timestamp):
        """
        Complete the current lap and start the next one.

        Args:
            timestamp: Lap completion time
        """
        # Calculate lap time
        lap_time = (time_to_nanoseconds(timestamp) - time_to_nanoseconds(self.last_lap_time)) / 1e9
        self.lap_times.append(lap_time)

        if self.log_level == "minimal":
            self.logger.info(f"Lap {self.current_lap} completed: {lap_time:.3f}s")
        else:
            self.logger.info(f"Lap {self.current_lap} completed in {lap_time:.3f}s")

        # Trigger lap completion callback
        if self.on_lap_complete:
            self.on_lap_complete(self.current_lap, lap_time)

        # Check if race is complete based on ending mode
        if self.race_ending_mode == "lap_complete" and self.current_lap >= self.required_laps:
            self._complete_race(timestamp)
        elif self.race_ending_mode == "lap_complete":
            # Start next lap
            self.current_lap += 1
            self.last_lap_time = timestamp
            if self.log_level == "minimal":
                self.logger.info(f"Starting lap {self.current_lap}")
            else:
                self.logger.info(f"Starting lap {self.current_lap}")
        else:
            # For crash and manual modes, continue laps indefinitely
            self.current_lap += 1
            self.last_lap_time = timestamp
            if self.log_level == "minimal":
                self.logger.info(f"Starting lap {self.current_lap}")
            else:
                self.logger.info(f"Starting lap {self.current_lap} (mode: {self.race_ending_mode})")

    def _complete_race(self, timestamp):
        """
        Complete the race.

        Args:
            timestamp: Race completion time
        """
        self.race_completed = True
        self.total_race_time = (time_to_nanoseconds(timestamp) - time_to_nanoseconds(self.race_start_time)) / 1e9

        # Calculate race statistics
        avg_lap_time = np.mean(self.lap_times)
        best_lap_time = min(self.lap_times)
        worst_lap_time = max(self.lap_times)

        if self.log_level == "minimal":
            self.logger.info(
                f"Race completed - Time: {self.total_race_time:.3f}s, Laps: {len(self.lap_times)}, Best: {best_lap_time:.3f}s")
        else:
            self.logger.info(f"Race completed!")
            self.logger.info(f"  Total time: {self.total_race_time:.3f}s")
            self.logger.info(f"  Total laps: {len(self.lap_times)}")
            self.logger.info(f"  Average lap: {avg_lap_time:.3f}s")
            self.logger.info(f"  Best lap: {best_lap_time:.3f}s")
            self.logger.info(f"  Worst lap: {worst_lap_time:.3f}s")

        # Trigger race completion callback
        if self.on_race_complete:
            self.on_race_complete(self.total_race_time, self.lap_times)

    def reset_race(self):
        """Reset race state for a new race."""
        self.race_started = False
        self.race_completed = False
        self.race_ended_by_crash = False
        self.race_ended_manually = False
        self.current_lap = 0
        self.lap_times = []
        self.race_start_time = None
        self.last_lap_time = None
        self.total_race_time = 0.0
        self.last_crossing_time = None
        self.previous_side = None
        self.current_heading = 0.0

        # Reset crash detection state
        self.last_odometry_time = None
        self.last_velocity = 0.0
        self.stationary_start_time = None
        self.velocity_history = []
        self.last_position_update = None

        self.logger.info("Race state reset")

    def get_race_stats(self) -> dict:
        """
        Get current race statistics.

        Returns:
            dict: Race statistics including times, lap count, etc.
        """
        return {
            'race_started': self.race_started,
            'race_completed': self.race_completed,
            'race_ended_by_crash': self.race_ended_by_crash,
            'race_ended_manually': self.race_ended_manually,
            'race_ending_mode': self.race_ending_mode,
            'race_ending_reason': self.get_race_ending_reason(),
            'current_lap': self.current_lap,
            'required_laps': self.required_laps,
            'lap_times': self.lap_times.copy(),
            'total_race_time': self.total_race_time,
            'average_lap_time': np.mean(self.lap_times) if self.lap_times else 0.0,
            'best_lap_time': min(self.lap_times) if self.lap_times else 0.0,
            'worst_lap_time': max(self.lap_times) if self.lap_times else 0.0,
            'laps_completed': len(self.lap_times),
            'current_velocity': self.last_velocity,
            'crash_detection_enabled': self.crash_detection_enabled
        }

    def is_race_active(self) -> bool:
        """Check if race is currently active."""
        return self.race_started and not self.race_completed

    def get_progress_percentage(self) -> float:
        """Get race progress as percentage."""
        if not self.race_started:
            return 0.0
        if self.race_ending_mode == "lap_complete":
            return min(100.0, (len(self.lap_times) / self.required_laps) * 100.0)
        else:
            # For crash and manual modes, show lap count instead
            return float(len(self.lap_times))

    def _update_velocity_history(self, velocity: float, timestamp):
        """
        Update velocity history for collision detection.

        Args:
            velocity: Current velocity
            timestamp: Current timestamp
        """
        current_time_ns = time_to_nanoseconds(timestamp)

        # Add current velocity to history
        self.velocity_history.append((current_time_ns, velocity))

        # Remove old entries outside the detection window
        cutoff_time = current_time_ns - (self.collision_detection_window * 1e9)
        self.velocity_history = [(t, v) for t, v in self.velocity_history if t >= cutoff_time]

    def _check_crash_conditions(self, timestamp) -> bool:
        """
        Check for crash conditions based on configured detection methods.

        Args:
            timestamp: Current timestamp

        Returns:
            bool: True if crash is detected
        """
        if not self.crash_detection_enabled:
            return False

        current_time_ns = time_to_nanoseconds(timestamp)

        # Check for odometry timeout
        if self._check_odometry_timeout(current_time_ns):
            self._end_race_by_crash("Odometry timeout - vehicle may have crashed", timestamp)
            return True

        # Check for stationary vehicle
        if self._check_stationary_condition(current_time_ns):
            self._end_race_by_crash("Vehicle stationary for too long - possible crash", timestamp)
            return True

        # Check for collision (sudden velocity change)
        if self.enable_collision_detection and self._check_collision_condition():
            self._end_race_by_crash("Sudden velocity change detected - possible collision", timestamp)
            return True

        return False

    def _check_odometry_timeout(self, current_time_ns: int) -> bool:
        """Check if odometry updates have timed out."""
        if self.last_odometry_time is None:
            return False

        time_since_last_update = (current_time_ns - time_to_nanoseconds(self.last_odometry_time)) / 1e9
        return time_since_last_update > self.max_odometry_timeout

    def _check_stationary_condition(self, current_time_ns: int) -> bool:
        """Check if vehicle has been stationary for too long."""
        # Check if vehicle is currently stationary
        if self.last_velocity > self.min_velocity_threshold:
            # Vehicle is moving, reset stationary timer
            self.stationary_start_time = None
            return False

        # Vehicle is stationary
        if self.stationary_start_time is None:
            # Start tracking stationary time
            self.stationary_start_time = current_time_ns
            return False

        # Check if stationary for too long
        stationary_duration = (current_time_ns - self.stationary_start_time) / 1e9
        return stationary_duration > self.max_stationary_time

    def _check_collision_condition(self) -> bool:
        """Check for collision based on sudden velocity changes."""
        if len(self.velocity_history) < 2:
            return False

        # Find the maximum velocity change in the detection window
        max_velocity_change = 0.0
        for i in range(1, len(self.velocity_history)):
            velocity_change = abs(self.velocity_history[i][1] - self.velocity_history[i - 1][1])
            max_velocity_change = max(max_velocity_change, velocity_change)

        return max_velocity_change > self.collision_velocity_threshold

    def _end_race_by_crash(self, crash_reason: str, timestamp):
        """
        End race due to crash detection.

        Args:
            crash_reason: Reason for crash detection
            timestamp: Time of crash detection
        """
        self.race_completed = True
        self.race_ended_by_crash = True
        self.total_race_time = (time_to_nanoseconds(timestamp) - time_to_nanoseconds(self.race_start_time)) / 1e9

        if self.log_level == "minimal":
            self.logger.warning(f"Race ended - crash detected: {crash_reason} (Duration: {self.total_race_time:.3f}s)")
        else:
            self.logger.warning(f"Race ended due to crash: {crash_reason}")
            self.logger.info(f"  Race duration: {self.total_race_time:.3f}s")
            self.logger.info(f"  Laps completed: {len(self.lap_times)}")
            if self.lap_times:
                self.logger.info(f"  Last lap time: {self.lap_times[-1]:.3f}s")

        # Trigger crash callback
        if self.on_race_crash:
            self.on_race_crash(crash_reason, self.total_race_time, self.lap_times)

    def _end_race_by_timeout(self, timestamp):
        """
        End race due to maximum duration timeout (safety feature).

        Args:
            timestamp: Time of timeout
        """
        self.race_completed = True
        self.race_ended_manually = True
        self.total_race_time = (time_to_nanoseconds(timestamp) - time_to_nanoseconds(self.race_start_time)) / 1e9

        if self.log_level == "minimal":
            self.logger.info(f"Race ended - timeout limit ({self.max_race_duration}s) reached")
        else:
            self.logger.info(f"Race ended due to maximum duration limit ({self.max_race_duration}s)")
            self.logger.info(f"  Total race time: {self.total_race_time:.3f}s")
            self.logger.info(f"  Laps completed: {len(self.lap_times)}")

        # Trigger race completion callback
        if self.on_race_complete:
            self.on_race_complete(self.total_race_time, self.lap_times)

    def get_race_ending_reason(self) -> str:
        """Get the reason why the race ended."""
        if not self.race_completed:
            return "Race still active"
        elif self.race_ended_by_crash:
            return "Crash detected"
        elif self.race_ended_manually:
            return "Manual timeout"
        elif len(self.lap_times) >= self.required_laps:
            return "Laps completed"
        else:
            return "Unknown"
