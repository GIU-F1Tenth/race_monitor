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
        self.start_line_p1 = [0.0, -1.0]  # Default start line points
        self.start_line_p2 = [0.0, 1.0]
        self.debounce_time = 2.0  # seconds
        self.required_laps = 5

        # Race state
        self.race_started = False
        self.race_completed = False
        self.current_lap = 0
        self.lap_times = []
        self.race_start_time = None
        self.last_lap_time = None
        self.total_race_time = 0.0

        # Lap detection state
        self.last_crossing_time = None
        self.previous_side = None
        self.current_position = None

        # Callbacks for events
        self.on_race_start = None
        self.on_lap_complete = None
        self.on_race_complete = None

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

        self.logger.info(f"Lap detector configured: start_line=({self.start_line_p1}, {self.start_line_p2}), "
                         f"required_laps={self.required_laps}, debounce={self.debounce_time}s")

    def set_callbacks(self, on_race_start: Callable = None,
                      on_lap_complete: Callable = None,
                      on_race_complete: Callable = None):
        """
        Set callback functions for race events.

        Args:
            on_race_start: Called when race starts (lap 1 begins)
            on_lap_complete: Called when a lap is completed (lap_number, lap_time)
            on_race_complete: Called when race is finished (total_time, lap_times)
        """
        self.on_race_start = on_race_start
        self.on_lap_complete = on_lap_complete
        self.on_race_complete = on_race_complete

    def update_position(self, x: float, y: float, timestamp=None):
        """
        Update current position and check for lap crossing.

        Args:
            x: X coordinate
            y: Y coordinate
            timestamp: Optional timestamp, uses current time if None
        """
        self.current_position = [x, y]

        if timestamp is None:
            timestamp = self.clock.now()

        # Check for lap crossing
        if self._detect_lap_crossing(timestamp):
            self._handle_lap_crossing(timestamp)

    def _detect_lap_crossing(self, current_time) -> bool:
        """
        Detect if the vehicle has crossed the start/finish line.

        Args:
            current_time: Current timestamp

        Returns:
            bool: True if a lap crossing is detected
        """
        if self.current_position is None:
            return False

        # Check debounce time
        if (self.last_crossing_time is not None and (time_to_nanoseconds(current_time) - \
            time_to_nanoseconds(self.last_crossing_time)) / 1e9 < self.debounce_time):
            return False

        current_side = self._get_side_of_line(self.current_position)

        # Initialize if this is the first position
        if self.previous_side is None:
            self.previous_side = current_side
            return False

        # Check for side change (crossing)
        if current_side != self.previous_side:
            self.previous_side = current_side
            return True

        return False

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

        self.logger.info(f"🏁 Race started! Beginning lap {self.current_lap}")

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

        self.logger.info(f"🏎️  Lap {self.current_lap} completed in {lap_time:.3f}s")

        # Trigger lap completion callback
        if self.on_lap_complete:
            self.on_lap_complete(self.current_lap, lap_time)

        # Check if race is complete
        if self.current_lap >= self.required_laps:
            self._complete_race(timestamp)
        else:
            # Start next lap
            self.current_lap += 1
            self.last_lap_time = timestamp
            self.logger.info(f"🚀 Starting lap {self.current_lap}")

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

        self.logger.info(f"🏆 Race completed!")
        self.logger.info(f"   Total time: {self.total_race_time:.3f}s")
        self.logger.info(f"   Total laps: {len(self.lap_times)}")
        self.logger.info(f"   Average lap: {avg_lap_time:.3f}s")
        self.logger.info(f"   Best lap: {best_lap_time:.3f}s")
        self.logger.info(f"   Worst lap: {worst_lap_time:.3f}s")

        # Trigger race completion callback
        if self.on_race_complete:
            self.on_race_complete(self.total_race_time, self.lap_times)

    def reset_race(self):
        """Reset race state for a new race."""
        self.race_started = False
        self.race_completed = False
        self.current_lap = 0
        self.lap_times = []
        self.race_start_time = None
        self.last_lap_time = None
        self.total_race_time = 0.0
        self.last_crossing_time = None
        self.previous_side = None

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
            'current_lap': self.current_lap,
            'required_laps': self.required_laps,
            'lap_times': self.lap_times.copy(),
            'total_race_time': self.total_race_time,
            'average_lap_time': np.mean(self.lap_times) if self.lap_times else 0.0,
            'best_lap_time': min(self.lap_times) if self.lap_times else 0.0,
            'worst_lap_time': max(self.lap_times) if self.lap_times else 0.0,
            'laps_completed': len(self.lap_times)
        }

    def is_race_active(self) -> bool:
        """Check if race is currently active."""
        return self.race_started and not self.race_completed

    def get_progress_percentage(self) -> float:
        """Get race progress as percentage."""
        if not self.race_started:
            return 0.0
        return min(100.0, (len(self.lap_times) / self.required_laps) * 100.0)
