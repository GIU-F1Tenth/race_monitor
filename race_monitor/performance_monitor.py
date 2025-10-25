#!/usr/bin/env python3

"""
Performance Monitor

Computational performance monitoring system for race monitoring operations.
Tracks CPU usage, memory consumption, control loop latency, and performance
metrics during autonomous racing with configurable thresholds and logging.

Core Features:
    - Real-time CPU and memory monitoring
    - Control loop latency tracking
    - Performance data logging and analysis
    - Configurable performance thresholds
    - Multi-format data export capabilities

Monitoring Capabilities:
    - System resource utilization (CPU/Memory)
    - Control loop timing and frequency analysis
    - Topic message timing and latency
    - Performance threshold violations
    - Statistical performance analysis

Output Formats:
    - CSV data export for analysis
    - Real-time performance logging
    - Statistical summaries

License: MIT
"""

import os
import csv
import time
import psutil
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any
import rclpy
from rclpy.node import Node
from race_monitor.logger_utils import RaceMonitorLogger, LogLevel


def time_to_nanoseconds(time_obj):
    """
    Convert time object to nanoseconds.
    
    Args:
        time_obj: ROS2 Time or builtin_interfaces Time object
        
    Returns:
        int: Time in nanoseconds
        
    Raises:
        ValueError: If time_obj is not a recognized time type
    """
    if hasattr(time_obj, 'nanoseconds'):
        return time_obj.nanoseconds
    elif hasattr(time_obj, 'sec') and hasattr(time_obj, 'nanosec'):
        return time_obj.sec * 1e9 + time_obj.nanosec
    else:
        raise ValueError(f"Unknown time object type: {type(time_obj)}")


class PerformanceMonitor:
    """
    Computational performance monitoring for racing operations.

    Tracks CPU usage, memory consumption, control loop timing,
    and other performance metrics with configurable thresholds.
    """

    def __init__(self, logger, clock):
        """
        Initialize the performance monitor.

        Args:
            logger: ROS2 logger instance
            clock: ROS2 clock instance
        """
        self.logger = logger
        self.clock = clock

        # Configuration
        self.enable_monitoring = True
        self.monitoring_window_size = 100
        self.cpu_monitoring_interval = 0.1
        self.enable_performance_logging = True
        self.performance_log_interval = 5.0
        self.max_acceptable_latency_ms = 50.0
        self.target_control_frequency_hz = 50.0
        self.max_acceptable_cpu_usage = 80.0
        self.max_acceptable_memory_mb = 512.0

        # Topic configuration for monitoring
        self.odometry_topics = ['car_state/odom']
        self.control_command_topics = ['/drive']

        # Performance data storage
        self.cpu_usage_history = deque(maxlen=self.monitoring_window_size)
        self.memory_usage_history = deque(maxlen=self.monitoring_window_size)
        self.control_latency_history = deque(maxlen=self.monitoring_window_size)
        self.control_frequency_history = deque(maxlen=self.monitoring_window_size)

        # Timing data
        self.odometry_timestamps = {}
        self.control_command_timestamps = {}
        self.last_performance_log = None

        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.performance_data = []

        # Process reference
        self.process = psutil.Process()

    def configure(self, config: dict):
        """
        Configure performance monitor parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.enable_monitoring = config.get('enable_computational_monitoring', self.enable_monitoring)
        self.monitoring_window_size = config.get('monitoring_window_size', self.monitoring_window_size)
        self.cpu_monitoring_interval = config.get('cpu_monitoring_interval', self.cpu_monitoring_interval)
        self.enable_performance_logging = config.get('enable_performance_logging', True)
        self.performance_log_interval = config.get('performance_log_interval', self.performance_log_interval)

        # Performance thresholds
        self.max_acceptable_latency_ms = config.get('max_acceptable_latency_ms', self.max_acceptable_latency_ms)
        self.target_control_frequency_hz = config.get('target_control_frequency_hz', self.target_control_frequency_hz)
        self.max_acceptable_cpu_usage = config.get('max_acceptable_cpu_usage', self.max_acceptable_cpu_usage)
        self.max_acceptable_memory_mb = config.get('max_acceptable_memory_mb', self.max_acceptable_memory_mb)

        # Topic configuration for monitoring
        self.odometry_topics = config.get('odometry_topics', ['car_state/odom'])
        self.control_command_topics = config.get('control_command_topics', ['/drive'])

        # Update deque sizes if window size changed
        if len(self.cpu_usage_history) != self.monitoring_window_size:
            self.cpu_usage_history = deque(maxlen=self.monitoring_window_size)
            self.memory_usage_history = deque(maxlen=self.monitoring_window_size)
            self.control_latency_history = deque(maxlen=self.monitoring_window_size)
            self.control_frequency_history = deque(maxlen=self.monitoring_window_size)

        if self.enable_monitoring:
            self.start_monitoring()

    def start_monitoring(self):
        """Start the performance monitoring thread."""
        if not self.enable_monitoring:
            return

        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop the performance monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                # Store metrics
                current_time = time.time()
                self.cpu_usage_history.append((current_time, cpu_percent))
                self.memory_usage_history.append((current_time, memory_mb))

                # Check for performance issues
                if cpu_percent > self.max_acceptable_cpu_usage:
                    self.logger.warn(f"High CPU usage detected: {cpu_percent:.1f}%", LogLevel.NORMAL)

                if memory_mb > self.max_acceptable_memory_mb:
                    self.logger.warn(f"High memory usage detected: {memory_mb:.1f} MB", LogLevel.NORMAL)

                # Periodically log performance stats
                if (self.last_performance_log is None or
                        current_time - self.last_performance_log > self.performance_log_interval):
                    self._log_performance_stats()
                    self.last_performance_log = current_time

            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}", LogLevel.DEBUG)

            time.sleep(self.cpu_monitoring_interval)

    def record_odometry_timestamp(self, topic_name: str, timestamp):
        """
        Record timestamp for odometry message reception.

        Args:
            topic_name: Name of the odometry topic
            timestamp: Message timestamp
        """
        if not self.enable_monitoring:
            return

        current_time = self.clock.now()
        self.odometry_timestamps[topic_name] = {
            'message_time': timestamp,
            'received_time': current_time
        }

    def record_control_command_timestamp(self, topic_name: str, timestamp):
        """
        Record timestamp for control command publication.

        Args:
            topic_name: Name of the control topic
            timestamp: Command timestamp
        """
        if not self.enable_monitoring:
            return

        current_time = self.clock.now()
        self.control_command_timestamps[topic_name] = {
            'command_time': timestamp,
            'published_time': current_time
        }

        # Calculate control loop latency if we have corresponding odometry
        self._calculate_control_latency(topic_name, current_time)

    def _calculate_control_latency(self, control_topic: str, control_time):
        """
        Calculate control loop latency.

        Args:
            control_topic: Name of control topic
            control_time: Control command timestamp
        """
        # Find the most recent odometry timestamp
        most_recent_odom_time = None
        for odom_topic, odom_data in self.odometry_timestamps.items():
            if most_recent_odom_time is None or odom_data['received_time'] > most_recent_odom_time:
                most_recent_odom_time = odom_data['received_time']

        if most_recent_odom_time is not None:
            latency_ns = time_to_nanoseconds(control_time) - time_to_nanoseconds(most_recent_odom_time)
            latency_ms = latency_ns / 1e6

            self.control_latency_history.append((time.time(), latency_ms))

            if latency_ms > self.max_acceptable_latency_ms:
                self.logger.warn(f"High control loop latency: {latency_ms:.1f}ms", LogLevel.DEBUG)

    def _log_performance_stats(self):
        """Log current performance statistics."""
        if not self.cpu_usage_history or not self.memory_usage_history:
            return

        # Calculate averages
        recent_cpu = [cpu for _, cpu in self.cpu_usage_history]
        recent_memory = [mem for _, mem in self.memory_usage_history]
        recent_latency = [lat for _, lat in self.control_latency_history] if self.control_latency_history else []

        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        avg_memory = sum(recent_memory) / len(recent_memory)
        avg_latency = sum(recent_latency) / len(recent_latency) if recent_latency else 0.0

        # Store for CSV export
        self.performance_data.append({
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': avg_cpu,
            'memory_mb': avg_memory,
            'control_latency_ms': avg_latency,
            'cpu_max': max(recent_cpu) if recent_cpu else 0,
            'memory_max': max(recent_memory) if recent_memory else 0,
            'latency_max': max(recent_latency) if recent_latency else 0
        })

    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Calculate and return comprehensive performance statistics.

        Returns:
            Dictionary containing performance statistics
        """
        if not self.cpu_usage_history:
            return {"monitoring_active": False}

        # Extract recent data
        recent_cpu = [cpu for _, cpu in self.cpu_usage_history]
        recent_memory = [mem for _, mem in self.memory_usage_history]
        recent_latency = [lat for _, lat in self.control_latency_history] if self.control_latency_history else []

        stats = {
            "monitoring_active": True,
            "window_size": len(recent_cpu),
            "cpu_usage": {
                "current": recent_cpu[-1] if recent_cpu else 0,
                "average": sum(recent_cpu) / len(recent_cpu),
                "max": max(recent_cpu),
                "min": min(recent_cpu),
                "threshold_exceeded": any(cpu > self.max_acceptable_cpu_usage for cpu in recent_cpu)
            },
            "memory_usage": {
                "current_mb": recent_memory[-1] if recent_memory else 0,
                "average_mb": sum(recent_memory) / len(recent_memory),
                "max_mb": max(recent_memory),
                "min_mb": min(recent_memory),
                "threshold_exceeded": any(mem > self.max_acceptable_memory_mb for mem in recent_memory)
            }
        }

        if recent_latency:
            stats["control_latency"] = {
                "current_ms": recent_latency[-1],
                "average_ms": sum(recent_latency) / len(recent_latency),
                "max_ms": max(recent_latency),
                "min_ms": min(recent_latency),
                "threshold_exceeded": any(lat > self.max_acceptable_latency_ms for lat in recent_latency)
            }

        return stats

    def save_performance_data_to_csv(self, output_dir: str, filename_prefix: str = "computational_performance"):
        """
        Save performance data to CSV file and generate summary.

        Args:
            output_dir: Directory to save the file
            filename_prefix: Prefix for the filename
        """
        if not self.performance_data:
            self.logger.warn("No performance data to save", LogLevel.DEBUG)
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{filename_prefix}.csv"

            # Organize file by extension
            csv_dir = os.path.join(output_dir, "csv")
            os.makedirs(csv_dir, exist_ok=True)
            filepath = os.path.join(csv_dir, filename)

            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'cpu_percent', 'memory_mb', 'control_latency_ms',
                              'cpu_max', 'memory_max', 'latency_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for data in self.performance_data:
                    writer.writerow(data)

            self.logger.success(f"Performance data saved to: {filepath}", LogLevel.NORMAL)

            # Generate performance summary
            self._generate_performance_summary(filepath)

        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}", LogLevel.NORMAL)

    def _generate_performance_summary(self, csv_filepath: str):
        """Generate a JSON summary of computational performance data."""
        try:
            import json
            import numpy as np
            from datetime import datetime

            if not self.performance_data:
                return

            # Extract data for analysis
            cpu_data = [d['cpu_percent'] for d in self.performance_data]
            memory_data = [d['memory_mb'] for d in self.performance_data]
            latency_data = [d['control_latency_ms'] for d in self.performance_data]
            cpu_max_data = [d['cpu_max'] for d in self.performance_data]
            memory_max_data = [d['memory_max'] for d in self.performance_data]
            latency_max_data = [d['latency_max'] for d in self.performance_data]

            # Calculate timestamps if available
            timestamps = [d.get('timestamp') for d in self.performance_data if d.get('timestamp')]
            duration = 0
            sample_rate = 0

            if len(timestamps) >= 2:
                try:
                    from datetime import datetime
                    first_time = datetime.fromisoformat(timestamps[0])
                    last_time = datetime.fromisoformat(timestamps[-1])
                    duration = (last_time - first_time).total_seconds()
                    sample_rate = len(self.performance_data) / duration if duration > 0 else 0
                except BaseException:
                    pass

            def calculate_grade(avg_val, max_val, thresholds):
                """Calculate performance grade based on thresholds."""
                for grade, (avg_thresh, max_thresh) in thresholds.items():
                    if avg_val <= avg_thresh and max_val <= max_thresh:
                        return grade
                return "F"

            # Grade calculations
            cpu_grade = calculate_grade(np.mean(cpu_data), np.max(cpu_data), {
                "A": (15, 50), "B": (30, 70), "C": (50, 85), "D": (70, 95)
            })

            memory_grade = calculate_grade(np.mean(memory_data), np.max(memory_data), {
                "A": (200, 300), "B": (300, 400), "C": (400, 500), "D": (500, 600)
            })

            latency_grade = calculate_grade(np.mean(latency_data), np.max(latency_data), {
                "A": (5, 15), "B": (10, 25), "C": (20, 40), "D": (30, 60)
            })

            # Overall grade
            grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
            avg_grade_value = np.mean(
                [grade_values[cpu_grade], grade_values[memory_grade], grade_values[latency_grade]])
            overall_grade = ["F", "D", "C", "B", "A"][min(4, int(avg_grade_value))]

            # Create performance summary
            performance_summary = {
                "metadata": {
                    "source_file": csv_filepath,
                    "generated_at": datetime.now().isoformat(),
                    "total_samples": len(self.performance_data),
                    "duration_seconds": duration,
                    "sample_rate_hz": sample_rate
                },
                "cpu_performance": {
                    "average_percent": float(np.mean(cpu_data)),
                    "best_percent": float(np.min(cpu_data)),
                    "worst_percent": float(np.max(cpu_data)),
                    "std_deviation": float(np.std(cpu_data)),
                    "peak_cpu_max": float(np.max(cpu_max_data))
                },
                "memory_performance": {
                    "average_mb": float(np.mean(memory_data)),
                    "best_mb": float(np.min(memory_data)),
                    "worst_mb": float(np.max(memory_data)),
                    "std_deviation": float(np.std(memory_data)),
                    "peak_memory_max": float(np.max(memory_max_data))
                },
                "control_latency": {
                    "average_ms": float(np.mean(latency_data)),
                    "best_ms": float(np.min(latency_data)),
                    "worst_ms": float(np.max(latency_data)),
                    "std_deviation": float(np.std(latency_data)),
                    "peak_latency_max": float(np.max(latency_max_data))
                },
                "performance_grades": {
                    "cpu_grade": cpu_grade,
                    "memory_grade": memory_grade,
                    "latency_grade": latency_grade,
                    "overall_grade": overall_grade
                }
            }

            # Save summary JSON file with organized structure
            json_filename = csv_filepath.replace('.csv', '_summary.json').split('/')[-1]  # Get just filename
            csv_dir = os.path.dirname(csv_filepath)
            base_dir = os.path.dirname(csv_dir)  # Go up from csv/ to parent
            json_dir = os.path.join(base_dir, "json")
            os.makedirs(json_dir, exist_ok=True)
            summary_path = os.path.join(json_dir, json_filename)

            with open(summary_path, 'w') as f:
                json.dump(performance_summary, f, indent=2)

            self.logger.debug(f"Performance summary generated: {summary_path}")

        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}", LogLevel.DEBUG)

    def reset_performance_data(self):
        """Clear all performance monitoring data."""
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()
        self.control_latency_history.clear()
        self.control_frequency_history.clear()
        self.performance_data.clear()
        self.odometry_timestamps.clear()
        self.control_command_timestamps.clear()

    def is_monitoring_active(self) -> bool:
        """Check if performance monitoring is active."""
        return self.monitoring_active and self.enable_monitoring
