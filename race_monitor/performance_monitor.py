#!/usr/bin/env python3

"""
Performance Monitor

Handles computational performance monitoring for the race monitor system.
Tracks CPU usage, memory consumption, control loop latency, and other
performance metrics during racing operations.

Features:
    - Real-time CPU and memory monitoring
    - Control loop latency tracking
    - Performance data logging and analysis
    - Configurable performance thresholds
    - CSV data export for analysis

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
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


class PerformanceMonitor:
    """
    Monitors computational performance during racing operations.
    
    Tracks various performance metrics including CPU usage, memory consumption,
    control loop timing, and topic frequencies for performance analysis.
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
        self.performance_log_interval = 5.0
        self.max_acceptable_latency_ms = 50.0
        self.target_control_frequency_hz = 50.0
        self.max_acceptable_cpu_usage = 80.0
        self.max_acceptable_memory_mb = 512.0
        
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
        self.performance_log_interval = config.get('performance_log_interval', self.performance_log_interval)
        self.max_acceptable_latency_ms = config.get('max_acceptable_latency_ms', self.max_acceptable_latency_ms)
        self.target_control_frequency_hz = config.get('target_control_frequency_hz', self.target_control_frequency_hz)
        self.max_acceptable_cpu_usage = config.get('max_acceptable_cpu_usage', self.max_acceptable_cpu_usage)
        self.max_acceptable_memory_mb = config.get('max_acceptable_memory_mb', self.max_acceptable_memory_mb)
        
        # Update deque sizes
        self.cpu_usage_history = deque(maxlen=self.monitoring_window_size)
        self.memory_usage_history = deque(maxlen=self.monitoring_window_size)
        self.control_latency_history = deque(maxlen=self.monitoring_window_size)
        self.control_frequency_history = deque(maxlen=self.monitoring_window_size)
        
        self.logger.info(f"Performance monitor configured: monitoring={self.enable_monitoring}, "
                        f"window_size={self.monitoring_window_size}, interval={self.cpu_monitoring_interval}s")
    
    def start_monitoring(self):
        """Start the performance monitoring thread."""
        if not self.enable_monitoring:
            return
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
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
                    self.logger.warn(f"High CPU usage detected: {cpu_percent:.1f}%")
                
                if memory_mb > self.max_acceptable_memory_mb:
                    self.logger.warn(f"High memory usage detected: {memory_mb:.1f} MB")
                
                # Periodically log performance stats
                if (self.last_performance_log is None or 
                    current_time - self.last_performance_log > self.performance_log_interval):
                    self._log_performance_stats()
                    self.last_performance_log = current_time
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
            
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
            latency_ns = (control_time - most_recent_odom_time).nanoseconds
            latency_ms = latency_ns / 1e6
            
            self.control_latency_history.append((time.time(), latency_ms))
            
            if latency_ms > self.max_acceptable_latency_ms:
                self.logger.warn(f"High control loop latency: {latency_ms:.1f}ms")
    
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
        
        self.logger.info(f"Performance Stats - CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}MB, "
                        f"Latency: {avg_latency:.1f}ms")
        
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
        Save performance data to CSV file.
        
        Args:
            output_dir: Directory to save the file
            filename_prefix: Prefix for the filename
        """
        if not self.performance_data:
            self.logger.warn("No performance data to save")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'cpu_percent', 'memory_mb', 'control_latency_ms',
                             'cpu_max', 'memory_max', 'latency_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for data in self.performance_data:
                    writer.writerow(data)
            
            self.logger.info(f"Saved performance data to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def reset_performance_data(self):
        """Clear all performance monitoring data."""
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()
        self.control_latency_history.clear()
        self.control_frequency_history.clear()
        self.performance_data.clear()
        self.odometry_timestamps.clear()
        self.control_command_timestamps.clear()
        self.logger.info("Performance monitoring data reset")
    
    def is_monitoring_active(self) -> bool:
        """Check if performance monitoring is active."""
        return self.monitoring_active and self.enable_monitoring