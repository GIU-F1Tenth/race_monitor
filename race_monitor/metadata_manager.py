#!/usr/bin/env python3

"""
Metadata Manager for Race Monitor

Handles creation and management of metadata files that contain
experiment information, timestamps, system details, and other
contextual information without cluttering filenames.

This replaces timestamp-based filename generation with clean,
consistent filenames while preserving all temporal and system
information in dedicated metadata files.

License: MIT
"""

import os
import json
import platform
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class MetadataManager:
    """
    Manages metadata creation and storage for race monitoring experiments.

    Creates clean metadata files with timestamp, experiment info, and system details
    while keeping actual data filenames clean and consistent.
    """

    def __init__(self, output_directory: str, logger=None):
        """
        Initialize metadata manager.

        Args:
            output_directory: Base directory for metadata files
            logger: Optional logger instance
        """
        self.output_directory = output_directory
        self.logger = logger
        self.metadata = {}
        self._initialize_system_info()

    def _initialize_system_info(self):
        """Initialize system information that stays constant during execution."""
        self.system_info = {
            'os': platform.system(),
            'os_release': platform.release(),
            'os_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable,
            'hostname': platform.node(),
            'architecture': platform.architecture()[0],
        }

        # Add ROS2 information if available
        try:
            import rclpy
            self.system_info['rclpy_version'] = rclpy.__version__ if hasattr(rclpy, '__version__') else 'unknown'
            self.system_info['ros_distro'] = os.environ.get('ROS_DISTRO', 'unknown')
        except ImportError:
            self.system_info['rclpy_version'] = 'not_available'
            self.system_info['ros_distro'] = 'not_available'

    def create_experiment_metadata(self, experiment_id: str, controller_name: str = '',
                                   custom_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create metadata for a new experiment.

        Args:
            experiment_id: Experiment identifier (e.g., 'exp_001')
            controller_name: Name of the controller being tested
            custom_data: Additional custom metadata

        Returns:
            Dictionary containing all metadata
        """
        timestamp = datetime.now()

        metadata = {
            'experiment_info': {
                'experiment_id': experiment_id,
                'controller_name': controller_name,
                'timestamp': timestamp.isoformat(),
                'date': timestamp.strftime('%Y-%m-%d'),
                'time': timestamp.strftime('%H:%M:%S'),
                'unix_timestamp': timestamp.timestamp(),
            },
            'system_info': self.system_info.copy(),
            'file_info': {
                'created_by': 'race_monitor_metadata_manager',
                'metadata_version': '1.0',
                'format': 'json'
            }
        }

        # Add custom data if provided
        if custom_data:
            metadata.update(custom_data)

        self.metadata = metadata
        return metadata

    def save_metadata_file(self, filename: str = 'experiment_metadata.txt') -> str:
        """
        Save metadata to a text file.

        Args:
            filename: Name of the metadata file

        Returns:
            Path to the saved metadata file
        """
        if not self.metadata:
            if self.logger:
                self.logger.warning("No metadata to save")
            return ""

        os.makedirs(self.output_directory, exist_ok=True)
        filepath = os.path.join(self.output_directory, filename)

        try:
            with open(filepath, 'w') as f:
                # Write header
                f.write("=" * 60 + "\n")
                f.write("RACE MONITOR EXPERIMENT METADATA\n")
                f.write("=" * 60 + "\n\n")

                # Experiment info
                exp_info = self.metadata.get('experiment_info', {})
                f.write("EXPERIMENT INFORMATION:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Experiment ID: {exp_info.get('experiment_id', 'unknown')}\n")
                f.write(f"Controller: {exp_info.get('controller_name', 'unknown')}\n")
                f.write(f"Date: {exp_info.get('date', 'unknown')}\n")
                f.write(f"Time: {exp_info.get('time', 'unknown')}\n")
                f.write(f"Full Timestamp: {exp_info.get('timestamp', 'unknown')}\n")
                f.write("\n")

                # System info
                sys_info = self.metadata.get('system_info', {})
                f.write("SYSTEM INFORMATION:\n")
                f.write("-" * 20 + "\n")
                f.write(f"OS: {sys_info.get('os', 'unknown')} {sys_info.get('os_release', '')}\n")
                f.write(f"Machine: {sys_info.get('machine', 'unknown')}\n")
                f.write(f"Architecture: {sys_info.get('architecture', 'unknown')}\n")
                f.write(f"Hostname: {sys_info.get('hostname', 'unknown')}\n")
                f.write(f"Python Version: {sys_info.get('python_version', 'unknown')}\n")
                f.write(f"ROS2 Distro: {sys_info.get('ros_distro', 'unknown')}\n")
                f.write("\n")

                # Additional metadata
                f.write("ADDITIONAL METADATA:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.metadata.items():
                    if key not in ['experiment_info', 'system_info', 'file_info']:
                        f.write(f"{key}: {value}\n")

                f.write("\n")
                f.write("=" * 60 + "\n")
                f.write("End of metadata file\n")
                f.write("=" * 60 + "\n")

            if self.logger:
                self.logger.info(f"Saved experiment metadata to: {filepath}")

            return filepath

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving metadata file: {e}")
            return ""

    def save_metadata_json(self, filename: str = 'experiment_metadata.json') -> str:
        """
        Save metadata to a JSON file.

        Args:
            filename: Name of the JSON metadata file

        Returns:
            Path to the saved JSON file
        """
        if not self.metadata:
            if self.logger:
                self.logger.warning("No metadata to save")
            return ""

        os.makedirs(self.output_directory, exist_ok=True)
        filepath = os.path.join(self.output_directory, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            if self.logger:
                self.logger.info(f"Saved experiment metadata JSON to: {filepath}")

            return filepath

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving metadata JSON: {e}")
            return ""

    def update_metadata(self, key: str, value: Any):
        """
        Update metadata with additional information.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value

    def get_clean_filename(self, base_name: str, extension: str = "") -> str:
        """
        Generate a clean filename without timestamps.

        Args:
            base_name: Base name for the file
            extension: File extension (with or without dot)

        Returns:
            Clean filename
        """
        if extension and not extension.startswith('.'):
            extension = '.' + extension
        return f"{base_name}{extension}"

    def get_experiment_summary(self) -> str:
        """
        Get a brief summary of the experiment for logging.

        Returns:
            Formatted summary string
        """
        if not self.metadata:
            return "No experiment metadata available"

        exp_info = self.metadata.get('experiment_info', {})
        sys_info = self.metadata.get('system_info', {})

        return (f"Experiment: {exp_info.get('experiment_id', 'unknown')} | "
                f"Controller: {exp_info.get('controller_name', 'unknown')} | "
                f"Date: {exp_info.get('date', 'unknown')} | "
                f"OS: {sys_info.get('os', 'unknown')}")
