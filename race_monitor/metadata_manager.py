#!/usr/bin/env python3

"""
Metadata Manager for Race Monitor

Handles creation and management of metadata files that contain
experiment information, timestamps, system details, and other
contextual information without cluttering filenames.

License: MIT
"""

import os
import platform
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from race_monitor.logger_utils import RaceMonitorLogger, LogLevel


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

        # Load race monitor data
        self.race_monitor_data = self._load_race_monitor_data()

    def _load_race_monitor_data(self) -> Dict[str, Any]:
        """Load race monitor maintainer and version information from data file."""
        race_monitor_data = {
            'maintainer': {
                'name': 'Mohammed Abdelazim',
                'email': 'mohammed@azab.io'
            },
            'version': {
                'race_monitor_version': '1.0.0',
                'build_date': 'unknown',
                'license': 'MIT'
            },
            'system': {
                'package_name': 'race_monitor',
                'description': 'Race Monitoring System',
                'repository': 'https://github.com/GIU-F1Tenth/race_monitor/'
            }
        }

        try:
            # Find the race_monitor_data.txt file
            # Look in the parent directory of this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_file_path = os.path.join(os.path.dirname(current_dir), 'race_monitor_data.txt')

            if os.path.exists(data_file_path):
                with open(data_file_path, 'r') as f:
                    current_section = None
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        # Check for section headers
                        if line.startswith('[') and line.endswith(']'):
                            current_section = line[1:-1].lower()
                            continue

                        # Parse key=value pairs
                        if '=' in line and current_section:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            if current_section in race_monitor_data:
                                race_monitor_data[current_section][key] = value

                if self.logger:
                    self.logger.debug(f"Loaded race monitor data from: {data_file_path}")
            else:
                if self.logger:
                    self.logger.warn(f"Race monitor data file not found: {data_file_path}", LogLevel.DEBUG)

        except Exception as e:
            if self.logger:
                self.logger.warn(f"Error loading race monitor data: {e}", LogLevel.DEBUG)

        return race_monitor_data

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

        # Create the metadata structure
        metadata = {
            'experiment_info': {
                'experiment_id': experiment_id,
                'controller_name': controller_name,
                'timestamp': timestamp.isoformat(),
                'date': timestamp.strftime('%Y-%m-%d'),
                'time': timestamp.strftime('%H:%M:%S'),
            },
            'system_info': self.system_info.copy(),
            'race_monitor_info': self.race_monitor_data.copy(),
            'file_info': {
                'metadata_created': timestamp.isoformat(),
                'race_monitor_version': self.race_monitor_data['version']['race_monitor_version']
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
                self.logger.warn("No metadata to save", LogLevel.DEBUG)
            return ""

        os.makedirs(self.output_directory, exist_ok=True)

        # Check for exceptions that should NOT be organized
        if filename == 'experiment_metadata.txt' or filename.lower().endswith('.md'):
            filepath = os.path.join(self.output_directory, filename)
        else:
            # Organize by extension
            _, ext = os.path.splitext(filename)
            ext = ext.lstrip('.').lower()
            if ext:
                ext_dir = os.path.join(self.output_directory, ext)
                os.makedirs(ext_dir, exist_ok=True)
                filepath = os.path.join(ext_dir, filename)
            else:
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
                f.write(f"ROS2 Distro: {sys_info.get('ros_distro', 'unknown')}\n")
                f.write("\n")

                # Race Monitor info
                rm_info = self.metadata.get('race_monitor_info', {})
                f.write("RACE MONITOR INFORMATION:\n")
                f.write("-" * 30 + "\n")
                maintainer = rm_info.get('maintainer', {})
                version = rm_info.get('version', {})
                system = rm_info.get('system', {})
                f.write(f"Maintainer: {maintainer.get('name', 'unknown')}\n")
                f.write(f"Email: {maintainer.get('email', 'unknown')}\n")
                f.write(f"Version: {version.get('race_monitor_version', 'unknown')}\n")
                f.write(f"Build Date: {version.get('build_date', 'unknown')}\n")
                f.write(f"License: {version.get('license', 'unknown')}\n")
                f.write(f"Package: {system.get('package_name', 'unknown')}\n")
                f.write(f"Description: {system.get('description', 'unknown')}\n")
                f.write(f"Repository: {system.get('repository', 'unknown')}\n")
                f.write("\n")

                # Additional metadata
                f.write("ADDITIONAL METADATA:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.metadata.items():
                    if key not in ['experiment_info', 'system_info', 'race_monitor_info', 'file_info']:
                        f.write(f"{key}: {value}\n")

                f.write("\n")
                f.write("=" * 60 + "\n")
                f.write("End of metadata file\n")
                f.write("=" * 60 + "\n")

            if self.logger:
                self.logger.success(f"Saved experiment metadata to: {filepath}", LogLevel.NORMAL)

            return filepath

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving metadata file: {e}", LogLevel.NORMAL)
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
