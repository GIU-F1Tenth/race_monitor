#!/usr/bin/env python3

"""
Data Manager

Handles data storage, file I/O operations, and trajectory management
for the race monitor system. Provides centralized data management
with support for multiple file formats and data export capabilities.

Features:
    - Trajectory data storage and management
    - Multi-format file export (CSV, TUM, JSON, Pickle)
    - Research data compilation and analysis
    - File organization and directory management
    - Data validation and error handling

Author: Mohammed Abdelazim (mohammed@azab.io)
License: MIT License
"""

import os
import csv
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import rclpy

# EVO imports for trajectory handling
try:
    from evo.core import trajectory
    from evo.tools import file_interface
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


class DataManager:
    """
    Manages data storage and file operations for race monitoring.
    
    Handles trajectory data storage, file export in multiple formats,
    and provides centralized data management for all race monitor components.
    """
    
    def __init__(self, logger):
        """
        Initialize the data manager.
        
        Args:
            logger: ROS2 logger instance
        """
        self.logger = logger
        
        # Configuration
        self.trajectory_output_directory = ""
        self.output_formats = ["csv", "json"]
        self.save_trajectories = True
        self.include_timestamps = True
        self.save_intermediate_results = True
        
        # Data storage
        self.current_lap_trajectory = []
        self.completed_trajectories = {}  # lap_number -> trajectory_data
        self.evaluation_summaries = []
        self.race_results = []
        
        # File paths
        self.base_output_dir = ""
        self.trajectory_dir = ""
        self.results_dir = ""
        
    def configure(self, config: dict):
        """
        Configure data manager parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.trajectory_output_directory = config.get('trajectory_output_directory', "")
        self.output_formats = config.get('output_formats', self.output_formats)
        self.save_trajectories = config.get('save_trajectories', self.save_trajectories)
        self.include_timestamps = config.get('include_timestamps', self.include_timestamps)
        self.save_intermediate_results = config.get('save_intermediate_results', self.save_intermediate_results)
        
        # Set up directory structure
        if self.trajectory_output_directory:
            self._setup_directories()
        
        self.logger.info(f"Data manager configured: output_dir={self.trajectory_output_directory}, "
                        f"formats={self.output_formats}")
    
    def _setup_directories(self):
        """Create directory structure for data storage."""
        try:
            self.base_output_dir = self.trajectory_output_directory
            self.trajectory_dir = self.base_output_dir
            self.results_dir = os.path.join(self.base_output_dir, "results")
            
            # Create directories
            os.makedirs(self.trajectory_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            
            self.logger.info(f"Data directories created: {self.base_output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")
    
    def start_new_lap_trajectory(self, lap_number: int):
        """
        Start recording a new lap trajectory.
        
        Args:
            lap_number: Lap number to start recording
        """
        self.current_lap_trajectory = []
        self.logger.debug(f"Started recording trajectory for lap {lap_number}")
    
    def add_trajectory_point(self, x: float, y: float, z: float = 0.0, 
                           timestamp=None, additional_data: Dict = None):
        """
        Add a point to the current lap trajectory.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            timestamp: Point timestamp
            additional_data: Additional data to store with the point
        """
        point_data = {
            'x': float(x),
            'y': float(y),
            'z': float(z)
        }
        
        if timestamp and self.include_timestamps:
            point_data['timestamp'] = timestamp.nanoseconds / 1e9
        
        if additional_data:
            point_data.update(additional_data)
        
        self.current_lap_trajectory.append(point_data)
    
    def complete_lap_trajectory(self, lap_number: int, lap_time: float) -> bool:
        """
        Complete the current lap trajectory and save it.
        
        Args:
            lap_number: Completed lap number
            lap_time: Lap time in seconds
            
        Returns:
            bool: True if successfully saved
        """
        if not self.current_lap_trajectory:
            self.logger.warn(f"No trajectory data for lap {lap_number}")
            return False
        
        # Store completed trajectory
        trajectory_data = {
            'lap_number': lap_number,
            'lap_time': lap_time,
            'points': self.current_lap_trajectory.copy(),
            'timestamp': datetime.now().isoformat(),
            'point_count': len(self.current_lap_trajectory)
        }
        
        self.completed_trajectories[lap_number] = trajectory_data
        
        # Save to file if enabled
        if self.save_trajectories:
            success = self.save_trajectory_to_file(trajectory_data)
            if success:
                self.logger.info(f"Saved trajectory for lap {lap_number} ({len(self.current_lap_trajectory)} points)")
            return success
        
        return True
    
    def save_trajectory_to_file(self, trajectory_data: Dict) -> bool:
        """
        Save trajectory data to file in multiple formats.
        
        Args:
            trajectory_data: Trajectory data to save
            
        Returns:
            bool: True if successfully saved
        """
        lap_number = trajectory_data['lap_number']
        points = trajectory_data['points']
        
        try:
            # Save in requested formats
            success = True
            
            if 'csv' in self.output_formats:
                success &= self._save_trajectory_csv(lap_number, points)
            
            if 'tum' in self.output_formats:
                success &= self._save_trajectory_tum(lap_number, points)
            
            if 'json' in self.output_formats:
                success &= self._save_trajectory_json(lap_number, trajectory_data)
            
            if 'pickle' in self.output_formats:
                success &= self._save_trajectory_pickle(lap_number, trajectory_data)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving trajectory for lap {lap_number}: {e}")
            return False
    
    def _save_trajectory_csv(self, lap_number: int, points: List[Dict]) -> bool:
        """Save trajectory as CSV file."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.csv"
            filepath = os.path.join(self.trajectory_dir, filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                if points:
                    fieldnames = list(points[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(points)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving CSV trajectory: {e}")
            return False
    
    def _save_trajectory_tum(self, lap_number: int, points: List[Dict]) -> bool:
        """Save trajectory in TUM format."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.tum"
            filepath = os.path.join(self.trajectory_dir, filename)
            
            with open(filepath, 'w') as f:
                for point in points:
                    timestamp = point.get('timestamp', 0.0)
                    x = point['x']
                    y = point['y']
                    z = point['z']
                    # Use identity quaternion if no orientation data
                    qx = point.get('qx', 0.0)
                    qy = point.get('qy', 0.0)
                    qz = point.get('qz', 0.0)
                    qw = point.get('qw', 1.0)
                    
                    f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving TUM trajectory: {e}")
            return False
    
    def _save_trajectory_json(self, lap_number: int, trajectory_data: Dict) -> bool:
        """Save trajectory as JSON file."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.json"
            filepath = os.path.join(self.trajectory_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving JSON trajectory: {e}")
            return False
    
    def _save_trajectory_pickle(self, lap_number: int, trajectory_data: Dict) -> bool:
        """Save trajectory as Pickle file."""
        try:
            filename = f"lap_{lap_number:03d}_trajectory.pkl"
            filepath = os.path.join(self.trajectory_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(trajectory_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving Pickle trajectory: {e}")
            return False
    
    def save_race_results_to_csv(self, race_data: Dict) -> bool:
        """
        Save race results to CSV file.
        
        Args:
            race_data: Race data including lap times, statistics, etc.
            
        Returns:
            bool: True if successfully saved
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"race_results_{timestamp}.csv"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(['Race Results', datetime.now().isoformat()])
                writer.writerow([])
                
                # Basic race info
                writer.writerow(['Total Race Time (s)', race_data.get('total_race_time', 0.0)])
                writer.writerow(['Total Laps', len(race_data.get('lap_times', []))])
                writer.writerow(['Controller', race_data.get('controller_name', 'unknown')])
                writer.writerow([])
                
                # Lap times
                writer.writerow(['Lap Number', 'Lap Time (s)'])
                for i, lap_time in enumerate(race_data.get('lap_times', []), 1):
                    writer.writerow([i, f"{lap_time:.4f}"])
                
                writer.writerow([])
                
                # Statistics
                lap_times = race_data.get('lap_times', [])
                if lap_times:
                    writer.writerow(['best_lap_s', f"{min(lap_times):.4f}"])
                    writer.writerow(['worst_lap_s', f"{max(lap_times):.4f}"])
                    writer.writerow(['average_lap_s', f"{np.mean(lap_times):.4f}"])
            
            self.logger.info(f"Saved race results to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving race results: {e}")
            return False
    
    def save_evaluation_summary(self, evaluation_data: Dict) -> bool:
        """
        Save evaluation summary data.
        
        Args:
            evaluation_data: Evaluation summary data
            
        Returns:
            bool: True if successfully saved
        """
        try:
            filename = "evaluation_summary.csv"
            filepath = os.path.join(self.base_output_dir, filename)
            
            # Store evaluation data
            self.evaluation_summaries.append(evaluation_data)
            
            # Write to CSV
            with open(filepath, 'w', newline='') as csvfile:
                if self.evaluation_summaries:
                    fieldnames = list(self.evaluation_summaries[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.evaluation_summaries)
            
            self.logger.info(f"Saved evaluation summary to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation summary: {e}")
            return False
    
    def get_trajectory_data(self, lap_number: int = None) -> Optional[Dict]:
        """
        Get trajectory data for a specific lap or all laps.
        
        Args:
            lap_number: Specific lap number, or None for all laps
            
        Returns:
            Trajectory data or None if not found
        """
        if lap_number is not None:
            return self.completed_trajectories.get(lap_number)
        else:
            return self.completed_trajectories
    
    def get_current_trajectory_points(self) -> List[Dict]:
        """Get current lap trajectory points."""
        return self.current_lap_trajectory.copy()
    
    def clear_trajectory_data(self):
        """Clear all stored trajectory data."""
        self.current_lap_trajectory.clear()
        self.completed_trajectories.clear()
        self.evaluation_summaries.clear()
        self.race_results.clear()
        self.logger.info("Cleared all trajectory data")
    
    def create_evo_trajectory(self, lap_number: int):
        """
        Create EVO trajectory object from stored trajectory data.
        
        Args:
            lap_number: Lap number to create trajectory for
            
        Returns:
            EVO trajectory object or None
        """
        if not EVO_AVAILABLE:
            return None
        
        trajectory_data = self.completed_trajectories.get(lap_number)
        if not trajectory_data:
            return None
        
        try:
            points = trajectory_data['points']
            timestamps = np.array([p.get('timestamp', i) for i, p in enumerate(points)])
            positions = np.array([[p['x'], p['y'], p['z']] for p in points])
            
            if all('qx' in p for p in points):
                # Has orientation data
                orientations = np.array([[p['qx'], p['qy'], p['qz'], p['qw']] for p in points])
                return trajectory.PoseTrajectory3D(positions, orientations, timestamps)
            else:
                # Position only
                return trajectory.PosePath3D(positions, timestamps)
                
        except Exception as e:
            self.logger.error(f"Error creating EVO trajectory: {e}")
            return None
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        return {
            "completed_laps": len(self.completed_trajectories),
            "current_lap_points": len(self.current_lap_trajectory),
            "evaluation_summaries": len(self.evaluation_summaries),
            "output_directory": self.base_output_dir,
            "output_formats": self.output_formats,
            "total_trajectory_points": sum(
                len(traj['points']) for traj in self.completed_trajectories.values()
            )
        }