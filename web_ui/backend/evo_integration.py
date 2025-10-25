"""
EVO Integration Module

Provides integration with the EVO trajectory evaluation library for analyzing
autonomous vehicle trajectories, computing trajectory metrics (APE, RPE), and
comparing experimental results against reference trajectories.
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import os
from datetime import datetime

class EvoIntegration:
    def __init__(self):
        self.evo_dir = Path("../../evo")
        self.data_dir = Path("../../race_monitor/evaluation_results")
        self.temp_dir = Path(tempfile.gettempdir()) / "race_monitor_evo"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check if EVO is available
        self.evo_available = self._check_evo_availability()

    def _check_evo_availability(self) -> bool:
        """
        Check if EVO library is available and properly installed.
        
        Returns:
            True if EVO can be imported, False otherwise
        """
        try:
            # Check if we can import evo
            result = subprocess.run(['python', '-c', 'import evo'], 
                                  capture_output=True, text=True, cwd=self.evo_dir)
            return result.returncode == 0
        except:
            return False

    def get_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get EVO trajectory evaluation metrics for experiment.
        
        Args:
            experiment_id: Identifier of the experiment to analyze
            
        Returns:
            Dictionary containing metrics and analysis status
        """
        if not self.evo_available:
            return {"error": "EVO not available", "available": False}
        
        metrics = {
            "experiment_id": experiment_id,
            "evo_available": True,
            "metrics": {},
            "last_analysis": None
        }
        
        if experiment_id == "current_experiment":
            # Look for existing EVO analysis results
            analysis_files = list(self.data_dir.glob("*evo*"))
            if analysis_files:
                metrics["last_analysis"] = {
                    "files": [f.name for f in analysis_files],
                    "timestamp": max([datetime.fromtimestamp(f.stat().st_mtime) for f in analysis_files]).isoformat()
                }
            
            # Check if we have trajectory data to analyze
            trajectory_files = list(self.data_dir.glob("lap_*_trajectory.txt"))
            ref_file = self.data_dir / "horizon_reference_trajectory.txt"
            
            metrics["available_data"] = {
                "trajectory_files": len(trajectory_files),
                "reference_available": ref_file.exists(),
                "can_analyze": len(trajectory_files) > 0 and ref_file.exists()
            }
        
        return metrics

    async def run_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """
        Run EVO trajectory analysis on experiment data.
        
        Args:
            experiment_id: Identifier of the experiment to analyze
            
        Returns:
            Dictionary containing analysis results and metrics
        """
        if not self.evo_available:
            return {"error": "EVO not available"}
        
        if experiment_id != "current_experiment":
            return {"error": "Only current experiment analysis supported for now"}
        
        try:
            # Prepare data for EVO analysis
            trajectory_files = list(self.data_dir.glob("lap_*_trajectory.txt"))
            ref_file = self.data_dir / "horizon_reference_trajectory.txt"
            
            if not trajectory_files:
                return {"error": "No trajectory files found"}
            
            if not ref_file.exists():
                return {"error": "No reference trajectory found"}
            
            results = {
                "experiment_id": experiment_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "trajectories_analyzed": len(trajectory_files),
                "metrics": {}
            }
            
            # Convert trajectories to EVO format and run analysis
            for i, traj_file in enumerate(trajectory_files[:5]):  # Limit for performance
                lap_num = self._extract_lap_number(traj_file.name)
                
                # Convert to TUM format for EVO
                tum_file = await self._convert_to_tum_format(traj_file, lap_num)
                ref_tum_file = await self._convert_reference_to_tum_format(ref_file)
                
                if tum_file and ref_tum_file:
                    # Run APE (Absolute Pose Error) analysis
                    ape_results = await self._run_evo_ape(ref_tum_file, tum_file, lap_num)
                    if ape_results:
                        results["metrics"][f"lap_{lap_num}_ape"] = ape_results
                    
                    # Run RPE (Relative Pose Error) analysis
                    rpe_results = await self._run_evo_rpe(ref_tum_file, tum_file, lap_num)
                    if rpe_results:
                        results["metrics"][f"lap_{lap_num}_rpe"] = rpe_results
            
            # Save results
            results_file = self.data_dir / f"evo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    async def _convert_to_tum_format(self, trajectory_file: Path, lap_num: int) -> Optional[Path]:
        """
        Convert trajectory file to TUM format for EVO.
        
        Args:
            trajectory_file: Path to trajectory data file
            lap_num: Lap number identifier
            
        Returns:
            Path to converted TUM file or None if conversion fails
        """
        try:
            # Load trajectory data
            data = np.loadtxt(trajectory_file, delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # Create TUM format: timestamp x y z qx qy qz qw
            # For 2D data, assume z=0 and no rotation
            tum_data = []
            for i, point in enumerate(data):
                timestamp = i * 0.1  # Assume 10Hz sampling
                x = point[0] if len(point) > 0 else 0
                y = point[1] if len(point) > 1 else 0
                z = point[2] if len(point) > 2 else 0
                # No rotation: qx=0, qy=0, qz=0, qw=1
                tum_data.append([timestamp, x, y, z, 0, 0, 0, 1])
            
            # Save TUM file
            tum_file = self.temp_dir / f"lap_{lap_num}_tum.txt"
            np.savetxt(tum_file, tum_data, fmt='%.6f')
            
            return tum_file
            
        except Exception as e:
            print(f"Error converting trajectory to TUM format: {e}")
            return None

    async def _convert_reference_to_tum_format(self, ref_file: Path) -> Optional[Path]:
        """
        Convert reference trajectory to TUM format.
        
        Args:
            ref_file: Path to reference trajectory file
            
        Returns:
            Path to converted TUM file or None if conversion fails
        """
        try:
            tum_ref_file = self.temp_dir / "reference_tum.txt"
            
            # Check if already converted
            if tum_ref_file.exists() and tum_ref_file.stat().st_mtime > ref_file.stat().st_mtime:
                return tum_ref_file
            
            # Load and convert reference data
            data = np.loadtxt(ref_file, delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            tum_data = []
            for i, point in enumerate(data):
                timestamp = i * 0.1
                x = point[0] if len(point) > 0 else 0
                y = point[1] if len(point) > 1 else 0
                z = point[2] if len(point) > 2 else 0
                tum_data.append([timestamp, x, y, z, 0, 0, 0, 1])
            
            np.savetxt(tum_ref_file, tum_data, fmt='%.6f')
            return tum_ref_file
            
        except Exception as e:
            print(f"Error converting reference to TUM format: {e}")
            return None

    async def _run_evo_ape(self, ref_file: Path, traj_file: Path, lap_num: int) -> Optional[Dict]:
        """
        Run EVO APE (Absolute Pose Error) analysis.
        
        Args:
            ref_file: Reference trajectory file path
            traj_file: Test trajectory file path
            lap_num: Lap number identifier
            
        Returns:
            Dictionary with APE metrics or None if analysis fails
        """
        try:
            output_file = self.temp_dir / f"ape_lap_{lap_num}.json"
            
            cmd = [
                'python', '-m', 'evo.main_ape',
                'tum', str(ref_file), str(traj_file),
                '--save_results', str(output_file),
                '--no_plot'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.evo_dir)
            
            if result.returncode == 0 and output_file.exists():
                with open(output_file, 'r') as f:
                    ape_data = json.load(f)
                
                return {
                    "lap": lap_num,
                    "type": "APE",
                    "rmse": ape_data.get("rmse", 0),
                    "mean": ape_data.get("mean", 0),
                    "median": ape_data.get("median", 0),
                    "std": ape_data.get("std", 0),
                    "min": ape_data.get("min", 0),
                    "max": ape_data.get("max", 0),
                    "sse": ape_data.get("sse", 0)
                }
            else:
                print(f"EVO APE failed for lap {lap_num}: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error running EVO APE: {e}")
            return None

    async def _run_evo_rpe(self, ref_file: Path, traj_file: Path, lap_num: int) -> Optional[Dict]:
        """
        Run EVO RPE (Relative Pose Error) analysis.
        
        Args:
            ref_file: Reference trajectory file path
            traj_file: Test trajectory file path
            lap_num: Lap number identifier
            
        Returns:
            Dictionary with RPE metrics or None if analysis fails
        """
        try:
            output_file = self.temp_dir / f"rpe_lap_{lap_num}.json"
            
            cmd = [
                'python', '-m', 'evo.main_rpe',
                'tum', str(ref_file), str(traj_file),
                '--save_results', str(output_file),
                '--no_plot'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.evo_dir)
            
            if result.returncode == 0 and output_file.exists():
                with open(output_file, 'r') as f:
                    rpe_data = json.load(f)
                
                return {
                    "lap": lap_num,
                    "type": "RPE",
                    "rmse": rpe_data.get("rmse", 0),
                    "mean": rpe_data.get("mean", 0),
                    "median": rpe_data.get("median", 0),
                    "std": rpe_data.get("std", 0),
                    "min": rpe_data.get("min", 0),
                    "max": rpe_data.get("max", 0)
                }
            else:
                print(f"EVO RPE failed for lap {lap_num}: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error running EVO RPE: {e}")
            return None

    async def compare_experiments(self, exp1_id: str, exp2_id: str) -> Dict[str, Any]:
        """
        Compare two experiments using EVO.
        
        Args:
            exp1_id: First experiment identifier
            exp2_id: Second experiment identifier
            
        Returns:
            Dictionary containing comparison results
        """
        if not self.evo_available:
            return {"error": "EVO not available"}
        
        # For now, only support comparing with current experiment
        if exp1_id != "current_experiment" and exp2_id != "current_experiment":
            return {"error": "Experiment comparison not yet fully implemented"}
        
        comparison = {
            "experiment_1": exp1_id,
            "experiment_2": exp2_id,
            "comparison_timestamp": datetime.now().isoformat(),
            "metrics_comparison": {},
            "summary": {}
        }
        
        # Get metrics for both experiments
        metrics1 = self.get_metrics(exp1_id)
        metrics2 = self.get_metrics(exp2_id)
        
        comparison["summary"] = {
            "both_have_data": metrics1.get("available_data", {}).get("can_analyze", False),
            "comparison_possible": False,
            "note": "Full experiment comparison will be implemented in future versions"
        }
        
        return comparison

    def _extract_lap_number(self, filename: str) -> int:
        """
        Extract lap number from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Lap number or 0 if not found
        """
        import re
        match = re.search(r'lap_(\d+)', filename)
        return int(match.group(1)) if match else 0

    def get_available_analyses(self) -> Dict[str, Any]:
        """
        Get list of available EVO analysis types.
        
        Returns:
            Dictionary describing available analysis methods and requirements
        """
        return {
            "evo_available": self.evo_available,
            "analyses": {
                "ape": {
                    "name": "Absolute Pose Error",
                    "description": "Measures absolute position error between trajectories",
                    "metrics": ["RMSE", "Mean", "Median", "Std", "Min", "Max"]
                },
                "rpe": {
                    "name": "Relative Pose Error", 
                    "description": "Measures relative position error over trajectory segments",
                    "metrics": ["RMSE", "Mean", "Median", "Std", "Min", "Max"]
                },
                "comparison": {
                    "name": "Trajectory Comparison",
                    "description": "Compare multiple trajectories side-by-side",
                    "metrics": ["Statistical comparison", "Visual overlay"]
                }
            },
            "requirements": {
                "reference_trajectory": "Required for meaningful analysis",
                "trajectory_format": "Supports TUM, KITTI, and CSV formats",
                "minimum_points": "At least 10 points recommended for reliable results"
            }
        }