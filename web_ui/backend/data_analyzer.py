"""
Data Analysis Module
Handles analysis of race data, lap times, trajectories, and performance metrics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import glob

class DataAnalyzer:
    def __init__(self):
        # Use absolute path to avoid issues when running from different directories
        base_path = Path(__file__).parent.parent.parent
        self.data_dir = base_path / "race_monitor" / "evaluation_results"
        self.graphs_dir = self.data_dir / "graphs"
        self.performance_dir = self.data_dir / "cpu_performance_data"
        self.research_dir = self.data_dir / "research_data"

    def get_experiments(self, filter_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Get list of experiments with metadata"""
        experiments = []
        
        # Look for trajectory files to identify experiments
        trajectory_files = list(self.data_dir.glob("lap_*_trajectory.txt"))
        
        # Group by experiment (assuming experiment data is in subdirectories or has common prefixes)
        experiment_data = {}
        
        # Check for evaluation summary
        summary_file = self.data_dir / "evaluation_summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            
            experiment_data["current_experiment"] = {
                "id": "current_experiment",
                "name": "Current Experiment",
                "date": datetime.fromtimestamp(summary_file.stat().st_mtime).isoformat(),
                "laps": len(df),
                "total_trajectory_files": len(trajectory_files),
                "status": "completed" if len(df) > 0 else "in_progress",
                "metrics": {
                    "avg_consistency": float(df['consistency'].mean()) if 'consistency' in df.columns else 0,
                    "avg_path_length": float(df['path_length'].mean()) if 'path_length' in df.columns else 0,
                    "avg_smoothness": float(df['smoothness'].mean()) if 'smoothness' in df.columns else 0
                }
            }
        
        # Check research data directory for other experiments
        if self.research_dir.exists():
            for controller_dir in self.research_dir.iterdir():
                if controller_dir.is_dir():
                    for exp_dir in controller_dir.iterdir():
                        if exp_dir.is_dir():
                            exp_id = f"{controller_dir.name}_{exp_dir.name}"
                            
                            # Look for analysis files
                            analysis_files = list(exp_dir.glob("*.json"))
                            csv_files = list(exp_dir.glob("*.csv"))
                            
                            if analysis_files or csv_files:
                                stat = exp_dir.stat()
                                experiment_data[exp_id] = {
                                    "id": exp_id,
                                    "name": f"{controller_dir.name} - {exp_dir.name}",
                                    "controller": controller_dir.name,
                                    "date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                    "status": "completed",
                                    "files": len(analysis_files) + len(csv_files)
                                }
        
        # Apply filters if provided
        if filter_params:
            filtered_data = {}
            for exp_id, exp_data in experiment_data.items():
                include = True
                
                if filter_params.get("controller_name") and filter_params["controller_name"] not in exp_data.get("controller", ""):
                    include = False
                
                if filter_params.get("date_from"):
                    try:
                        exp_date = datetime.fromisoformat(exp_data["date"].replace('Z', '+00:00'))
                        filter_date = datetime.fromisoformat(filter_params["date_from"])
                        if exp_date < filter_date:
                            include = False
                    except:
                        pass
                
                if include:
                    filtered_data[exp_id] = exp_data
            
            experiment_data = filtered_data
        
        return {
            "experiments": list(experiment_data.values()),
            "total_count": len(experiment_data),
            "data_directory": str(self.data_dir)
        }

    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed data for specific experiment"""
        if experiment_id == "current_experiment":
            return self._get_current_experiment_details()
        
        # Look in research data
        for controller_dir in self.research_dir.iterdir():
            if controller_dir.is_dir():
                for exp_dir in controller_dir.iterdir():
                    if exp_dir.is_dir() and f"{controller_dir.name}_{exp_dir.name}" == experiment_id:
                        return self._get_research_experiment_details(exp_dir)
        
        raise FileNotFoundError(f"Experiment {experiment_id} not found")

    def _get_current_experiment_details(self) -> Dict[str, Any]:
        """Get details for current experiment"""
        details = {
            "id": "current_experiment",
            "name": "Current Experiment",
            "laps": [],
            "summary": {},
            "trajectories": [],
            "graphs": []
        }
        
        # Load evaluation summary
        summary_file = self.data_dir / "evaluation_summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            details["summary"] = {
                "total_laps": len(df),
                "statistics": {
                    "consistency": {
                        "mean": float(df['consistency'].mean()) if 'consistency' in df.columns else 0,
                        "std": float(df['consistency'].std()) if 'consistency' in df.columns else 0,
                        "min": float(df['consistency'].min()) if 'consistency' in df.columns else 0,
                        "max": float(df['consistency'].max()) if 'consistency' in df.columns else 0
                    },
                    "path_length": {
                        "mean": float(df['path_length'].mean()) if 'path_length' in df.columns else 0,
                        "std": float(df['path_length'].std()) if 'path_length' in df.columns else 0
                    },
                    "smoothness": {
                        "mean": float(df['smoothness'].mean()) if 'smoothness' in df.columns else 0,
                        "std": float(df['smoothness'].std()) if 'smoothness' in df.columns else 0
                    }
                }
            }
            
            # Convert lap data
            for _, row in df.iterrows():
                lap_data = {
                    "lap_number": int(row['lap_number']),
                    "consistency": float(row['consistency']) if pd.notna(row['consistency']) else None,
                    "path_length": float(row['path_length']) if pd.notna(row['path_length']) else None,
                    "smoothness": float(row['smoothness']) if pd.notna(row['smoothness']) else None
                }
                details["laps"].append(lap_data)
        
        # Find trajectory files
        trajectory_files = list(self.data_dir.glob("lap_*_trajectory.txt"))
        for traj_file in sorted(trajectory_files):
            details["trajectories"].append({
                "filename": traj_file.name,
                "lap_number": self._extract_lap_number(traj_file.name),
                "size": traj_file.stat().st_size,
                "modified": datetime.fromtimestamp(traj_file.stat().st_mtime).isoformat()
            })
        
        # Find graph files
        if self.graphs_dir.exists():
            graph_files = list(self.graphs_dir.glob("*"))
            for graph_file in graph_files:
                if graph_file.is_file():
                    details["graphs"].append({
                        "filename": graph_file.name,
                        "type": graph_file.suffix[1:],  # Remove the dot
                        "size": graph_file.stat().st_size,
                        "modified": datetime.fromtimestamp(graph_file.stat().st_mtime).isoformat()
                    })
        
        return details

    def _get_research_experiment_details(self, exp_dir: Path) -> Dict[str, Any]:
        """Get details for research experiment"""
        details = {
            "id": exp_dir.parent.name + "_" + exp_dir.name,
            "name": f"{exp_dir.parent.name} - {exp_dir.name}",
            "controller": exp_dir.parent.name,
            "files": [],
            "analysis": {}
        }
        
        # List all files in experiment directory
        for file_path in exp_dir.iterdir():
            if file_path.is_file():
                details["files"].append({
                    "filename": file_path.name,
                    "type": file_path.suffix[1:] if file_path.suffix else "unknown",
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
                
                # Try to load JSON analysis files
                if file_path.suffix == ".json":
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            details["analysis"][file_path.stem] = data
                    except:
                        pass
        
        return details

    def _extract_lap_number(self, filename: str) -> int:
        """Extract lap number from filename"""
        import re
        match = re.search(r'lap_(\d+)', filename)
        return int(match.group(1)) if match else 0

    def get_summary(self) -> Dict[str, Any]:
        """Get overall data summary and statistics"""
        summary = {
            "experiments_count": 0,
            "total_laps": 0,
            "total_trajectories": 0,
            "latest_experiment": None,
            "data_health": {
                "missing_files": [],
                "recent_activity": []
            }
        }
        
        # Count trajectory files
        trajectory_files = list(self.data_dir.glob("lap_*_trajectory.txt"))
        summary["total_trajectories"] = len(trajectory_files)
        
        # Check evaluation summary
        summary_file = self.data_dir / "evaluation_summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            summary["total_laps"] = len(df)
            summary["latest_experiment"] = {
                "name": "Current Experiment",
                "last_modified": datetime.fromtimestamp(summary_file.stat().st_mtime).isoformat(),
                "laps": len(df)
            }
        
        # Count research experiments
        if self.research_dir.exists():
            exp_count = 0
            for controller_dir in self.research_dir.iterdir():
                if controller_dir.is_dir():
                    exp_count += len([d for d in controller_dir.iterdir() if d.is_dir()])
            summary["experiments_count"] = exp_count + (1 if summary_file.exists() else 0)
        
        return summary

    def get_lap_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed lap analysis for experiment"""
        if experiment_id == "current_experiment":
            summary_file = self.data_dir / "evaluation_summary.csv"
            if not summary_file.exists():
                raise FileNotFoundError("No evaluation summary found")
            
            df = pd.read_csv(summary_file)
            
            analysis = {
                "lap_times": [],
                "consistency_analysis": {},
                "performance_trends": {},
                "outlier_detection": {}
            }
            
            # Lap-by-lap analysis
            for _, row in df.iterrows():
                analysis["lap_times"].append({
                    "lap": int(row['lap_number']),
                    "consistency": float(row['consistency']) if pd.notna(row['consistency']) else None,
                    "path_length": float(row['path_length']) if pd.notna(row['path_length']) else None,
                    "smoothness": float(row['smoothness']) if pd.notna(row['smoothness']) else None
                })
            
            # Consistency analysis
            if 'consistency' in df.columns:
                consistency_values = df['consistency'].dropna()
                analysis["consistency_analysis"] = {
                    "mean": float(consistency_values.mean()),
                    "std": float(consistency_values.std()),
                    "cv": float(consistency_values.std() / consistency_values.mean()) if consistency_values.mean() != 0 else 0,
                    "trend": self._calculate_trend(consistency_values.values)
                }
            
            # Performance trends
            for column in ['consistency', 'path_length', 'smoothness']:
                if column in df.columns:
                    values = df[column].dropna()
                    analysis["performance_trends"][column] = {
                        "trend": self._calculate_trend(values.values),
                        "improvement": float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0
                    }
            
            return analysis
        
        raise NotImplementedError(f"Lap analysis for {experiment_id} not yet implemented")

    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:  # Threshold for "stable"
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"

    def get_trajectory_plot_data(self, experiment_id: str) -> Dict[str, Any]:
        """Get trajectory visualization data"""
        plot_data = {
            "trajectories": [],
            "reference": None,
            "metadata": {}
        }
        
        if experiment_id == "current_experiment":
            # Load trajectory files
            trajectory_files = sorted(list(self.data_dir.glob("lap_*_trajectory.txt")))
            
            for traj_file in trajectory_files[:10]:  # Limit to first 10 laps for performance
                lap_num = self._extract_lap_number(traj_file.name)
                try:
                    # Assume trajectory files have x,y,z or x,y format
                    data = np.loadtxt(traj_file, delimiter=',')
                    if len(data.shape) == 1:
                        data = data.reshape(1, -1)
                    
                    trajectory = {
                        "lap": lap_num,
                        "name": f"Lap {lap_num}",
                        "x": data[:, 0].tolist() if data.shape[1] > 0 else [],
                        "y": data[:, 1].tolist() if data.shape[1] > 1 else [],
                        "z": data[:, 2].tolist() if data.shape[1] > 2 else None
                    }
                    plot_data["trajectories"].append(trajectory)
                except Exception as e:
                    print(f"Error loading trajectory {traj_file}: {e}")
            
            # Load reference trajectory if available
            ref_file = self.data_dir / "horizon_reference_trajectory.txt"
            if ref_file.exists():
                try:
                    ref_data = np.loadtxt(ref_file, delimiter=',')
                    if len(ref_data.shape) == 1:
                        ref_data = ref_data.reshape(1, -1)
                    
                    plot_data["reference"] = {
                        "name": "Reference Trajectory",
                        "x": ref_data[:, 0].tolist() if ref_data.shape[1] > 0 else [],
                        "y": ref_data[:, 1].tolist() if ref_data.shape[1] > 1 else [],
                        "z": ref_data[:, 2].tolist() if ref_data.shape[1] > 2 else None
                    }
                except Exception as e:
                    print(f"Error loading reference trajectory: {e}")
        
        return plot_data

    def get_performance_plot_data(self, experiment_id: str) -> Dict[str, Any]:
        """Get performance metrics visualization data"""
        if experiment_id == "current_experiment":
            summary_file = self.data_dir / "evaluation_summary.csv"
            if not summary_file.exists():
                return {"error": "No performance data available"}
            
            df = pd.read_csv(summary_file)
            
            plot_data = {
                "lap_numbers": df['lap_number'].tolist(),
                "metrics": {}
            }
            
            for column in ['consistency', 'path_length', 'smoothness']:
                if column in df.columns:
                    plot_data["metrics"][column] = {
                        "values": df[column].tolist(),
                        "mean": float(df[column].mean()),
                        "std": float(df[column].std())
                    }
            
            return plot_data
        
        return {"error": f"Performance data for {experiment_id} not available"}

    def get_comparison_plot_data(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Get comparison visualization for multiple experiments"""
        comparison_data = {
            "experiments": {},
            "metrics": ["consistency", "path_length", "smoothness"]
        }
        
        for exp_id in experiment_ids:
            try:
                exp_data = self.get_performance_plot_data(exp_id)
                if "error" not in exp_data:
                    comparison_data["experiments"][exp_id] = exp_data
            except:
                pass
        
        return comparison_data

    def export_experiment(self, experiment_id: str, format: str = "csv") -> Path:
        """Export experiment data in specified format"""
        if experiment_id == "current_experiment":
            summary_file = self.data_dir / "evaluation_summary.csv"
            if format == "csv" and summary_file.exists():
                return summary_file
            
            # Create export in requested format
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                export_path = self.data_dir / f"export_{experiment_id}.{format}"
                
                if format == "json":
                    df.to_json(export_path, orient='records', indent=2)
                elif format == "xlsx":
                    df.to_excel(export_path, index=False)
                else:  # Default to CSV
                    df.to_csv(export_path, index=False)
                
                return export_path
        
        raise FileNotFoundError(f"Cannot export {experiment_id} in {format} format")