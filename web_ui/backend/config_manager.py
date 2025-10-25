"""
Configuration Management Module

Manages loading, saving, and validating YAML configuration files for the race monitor.
Supports configuration templates, automatic backups, and structured configuration editing.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any
import shutil
from datetime import datetime

class ConfigManager:
    def __init__(self):
        self.config_dir = Path("../../config")
        self.backup_dir = Path("../../config/backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Templates for different use cases
        self.templates = {
            "basic_race": "Basic race monitoring configuration",
            "research_mode": "Research and analysis focused configuration", 
            "performance_testing": "Performance monitoring focused configuration",
            "evo_analysis": "EVO trajectory analysis configuration"
        }

    def list_configs(self) -> Dict[str, Any]:
        """
        List all available configuration files.
        
        Returns:
            Dictionary with configuration file metadata and templates
        """
        configs = []
        
        if self.config_dir.exists():
            for file_path in self.config_dir.glob("*.yaml"):
                if file_path.name != "backups":
                    stat = file_path.stat()
                    configs.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_template": file_path.name.startswith("template_")
                    })
        
        return {
            "configs": configs,
            "templates": self.templates,
            "config_dir": str(self.config_dir)
        }

    def get_config(self, filename: str) -> str:
        """
        Get configuration file content.
        
        Args:
            filename: Name of configuration file
            
        Returns:
            Configuration file content as string
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {filename} not found")
        
        with open(file_path, 'r') as f:
            return f.read()

    def save_config(self, filename: str, content: str) -> None:
        """
        Save configuration file with automatic backup.
        
        Args:
            filename: Name of configuration file
            content: YAML configuration content
            
        Raises:
            ValueError: If YAML content is invalid
        """
        file_path = self.config_dir / filename
        
        # Create backup if file exists
        if file_path.exists():
            self._create_backup(filename)
        
        # Validate YAML content
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML content: {str(e)}")
        
        # Save the file
        with open(file_path, 'w') as f:
            f.write(content)

    def delete_config(self, filename: str) -> None:
        """
        Delete configuration file with backup.
        
        Args:
            filename: Name of configuration file to delete
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {filename} not found")
        
        # Create backup before deletion
        self._create_backup(filename)
        file_path.unlink()

    def _create_backup(self, filename: str) -> None:
        """
        Create timestamped backup of configuration file.
        
        Args:
            filename: Name of configuration file to backup
        """
        file_path = self.config_dir / filename
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filename}.backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            shutil.copy2(file_path, backup_path)

    def get_config_structure(self, filename: str) -> Dict[str, Any]:
        """
        Get structured configuration data for form editing.
        
        Args:
            filename: Name of configuration file
            
        Returns:
            Dictionary with structured configuration parameters
        """
        content = self.get_config(filename)
        config_data = yaml.safe_load(content)
        
        # Extract race_monitor parameters
        if 'race_monitor' in config_data and 'ros__parameters' in config_data['race_monitor']:
            params = config_data['race_monitor']['ros__parameters']
            
            return {
                "basic_settings": {
                    "start_line_p1": params.get("start_line_p1", [0.0, -1.0]),
                    "start_line_p2": params.get("start_line_p2", [0.0, 1.0]),
                    "required_laps": params.get("required_laps", 5),
                    "debounce_time": params.get("debounce_time", 2.0),
                    "frame_id": params.get("frame_id", "map")
                },
                "race_ending": {
                    "race_ending_mode": params.get("race_ending_mode", "lap_complete"),
                    "crash_detection": params.get("crash_detection", {}),
                    "manual_mode": params.get("manual_mode", {})
                },
                "research_settings": {
                    "controller_name": params.get("controller_name", "custom_controller"),
                    "experiment_id": params.get("experiment_id", "exp_001"),
                    "test_description": params.get("test_description", ""),
                    "enable_advanced_metrics": params.get("enable_advanced_metrics", True)
                },
                "evo_integration": {
                    "enable_trajectory_evaluation": params.get("enable_trajectory_evaluation", True),
                    "evaluation_interval_laps": params.get("evaluation_interval_laps", 1),
                    "reference_trajectory_file": params.get("reference_trajectory_file", ""),
                    "pose_relations": params.get("pose_relations", []),
                    "statistics_types": params.get("statistics_types", [])
                },
                "advanced_settings": {
                    "trajectory_analysis": {
                        "save_trajectories": params.get("save_trajectories", True),
                        "trajectory_output_directory": params.get("trajectory_output_directory", ""),
                        "evaluate_smoothness": params.get("evaluate_smoothness", True),
                        "evaluate_consistency": params.get("evaluate_consistency", True)
                    },
                    "performance_monitoring": {
                        "enable_computational_monitoring": params.get("enable_computational_monitoring", True),
                        "monitoring_window_size": params.get("monitoring_window_size", 100),
                        "cpu_monitoring_interval": params.get("cpu_monitoring_interval", 0.1)
                    },
                    "visualization": {
                        "auto_generate_graphs": params.get("auto_generate_graphs", True),
                        "graph_output_directory": params.get("graph_output_directory", ""),
                        "graph_formats": params.get("graph_formats", ["png", "pdf"])
                    }
                },
                "raw_content": content
            }
        
        return {"raw_content": content}

    def save_structured_config(self, filename: str, structured_data: Dict[str, Any]) -> None:
        """
        Save configuration from structured form data.
        
        Args:
            filename: Name of configuration file
            structured_data: Structured configuration dictionary
            
        Raises:
            NotImplementedError: Structured data conversion not yet implemented
        """
        # This would convert the structured data back to YAML format
        # For now, we'll use the raw content if provided
        if "raw_content" in structured_data:
            self.save_config(filename, structured_data["raw_content"])
        else:
            # TODO: Implement structured data to YAML conversion
            raise NotImplementedError("Structured data conversion not yet implemented")

    def create_template(self, template_name: str) -> str:
        """
        Create a new configuration from template.
        
        Args:
            template_name: Name of template to use
            
        Returns:
            Template configuration content as string
        """
        templates = {
            "basic_race": self._get_basic_race_template(),
            "research_mode": self._get_research_template(),
            "performance_testing": self._get_performance_template(),
            "evo_analysis": self._get_evo_template()
        }
        
        return templates.get(template_name, templates["basic_race"])

    def _get_basic_race_template(self) -> str:
        """Basic race monitoring template"""
        return """race_monitor:
  ros__parameters:
    # Basic race configuration
    start_line_p1: [0.0, -1.0]
    start_line_p2: [0.0, 1.0] 
    required_laps: 5
    debounce_time: 2.0
    output_file: "race_results.csv"
    frame_id: "map"
    
    # Race ending
    race_ending_mode: "lap_complete"
    
    # Basic settings
    controller_name: "my_controller"
    experiment_id: "exp_001"
    enable_trajectory_evaluation: true
    evaluation_interval_laps: 1
"""

    def _get_research_template(self) -> str:
        """Research-focused template with full analysis"""
        # Return the current race_monitor.yaml as research template
        try:
            return self.get_config("race_monitor.yaml")
        except FileNotFoundError:
            return self._get_basic_race_template()

    def _get_performance_template(self) -> str:
        """Performance monitoring focused template"""
        return """race_monitor:
  ros__parameters:
    # Basic race configuration
    start_line_p1: [0.0, -1.0]
    start_line_p2: [0.0, 1.0]
    required_laps: 10
    debounce_time: 1.0
    output_file: "performance_results.csv"
    
    # Performance monitoring focus
    enable_computational_monitoring: true
    monitoring_window_size: 200
    cpu_monitoring_interval: 0.05
    enable_performance_logging: true
    performance_log_interval: 1.0
    
    # Detailed performance metrics
    max_acceptable_latency_ms: 25.0
    target_control_frequency_hz: 100.0
    max_acceptable_cpu_usage: 70.0
"""

    def _get_evo_template(self) -> str:
        """EVO analysis focused template"""
        return """race_monitor:
  ros__parameters:
    # Basic settings
    start_line_p1: [0.0, -1.0]
    start_line_p2: [0.0, 1.0]
    required_laps: 3
    output_file: "evo_analysis_results.csv"
    
    # EVO Integration
    enable_trajectory_evaluation: true
    evaluation_interval_laps: 1
    reference_trajectory_file: "horizon_mapper/horizon_mapper/ref_trajectory.csv"
    
    # Advanced EVO metrics
    pose_relations: ["translation_part", "rotation_part", "full_transformation"]
    statistics_types: ["rmse", "mean", "median", "std", "min", "max", "sse"]
    
    # Detailed analysis
    enable_advanced_metrics: true
    calculate_all_statistics: true
    analyze_rotation_errors: true
    enable_geometric_analysis: true
    
    # Graph generation
    auto_generate_graphs: true
    generate_trajectory_plots: true
    generate_error_plots: true
    generate_metrics_plots: true
"""