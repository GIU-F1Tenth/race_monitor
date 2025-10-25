"""
Live Monitor Module

Provides real-time monitoring of race data through ROS2 integration.
Supports live data streaming, system health monitoring, and WebSocket communication.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess
import os
from pathlib import Path

class LiveMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.ros_available = False
        self.current_data = {}
        self.data_dir = Path("../../race_monitor/evaluation_results")
        
        # Check ROS2 availability
        self._check_ros2_availability()

    def _check_ros2_availability(self):
        """Check if ROS2 is installed and environment is sourced."""
        try:
            # Check if ros2 command is available
            result = subprocess.run(['which', 'ros2'], capture_output=True, text=True)
            if result.returncode == 0:
                # Check if ROS2 is properly sourced
                env_result = subprocess.run(['ros2', 'node', 'list'], 
                                          capture_output=True, text=True, timeout=5)
                self.ros_available = env_result.returncode == 0
            else:
                self.ros_available = False
        except Exception:
            self.ros_available = False

    async def get_live_data(self) -> Dict[str, Any]:
        """
        Get current live monitoring data.
        
        Returns:
            Dictionary with race status, system health, and recent activity
        """
        live_data = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.is_monitoring,
            "ros_available": self.ros_available,
            "race_status": await self._get_race_status(),
            "system_health": await self._get_system_health(),
            "recent_activity": await self._get_recent_activity()
        }
        
        if self.is_monitoring and self.ros_available:
            live_data.update(await self._get_ros_data())
        
        return live_data

    async def _get_race_status(self) -> Dict[str, Any]:
        """
        Get current race status from ROS2.
        
        Returns:
            Dictionary with race activity and lap information
        """
        status = {
            "race_active": False,
            "current_lap": 0,
            "total_laps": 0,
            "last_lap_time": None,
            "current_position": None
        }
        
        # Check if race monitor node is active
        if self.ros_available:
            try:
                result = subprocess.run(['ros2', 'node', 'list'], 
                                      capture_output=True, text=True, timeout=3)
                if 'race_monitor' in result.stdout:
                    status["race_active"] = True
                    
                    # Try to get topic info
                    topic_result = subprocess.run(['ros2', 'topic', 'list'], 
                                                capture_output=True, text=True, timeout=3)
                    if '/race_monitor' in topic_result.stdout:
                        status["topics_available"] = True
            except Exception as e:
                print(f"Error checking race status: {e}")
        
        return status

    async def _get_system_health(self) -> Dict[str, Any]:
        """
        Get system health information.
        
        Returns:
            Dictionary with CPU, memory, disk usage, and ROS2 nodes/topics
        """
        health = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_space": 0,
            "ros2_nodes": [],
            "active_topics": []
        }
        
        try:
            # Get CPU and memory usage
            import psutil
            health["cpu_usage"] = psutil.cpu_percent(interval=0.1)
            health["memory_usage"] = psutil.virtual_memory().percent
            health["disk_space"] = psutil.disk_usage('/').percent
        except ImportError:
            # Fallback to system commands
            try:
                # Get CPU usage
                cpu_result = subprocess.run(['top', '-bn1'], capture_output=True, text=True, timeout=2)
                if cpu_result.returncode == 0:
                    lines = cpu_result.stdout.split('\n')
                    for line in lines:
                        if 'Cpu(s):' in line:
                            # Parse CPU usage from top output
                            import re
                            match = re.search(r'(\d+\.\d+)%us', line)
                            if match:
                                health["cpu_usage"] = float(match.group(1))
            except Exception:
                pass
        
        # Get ROS2 nodes and topics if available
        if self.ros_available:
            try:
                nodes_result = subprocess.run(['ros2', 'node', 'list'], 
                                            capture_output=True, text=True, timeout=3)
                if nodes_result.returncode == 0:
                    health["ros2_nodes"] = [node.strip() for node in nodes_result.stdout.split('\n') if node.strip()]
                
                topics_result = subprocess.run(['ros2', 'topic', 'list'], 
                                             capture_output=True, text=True, timeout=3)
                if topics_result.returncode == 0:
                    health["active_topics"] = [topic.strip() for topic in topics_result.stdout.split('\n') if topic.strip()]
            except Exception as e:
                print(f"Error getting ROS2 info: {e}")
        
        return health

    async def _get_recent_activity(self) -> Dict[str, Any]:
        """
        Get recent file system activity.
        
        Returns:
            Dictionary with recent files and modification timestamps
        """
        activity = {
            "recent_files": [],
            "file_changes": 0,
            "last_update": None
        }
        
        try:
            # Check for recent files in evaluation results
            if self.data_dir.exists():
                recent_files = []
                for file_path in self.data_dir.rglob("*"):
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        recent_files.append({
                            "name": file_path.name,
                            "path": str(file_path.relative_to(self.data_dir)),
                            "modified": mtime.isoformat(),
                            "size": file_path.stat().st_size
                        })
                
                # Sort by modification time, get latest 10
                recent_files.sort(key=lambda x: x["modified"], reverse=True)
                activity["recent_files"] = recent_files[:10]
                activity["file_changes"] = len(recent_files)
                
                if recent_files:
                    activity["last_update"] = recent_files[0]["modified"]
        
        except Exception as e:
            print(f"Error getting recent activity: {e}")
        
        return activity

    async def _get_ros_data(self) -> Dict[str, Any]:
        """
        Get live ROS2 topic data.
        
        Returns:
            Dictionary with latest messages from key ROS2 topics
        """
        ros_data = {
            "odometry": None,
            "vehicle_state": None,
            "lap_timing": None,
            "performance_metrics": None
        }
        
        if not self.ros_available:
            return ros_data
        
        try:
            # Try to echo recent messages from key topics
            topics_to_check = [
                ("/car_state/odom", "odometry"),
                ("/vehicle_state", "vehicle_state"),
                ("/race_monitor/lap_time", "lap_timing"),
                ("/race_monitor/performance", "performance_metrics")
            ]
            
            for topic, key in topics_to_check:
                try:
                    # Use timeout to avoid blocking
                    result = subprocess.run(['ros2', 'topic', 'echo', topic, '--once'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0 and result.stdout.strip():
                        ros_data[key] = {
                            "available": True,
                            "last_message_preview": result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        ros_data[key] = {"available": False}
                except subprocess.TimeoutExpired:
                    ros_data[key] = {"available": False, "status": "timeout"}
                except Exception as e:
                    ros_data[key] = {"available": False, "error": str(e)}
        
        except Exception as e:
            print(f"Error getting ROS data: {e}")
        
        return ros_data

    async def start(self) -> Dict[str, Any]:
        """
        Start live monitoring.
        
        Returns:
            Dictionary with monitoring status and warnings
        """
        if self.is_monitoring:
            return {"status": "already_running"}
        
        self.is_monitoring = True
        
        # Check system readiness
        status = {
            "monitoring_started": True,
            "timestamp": datetime.now().isoformat(),
            "ros_available": self.ros_available,
            "warnings": []
        }
        
        if not self.ros_available:
            status["warnings"].append("ROS2 not available - limited monitoring functionality")
        
        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())
        
        return status

    async def stop(self) -> None:
        """Stop live monitoring"""
        self.is_monitoring = False

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Update current data
                self.current_data = await self.get_live_data()
                
                # Sleep for 1 second
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error

    def get_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.
        
        Returns:
            Dictionary with monitoring state and capabilities
        """
        return {
            "monitoring_active": self.is_monitoring,
            "ros_available": self.ros_available,
            "last_update": datetime.now().isoformat(),
            "capabilities": {
                "real_time_data": self.ros_available,
                "file_monitoring": True,
                "system_health": True,
                "race_status": self.ros_available
            }
        }

    async def get_topic_info(self, topic_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific ROS2 topic.
        
        Args:
            topic_name: Name of the ROS2 topic
            
        Returns:
            Dictionary with topic info, type, and recent message
        """
        if not self.ros_available:
            return {"error": "ROS2 not available"}
        
        try:
            # Get topic info
            info_result = subprocess.run(['ros2', 'topic', 'info', topic_name], 
                                       capture_output=True, text=True, timeout=3)
            
            # Get topic type
            type_result = subprocess.run(['ros2', 'topic', 'type', topic_name], 
                                       capture_output=True, text=True, timeout=3)
            
            # Get recent message
            echo_result = subprocess.run(['ros2', 'topic', 'echo', topic_name, '--once'], 
                                       capture_output=True, text=True, timeout=5)
            
            return {
                "topic": topic_name,
                "info": info_result.stdout if info_result.returncode == 0 else "Not available",
                "type": type_result.stdout.strip() if type_result.returncode == 0 else "Unknown",
                "recent_message": echo_result.stdout if echo_result.returncode == 0 else "No recent message",
                "available": info_result.returncode == 0
            }
            
        except Exception as e:
            return {"error": f"Failed to get topic info: {str(e)}"}

    async def trigger_emergency_stop(self) -> Dict[str, Any]:
        """
        Trigger emergency stop via ROS2 topic.
        
        Returns:
            Dictionary with emergency stop status
        """
        if not self.ros_available:
            return {"error": "ROS2 not available"}
        
        try:
            # Try to publish emergency stop message
            result = subprocess.run(['ros2', 'topic', 'pub', '--once', 
                                   '/race_monitor/emergency_stop', 
                                   'std_msgs/msg/Bool', 
                                   'data: true'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return {"status": "emergency_stop_sent", "timestamp": datetime.now().isoformat()}
            else:
                return {"error": "Failed to send emergency stop", "details": result.stderr}
                
        except Exception as e:
            return {"error": f"Emergency stop failed: {str(e)}"}