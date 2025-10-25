"""
Centralized logging utility for Race Monitor.

Provides consistent, professional logging across all modules with configurable
verbosity levels and structured message formatting.
"""

from enum import Enum
from typing import Optional
from rclpy.node import Node


class LogLevel(Enum):
    """Logging verbosity levels."""
    MINIMAL = "minimal"  # Only critical events and final results
    NORMAL = "normal"    # Standard operational messages
    DEBUG = "debug"      # Detailed diagnostic information
    VERBOSE = "verbose"  # Maximum detail for troubleshooting


class RaceMonitorLogger:
    """
    Professional logging wrapper for Race Monitor components.
    
    Provides consistent, structured logging with context-aware messages
    and configurable verbosity levels.
    """

    def __init__(self, node: Node, component_name: str, log_level: str = "normal"):
        """
        Initialize logger for a specific component.
        
        Args:
            node: ROS2 node instance
            component_name: Name of the component (e.g., "LapDetector", "TrajectoryAnalyzer")
            log_level: Logging verbosity ("minimal", "normal", "debug", "verbose")
        """
        self.node = node
        self.component = component_name
        self._set_log_level(log_level)

    def _set_log_level(self, level: str):
        """Set and validate the logging level."""
        level_lower = level.lower()
        try:
            self.log_level = LogLevel(level_lower)
        except ValueError:
            self.node.get_logger().warn(
                f"[{self.component}] Invalid log level '{level}'. Using 'normal'."
            )
            self.log_level = LogLevel.NORMAL

    def _format_message(self, message: str, prefix: str = "") -> str:
        """Format message with component context."""
        component_tag = f"[{self.component}]"
        if prefix:
            return f"{component_tag} {prefix} {message}"
        return f"{component_tag} {message}"

    def _should_log(self, required_level: LogLevel) -> bool:
        """Check if message should be logged based on current level."""
        level_order = {
            LogLevel.MINIMAL: 0,
            LogLevel.NORMAL: 1,
            LogLevel.DEBUG: 2,
            LogLevel.VERBOSE: 3
        }
        return level_order[self.log_level] >= level_order[required_level]

    # Critical/Error messages - Always logged
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message (always shown)."""
        formatted = self._format_message(message, "❌ ERROR:")
        if exception:
            formatted += f" | Exception: {str(exception)}"
        self.node.get_logger().error(formatted)

    def critical(self, message: str):
        """Log critical message (always shown)."""
        self.node.get_logger().error(
            self._format_message(message, "🚨 CRITICAL:")
        )

    def warn(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log warning message."""
        if self._should_log(level):
            self.node.get_logger().warn(
                self._format_message(message, "⚠️  WARNING:")
            )

    # Informational messages - Level dependent
    def info(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log informational message."""
        if self._should_log(level):
            self.node.get_logger().info(self._format_message(message))

    def success(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log success message."""
        if self._should_log(level):
            self.node.get_logger().info(
                self._format_message(message, "✓")
            )

    def debug(self, message: str):
        """Log debug message (only in debug/verbose mode)."""
        if self._should_log(LogLevel.DEBUG):
            self.node.get_logger().info(
                self._format_message(message, "🔍 DEBUG:")
            )

    def verbose(self, message: str):
        """Log verbose message (only in verbose mode)."""
        if self._should_log(LogLevel.VERBOSE):
            self.node.get_logger().info(
                self._format_message(message, "📝 VERBOSE:")
            )

    # Specialized logging methods
    def event(self, event_name: str, details: str = "", level: LogLevel = LogLevel.NORMAL):
        """Log an event with optional details."""
        if self._should_log(level):
            message = f"📍 EVENT: {event_name}"
            if details:
                message += f" | {details}"
            self.node.get_logger().info(self._format_message(message))

    def metric(self, metric_name: str, value: any, unit: str = "", level: LogLevel = LogLevel.NORMAL):
        """Log a metric value."""
        if self._should_log(level):
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            message = f"-> {metric_name}: {value_str}"
            if unit:
                message += f" {unit}"
            self.node.get_logger().info(self._format_message(message))

    def status(self, status: str, level: LogLevel = LogLevel.NORMAL):
        """Log status update."""
        if self._should_log(level):
            self.node.get_logger().info(
                self._format_message(status, "🔄 STATUS:")
            )

    def progress(self, current: int, total: int, description: str = "", level: LogLevel = LogLevel.NORMAL):
        """Log progress information."""
        if self._should_log(level):
            percentage = (current / total * 100) if total > 0 else 0
            message = f"⏳ PROGRESS: {current}/{total} ({percentage:.1f}%)"
            if description:
                message += f" | {description}"
            self.node.get_logger().info(self._format_message(message))

    def section(self, section_name: str, level: LogLevel = LogLevel.NORMAL):
        """Log section header for better organization."""
        if self._should_log(level):
            self.node.get_logger().info(
                self._format_message(f"{'='*50}")
            )
            self.node.get_logger().info(
                self._format_message(f"  {section_name}")
            )
            self.node.get_logger().info(
                self._format_message(f"{'='*50}")
            )

    def startup(self, message: str):
        """Log startup message (always shown)."""
        self.node.get_logger().info(
            self._format_message(message, "STARTUP:")
        )

    def shutdown(self, message: str):
        """Log shutdown message (always shown)."""
        self.node.get_logger().info(
            self._format_message(message, "🛑 SHUTDOWN:")
        )

    def config(self, parameter: str, value: any, level: LogLevel = LogLevel.DEBUG):
        """Log configuration parameter."""
        if self._should_log(level):
            self.node.get_logger().info(
                self._format_message(f"⚙️  CONFIG: {parameter} = {value}")
            )
