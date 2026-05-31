#!/usr/bin/env python3

"""
Race Monitor Control Node

Provides keyboard and joystick control over race_monitor services.
Supports configurable key/button mappings and optional input sources.
"""

import threading
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Joy
from .logger_utils import RaceMonitorLogger, LogLevel


class RaceMonitorControl(Node):
    """Control node for race_monitor services via keyboard or joystick."""

    def __init__(self):
        super().__init__('ctrl_node', automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('log_level', 'normal')
        self.declare_parameter('enable_keyboard', True)
        self.declare_parameter('enable_joy', True)
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('reset_race_service', '/race_monitor/reset_race')
        self.declare_parameter('force_lap_complete_service', '/race_monitor/force_lap_complete')
        self.declare_parameter('pause_race_service', '/race_monitor/pause_race')
        self.declare_parameter('resume_race_service', '/race_monitor/resume_race')
        self.declare_parameter('reset_lap_time_service', '/race_monitor/reset_lap_time')

        log_level = str(self.get_parameter('log_level').value)
        self.logger = RaceMonitorLogger(self, "CtrlNode", log_level)
        self.logger.startup("Race Monitor Control Node")

        self.enable_keyboard = bool(self.get_parameter('enable_keyboard').value)
        self.enable_joy = bool(self.get_parameter('enable_joy').value)
        self.joy_topic = str(self.get_parameter('joy_topic').value)

        # ROS2 doesn't support dict parameters — read nested keys via prefix
        kb = self.get_parameters_by_prefix('keyboard_bindings')
        self.keyboard_bindings = {k: v.value for k, v in kb.items()} if kb else {
            'r': 'reset_race', 'f': 'force_lap_complete', 'p': 'pause_race',
            'u': 'resume_race', 't': 'reset_lap_time'
        }
        joy = self.get_parameters_by_prefix('joy_bindings')
        self.joy_bindings = {k: v.value for k, v in joy.items()} if joy else {
            'reset_race': 1, 'force_lap_complete': 0, 'pause_race': 2,
            'resume_race': 3, 'reset_lap_time': 4
        }

        self.service_names = {
            'reset_race': str(self.get_parameter('reset_race_service').value),
            'force_lap_complete': str(self.get_parameter('force_lap_complete_service').value),
            'pause_race': str(self.get_parameter('pause_race_service').value),
            'resume_race': str(self.get_parameter('resume_race_service').value),
            'reset_lap_time': str(self.get_parameter('reset_lap_time_service').value)
        }

        self.logger.debug(f"Keyboard bindings: {dict(self.keyboard_bindings)}")
        self.logger.debug(f"Joy bindings: {dict(self.joy_bindings)}")

        self._service_clients = {
            action: self.create_client(Trigger, service_name)
            for action, service_name in self.service_names.items()
        }

        self._last_buttons = []
        self._keyboard_thread = None
        self._keyboard_stop = threading.Event()
        self._keyboard_settings = None
        self._tty_file = None

        if self.enable_joy:
            self.create_subscription(Joy, self.joy_topic, self._joy_callback, 10)
            self.logger.info(f"Joy control enabled on topic: {self.joy_topic}")

        if self.enable_keyboard:
            self._start_keyboard_listener()

        self.logger.success("Control node ready", LogLevel.NORMAL)

    def _start_keyboard_listener(self):
        # Open /dev/tty directly so keyboard input works even when stdin is
        # redirected (e.g. when launched via ros2 launch instead of ros2 run)
        try:
            self._tty_file = open('/dev/tty', 'rb', buffering=0)
        except OSError as e:
            self.logger.warn(f"Keyboard control disabled: cannot open /dev/tty: {e}")
            return

        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()
        self.logger.success("Keyboard control enabled", LogLevel.NORMAL)

    def _keyboard_loop(self):
        fd = self._tty_file.fileno()
        self._keyboard_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        try:
            while rclpy.ok() and not self._keyboard_stop.is_set():
                rlist, _, _ = select.select([self._tty_file], [], [], 0.1)
                if not rlist:
                    continue

                ch = self._tty_file.read(1)
                if not ch:
                    continue

                char = ch.decode('utf-8', errors='replace')
                action = self.keyboard_bindings.get(char.lower())
                if action:
                    self._invoke_action(action, source=f"key:{char}")
        finally:
            if self._keyboard_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._keyboard_settings)
            self._tty_file.close()
            self._tty_file = None

    def _joy_callback(self, msg: Joy):
        if not self.joy_bindings:
            return

        if not self._last_buttons:
            self._last_buttons = [0] * len(msg.buttons)

        for action, index in self.joy_bindings.items():
            if index < 0 or index >= len(msg.buttons):
                continue

            if msg.buttons[index] == 1 and self._last_buttons[index] == 0:
                self._invoke_action(action, source=f"joy:{index}")

        self._last_buttons = list(msg.buttons)

    def _invoke_action(self, action: str, source: str = ""):
        client = self._service_clients.get(action)
        if client is None:
            self.logger.warn(f"Unknown action: {action}")
            return

        if not client.wait_for_service(timeout_sec=0.2):
            self.logger.warn(
                f"Service unavailable for action '{action}': {self.service_names.get(action)}",
                LogLevel.NORMAL
            )
            return

        self.logger.event(action, f"triggered by {source}", LogLevel.NORMAL)
        request = Trigger.Request()
        future = client.call_async(request)
        future.add_done_callback(lambda fut: self._handle_response(action, source, fut))

    def _handle_response(self, action: str, source: str, future):
        try:
            response = future.result()
            if response.success:
                self.logger.success(
                    f"{action} -> {response.message or 'OK'}", LogLevel.NORMAL)
            else:
                self.logger.warn(
                    f"{action} failed: {response.message or 'no message'}", LogLevel.NORMAL)
        except Exception as exc:
            self.logger.error(f"Service call failed for action '{action}' ({source})", exception=exc)

    def destroy_node(self):
        self.logger.shutdown("Race Monitor Control Node")
        self._keyboard_stop.set()
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=1.0)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RaceMonitorControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
