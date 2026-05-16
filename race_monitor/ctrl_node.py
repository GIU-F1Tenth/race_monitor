#!/usr/bin/env python3

"""
Race Monitor Control Node

Provides keyboard and joystick control over race_monitor services.
Supports configurable key/button mappings and optional input sources.
"""

import sys
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
        super().__init__('ctrl_node')

        self.declare_parameter('log_level', 'normal')
        self.declare_parameter('enable_keyboard', True)
        self.declare_parameter('enable_joy', True)
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('keyboard_bindings', {
            'r': 'reset_race',
            'f': 'force_lap_complete',
            'p': 'pause_race',
            'u': 'resume_race',
            't': 'reset_lap_time'
        })
        self.declare_parameter('joy_bindings', {
            'reset_race': 1,
            'force_lap_complete': 0,
            'pause_race': 2,
            'resume_race': 3,
            'reset_lap_time': 4
        })
        self.declare_parameter('reset_race_service', '/race_monitor/reset_race')
        self.declare_parameter('force_lap_complete_service', '/race_monitor/force_lap_complete')
        self.declare_parameter('pause_race_service', '/race_monitor/pause_race')
        self.declare_parameter('resume_race_service', '/race_monitor/resume_race')
        self.declare_parameter('reset_lap_time_service', '/race_monitor/reset_lap_time')

        log_level = str(self.get_parameter('log_level').value)
        self.logger = RaceMonitorLogger(self, "CtrlNode", log_level)

        self.enable_keyboard = bool(self.get_parameter('enable_keyboard').value)
        self.enable_joy = bool(self.get_parameter('enable_joy').value)
        self.joy_topic = str(self.get_parameter('joy_topic').value)

        self.keyboard_bindings = dict(self.get_parameter('keyboard_bindings').value)
        self.joy_bindings = dict(self.get_parameter('joy_bindings').value)

        self.service_names = {
            'reset_race': str(self.get_parameter('reset_race_service').value),
            'force_lap_complete': str(self.get_parameter('force_lap_complete_service').value),
            'pause_race': str(self.get_parameter('pause_race_service').value),
            'resume_race': str(self.get_parameter('resume_race_service').value),
            'reset_lap_time': str(self.get_parameter('reset_lap_time_service').value)
        }

        self._service_clients = {
            action: self.create_client(Trigger, service_name)
            for action, service_name in self.service_names.items()
        }

        self._last_buttons = []
        self._keyboard_thread = None
        self._keyboard_stop = threading.Event()
        self._keyboard_settings = None

        if self.enable_joy:
            self.create_subscription(Joy, self.joy_topic, self._joy_callback, 10)
            self.logger.info(f"Joy control enabled on {self.joy_topic}")

        if self.enable_keyboard:
            self._start_keyboard_listener()

        self.logger.info("Race monitor control node ready")

    def _start_keyboard_listener(self):
        if not sys.stdin.isatty():
            self.logger.warn("Keyboard control disabled: stdin is not a TTY")
            return

        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()
        self.logger.info("Keyboard control enabled")

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        self._keyboard_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        try:
            while rclpy.ok() and not self._keyboard_stop.is_set():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not rlist:
                    continue

                ch = sys.stdin.read(1)
                if not ch:
                    continue

                action = self.keyboard_bindings.get(ch.lower())
                if action:
                    self._invoke_action(action, source=f"key:{ch}")
        finally:
            if self._keyboard_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._keyboard_settings)

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
            self.logger.warn(f"Service unavailable for action {action}: {self.service_names.get(action)}")
            return

        request = Trigger.Request()
        future = client.call_async(request)
        future.add_done_callback(lambda fut: self._handle_response(action, source, fut))

    def _handle_response(self, action: str, source: str, future):
        try:
            response = future.result()
            if response.success:
                msg = response.message or "OK"
                self.logger.info(f"Action {action} ({source}) -> {msg}")
            else:
                msg = response.message or "Failed"
                self.logger.warn(f"Action {action} ({source}) failed: {msg}")
        except Exception as exc:
            self.logger.error(f"Action {action} ({source}) error", exception=exc)

    def destroy_node(self):
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
