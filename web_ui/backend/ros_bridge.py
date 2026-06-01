"""
ROS2 bridge for the Race Monitor dashboard.

Runs a rclpy node in a background thread and maintains a thread-safe state dict
that the FastAPI layer can read at 10 Hz.  Service calls are executed via the
ros2 CLI so they don't conflict with the spinning thread.
"""

import math
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

# ── ROS2 optional import ──────────────────────────────────────────────────────
ROS_AVAILABLE = False
try:
    import rclpy                                        # type: ignore
    from rclpy.node import Node                         # type: ignore
    from std_msgs.msg import Bool, Float32, Int32, String  # type: ignore
    from nav_msgs.msg import Odometry                   # type: ignore

    try:
        import base64
        import numpy as np
        from sensor_msgs.msg import Image               # type: ignore
        CAMERA_AVAILABLE = True
    except ImportError:
        CAMERA_AVAILABLE = False

    ROS_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False


_EMPTY_STATE: Dict[str, Any] = {
    "ros_connected": False,
    "race_monitor_connected": False,
    "race_running": False,
    "race_status": "Waiting...",
    "lap_count": 0,
    "lap_time": 0.0,
    "lap_times": [],
    "controller_name": "",
    "lap_timer_reset_ts": 0.0,   # bumped when reset_lap_time is called
    "position": None,
    "velocity": 0.0,
    "heading": 0.0,
    "camera_frame": None,
    "ts": 0.0,
}


class RaceBridge:
    """Thread-safe bridge between rclpy callbacks and the FastAPI event loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = dict(_EMPTY_STATE)
        self._node: Optional[Any] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not ROS_AVAILABLE:
            print("[RosBridge] rclpy not available — running in offline mode")
            return
        threading.Thread(target=self._spin, daemon=True, name="ros_bridge").start()
        threading.Thread(target=self._poll_controller_name, daemon=True, name="ctrl_name_poll").start()

    def _spin(self) -> None:
        try:
            rclpy.init(args=None)
            self._node = _RaceNode(self)
            self._update({"ros_connected": True})
            rclpy.spin(self._node)
        except Exception as exc:
            print(f"[RosBridge] spin error: {exc}")
        finally:
            self._update({"ros_connected": False, "race_monitor_connected": False})

    def _poll_controller_name(self) -> None:
        """Poll race_monitor parameter for controller_name every 5 s."""
        while True:
            time.sleep(5)
            try:
                proc = subprocess.run(
                    ["ros2", "param", "get", "/race_monitor", "controller_name"],
                    capture_output=True, text=True, timeout=3,
                )
                if proc.returncode == 0:
                    line = proc.stdout.strip()
                    if "value is:" in line:
                        name = line.split("value is:")[-1].strip().strip("'\"")
                        self._update({"controller_name": name})
            except Exception:
                pass

    # ── thread-safe state access ──────────────────────────────────────────────

    def _update(self, patch: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(patch)
            self._state["ts"] = datetime.now().timestamp()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # ── service calls ─────────────────────────────────────────────────────────

    def reset_race(self) -> Dict[str, Any]:
        result = self._ros2_call("/race_monitor/reset_race")
        if result["success"]:
            self._update({
                "lap_times": [], "lap_count": 0, "lap_time": 0.0,
                "race_running": False, "race_status": "WAITING",
                "lap_timer_reset_ts": time.time(),
            })
        return result

    def force_race_complete(self) -> Dict[str, Any]:
        result = self._ros2_call("/race_monitor/force_race_complete")
        if result["success"]:
            self._update({
                "lap_times": [], "lap_count": 0, "lap_time": 0.0,
                "race_running": False, "race_status": "WAITING",
                "lap_timer_reset_ts": time.time(),
            })
        return result

    def pause_race(self) -> Dict[str, Any]:
        return self._ros2_call("/race_monitor/pause_race")

    def resume_race(self) -> Dict[str, Any]:
        return self._ros2_call("/race_monitor/resume_race")

    def reset_lap_time(self) -> Dict[str, Any]:
        result = self._ros2_call("/race_monitor/reset_lap_time")
        if result["success"]:
            self._update({"lap_timer_reset_ts": time.time()})
        return result

    def _ros2_call(self, service: str) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                ["ros2", "service", "call", service, "std_srvs/srv/Trigger", "{}"],
                capture_output=True, text=True, timeout=5,
            )
            output = proc.stdout + proc.stderr
            if proc.returncode == 0:
                # Parse "message='...' " from ros2 service call output
                import re
                m = re.search(r"message='([^']*)'", output)
                msg = m.group(1) if m else "OK"
                return {"success": True, "message": msg}
            return {"success": False, "message": (proc.stderr or proc.stdout)[:200].strip()}
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Service call timed out"}
        except FileNotFoundError:
            return {"success": False, "message": "ros2 CLI not found — is ROS2 sourced?"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}


# ── ROS2 node ─────────────────────────────────────────────────────────────────

if ROS_AVAILABLE:
    class _RaceNode(Node):  # type: ignore
        def __init__(self, bridge: RaceBridge):
            super().__init__("race_monitor_ui")
            self.bridge = bridge

            self.create_subscription(Bool,    "/race_monitor/race_running", self._cb_running,   10)
            self.create_subscription(String,  "/race_monitor/race_status",  self._cb_status,    10)
            self.create_subscription(Int32,   "/race_monitor/lap_count",    self._cb_lap_count, 10)
            self.create_subscription(Float32, "/race_monitor/lap_time",     self._cb_lap_time,  10)

            odom_topics = os.getenv("ODOM_TOPIC", "/odom,/ego_racecar/odom,/car_state/odom").split(",")
            for odom_topic in odom_topics:
                self.create_subscription(Odometry, odom_topic.strip(), self._cb_odom, 10)

            cam_topic = os.getenv("CAMERA_TOPIC", "")
            if cam_topic and CAMERA_AVAILABLE:
                self.create_subscription(Image, cam_topic, self._cb_image, 1)

            self.create_timer(2.0, self._check_connection)
            self.create_timer(1.0, self._reconcile_lap_history)

        def _check_connection(self) -> None:
            node_names = self.get_node_names()
            connected = 'race_monitor' in node_names
            self.bridge._update({"race_monitor_connected": connected})

        def _reconcile_lap_history(self) -> None:
            """Catch the final lap missed when race ends naturally."""
            s = self.bridge.get_state()
            if s.get("race_running"):
                return
            times     = list(s.get("lap_times", []))
            last_t    = round(s.get("lap_time", 0.0), 3)
            lap_count = s.get("lap_count", 0)
            already_have = bool(times) and abs(times[-1] - last_t) < 0.001
            if (last_t > 0 and lap_count > 0
                    and len(times) < lap_count
                    and not already_have):
                times.append(last_t)
                self.bridge._update({"lap_times": times})

        def _cb_running(self, msg) -> None:
            s = self.bridge.get_state()
            was_running = s.get("race_running", False)
            running = bool(msg.data)

            if was_running and not running:
                # Transition to not-running (race ended OR paused).
                # Only append the final lap if the lap_time is genuinely new —
                # i.e. not already the last entry in times.
                # During a mid-lap pause, lap_time == times[-1] (last completed lap),
                # so the guard prevents a duplicate. On natural race end, lap_time
                # is the just-finished final lap and differs from times[-1].
                times     = list(s.get("lap_times", []))
                last_time = round(s.get("lap_time", 0.0), 3)
                lap_count = s.get("lap_count", 0)
                already_have = bool(times) and abs(times[-1] - last_time) < 0.001
                if (last_time > 0 and lap_count > 0
                        and len(times) < lap_count
                        and not already_have):
                    times.append(last_time)
                    self.bridge._update({"race_running": running, "lap_times": times})
                    return

            self.bridge._update({"race_running": running})

        def _cb_status(self, msg) -> None:
            self.bridge._update({"race_status": str(msg.data)})

        def _cb_lap_count(self, msg) -> None:
            self.bridge._update({"lap_count": int(msg.data)})

        def _cb_lap_time(self, msg) -> None:
            new_time = round(float(msg.data), 3)
            s = self.bridge.get_state()
            times = list(s.get("lap_times", []))
            lap_count = s.get("lap_count", 0)

            # race_monitor publishes lap_count = current lap being driven (1-indexed)
            # and lap_time = last *completed* lap's time.
            # So at any point: len(completed laps) == lap_count - 1.
            # Only append when our list is shorter than expected — prevents
            # duplicates from the topic firing on every odom tick.
            expected = max(0, lap_count - 1)
            if new_time > 0 and len(times) < expected:
                times.append(new_time)
                if len(times) > 50:
                    times = times[-50:]
                self.bridge._update({"lap_time": new_time, "lap_times": times})
            else:
                self.bridge._update({"lap_time": new_time})

        def _cb_odom(self, msg) -> None:
            p = msg.pose.pose.position
            t = msg.twist.twist.linear
            v = math.sqrt(t.x ** 2 + t.y ** 2)
            q = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
            )
            self.bridge._update({
                "position": {"x": round(p.x, 3), "y": round(p.y, 3)},
                "velocity": round(v, 3),
                "heading": round(yaw, 4),
            })

        def _cb_image(self, msg) -> None:
            try:
                arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                img = arr.reshape((msg.height, msg.width, 3))
                from PIL import Image as PilImage
                import io
                pil = PilImage.fromarray(img if msg.encoding == "rgb8" else img[:, :, ::-1])
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=70)
                b64 = base64.b64encode(buf.getvalue()).decode()
                self.bridge._update({"camera_frame": b64})
            except Exception:
                pass
