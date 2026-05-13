"""
ROS2 bridge for the Race Monitor dashboard.

Runs a rclpy node in a background thread and maintains a thread-safe state dict
that the FastAPI layer can read at 10 Hz.  Service calls (reset / force-lap) are
executed via the ros2 CLI so they don't conflict with the spinning thread.
"""

import math
import os
import subprocess
import threading
from datetime import datetime
from typing import Any, Dict, Optional

# ── ROS2 optional import ──────────────────────────────────────────────────────
ROS_AVAILABLE = False
try:
    import rclpy                                        # type: ignore
    from rclpy.node import Node                         # type: ignore
    from std_msgs.msg import Bool, Float32, Int32, String  # type: ignore
    from nav_msgs.msg import Odometry                   # type: ignore

    # Optional camera support
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
        t = threading.Thread(target=self._spin, daemon=True, name="ros_bridge")
        t.start()

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

    # ── thread-safe state access ──────────────────────────────────────────────

    def _update(self, patch: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(patch)
            self._state["ts"] = datetime.now().timestamp()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # ── service calls via ros2 CLI ────────────────────────────────────────────

    def reset_race(self) -> Dict[str, Any]:
        result = self._ros2_call("/race_monitor/reset_race")
        if result["success"]:
            self._update({
                "lap_times": [], "lap_count": 0,
                "race_running": False, "race_status": "Waiting...",
            })
        return result

    def force_lap(self) -> Dict[str, Any]:
        return self._ros2_call("/race_monitor/force_lap_complete")

    def _ros2_call(self, service: str) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                ["ros2", "service", "call", service, "std_srvs/srv/Trigger", "{}"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0:
                return {"success": True, "message": "OK"}
            return {"success": False, "message": (proc.stderr or proc.stdout)[:200]}
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Service call timed out"}
        except FileNotFoundError:
            return {"success": False, "message": "ros2 CLI not found — is ROS2 sourced?"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}


# ── ROS2 node (only defined when rclpy is available) ─────────────────────────

if ROS_AVAILABLE:
    class _RaceNode(Node):  # type: ignore
        def __init__(self, bridge: RaceBridge):
            super().__init__("race_monitor_ui")
            self.bridge = bridge

            # Race monitor topic subscriptions
            self.create_subscription(Bool,    "/race_monitor/race_running", self._cb_running,   10)
            self.create_subscription(String,  "/race_monitor/race_status",  self._cb_status,    10)
            self.create_subscription(Int32,   "/race_monitor/lap_count",    self._cb_lap_count, 10)
            self.create_subscription(Float32, "/race_monitor/lap_time",     self._cb_lap_time,  10)

            # Odometry (configurable via env)
            odom_topic = os.getenv("ODOM_TOPIC", "/odom")
            self.create_subscription(Odometry, odom_topic, self._cb_odom, 10)

            # Optional camera
            cam_topic = os.getenv("CAMERA_TOPIC", "")
            if cam_topic and CAMERA_AVAILABLE:
                self.create_subscription(Image, cam_topic, self._cb_image, 1)

            # Heartbeat: check if race_monitor topics are active
            self.create_timer(2.0, self._check_connection)

        # ── topic callbacks ───────────────────────────────────────────────────

        def _check_connection(self) -> None:
            names = [n for n, _ in self.get_topic_names_and_types()]
            connected = any("/race_monitor/" in n for n in names)
            self.bridge._update({"race_monitor_connected": connected})

        def _cb_running(self, msg) -> None:
            self.bridge._update({"race_running": bool(msg.data)})

        def _cb_status(self, msg) -> None:
            self.bridge._update({"race_status": str(msg.data)})

        def _cb_lap_count(self, msg) -> None:
            self.bridge._update({"lap_count": int(msg.data)})

        def _cb_lap_time(self, msg) -> None:
            s = self.bridge.get_state()
            times = list(s.get("lap_times", []))
            times.append(round(float(msg.data), 3))
            if len(times) > 50:
                times = times[-50:]
            self.bridge._update({"lap_time": float(msg.data), "lap_times": times})

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
                h, w, c = msg.height, msg.width, 3
                img = arr.reshape((h, w, c))
                # Encode to JPEG
                from PIL import Image as PilImage
                pil = PilImage.fromarray(img if msg.encoding == "rgb8" else img[:, :, ::-1])
                import io
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=70)
                b64 = base64.b64encode(buf.getvalue()).decode()
                self.bridge._update({"camera_frame": b64})
            except Exception:
                pass
