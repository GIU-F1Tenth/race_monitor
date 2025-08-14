#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, Bool, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

import numpy as np
import csv
import os
from datetime import datetime
import math


class RaceMonitor(Node):
    """Race Monitor node

    - Counts laps using a start/finish line set in parameters or by clicking two points in RViz
    - Tracks lap times, best/worst/average, total race time
    - Publishes live lap count and race_running flag
    - Saves results to CSV when race completes (or on shutdown)
    - Publishes a nicer visualization of the start/finish line (checkered pattern + center line)
    """

    def __init__(self):
        super().__init__('race_monitor',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.get_logger().info('Race Monitor node started')

        # Declare parameters with defaults (only if not already declared)
        if not self.has_parameter('start_line_p1'):
            self.declare_parameter('start_line_p1', [0.0, -1.0])  # Bottom of vertical line
        if not self.has_parameter('start_line_p2'):
            self.declare_parameter('start_line_p2', [0.0, 1.0])   # Top of vertical line
        if not self.has_parameter('required_laps'):
            self.declare_parameter('required_laps', 5)
        if not self.has_parameter('debounce_time'):
            self.declare_parameter('debounce_time', 2.0)
        if not self.has_parameter('output_file'):
            self.declare_parameter('output_file', 'race_results.csv')
        if not self.has_parameter('frame_id'):
            self.declare_parameter('frame_id', 'map')

        # Read parameters
        self.start_line_p1 = np.array(self.get_parameter('start_line_p1').value, dtype=float)
        self.start_line_p2 = np.array(self.get_parameter('start_line_p2').value, dtype=float)
        self.required_laps = int(self.get_parameter('required_laps').value)
        self.debounce_time = float(self.get_parameter('debounce_time').value)
        self.output_file = str(self.get_parameter('output_file').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        self.get_logger().info(f"Initial start line: P1={self.start_line_p1}, P2={self.start_line_p2}")
        self.get_logger().info(f"Required laps: {self.required_laps}, Debounce: {self.debounce_time}s")

        # Race state
        self.lap_count = 0
        self.race_running = False
        self.race_started = False
        self.lap_times = []
        self.race_start_time = None
        self.lap_start_time = None
        self.last_crossing_time = None
        self.car_running = False
        self.race_finished = False  # Add this flag

        # Position tracking
        self.current_position = np.array([0.0, 0.0])
        self.last_position = np.array([0.0, 0.0])
        self.current_heading = 0.0
        self.position_initialized = False

        # Clicked point handling
        self.pending_point = None  # holds first clicked point until second is given

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, 'car_state/odom', self.odom_callback, 20)
        self.clicked_point_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.clicked_point_callback, 10)

        # Publishers
        self.lap_count_pub = self.create_publisher(Int32, '/race_monitor/lap_count', 10)
        self.lap_time_pub = self.create_publisher(Float32, '/race_monitor/lap_time', 10)
        self.best_lap_time_pub = self.create_publisher(Float32, '/race_monitor/best_lap_time', 10)
        self.race_running_pub = self.create_publisher(Bool, '/race_monitor/race_running', 10)
        self.start_line_marker_pub = self.create_publisher(Marker, '/race_monitor/start_line_marker', 10)
        self.race_state_pub = self.create_publisher(String, '/race_monitor/state', 10)

        # Timers
        self.status_timer = self.create_timer(0.1, self.publish_race_status)
        self.marker_timer = self.create_timer(1.0, self.publish_start_line_marker)

        # Remember last published line to avoid re-publishing identical markers unnecessarily
        self._last_line = (None, None)

        self.get_logger().info('Race Monitor initialized. Use RViz Publish Point to set start/finish line (click two points).')

    # --------------------------- Callbacks ---------------------------------
    def odom_callback(self, msg: Odometry):
        """Handle odometry; update position and heading, then check for crossings."""
        pos = msg.pose.pose.position
        self.last_position = self.current_position.copy()
        self.current_position = np.array([pos.x, pos.y], dtype=float)

        orientation = msg.pose.pose.orientation
        self.current_heading = self.quaternion_to_yaw(orientation)

        self.car_running = True

        if not self.position_initialized:
            self.position_initialized = True
            self.last_position = self.current_position.copy()
            return

        # Check if car is stopped (velocity magnitude is zero)
        linear = msg.twist.twist.linear
        velocity_mag = math.sqrt(linear.x**2 + linear.y**2 + linear.z**2)
        if velocity_mag == 0 and self.lap_count > 0:
            import time
            time.sleep(3)
            linear = msg.twist.twist.linear
            velocity_mag = math.sqrt(linear.x**2 + linear.y**2 + linear.z**2)

            if velocity_mag == 0:
                self.car_running = False
                self.finish_race()

        self.check_lap_crossing()

    def clicked_point_callback(self, msg: PointStamped):
        """Receive points from RViz Publish Point tool and set start/finish line.

        Behavior:
        - First click sets pending_point (P1)
        - Second click sets P2 and updates stored line immediately
        - If user wants to set only one point and keep previous other point, they may click once and wait
        """
        point = np.array([msg.point.x, msg.point.y], dtype=float)

        if self.pending_point is None:
            self.pending_point = point
            self.get_logger().info(
                f"Pending start/finish point set to ({point[0]:.3f}, {point[1]:.3f}). Click second point to complete the line.")
        else:
            # Use previous P2 if pending_point is only update for P1 -- but here we set both
            new_p1 = self.pending_point
            new_p2 = point
            self.start_line_p1 = new_p1
            self.start_line_p2 = new_p2
            self.pending_point = None
            self.get_logger().info(
                f"Start/finish line updated to P1=({new_p1[0]:.3f},{new_p1[1]:.3f}) P2=({new_p2[0]:.3f},{new_p2[1]:.3f})")

            # reset race start if user wants (optional): here we keep state but you can re-zero lap_count if desired
            # self.lap_count = 0
            # self.lap_times = []

            # Immediately publish updated marker
            self.publish_start_line_marker()

    # --------------------------- Lap detection ------------------------------
    def check_lap_crossing(self):
        if not self.position_initialized:
            return

        # If line is degenerate, skip
        if np.allclose(self.start_line_p1, self.start_line_p2):
            return

        # Check intersection between the segment traveled since last odom and the start line
        if self.line_intersection(self.last_position, self.current_position, self.start_line_p1, self.start_line_p2):
            now = self.get_clock().now()

            # debounce
            if self.last_crossing_time is not None:
                elapsed = (now - self.last_crossing_time).nanoseconds / 1e9
                if elapsed < self.debounce_time:
                    return

            # direction check
            if not self.heading_check(self.current_heading, self.start_line_p2 - self.start_line_p1):
                # return
                pass

            # record crossing
            self.last_crossing_time = now

            if not self.race_started:
                # Start race on first valid crossing
                self.race_started = True
                self.race_running = True
                self.race_start_time = now
                self.lap_start_time = now
                self.get_logger().info('Race started')
            else:
                # complete lap
                self.complete_lap(now)

    def complete_lap(self, current_time):
        if self.lap_start_time is None:
            return

        lap_time = (current_time - self.lap_start_time).nanoseconds / 1e9
        self.lap_times.append(lap_time)
        self.lap_count += 1

        self.get_logger().info(f'Lap {self.lap_count} completed in {lap_time:.3f}s')

        # Publish lap time and best lap
        self.publish_value(self.lap_time_pub, float(lap_time))
        best = float(min(self.lap_times))
        self.publish_value(self.best_lap_time_pub, best)

        self.lap_start_time = current_time

        if self.lap_count == self.required_laps or not self.car_running:
            self.finish_race()

    def finish_race(self):
        # Prevent running twice
        if self.race_finished:
            return
        self.race_finished = True
        self.race_running = False
        if self.lap_count > self.required_laps:
            return
        total_time = 0.0
        if self.race_start_time is not None:
            total_time = (self.get_clock().now() - self.race_start_time).nanoseconds / 1e9

        if len(self.lap_times) > 0:
            best = min(self.lap_times)
            worst = max(self.lap_times)
            avg = float(np.mean(self.lap_times))
        else:
            best = worst = avg = 0.0

        self.get_logger().info(f'Race finished. Total time: {total_time:.3f}s')
        self.get_logger().info(f'Best: {best:.3f}s Worst: {worst:.3f}s Avg: {avg:.3f}s')

        # save results
        self.save_results_to_csv(total_time)

    # --------------------------- Utilities ---------------------------------
    def publish_value(self, pub, value):
        msg_type = pub.msg_type if hasattr(pub, 'msg_type') else None
        # Workaround: create Float32 or Int32 depending on publisher
        if pub is self.best_lap_time_pub or pub is self.lap_time_pub:
            m = Float32()
            m.data = float(value)
            pub.publish(m)
        elif pub is self.lap_count_pub:
            m = Int32()
            m.data = int(value)
            pub.publish(m)

    def publish_race_status(self):
        # lap count
        lap_msg = Int32()
        lap_msg.data = int(self.lap_count)
        self.lap_count_pub.publish(lap_msg)

        # race running
        rmsg = Bool()
        rmsg.data = bool(self.race_running)
        self.race_running_pub.publish(rmsg)

        # best lap publish even if not set
        if len(self.lap_times) > 0:
            best_msg = Float32()
            best_msg.data = float(min(self.lap_times))
            self.best_lap_time_pub.publish(best_msg)

        # also publish string state
        state = String()
        state.data = 'running' if self.race_running else ('staged' if self.race_started else 'idle')
        self.race_state_pub.publish(state)

    def line_intersection(self, p1, p2, q1, q2):
        """Return True if segment p1-p2 intersects q1-q2."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A = p1
        B = p2
        C = q1
        D = q2
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

    def heading_check(self, heading, line_vector):
        """Check that vehicle heading points roughly in the same direction as the line's normal.

        We define the "forward" direction of the line as the normal (perpendicular) pointing from P1->P2 rotated +90deg.
        Crossing is valid if the dot(heading_vector, line_normal) > 0.
        """
        lv = np.array(line_vector, dtype=float)
        if np.linalg.norm(lv) == 0.0:
            return False
        line_normal = np.array([-lv[1], lv[0]])
        norm = np.linalg.norm(line_normal)
        if norm == 0.0:
            return False
        line_normal = line_normal / norm

        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        dot = float(np.dot(heading_vector, line_normal))
        return dot > 0.0

    def quaternion_to_yaw(self, q):
        # yaw (z) from quaternion
        x, y, z, w = q.x, q.y, q.z, q.w
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)

    # --------------------------- Visualization ------------------------------
    def publish_start_line_marker(self):
        """Publish a simple colored line visualization for the start/finish line.

        Markers are published in frame self.frame_id (default 'map').
        """
        # Avoid republishing if identical
        if self._last_line[0] is not None and np.allclose(
                self._last_line[0], self.start_line_p1) and np.allclose(
                self._last_line[1], self.start_line_p2):
            return

        # store
        self._last_line = (self.start_line_p1.copy(), self.start_line_p2.copy())

        # Delete existing markers first
        clear = Marker()
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.header.frame_id = self.frame_id
        clear.ns = 'race_monitor'
        clear.action = Marker.DELETEALL
        self.start_line_marker_pub.publish(clear)

        # Create a sleek green line (LINE_STRIP)
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = self.frame_id
        line_marker.ns = 'race_monitor'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.15  # Reduced thickness (15cm) for cleaner look

        # Bright green color for start/finish line with reduced opacity
        line_marker.color.r = 0.0
        line_marker.color.g = 0.8
        line_marker.color.b = 0.2
        line_marker.color.a = 0.6  # More transparent for subtle appearance

        # Add two points (endpoints) to the line strip
        from geometry_msgs.msg import Point
        pt1 = Point()
        pt1.x = float(self.start_line_p1[0])
        pt1.y = float(self.start_line_p1[1])
        pt1.z = 0.02  # Lower to ground for more realistic appearance

        pt2 = Point()
        pt2.x = float(self.start_line_p2[0])
        pt2.y = float(self.start_line_p2[1])
        pt2.z = 0.02  # Lower to ground for more realistic appearance

        line_marker.points = [pt1, pt2]
        self.start_line_marker_pub.publish(line_marker)

        # Add subtle endpoint markers for better visibility
        for i, point in enumerate([self.start_line_p1, self.start_line_p2]):
            endpoint_marker = Marker()
            endpoint_marker.header.stamp = self.get_clock().now().to_msg()
            endpoint_marker.header.frame_id = self.frame_id
            endpoint_marker.ns = 'race_monitor'
            endpoint_marker.id = i + 1
            endpoint_marker.type = Marker.SPHERE
            endpoint_marker.action = Marker.ADD

            # Position
            endpoint_marker.pose.position.x = float(point[0])
            endpoint_marker.pose.position.y = float(point[1])
            endpoint_marker.pose.position.z = 0.03
            endpoint_marker.pose.orientation.w = 1.0

            # Small sphere size
            endpoint_marker.scale.x = 0.1
            endpoint_marker.scale.y = 0.1
            endpoint_marker.scale.z = 0.1

            # Slightly darker green for endpoints with reduced opacity
            endpoint_marker.color.r = 0.0
            endpoint_marker.color.g = 0.6
            endpoint_marker.color.b = 0.1
            endpoint_marker.color.a = 0.5  # More transparent

            self.start_line_marker_pub.publish(endpoint_marker)

        # Calculate line length for logging
        line_length = np.linalg.norm(self.start_line_p2 - self.start_line_p1)
        self.get_logger().info(f"Published start line (length: {line_length:.2f}m) in frame '{self.frame_id}'")

    def save_results_to_csv(self, total_race_time):
        try:
            data_dir = os.path.join(os.getcwd(), 'race_monitor', 'data')
            os.makedirs(data_dir, exist_ok=True)
            filename = self.output_file
            filepath = os.path.join(data_dir, filename)

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['lap_number', 'lap_time_s'])
                for i, t in enumerate(self.lap_times, start=1):
                    writer.writerow([i, f"{t:.4f}"])

                writer.writerow(['total_time_s', f"{total_race_time:.4f}"])
                if len(self.lap_times) > 0:
                    writer.writerow(['best_lap_s', f"{min(self.lap_times):.4f}"])
                    writer.writerow(['worst_lap_s', f"{max(self.lap_times):.4f}"])
                    writer.writerow(['average_lap_s', f"{np.mean(self.lap_times):.4f}"])

            self.get_logger().info(f"Saved race results to: {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error saving CSV: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RaceMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received — shutting down')
        # Save partial results if running or any laps recorded
        if node.race_started and len(node.lap_times) > 0:
            total = 0.0
            if node.race_start_time is not None:
                total = (node.get_clock().now() - node.race_start_time).nanoseconds / 1e9
            node.save_results_to_csv(total)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
