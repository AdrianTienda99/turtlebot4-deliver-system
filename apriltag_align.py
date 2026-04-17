#!/usr/bin/env python3

import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class AprilTagAlignNode(Node):
    def __init__(self):
        super().__init__('apriltag_align')

        self.declare_parameter('target_tag_id', 0)
        self.declare_parameter('image_width', 512)
        self.declare_parameter('center_tolerance_px', 20.0)
        self.declare_parameter('target_tag_width_px', 120.0)
        self.declare_parameter('delivery_target_tag_width_px', 110.0)   # stop earlier for tag 0
        self.declare_parameter('linear_speed', 0.04)
        self.declare_parameter('angular_gain', 0.003)
        self.declare_parameter('max_angular_speed', 0.20)

        # Delivery-side only backoff after alignment
        self.declare_parameter('delivery_backoff_enabled', True)
        self.declare_parameter('delivery_backoff_speed', -0.05)         # reverse speed
        self.declare_parameter('delivery_backoff_duration', 1.0)        # seconds

        self.target_tag_id = int(self.get_parameter('target_tag_id').value)
        self.image_width = float(self.get_parameter('image_width').value)
        self.center_tolerance_px = float(self.get_parameter('center_tolerance_px').value)
        self.target_tag_width_px = float(self.get_parameter('target_tag_width_px').value)
        self.delivery_target_tag_width_px = float(
            self.get_parameter('delivery_target_tag_width_px').value
        )
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_gain = float(self.get_parameter('angular_gain').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        self.delivery_backoff_enabled = bool(
            self.get_parameter('delivery_backoff_enabled').value
        )
        self.delivery_backoff_speed = float(
            self.get_parameter('delivery_backoff_speed').value
        )
        self.delivery_backoff_duration = float(
            self.get_parameter('delivery_backoff_duration').value
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/apriltag_align_status', 10)

        self.det_sub = self.create_subscription(
            String,
            '/apriltag_detections_simple',
            self.detection_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.last_detection_time = self.get_clock().now()
        self.tag_visible = False
        self.current_center_x = None
        self.current_tag_width = None
        self.current_tag_id = None
        self.aligned = False

        # Backoff state
        self.backing_off = False
        self.backoff_done = False
        self.backoff_start_time = None

        self.get_logger().info(
            f"AprilTag align started for target tag id {self.target_tag_id}"
        )
        self.get_logger().info(
            f"Using stop width {self.get_active_target_tag_width():.1f}px for tag {self.target_tag_id}"
        )

        if self.target_tag_id == 0 and self.delivery_backoff_enabled:
            self.get_logger().info(
                f"Delivery backoff enabled: speed={self.delivery_backoff_speed:.2f} m/s, "
                f"duration={self.delivery_backoff_duration:.2f} s"
            )

    def get_active_target_tag_width(self) -> float:
        # Delivery side: stop a bit earlier
        if self.target_tag_id == 0:
            return self.delivery_target_tag_width_px
        return self.target_tag_width_px

    def publish_status(self, state: str, error_x=None, tag_width=None):
        payload = {
            "target_tag_id": self.target_tag_id,
            "current_tag_id": self.current_tag_id,
            "state": state,
            "tag_visible": self.tag_visible,
            "aligned": self.aligned,
            "error_x": error_x,
            "tag_width": tag_width,
            "active_target_tag_width": self.get_active_target_tag_width(),
            "backing_off": self.backing_off,
            "backoff_done": self.backoff_done
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def detection_callback(self, msg: String):
        # Ignore new detections once delivery backoff is complete
        # so the robot does not re-approach the delivery tag.
        if self.backoff_done:
            return

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse detection JSON: {e}")
            return

        detected_id = int(data.get("id", -1))
        if detected_id != self.target_tag_id:
            return

        center = data.get("center", None)
        corners = data.get("corners", None)

        if center is None or corners is None or len(corners) != 4:
            return

        self.current_tag_id = detected_id
        self.current_center_x = float(center[0])

        top_w = math.hypot(
            corners[1][0] - corners[0][0],
            corners[1][1] - corners[0][1]
        )
        bottom_w = math.hypot(
            corners[2][0] - corners[3][0],
            corners[2][1] - corners[3][1]
        )
        self.current_tag_width = 0.5 * (top_w + bottom_w)

        self.last_detection_time = self.get_clock().now()
        self.tag_visible = True

    def stop_robot(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def start_delivery_backoff(self):
        self.backing_off = True
        self.backoff_done = False
        self.backoff_start_time = self.get_clock().now()
        self.get_logger().info(
            f"Delivery alignment reached for tag 0. Starting backoff: "
            f"speed={self.delivery_backoff_speed:.2f}, duration={self.delivery_backoff_duration:.2f}s"
        )

    def handle_backoff(self):
        if not self.backing_off:
            return False

        now = self.get_clock().now()
        elapsed = (now - self.backoff_start_time).nanoseconds / 1e9

        if elapsed < self.delivery_backoff_duration:
            cmd = Twist()
            cmd.linear.x = self.delivery_backoff_speed
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.aligned = False
            self.publish_status("BACKING_OFF", tag_width=self.current_tag_width)
            return True

        self.stop_robot()
        self.backing_off = False
        self.backoff_done = True
        self.aligned = True
        self.publish_status("ALIGNED", tag_width=self.current_tag_width)
        self.get_logger().info("Delivery backoff complete. Final state: ALIGNED")
        return True

    def control_loop(self):
        # If we are in delivery backoff, handle it first
        if self.handle_backoff():
            return

        # If delivery backoff already finished, stay stopped and latched aligned
        if self.backoff_done:
            self.aligned = True
            self.stop_robot()
            self.publish_status("ALIGNED", tag_width=self.current_tag_width)
            return

        now = self.get_clock().now()
        dt = (now - self.last_detection_time).nanoseconds / 1e9

        if dt > 0.5:
            if self.tag_visible:
                self.get_logger().warn(
                    f"Target tag {self.target_tag_id} lost, stopping robot"
                )
            self.tag_visible = False
            self.aligned = False
            self.stop_robot()
            self.publish_status("LOST")
            return

        if self.current_center_x is None or self.current_tag_width is None:
            self.stop_robot()
            self.publish_status("SEARCHING")
            return

        image_center_x = self.image_width / 2.0
        error_x = self.current_center_x - image_center_x
        active_target_tag_width = self.get_active_target_tag_width()

        cmd = Twist()

        reached_alignment = (
            abs(error_x) < self.center_tolerance_px and
            self.current_tag_width >= active_target_tag_width
        )

        if reached_alignment:
            # Delivery side only: align, then back off a little
            if self.target_tag_id == 0 and self.delivery_backoff_enabled:
                self.start_delivery_backoff()
                self.publish_status(
                    "DELIVERY_ALIGNED_STARTING_BACKOFF",
                    error_x=error_x,
                    tag_width=self.current_tag_width
                )
                return

            # Pickup / UR side: unchanged behavior
            self.aligned = True
            self.stop_robot()
            self.publish_status(
                "ALIGNED",
                error_x=error_x,
                tag_width=self.current_tag_width
            )
            self.get_logger().info(
                f"Aligned to tag {self.target_tag_id}. "
                f"error_x={error_x:.1f}, "
                f"tag_width={self.current_tag_width:.1f}, "
                f"target_width={active_target_tag_width:.1f}"
            )
            return

        self.aligned = False

        if abs(error_x) > self.center_tolerance_px:
            angular = -self.angular_gain * error_x
            angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))
            cmd.angular.z = angular
            cmd.linear.x = 0.0
            state = "ALIGNING_YAW"
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
            state = "APPROACHING"

        self.cmd_pub.publish(cmd)
        self.publish_status(state, error_x=error_x, tag_width=self.current_tag_width)

        self.get_logger().info(
            f"Tracking tag {self.target_tag_id}: "
            f"center_x={self.current_center_x:.1f}, "
            f"error_x={error_x:.1f}, "
            f"tag_width={self.current_tag_width:.1f}, "
            f"target_width={active_target_tag_width:.1f}, "
            f"vx={cmd.linear.x:.2f}, wz={cmd.angular.z:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
