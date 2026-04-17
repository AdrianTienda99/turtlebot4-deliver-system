#!/usr/bin/env python3

import os
import json
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from pyapriltags import Detector


class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('apriltag_detector')

        self.declare_parameter('image_topic', '/depthai/ojb_tracker/rgb/image')
        self.declare_parameter('tag_family', 'tag36h11')
        self.declare_parameter('save_debug_image', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.tag_family = self.get_parameter('tag_family').value
        self.save_debug_image = self.get_parameter('save_debug_image').value

        self.bridge = CvBridge()
        self.detector = Detector(families=self.tag_family)

        self.pub = self.create_publisher(String, '/apriltag_detections_simple', 10)
        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.frame_count = 0
        self.last_log_frame = 0
        self.debug_dir = '/tmp/apriltag_debug'
        os.makedirs(self.debug_dir, exist_ok=True)

        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")
        self.get_logger().info(f"Using AprilTag family: {self.tag_family}")

    def image_callback(self, msg: Image):
        self.frame_count += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge failed: {e}")
            return

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Image received: shape={frame.shape}, encoding={msg.encoding}, frame_id={msg.header.frame_id}"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)

        if self.save_debug_image and self.frame_count == 30:
            cv2.imwrite(os.path.join(self.debug_dir, 'frame_debug.jpg'), frame)
            cv2.imwrite(os.path.join(self.debug_dir, 'frame_debug_gray.jpg'), gray)
            self.get_logger().info(f"Saved debug images to {self.debug_dir}")

        if not results:
            return

        for tag in results:
            center_x, center_y = float(tag.center[0]), float(tag.center[1])
            corners = [[float(c[0]), float(c[1])] for c in tag.corners]

            for i in range(4):
                p1 = tuple(map(int, tag.corners[i]))
                p2 = tuple(map(int, tag.corners[(i + 1) % 4]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID {tag.tag_id}",
                (int(center_x) + 10, int(center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            payload = {
                "id": int(tag.tag_id),
                "center": [center_x, center_y],
                "corners": corners
            }

            out = String()
            out.data = json.dumps(payload)
            self.pub.publish(out)

            cv2.imwrite(os.path.join(self.debug_dir, 'frame_detected.jpg'), frame)

            self.get_logger().info(
                f"Tag detected: id={tag.tag_id}, center=({center_x:.1f}, {center_y:.1f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
