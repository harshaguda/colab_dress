#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image



class ArucoAlignmentTest(Node):
    def __init__(self):
        super().__init__("aruco_alignment_test")
        self.bridge = CvBridge()

        # Topics for RealSense aligned depth and color
        color_topic = "/camera/camera/color/image_raw"
        depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"

        self.sub_color = Subscriber(self, Image, color_topic)
        self.sub_depth = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.callback)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.get_logger().info("ArucoAlignmentTest started.")

    def callback(self, color_msg: Image, depth_msg: Image):
        # Convert images
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Detect ArUco markers
        corners, ids, _ = self.detector.detectMarkers(color)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color, corners, ids)

            # Compute center of first marker
            c = corners[0][0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))

            # Draw center on RGB image
            cv2.circle(color, (center_x, center_y), 5, (0, 0, 255), -1)

            # Visualize depth: normalize for display
            depth_vis = self.normalize_depth(depth)
            cv2.circle(depth_vis, (center_x, center_y), 5, (0, 0, 255), -1)
        else:
            depth_vis = self.normalize_depth(depth)

        cv2.imshow("RGB with ArUco", color)
        cv2.imshow("Aligned Depth with Center", depth_vis)
        cv2.waitKey(1)

    @staticmethod
    def normalize_depth(depth):
        # Handle 16UC1 or 32FC1 depth
        if depth.dtype == np.uint16:
            depth_float = depth.astype(np.float32)
            max_val = np.percentile(depth_float, 99)
            depth_norm = np.clip(depth_float / max_val * 255.0, 0, 255).astype(np.uint8)
        else:
            max_val = np.percentile(depth, 99)
            depth_norm = np.clip(depth / max_val * 255.0, 0, 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return depth_color


def main():
    rclpy.init()
    node = ArucoAlignmentTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()