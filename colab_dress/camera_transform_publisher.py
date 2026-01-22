#!/usr/bin/env python3
"""ROS 2 node that publishes the transform from base_link to an external camera."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R


class CameraTransformPublisher(Node):
    """Continuously publish a transform based on a 4x4 homogeneous matrix."""

    def __init__(self) -> None:
        super().__init__("camera_transform_publisher")

        # Parameters allowing customization at runtime
        self.declare_parameter("matrix_path", "translation_matrix.npy")
        self.declare_parameter("parent_frame", "base")
        self.declare_parameter("child_frame", "camera_link")
        self.declare_parameter("publish_rate", 10.0)

        matrix_path_param = self.get_parameter("matrix_path").get_parameter_value().string_value
        parent_frame = self.get_parameter("parent_frame").get_parameter_value().string_value
        child_frame = self.get_parameter("child_frame").get_parameter_value().string_value
        publish_rate = float(self.get_parameter("publish_rate").get_parameter_value().double_value)

        self._tf_broadcaster = TransformBroadcaster(self)

        # Load transform matrix from disk
        self.transform_matrix: Optional[np.ndarray] = None
        try:
            matrix_path = Path(matrix_path_param).expanduser().resolve()
            loaded_matrix = np.load(matrix_path)
            transform_matrix = loaded_matrix.astype(float)
            # Apply extra rotation (color camera -> depth camera) as in ROS1 code
            r_matrix = R.from_euler('xyz', [math.radians(135.0), math.radians(-90.0), math.radians(-45.0)]).as_matrix()
            t_color_to_depth = np.eye(4)
            t_color_to_depth[:3, :3] = r_matrix
            transform_matrix = transform_matrix @ t_color_to_depth

            # x, y, z = 0.367, 0.017, -0.1
            x, y, z = 0.473, -0.06, 0.06 # Dirty fix for new camera position, investigate why original values are not working. 
                                         # Probably something to do with the transformations and their order.
                            

            offset = np.array([x, y, z])
            # tvec4 = np.array([ 0.367, 0.017, -0.027, 1.0])
            # qx, qy, qz, qw = 1.000, -0.008, 0.005, 0.014

            # rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            # T = np.eye(4)
            # # T[:3, :3] = rot
            # T[:3, 3] = [x, y, z]
            # transform_matrix = T @ transform_matrix
            # transform_matrix = np.linalg.inv(transform_matrix)
            print(transform_matrix)
            transform_matrix[0:3, 3] += offset  # Adjust for camera offset from robot base
            if transform_matrix.shape != (4, 4):
                raise ValueError(f"Matrix must be 4x4, got {transform_matrix.shape}")

            self.transform_matrix = transform_matrix

            self.get_logger().info("Successfully loaded translation matrix")
            self.get_logger().info(f"Matrix file: {matrix_path}")
            self.get_logger().debug(f"Loaded matrix (marker->camera)\n{loaded_matrix}")
            self.get_logger().debug(f"Combined matrix (camera frame)\n{transform_matrix}")
        except FileNotFoundError:
            self.get_logger().error(f"translation matrix file not found: {matrix_path_param}")
            raise
        except Exception as exc:  # pragma: no cover - best effort logging
            self.get_logger().error(f"Failed to load matrix: {exc}")
            raise

        if self.transform_matrix is None:
            raise RuntimeError("Transform matrix failed to load")

        self.translation_vector = self.transform_matrix[:3, 3]
        rotation_matrix = self.transform_matrix[:3, :3]
        self.quaternion = R.from_matrix(rotation_matrix).as_quat()

        self.get_logger().info(
            "Translation (x, y, z): (%.4f, %.4f, %.4f)"
            % (self.translation_vector[0], self.translation_vector[1], self.translation_vector[2])
        )
        self.get_logger().info(f"Rotation matrix:\n{rotation_matrix}")
        self.get_logger().info(
            "Quaternion (x, y, z, w): (%.4f, %.4f, %.4f, %.4f)"
            % (self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3])
        )

        self._transform_msg = TransformStamped()
        self._transform_msg.header.frame_id = parent_frame
        self._transform_msg.child_frame_id = child_frame
        self._transform_msg.transform.translation.x = float(self.translation_vector[0])
        self._transform_msg.transform.translation.y = float(self.translation_vector[1])
        self._transform_msg.transform.translation.z = float(self.translation_vector[2])
        self._transform_msg.transform.rotation.x = float(self.quaternion[0])
        self._transform_msg.transform.rotation.y = float(self.quaternion[1])
        self._transform_msg.transform.rotation.z = float(self.quaternion[2])
        self._transform_msg.transform.rotation.w = float(self.quaternion[3])

        # Print transform summary for the user
        self._log_transform_summary()

        period = 1.0 / publish_rate if publish_rate > 0.0 else 0.1
        self._timer = self.create_timer(period, self._publish_transform)

    def _publish_transform(self) -> None:
        self._transform_msg.header.stamp = self.get_clock().now().to_msg()
        self._tf_broadcaster.sendTransform(self._transform_msg)

    def _log_transform_summary(self) -> None:
        self.get_logger().info("=" * 50)
        self.get_logger().info("TRANSFORM INFORMATION")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Parent frame: {self._transform_msg.header.frame_id}")
        self.get_logger().info(f"Child frame: {self._transform_msg.child_frame_id}")
        self.get_logger().info(
            "Translation (x, y, z): (%.4f, %.4f, %.4f)"
            % (self.translation_vector[0], self.translation_vector[1], self.translation_vector[2])
        )
        self.get_logger().info(
            "Quaternion (x, y, z, w): (%.4f, %.4f, %.4f, %.4f)"
            % (self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3])
        )
        euler = R.from_quat(self.quaternion).as_euler('xyz', degrees=False)
        self.get_logger().info(
            "Euler angles (deg) roll, pitch, yaw: (%.2f, %.2f, %.2f)"
            % (math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2]))
        )
        self.get_logger().info("=" * 50)


def main() -> None:
    rclpy.init()
    node = CameraTransformPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()