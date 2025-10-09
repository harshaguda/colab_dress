#!/usr/bin/env python3
"""
Example subscriber for ArucoMarker messages.
Listens to detected ArUco markers and prints their information.
"""

import rclpy
from rclpy.node import Node
from colab_dress_interfaces.msg import ArucoMarker


class ArucoMarkerListener(Node):

    def __init__(self):
        super().__init__('aruco_marker_listener')
        self.subscription = self.create_subscription(
            ArucoMarker,
            'aruco_markers',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('ArUco Marker Listener started')

    def listener_callback(self, msg):
        """Process received ArUco marker messages"""
        self.get_logger().info(
            f'\n=== ArUco Marker Detected ===\n'
            f'ID: {msg.id}\n'
            f'Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}\n'
            f'Frame ID: {msg.header.frame_id}\n'
            f'Corners (x,y):\n'
            f'  Top-Left:     ({msg.corners[0]:.2f}, {msg.corners[1]:.2f})\n'
            f'  Top-Right:    ({msg.corners[2]:.2f}, {msg.corners[3]:.2f})\n'
            f'  Bottom-Right: ({msg.corners[4]:.2f}, {msg.corners[5]:.2f})\n'
            f'  Bottom-Left:  ({msg.corners[6]:.2f}, {msg.corners[7]:.2f})\n'
        )


def main(args=None):
    rclpy.init(args=args)
    listener = ArucoMarkerListener()
    
    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    
    listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
