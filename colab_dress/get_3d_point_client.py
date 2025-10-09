#!/usr/bin/env python3
"""
Example service client for Get3DPoint service.
Calls the service to convert 2D pixel coordinates to 3D coordinates.
"""

import sys
import rclpy
from rclpy.node import Node
from colab_dress_interfaces.srv import Get3DPoint


class Get3DPointClient(Node):

    def __init__(self):
        super().__init__('get_3d_point_client')
        self.cli = self.create_client(Get3DPoint, 'get_3d_point')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = Get3DPoint.Request()

    def send_request(self, px, py):
        self.req.px = px
        self.req.py = py
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 3:
        print('Usage: ros2 run colab_dress get_3d_point_client <px> <py>')
        print('Example: ros2 run colab_dress get_3d_point_client 320 240')
        sys.exit(1)

    px = int(sys.argv[1])
    py = int(sys.argv[2])

    client = Get3DPointClient()
    response = client.send_request(px, py)
    
    client.get_logger().info(
        f'Result: 2D({px}, {py}) -> 3D({response.rx:.3f}, {response.ry:.3f}, {response.rz:.3f})')

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
