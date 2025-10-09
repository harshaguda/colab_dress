#!/usr/bin/env python3
"""
Example service server for Get3DPoint service.
Converts 2D pixel coordinates to 3D real-world coordinates.
"""

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from rclpy.node import Node
from colab_dress_interfaces.srv import Get3DPoint
import pyrealsense2 as rs2
import cv2
from cv_bridge import CvBridge

class cameraIntrinsics:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.model = None
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.coeffs = None

class Get3DPointService(Node):

    def __init__(self,
                 depth_image_topic="/camera/camera/depth/image_rect_raw",
                 depth_info_topic="/camera/camera/depth/camera_info",
                 color_image_topic="/camera/camera/color/image_raw",
                 color_info_topic="/camera/camera/color/camera_info"):
        super().__init__('get_3d_point_service')
        self.srv = self.create_service(Get3DPoint, 'get_3d_point', self.get_3d_point_callback)
        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            qos_profile_sensor_data)
        self.color_image_subscription = self.create_subscription(
            Image,
            color_image_topic,
            self.color_image_callback,
            qos_profile_sensor_data)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            color_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data)
        self.depth_info_subscription = self.create_subscription(
            CameraInfo,
            depth_info_topic,
            self.depth_info_callback,
            qos_profile_sensor_data)
        self.bridge = CvBridge()
        self.get_logger().info('Get3DPoint service is ready')
        self.color_intrinsics = cameraIntrinsics()
        self.depth_intrinsics = cameraIntrinsics()
        self.matrix_coefficients = None
        self.depth_to_color_transform = None  # Assume aligned for now
        self.depth_image = None
        
        self.px = 480
        self.py = 320
    def camera_info_callback(self, msg):
        # self.get_logger().info(f"Received CameraInfo: msg=,{msg}")
        # print(msg)
        self.color_intrinsics = rs2.intrinsics()
        self.color_intrinsics.width = msg.width
        self.color_intrinsics.height = msg.height
        # self.color_intrinsics.model = msg.d
        self.color_intrinsics.fx = msg.k[0]
        self.color_intrinsics.fy = msg.k[4]
        self.color_intrinsics.ppx = msg.k[2]
        self.color_intrinsics.ppy = msg.k[5]
        self.matrix_coefficients = msg.k
        if msg.distortion_model == 'plumb_bob':
                self.color_intrinsics.model = rs2.distortion.brown_conrady
        elif msg.distortion_model == 'equidistant':
            self.color_intrinsics.model = rs2.distortion.kannala_brandt4
        self.color_intrinsics.coeffs = [i for i in msg.d]
        # print(msg.)
        # self.color_intrinsics.coeffs = msg.k
    def depth_info_callback(self, msg):
        # self.get_logger().info(f"Received Depth CameraInfo: msg=,{msg}")
        # print(msg)
        self.depth_intrinsics = rs2.intrinsics()

        self.depth_intrinsics.width = msg.width
        self.depth_intrinsics.height = msg.height
        # self.depth_intrinsics.model = msg.d
        self.depth_intrinsics.fx = msg.k[0]
        self.depth_intrinsics.fy = msg.k[4]
        self.depth_intrinsics.ppx = msg.k[2]
        self.depth_intrinsics.ppy = msg.k[5]

        if msg.distortion_model == 'plumb_bob':
            self.depth_intrinsics.model = rs2.distortion.brown_conrady
        elif msg.distortion_model == 'equidistant':
            self.depth_intrinsics.model = rs2.distortion.kannala_brandt4
        self.depth_intrinsics.coeffs = [i for i in msg.d]


    def color_image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.circle(self.color_image, (self.px, self.py), 5, (0, 0, 255), -1)
        # cv2.imshow("Camera Image", self.color_image)
        # cv2.waitKey(1)
    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        

    def get_3d_point_callback(self, request, response):
        """
        Convert 2D pixel coordinates to 3D coordinates.
        This is a simple example - in real use, you'd use camera calibration data.
        """
        self.get_logger().info(f'Received request: px={request.px}, py={request.py}')
        # response.rx, response.ry, response.rz = 
        self.px = request.px
        self.py = request.py
        x, y, z = self.get_3d_point_from_color_pixel(request.px, request.py)
        # print(x, y, z)
        response.rx = x if x is not None else 0.0
        response.ry = y if y is not None else 0.0
        response.rz = z if z is not None else 0.0
        
        self.get_logger().info(f'Sending response: rx={response.rx}, ry={response.ry}, rz={response.rz}')
        return response

    def get_depth_at_color_pixel(self, color_u, color_v):
        """
        Get depth value at a color pixel coordinate
        
        Args:
            color_u: x coordinate in color image
            color_v: y coordinate in color image
            
        Returns:
            float: depth value in meters, or None if no valid depth
        """
        if self.depth_image is None:
            self.get_logger().warn("No depth image available")
            return None, None, None
            
        # Check bounds for color image
        if (color_u < 0 or color_u >= self.color_intrinsics.width or
            color_v < 0 or color_v >= self.color_intrinsics.height):
            self.get_logger().warn("Color pixel out of bounds")
            return None, None, None
            
        if self.depth_to_color_transform is None:
            # Cameras are aligned - direct mapping
            # Scale coordinates if image sizes are different
            if (self.color_intrinsics.width != self.depth_intrinsics.width or 
                self.color_intrinsics.height != self.depth_intrinsics.height):
                
                scale_x = self.depth_intrinsics.width / self.color_intrinsics.width
                scale_y = self.depth_intrinsics.height / self.color_intrinsics.height
                depth_u = int(color_u * scale_x)
                depth_v = int(color_v * scale_y)
            else:
                depth_u, depth_v = color_u, color_v
        else:
            # Non-aligned cameras - need to estimate depth for projection
            # Use average depth in neighborhood for initial estimate
            neighborhood_size = 5
            total_depth = 0
            valid_pixels = 0
            
            for du in range(-neighborhood_size, neighborhood_size + 1):
                for dv in range(-neighborhood_size, neighborhood_size + 1):
                    test_u = color_u + du
                    test_v = color_v + dv
                    if (0 <= test_u < self.color_intrinsics.width and 
                        0 <= test_v < self.color_intrinsics.height):
                        # Rough estimate using aligned assumption for initial depth
                        scale_x = self.depth_intrinsics.width / self.color_intrinsics.width
                        scale_y = self.depth_intrinsics.height / self.color_intrinsics.height
                        est_depth_u = int(test_u * scale_x)
                        est_depth_v = int(test_v * scale_y)
                        
                        if (0 <= est_depth_u < self.depth_image.shape[1] and 
                            0 <= est_depth_v < self.depth_image.shape[0]):
                            depth_val = self.depth_image[est_depth_v, est_depth_u]
                            if depth_val > 0:
                                total_depth += depth_val
                                valid_pixels += 1
            
            if valid_pixels == 0:
                self.get_logger().warn("No valid depth pixels found")
                return None, None, None
                
            avg_depth = total_depth / valid_pixels
            
            # Use this average depth for projection
            depth_coords = self.project_color_pixel_to_depth_pixel(color_u, color_v, avg_depth)
            if depth_coords is None:
                self.get_logger().warn("Depth projection failed")
                return None, None, None
            depth_u, depth_v = depth_coords
        # Get depth value
        if (0 <= depth_u < self.depth_image.shape[1] and 
            0 <= depth_v < self.depth_image.shape[0]):
            depth_value = self.depth_image[depth_v, depth_u]
            if depth_value > 0:
                return depth_value / 1000.0, depth_u, depth_v  # Convert mm to meters
        
        self.get_logger().warn("No valid depth at projected pixel")
        return None, depth_u, depth_v

    def project_color_pixel_to_depth_pixel(self, color_u, color_v, depth_value):
        """
        Project a color pixel to depth pixel coordinates using camera calibration
        
        Args:
            color_u: x coordinate in color image
            color_v: y coordinate in color image  
            depth_value: depth value at the corresponding 3D point
            
        Returns:
            tuple: (depth_u, depth_v) coordinates in depth image, or None if projection fails
        """
        if self.color_intrinsics is None or self.depth_intrinsics is None:
            self.get_logger().warn("Camera intrinsics not available")
            return None
            
        try:
            if self.depth_to_color_transform is None:
                # Cameras are aligned, use direct mapping
                return (color_u, color_v)
            
        except Exception as e:
            self.get_logger().error(f"Error in pixel projection: {e}")
            return None

    def get_3d_point_from_color_pixel(self, color_u, color_v) -> list[float | None]:
        """
        Get 3D point coordinates from color pixel
        
        Args:
            color_u: x coordinate in color image
            color_v: y coordinate in color image
            
        Returns:
            list: [x, y, z] coordinates in meters in color camera frame, or None
        """
        depth, du, dv = self.get_depth_at_color_pixel(color_u, color_v)
        self.get_logger().info(f"Depth at color pixel ({color_u}, {color_v}) -> depth: {depth}, depth pixel: ({du}, {dv})")
        if depth is None:
            return [None, None, None]
            
        if self.color_intrinsics is None:
            self.get_logger().warn("Color intrinsics not available")
            return [None, None, None]

        try:
            # Convert depth back to mm for RealSense function
            depth_mm = depth * 1000.0
            point_3d = rs2.rs2_deproject_pixel_to_point(
                self.color_intrinsics, [color_u, color_v], depth_mm)
            # Convert back to meters
            return [point_3d[0]/1000.0, point_3d[1]/1000.0, point_3d[2]/1000.0]
        except Exception as e:
            self.get_logger().error(f"Error in deprojection: {e}")
            return [None, None, None]



def main(args=None):
    rclpy.init(args=args)
    service = Get3DPointService()
    rclpy.spin(service)
    service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
