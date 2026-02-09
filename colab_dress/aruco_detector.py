import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
from rclpy.node import Node
from colab_dress_interfaces.msg import ArucoMarker
import pyrealsense2 as rs2
import sys

# Define available ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

class ArucoDetector(Node):
    def __init__(self, 
                 color_image_topic="/camera/camera/color/image_raw",
                 depth_image_topic="/camera/camera/aligned_depth_to_color/image_raw",
                 color_info_topic="/camera/camera/color/camera_info",
                 aruco_dict_type="DICT_5X5_50"):
        super().__init__('aruco_detector')

        # Parameters
        self.aruco_dict_type = aruco_dict_type
        self.bridge = CvBridge()

        # Subscriptions
        if color_image_topic.endswith("/compressed"):
            self.color_image_subscription = self.create_subscription(
                CompressedImage,
                color_image_topic,
                self.color_compressed_image_callback,
                qos_profile_sensor_data)
        else:
            self.color_image_subscription = self.create_subscription(
                Image,
                color_image_topic,
                self.color_image_callback,
                qos_profile_sensor_data)

        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            qos_profile_sensor_data)

        self.color_info_subscription = self.create_subscription(
            CameraInfo,
            color_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data)

        # Publisher for detected markers (one publisher per topic base)
        # We'll publish all markers, but you can also create separate publishers per ID
        self.marker_publisher = self.create_publisher(
            ArucoMarker,
            'aruco_markers',
            10)

        # Camera calibration data
        self.matrix_coefficients = np.zeros((3, 3))
        self.coeffs = np.zeros((8,))

        # Latest aligned depth frame (same pixel frame as color)
        self._latest_depth = None
        self._latest_depth_stamp = None

        self.get_logger().info(f'ArUco detector initialized with {aruco_dict_type}')
        self.get_logger().info(f'Subscribing color: {color_image_topic}')
        self.get_logger().info(f'Subscribing depth: {depth_image_topic}')

        self.intrinsics = None

    def camera_info_callback(self, msg):
        if self.matrix_coefficients.sum() == 0 and self.coeffs.sum() == 0:
            self.matrix_coefficients = np.array(msg.k).reshape(3, 3)
            self.coeffs = np.array(msg.d)
            self.get_logger().info('Camera calibration data received')

        if self.intrinsics:
            return
        try:
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = msg.width
            self.intrinsics.height = msg.height
            self.intrinsics.fx = msg.k[0]
            self.intrinsics.fy = msg.k[4]
            self.intrinsics.ppx = msg.k[2]
            self.intrinsics.ppy = msg.k[5]
            if msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif msg.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in msg.d]
            self.get_logger().info("Camera Intrinsics Received (RS2)")
        except Exception as e:
            self.get_logger().error(f"Failed to parse camera info: {e}")

    def depth_image_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self._latest_depth = depth_image
            self._latest_depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def color_image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.detect_and_publish_aruco_markers(color_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def color_compressed_image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if color_image is None:
                raise ValueError("Failed to decode compressed image")
            self.detect_and_publish_aruco_markers(color_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {str(e)}')

    def get_3d_point_local(self, u, v):
        if not self.intrinsics or self._latest_depth is None:
            return None
        
        h, w = self._latest_depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
            
        dist = self._latest_depth[v, u]
        if self._latest_depth.dtype == np.uint16:
            dist_m = dist * 0.001
        else:
            dist_m = float(dist)
            
        if dist_m <= 0:
            return None
        self.get_logger().info(f"Deprojecting pixel ({u}, {v}) with depth {dist_m:.3f}m")
        p3d = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [float(u), float(v)], dist_m)
        self.get_logger().info(f"3D point in camera frame: x={p3d[0]:.3f}m, y={p3d[1]:.3f}m, z={p3d[2]:.3f}m")
        class Response:
            def __init__(self, x, y, z):
                self.rx = x
                self.ry = y
                self.rz = z
                
        return Response(p3d[0], p3d[1], p3d[2])

    def detect_and_publish_aruco_markers(self, frame, header):
        """
        Detect ArUco markers in the image and publish each one as a separate message
        
        Args:
            frame: Input image frame (BGR)
            header: ROS message header from the original image
        """        # Initialize storage for valid detection to be used in saving
        last_valid_rvec = None
        last_valid_response = None
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.aruco_dict_type])
        
        # Handle different OpenCV versions for ArUco detection
        try:
            # Try new OpenCV API (4.7+)
            if hasattr(cv2.aruco, 'ArucoDetector'):
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                # Use older OpenCV API (4.0-4.6)
                parameters = cv2.aruco.DetectorParameters_create()
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        except Exception as e:
            self.get_logger().error(f'ArUco detection failed: {str(e)}')
            return
        
        # Process and publish each detected marker
        if ids is not None and len(ids) > 0:
            # Compute center of the first marker for visualization (in pixel coords)
            marker_corners0 = corners[0][0]  # (4, 2)
            center_x, center_y = np.mean(marker_corners0, axis=0).astype(int)

            # Depth visualization (aligned depth -> same pixel coords as color)
            if self._latest_depth is not None:
                depth_vis = self.normalize_depth(self._latest_depth)
                h, w = self._latest_depth.shape[:2]
                center_x = int(np.clip(center_x, 0, w - 1))
                center_y = int(np.clip(center_y, 0, h - 1))
                cv2.circle(depth_vis, (center_x, center_y), 5, (0, 0, 255), -1)

                depth_value_m = None
                try:
                    if self._latest_depth.dtype == np.uint16:
                        depth_value_m = float(self._latest_depth[center_y, center_x]) / 1000.0
                    else:
                        dv = float(self._latest_depth[center_y, center_x])
                        if np.isfinite(dv):
                            depth_value_m = dv
                except Exception:
                    depth_value_m = None

                label = f"id={int(ids[0][0])}"
                if depth_value_m is not None:
                    label += f" z={depth_value_m:.3f}m"
                cv2.putText(
                    depth_vis,
                    label,
                    (max(center_x + 10, 0), max(center_y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Aligned Depth with Center", depth_vis)

            for i, marker_id in enumerate(ids):
                # Create ArucoMarker message
                marker_msg = ArucoMarker()
                marker_msg.header = header
                marker_msg.id = int(marker_id[0])
                
                # Flatten the corners array
                # corners[i] is shape (1, 4, 2) -> flatten to [x1, y1, x2, y2, x3, y3, x4, y4]
                marker_corners = corners[i][0]  # Shape: (4, 2)
                marker_msg.corners = [
                    float(marker_corners[0][0]), float(marker_corners[0][1]),  # Top-left
                    float(marker_corners[1][0]), float(marker_corners[1][1]),  # Top-right
                    float(marker_corners[2][0]), float(marker_corners[2][1]),  # Bottom-right
                    float(marker_corners[3][0]), float(marker_corners[3][1])   # Bottom-left
                ]
                
                # Publish the marker
                self.marker_publisher.publish(marker_msg)
                
                self.get_logger().debug(
                    f'Published marker ID {marker_msg.id} at corners: '
                    f'[{marker_msg.corners[0]:.1f}, {marker_msg.corners[1]:.1f}], '
                    f'[{marker_msg.corners[2]:.1f}, {marker_msg.corners[3]:.1f}], '
                    f'[{marker_msg.corners[4]:.1f}, {marker_msg.corners[5]:.1f}], '
                    f'[{marker_msg.corners[6]:.1f}, {marker_msg.corners[7]:.1f}]'
                )
                marker_size = 0.05  # Size of the marker in meters (adjust as needed)
                # Define marker corners in 3D space (marker coordinate system)
                objPoints = np.array([
                    [-marker_size/2, marker_size/2, 0],
                    [marker_size/2, marker_size/2, 0],
                    [marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ], dtype=np.float32)
                
                # Get 2D corners from detected marker
                imgPoints = corners[i][0].astype(np.float32)
                
                # Solve for pose
                success, rvec, tvec = cv2.solvePnP(
                                                    objPoints,
                                                    imgPoints, 
                                                    self.matrix_coefficients, 
                                                    self.coeffs
                                                )
                if success:
                    # Draw axis for the marker

                    cv2.drawFrameAxes(frame, self.matrix_coefficients, self.coeffs, 
                                     rvec, tvec, 0.03)
                    
                    # Display position information
                    marker_position = f"ID:{ids[i][0]} x:{tvec[0][0]:.2f} y:{tvec[1][0]:.2f} z:{tvec[2][0]:.2f}"
                    cv2.putText(frame, marker_position, 
                               (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    px, py = np.mean(marker_corners, axis=0).astype(int)
                    response = self.get_3d_point_local(int(px), int(py))
                    if response is not None:
                        last_valid_rvec = rvec
                        last_valid_response = response
            # Optional: Draw detected markers for visualization
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
            
            self.get_logger().info(f'Detected and published {len(ids)} ArUco marker(s)')
        cv2.imshow("ArUco Markers", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if last_valid_rvec is not None and last_valid_response is not None:
                self.save_translation_matrix(last_valid_rvec, last_valid_response)
                self.get_logger().info('Translation matrix saved.')
                cv2.destroyAllWindows()
                self.destroy_node()
                sys.exit(1)
                rclpy.shutdown()
            else:
                self.get_logger().warn('Cannot save: 3D point not available.')

    @staticmethod
    def normalize_depth(depth):
        """Normalize depth image to an 8-bit color-mapped visualization."""
        if depth.dtype == np.uint16:
            depth_float = depth.astype(np.float32)
            max_val = np.percentile(depth_float, 99)
            if not np.isfinite(max_val) or max_val <= 0:
                max_val = 1.0
            depth_norm = np.clip(depth_float / max_val * 255.0, 0, 255).astype(np.uint8)
        else:
            depth_float = depth.astype(np.float32)
            depth_float = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = np.percentile(depth_float, 99)
            if not np.isfinite(max_val) or max_val <= 0:
                max_val = 1.0
            depth_norm = np.clip(depth_float / max_val * 255.0, 0, 255).astype(np.uint8)

        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def save_translation_matrix(self, rvec, response):
        """
        Save the rotation and translation vectors to a file.
        
        Args:
            rvec: Rotation vector
            response: Translation vector from Get3DPoint service
        Returns:
            np.ndarray: The computed transformation matrix
        """

        # while not_saved:
        
        R, _ = cv2.Rodrigues(rvec)

        tvec = np.array([response.rx, response.ry, response.rz])
        self.get_logger().info(f"tvec (camera frame): x={tvec[0]:.3f}m, y={tvec[1]:.3f}m, z={tvec[2]:.3f}m")
        tvec4 = np.array([response.rx, response.ry, response.rz, 1.0])
        T_camera_to_marker = np.zeros((4, 4), dtype=np.float32)
        T_camera_to_marker[0:3, 0:3] = R.T   
        T_camera_to_marker[:-1,3] = R.T @ (-tvec)
        T_camera_to_marker[3, 3] = 1
        T_marker_to_base = np.eye(4)
        T_marker_to_base[:3, 3] = np.array([0.367, 0.0119, -0.024])
        Trans = T_marker_to_base @ T_camera_to_marker
        np.save("translation_matrix", Trans)
        self.get_logger().info(f"Translation matrix saved to 'translation_matrix.npy'")

        self.get_logger().info(f"Transformation matrix:\n{Trans}")
        self.get_logger().info(f"Transformed tvec4: {Trans @ tvec4.T}")
        
        return Trans

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    rclpy.spin(aruco_detector)
    aruco_detector.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
