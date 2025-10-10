import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.node import Node
from colab_dress_interfaces.msg import ArucoMarker
from colab_dress.get_3d_point_client import Get3DPointClient
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
                 color_info_topic="/camera/camera/color/camera_info",
                 aruco_dict_type="DICT_5X5_50"):
        super().__init__('aruco_detector')

        # Parameters
        self.aruco_dict_type = aruco_dict_type
        self.bridge = CvBridge()

        # Subscriptions
        self.color_image_subscription = self.create_subscription(
            Image,
            color_image_topic,
            self.color_image_callback,
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

        self.get_logger().info(f'ArUco detector initialized with {aruco_dict_type}')

        self.get_3d_point = Get3DPointClient()

    def camera_info_callback(self, msg):
        if self.matrix_coefficients.sum() == 0 and self.coeffs.sum() == 0:
            self.matrix_coefficients = np.array(msg.k).reshape(3, 3)
            self.coeffs = np.array(msg.d)
            self.get_logger().info('Camera calibration data received')

    def color_image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.detect_and_publish_aruco_markers(color_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_and_publish_aruco_markers(self, frame, header):
        """
        Detect ArUco markers in the image and publish each one as a separate message
        
        Args:
            frame: Input image frame (BGR)
            header: ROS message header from the original image
        """
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
                    response = self.get_3d_point.send_request(int(px), int(py))
            # Optional: Draw detected markers for visualization
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
            
            self.get_logger().info(f'Detected and published {len(ids)} ArUco marker(s)')
        cv2.imshow("ArUco Markers", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            
            if ids is not None and len(ids) > 0:
                self.save_translation_matrix(rvec, response)
                self.get_logger().info('Translation matrix saved.')
                cv2.destroyAllWindows()
                self.destroy_node()
                sys.exit(1)
                rclpy.shutdown()

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
        Trans = np.zeros((4, 4), dtype=np.float32)
        Trans[0:3, 0:3] = R.T
        Trans[:-1,3] = R.T @ (-tvec)
        Trans[3, 3] = 1
        # Trans[0, 3] += 0.15 # Adjust for arucco offset from robot base
        np.save("translation_matrix", Trans)
        print(Trans, R, tvec)
        # print(Trans @ tvec4.T, Trans, tvec4)
        
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
