from calendar import c
import enum
import mediapipe as mp

import numpy as np
import pyrealsense2 as rs
import cv2
import time
import math
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy

import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Pose2D, Pose, PoseArray
from colab_dress_interfaces.msg import Pose2DArray
from colab_dress.get_3d_point_client import Get3DPointClient
from visualization_msgs.msg import Marker

class PoseEstimator(Node):
    def __init__(self, 
                 color_image_topic="/camera/camera/color/image_raw",
                 debug=False, 
                 translate=False):
        
        super().__init__('pose_estimator')                    
    
        
        self.debug = debug
        self.no_smooth_landmarks = False
        self.static_image_mode = False
        self.model_complexity = 0
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.model_path = 'src/colab_dress/resource/pose_landmarker_full.task'

        # Define landmark indices for the new API
        self.WRIST = 16  # RIGHT_WRIST
        self.ELBOW = 14  # RIGHT_ELBOW  
        self.SHOULDER = 12  # RIGHT_SHOULDER
        
        self.declare_parameter('use_solutions', True)
        self.declare_parameter('model_complexity', 0)
        self.declare_parameter('static_image_mode', False)
        self.declare_parameter('smooth_landmarks', True)
        self.declare_parameter('depth_image_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')

        self.model_complexity = int(self.get_parameter('model_complexity').value)
        self.static_image_mode = bool(self.get_parameter('static_image_mode').value)
        self.smooth_landmarks = bool(self.get_parameter('smooth_landmarks').value)
        depth_image_topic = self.get_parameter('depth_image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        # Determine valid model path like in simple_pose_estimator.py
        import os
        possible_paths = [
            'src/colab_dress/resource/pose_landmarker_full.task',
            'share/colab_dress/resource/pose_landmarker_full.task',
            os.path.join(os.getcwd(), 'src/colab_dress/resource/pose_landmarker_full.task'),
            '/home/hguda/colab_dress_ws/src/colab_dress/resource/pose_landmarker_full.task' # Fallback
        ]
        self.model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                self.model_path = p
                break
        
        self.use_solutions = hasattr(mp, 'solutions')

        if self.use_solutions:
            self.get_logger().info("Using MediaPipe Solutions API")
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.get_logger().info("Using MediaPipe Tasks API")
            if not self.model_path:
                self.get_logger().error(f"Model file not found in paths: {possible_paths}.")
                raise FileNotFoundError("Model file not found")
            
            # Use tasks API with VIDEO mode for better performance
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
        
        self.bridge = CvBridge()
        
        self.intrinsics = None
        self.depth_image = None

        # Subscriptions
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data)
            
        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            qos_profile_sensor_data)

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

        self.pose_publisher = self.create_publisher(Pose2DArray, '/pose_estimator/pose_2d', 10)
        self.pose3d_publisher = self.create_publisher(PoseArray, '/pose_estimator/pose_3d', 10)
        self.arm_points_marker_pub = self.create_publisher(Marker, '/pose_estimator/arm_points', 10)

        self.declare_parameter('marker_frame_id', '')  # default: use incoming image header frame_id
        self.declare_parameter('marker_scale', 0.05)  # 5cm spheres - easier to see in RViz
        self.declare_parameter('use_3d_points', False)
        self.declare_parameter('enable_display', False)
        self.declare_parameter('resize_width', 320)
        self.declare_parameter('resize_height', 240)

        self.translate = bool(translate) and bool(self.get_parameter('use_3d_points').value)
        # self.get_3d_point = Get3DPointClient() if self.translate else None
        
        self._resize_w = int(self.get_parameter('resize_width').value)
        self._resize_h = int(self.get_parameter('resize_height').value)
        self._enable_display = bool(self.get_parameter('enable_display').value)

        self._last_frame_time = time.time()
        self._fps = 0.0
        self._fps_history = []
        self._fps_window = 30  # Rolling average over 30 frames
        
        if self.translate:
            self.get_logger().info('Publishing arm keypoints as markers on /pose_estimator/arm_points')
        else:
            self.get_logger().info('3D point service disabled; publishing 2D poses only')
        
        self.get_logger().info(f'Display enabled: {self._enable_display}')
        self.get_logger().info(f'Resize: {self._resize_w}x{self._resize_h}')
        self.get_logger().info('Pose estimator ready')

    def camera_info_callback(self, msg):
        if self.intrinsics:
            return
        
        try:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = msg.width
            self.intrinsics.height = msg.height
            self.intrinsics.fx = msg.k[0]
            self.intrinsics.fy = msg.k[4]
            self.intrinsics.ppx = msg.k[2]
            self.intrinsics.ppy = msg.k[5]
            
            if msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs.distortion.brown_conrady
            elif msg.distortion_model == 'equidistant':
                self.intrinsics.model = rs.distortion.kannala_brandt4
            
            # This mapping depends on the exact storage implementation of Rs2 in python
            # Usually K is 9 elements, D is array.
            self.intrinsics.coeffs = [i for i in msg.d]
            self.get_logger().info('Camera intrinsics received')
        except Exception as e:
            self.get_logger().error(f'Failed to parse camera info: {e}')

    def depth_image_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def color_image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.estimate_pose(color_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def color_compressed_image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if color_image is None:
                raise ValueError("Failed to decode compressed image")
            self.estimate_pose(color_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {str(e)}')

        
    def estimate_pose(self, image, header):
        frame_start = time.time()
        now = frame_start
        dt = now - self._last_frame_time
        if dt > 0:
            instantaneous_fps = 1.0 / dt
            self._fps_history.append(instantaneous_fps)
            if len(self._fps_history) > self._fps_window:
                self._fps_history.pop(0)
            self._fps = sum(self._fps_history) / len(self._fps_history)
        self._last_frame_time = now

        h, w, _ = image.shape
        # Convert BGR to RGB as MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmarks = None
        
        if self.use_solutions:
            # Legacy Solutions API
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
        else:
            # Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(time.time() * 1000)
            
            try:
                results = self.pose.detect_for_video(mp_image, timestamp_ms)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]
            except ValueError as e:
                self.get_logger().warn(f"Tasks API Value Error: {e}")
                landmarks = None

        shoulder_verts = []
        required_landmarks = [self.WRIST, self.ELBOW, self.SHOULDER]
        
        poses = Pose2DArray()
        poses3d = PoseArray()
        poses3d.header = header
        if landmarks:
            for landmark_idx in required_landmarks:
                pose = Pose2D()
                pose3d = Pose()
                landmark_px = landmarks[landmark_idx]
                x_px = min(math.floor(landmark_px.x * w), w - 1)
                y_px = min(math.floor(landmark_px.y * h), h - 1)
                shoulder_verts.append([x_px, y_px])
                pose.x = float(x_px)
                pose.y = float(y_px)
                poses.poses.append(pose)
                if self.translate:
                    # Logic to get 3D point locally
                    if self.intrinsics and self.depth_image is not None:
                        # Check bounds
                        d_h, d_w = self.depth_image.shape
                        if 0 <= x_px < d_w and 0 <= y_px < d_h:
                            depth_val = self.depth_image[y_px, x_px]
                            # Deproject
                            # Depth is likely in mm or meters. rs2 usually in MM.
                            # Standard ROS depth is float meters (32FC1) or uint16 mm (16UC1).
                            # If uint16, typically mm. If float, meters.
                            # rs.deproject expects meters.
                            
                            depth_meters = 0.0
                            if self.depth_image.dtype == np.uint16:
                                depth_meters = depth_val * 0.001
                            else:
                                depth_meters = float(depth_val)
                                
                            point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(x_px), float(y_px)], depth_meters)
                            
                            pose3d.position.x = point_3d[0]
                            pose3d.position.y = point_3d[1]
                            pose3d.position.z = point_3d[2]
                        else:
                            self.get_logger().warn(f"Pixel {x_px},{y_px} out of depth image bounds {d_w}x{d_h}")
                    else:
                        if not self.intrinsics:
                             self.get_logger().debug("Waiting for intrinsics...")
                        if self.depth_image is None:
                             self.get_logger().debug("Waiting for depth image...")

                    poses3d.poses.append(pose3d)

            if self.translate:
                self.pose3d_publisher.publish(poses3d)
                self.publish_arm_points_marker(poses3d, header)
            self.pose_publisher.publish(poses)

        for i, vert in enumerate(shoulder_verts):
            x_px, y_px = vert
            
        if self.debug and self._enable_display:
            image.flags.writeable = True            
            for i, vert in enumerate(shoulder_verts):
                x_px, y_px = vert
                # Draw a circle at the landmark position
                image = cv2.circle(image, center=(x_px, y_px), radius=5, color=(0, 0, 255), thickness=-1)
                if i != 0:
                    x_px1, y_px1 = shoulder_verts[i-1]
                    image = cv2.line(image, (x_px, y_px), (x_px1, y_px1), (0, 255, 0), 2)

            cv2.putText(
                image,
                f"FPS: {self._fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow('RealSense Pose Detector', image)

            if cv2.waitKey(1) & 0xFF == 27:
                exit()
        else:
            self.get_logger().info(f'FPS: {self._fps:.1f}')
        
        elapsed = time.time() - frame_start
        if elapsed > 0.05:  # Log if frame takes > 50ms
            self.get_logger().warn(f'Slow frame: {elapsed*1000:.1f}ms')

    def publish_arm_points_marker(self, poses3d: PoseArray, header):
        if not poses3d.poses:
            return

        marker = Marker()
        # Use poses3d header (base frame) not camera header - coordinates are already in base frame
        marker.header = poses3d.header
        self.get_logger().info('Marker frame id: ' + marker.header.frame_id)

        marker.ns = 'pose_estimator'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        scale = float(self.get_parameter('marker_scale').get_parameter_value().double_value)
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0

        marker.points = [Point(x=p.position.x, y=p.position.y, z=p.position.z) for p in poses3d.poses]
        self.arm_points_marker_pub.publish(marker)
def main():
    rclpy.init()
    pose_estimator = PoseEstimator(debug=True, translate=True)
    rclpy.spin(pose_estimator)
    pose_estimator.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()