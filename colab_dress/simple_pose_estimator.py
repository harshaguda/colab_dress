#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import pyrealsense2 as rs2  # Import pyrealsense2
from cv_bridge import CvBridge
from typing import Optional

class SimplePoseEstimator(Node):
    def __init__(self):
        super().__init__('simple_pose_estimator')
        
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('real_robot', True)
        self.declare_parameter('franka_pose_topic', '/franka_robot_state_broadcaster/current_pose')
        self.declare_parameter('display_image_topic', '/pose_estimator/display_image/compressed')
        
        topic_name = self.get_parameter('input_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self._real_robot = bool(self.get_parameter('real_robot').value)
        self._franka_pose_topic = self.get_parameter('franka_pose_topic').get_parameter_value().string_value
        self._display_image_topic = self.get_parameter('display_image_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.intrinsics = None
        self.depth_image = None
        
        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.pose_3d_pub = self.create_publisher(PoseArray, '/pose_estimator/pose_3d', 10)
        self.display_image_pub = self.create_publisher(CompressedImage, self._display_image_topic, 10)
        self._last_franka_pose: Optional[PoseStamped] = None
        if self._real_robot:
            self.franka_pose_sub = self.create_subscription(
                PoseStamped,
                self._franka_pose_topic,
                self.franka_pose_callback,
                10
            )
        
        self.get_logger().info(f'Simple Pose Estimator started. Subscribed to: {topic_name}')
        self.get_logger().info(f'Publishing display image on: {self._display_image_topic}')
        
        # Determine valid model path
        possible_paths = [
            # 'src/colab_dress/resource/pose_landmarker_full.task',
            # 'share/colab_dress/resource/pose_landmarker_full.task',
            os.path.join(os.getcwd(), 'src/colab_dress/resource/pose_landmarker_heavy.task')
        ]
        self.model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                self.model_path = p
                break
        
        # Initialize MediaPipe
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
                self.get_logger().error(f"Model file not found in paths: {possible_paths}. CWD: {os.getcwd()}")
                # Fallback to hardcoded absolute path if possible or raise error
                # For this specific user env, I know it exists in:
                fallback = '/home/hguda/colab_dress_ws/src/colab_dress/resource/pose_landmarker_heavy.task'
                if os.path.exists(fallback):
                    self.model_path = fallback
                else:
                    raise FileNotFoundError("MediaPipe Tasks API requires 'pose_landmarker_heavy.task'. Model not found.")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
        self.translate = True
        if self.translate:
            self.translation = np.load('translation_matrix.npy')  # Load translation matrix if needed
        

        self.prev_time = time.time()
        self.get_logger().info(f'Simple Pose Estimator started. Subscribed to: {topic_name}')

    def franka_pose_callback(self, msg: PoseStamped):
        self._last_franka_pose = msg

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def camera_info_callback(self, msg):
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
            self.get_logger().info("Camera Intrinsics Received")
        except Exception as e:
            self.get_logger().error(f"Failed to parse camera info: {e}")

    def listener_callback(self, msg):
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                self.get_logger().warn("Failed to decode image from compressed message")
                return

            # MediaPipe needs RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.use_solutions:
                image_rgb.flags.writeable = False
                results = self.pose.process(image_rgb)
                
                image.flags.writeable = True
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
            else:
                # Tasks API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                try:
                    results = self.pose.detect_for_video(mp_image, timestamp_ms)
                except ValueError as e:
                    self.get_logger().warn(f"Tasks API Value Error: {e}")
                    return

                if results.pose_landmarks:
                    for landmarks in results.pose_landmarks:
                        h, w, _ = image.shape
                        points = []
                        for lm in landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            points.append((cx, cy))
                            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            
                            # Get 3D Point
                            if self.intrinsics and self.depth_image is not None:
                                d_h, d_w = self.depth_image.shape
                                if 0 <= cx < d_w and 0 <= cy < d_h:
                                    dist = self.depth_image[cy, cx]
                                    # Handle uint16 (mm) vs float (m)
                                    if self.depth_image.dtype == np.uint16:
                                        dist_m = dist * 0.001
                                    else:
                                        dist_m = float(dist)
                                        
                                    if dist_m > 0:
                                        p3d = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [float(cx), float(cy)], dist_m)
                                        # Log select keypoints (e.g. wrist=16) 
                                        # But here we are looping all. Maybe just print one?
                                        # For verification, let's just draw text for the first point or wrist
                                        pass
                        
                        # Show 3D coords for right wrist (16) if available
                        if len(points) > 16 and self.intrinsics and self.depth_image is not None:
                            pose_msg = PoseArray()
                            pose_msg.header.stamp = self.get_clock().now().to_msg()
                            pose_msg.header.frame_id = "camera_link"

                            for idx in [16, 14, 12]:  # Right shoulder, elbow, wrist
                                rw_x, rw_y = points[idx]
                                d_h, d_w = self.depth_image.shape
                                if 0 <= rw_x < d_w and 0 <= rw_y < d_h:
                                    dist = self.depth_image[rw_y, rw_x]
                                    dist_m = dist * 0.001 if self.depth_image.dtype == np.uint16 else float(dist)
                                    if dist_m > 0:
                                        rw_3d = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [float(rw_x), float(rw_y)], dist_m)
                                        P = np.array([[rw_3d[0]], [rw_3d[1]], [rw_3d[2]], [1]])
                                        
                                        p_final = rw_3d
                                        if self.translate:
                                            rw_3d_trans = self.translation @ P
                                            p_final = [rw_3d_trans[0][0], rw_3d_trans[1][0], rw_3d_trans[2][0]]
                                            label = f"RW: {rw_3d_trans[0][0]:.2f}, {rw_3d_trans[1][0]:.2f}, {rw_3d_trans[2][0]:.2f}"
                                        else:
                                            label = f"RW: {rw_3d[0]:.2f}, {rw_3d[1]:.2f}, {rw_3d[2]:.2f}"
                                        cv2.putText(image, label, (rw_x, rw_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                                        p = Pose()
                                        p.position.x = float(p_final[0])
                                        p.position.y = float(p_final[1])
                                        p.position.z = float(p_final[2])
                                        pose_msg.poses.append(p)
                                    else:
                                        pose_msg.poses.append(Pose())
                                else:
                                    pose_msg.poses.append(Pose())

                            if self._real_robot and self._last_franka_pose is not None:
                                franka_pose = Pose()
                                franka_pose.position = self._last_franka_pose.pose.position
                                franka_pose.orientation = self._last_franka_pose.pose.orientation
                                pose_msg.poses.insert(0, franka_pose)
                            
                            self.pose_3d_pub.publish(pose_msg)

                        # Basic torso connections
                        connections = [
                            # (11, 12), (11, 13), (13, 15), 
                            (12, 14), (14, 16), # Arms
                            # (11, 23), (12, 24), (23, 24), # Torso
                            # (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32) # Legs
                        ]
                        
                        for p1, p2 in connections:
                            if p1 < len(points) and p2 < len(points):
                                cv2.line(image, points[p1], points[p2], (0, 255, 0), 2)

            # FPS
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            fps = 1.0 / dt if dt > 0 else 0
            
            cv2.putText(
                image,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            cv2.imshow('Simple Pose Estimator', image)
            ok, encoded_image = cv2.imencode('.jpg', image)
            if ok:
                image_msg = CompressedImage()
                image_msg.header.stamp = self.get_clock().now().to_msg()
                image_msg.format = 'jpeg'
                image_msg.data = encoded_image.tobytes()
                self.display_image_pub.publish(image_msg)
            key = cv2.waitKey(1)
            if key == 27: # ESC
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = SimplePoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
