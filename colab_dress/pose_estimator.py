from calendar import c
import enum
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
from sensor_msgs.msg import Image
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
        self.translate = translate
        self.no_smooth_landmarks = False
        self.static_image_mode = True
        self.model_complexity = 1
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.model_path = 'src/colab_dress/resource/pose_landmarker_heavy.task'

        # Define landmark indices for the new API
        self.WRIST = 16  # RIGHT_WRIST
        self.ELBOW = 14  # RIGHT_ELBOW  
        self.SHOULDER = 12  # RIGHT_SHOULDER
        
        # Create PoseLandmarker with new API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        
        self.bridge = CvBridge()

        # Subscriptions
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
        self.get_3d_point = Get3DPointClient()
        
        self.get_logger().info('Publishing arm keypoints as markers on /pose_estimator/arm_points')

    def color_image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
        
        self.estimate_pose(color_image, msg.header)

        
    def estimate_pose(self, image, header):
        h, w, _ = image.shape
        # Convert BGR to RGB as MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = self.pose.detect(mp_image)
        shoulder_verts = []
        required_landmarks = [self.WRIST, self.ELBOW, self.SHOULDER]
        
        poses = Pose2DArray()
        poses3d = PoseArray()
        poses3d.header = header
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]  # Get first detected pose
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
                    response = self.get_3d_point.send_request(int(x_px), int(y_px))
                    pose3d.position.x = response.rx
                    pose3d.position.y = response.ry
                    pose3d.position.z = response.rz
                    poses3d.poses.append(pose3d)

            if self.translate:
                self.pose3d_publisher.publish(poses3d)
                self.publish_arm_points_marker(poses3d, header)
            self.pose_publisher.publish(poses)

        for i, vert in enumerate(shoulder_verts):
            x_px, y_px = vert
            
        if self.debug:
            image.flags.writeable = True            
            for i, vert in enumerate(shoulder_verts):
                x_px, y_px = vert
                # Draw a circle at the landmark position
                image = cv2.circle(image, center=(x_px, y_px), radius=5, color=(0, 0, 255), thickness=-1)
                if i != 0:
                    x_px1, y_px1 = shoulder_verts[i-1]
                    image = cv2.line(image, (x_px, y_px), (x_px1, y_px1), (0, 255, 0), 2)



        cv2.imshow('RealSense Pose Detector', image)

        if cv2.waitKey(1) & 0xFF == 27:
            exit()

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