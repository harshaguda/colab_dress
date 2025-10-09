from calendar import c
import enum
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
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
from geometry_msgs.msg import Pose2D, Pose, PoseArray
from colab_dress_interfaces.msg import Pose2DArray
from colab_dress.get_3d_point_client import Get3DPointClient

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
        self.model_path = 'pose_landmarker.task'

        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            smooth_landmarks=self.no_smooth_landmarks,
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
            )
        
        self.bridge = CvBridge()

        # Subscriptions
        self.color_image_subscription = self.create_subscription(
            Image,
            color_image_topic,
            self.color_image_callback,
            qos_profile_sensor_data)

        self.pose_publisher = self.create_publisher(Pose2DArray, '/pose_estimator/pose_2d', 10)
        self.pose3d_publisher = self.create_publisher(PoseArray, '/pose_estimator/pose_3d', 10)
        self.get_3d_point = Get3DPointClient()

    def color_image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.estimate_pose(color_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # print(pose_landmarks, idx)
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def estimate_pose(self, image):
        h, w, _ = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.pose.process(image)
        shoulder_verts = []
        required_landmarks = [
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]
        poses = Pose2DArray()
        poses3d = PoseArray()
        if results.pose_landmarks is not None:
            for landmark in required_landmarks:
                pose = Pose2D()
                pose3d = Pose()
                landmark_px = results.pose_landmarks.landmark[landmark]
                x_px = min(math.floor(landmark_px.x * w), w - 1)
                y_px = min(math.floor(landmark_px.y * h), h - 1)
                shoulder_verts.append([x_px, y_px])
                pose.x = float(x_px)
                pose.y = float(y_px)
                poses.poses.append(pose)
            #     if self.translate:
            #         response = self.get_3d_point.send_request(int(x_px), int(y_px))
            #         pose3d.x = response.rx
            #         pose3d.y = response.ry
            #         pose3d.z = response.rz
            #         poses3d.poses.append(pose3d)

            # if self.translate:
            #     self.pose3d_publisher.publish(poses3d)
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
def main():
    rclpy.init()
    pose_estimator = PoseEstimator(debug=True, translate=True)
    rclpy.spin(pose_estimator)
    pose_estimator.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()