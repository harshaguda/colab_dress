import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from typing import List
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

class EngagementDetectorNode(Node):
    def __init__(self,
                 device='cpu', 
                 color_image_topic="/camera/camera/color/image_raw"
                 ):
        super().__init__('emotions_node')
        self.device = device
        self.fer = None  # Placeholder for the FER model
        self.all_scores = None
        self.engage_flag = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = get_model_list()[0]

        self.fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=self.device)

        self.all_frames = []
        self.all_scores = None

        engage_flag = False
        self.engagement = "none"
        self.emotion = "none"

        self.bridge = CvBridge()

        # Subscriptions
        self.color_image_subscription = self.create_subscription(
            Image,
            color_image_topic,
            self.color_image_callback,
            qos_profile_sensor_data)
        
        # Publishers
        self.emotions_publisher = self.create_publisher(String, '/engagement/emotions', 10)
        self.engagement_publisher = self.create_publisher(String, '/engagement/status', 10)
    
    def color_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        image, _, _ = self.predict_emotions(cv_image)
        cv2.imshow("Engagement Detector", image)
        cv2.waitKey(1)

    def recognize_faces(self, frame: np.ndarray, device: str) -> List[np.array]:
        # Placeholder for face recognition logic
        """
        Detects faces in the given image and returns the facial images cropped from the original.

        This function reads an image from the specified path, detects faces using the MTCNN
        face detection model, and returns a list of cropped face images.

        Args:
            frame (numpy.ndarray): The image frame in which faces need to be detected.
            device (str): The device to run the MTCNN face detection model on, e.g., 'cpu' or 'cuda'.

        Returns:
            list: A list of numpy arrays, representing a cropped face image from the original image.

        Example:
            faces = recognize_faces('image.jpg', 'cuda')
            # faces contains the cropped face images detected in 'image.jpg'.
        """

        def detect_face(frame: np.ndarray):
            mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
            bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
            if probs[0] is None:
                return []
            bounding_boxes = bounding_boxes[probs > 0.9]
            return bounding_boxes

        bounding_boxes = detect_face(frame)
        facial_images = []
        for bbox in bounding_boxes:
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            facial_images.append(frame[y1:y2, x1:x2, :])
        return facial_images, bounding_boxes
    
    

    def predict_emotions(self, image):
        # Placeholder for emotion prediction logic
    
        facial_images, bboxes = self.recognize_faces(image, self.device)
        
        if len(facial_images) != 0:
            for bbox in bboxes:
                if bbox.any() < 0:
                    continue
            emotions, scores = self.fer.predict_emotions(facial_images, logits=True)
            self.emotion = emotions[0]
            self.all_frames += facial_images
            if len(self.all_frames) > 10:
                self.all_frames = self.all_frames[-30:]
                engagements, scores = self.fer.predict_engagement(self.all_frames, sliding_window_width=10)
                self.all_frames = []
                self.engagement = engagements[0]
        
            for bbox in bboxes:
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{emotions[0]}, {self.engagement}", (0,410), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        self.emotions_publisher.publish(String(data=self.emotion))
        self.engagement_publisher.publish(String(data=self.engagement))
        
        return image, self.emotion, self.engagement

def main():
    rclpy.init()
    node = EngagementDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()