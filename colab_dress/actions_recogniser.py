from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ActionsPerf:
    def __init__(
        self,
        device: str = "cpu",
        camid: Union[int, str] = 0,
        model_path: str = "/home/hguda/colab_dress_ws/src/checkpoint-120",
        confidence_threshold: float = 0.9,
    ) -> None:
        self.device = device
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.confidence_threshold = confidence_threshold

        self.image_processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_path, attn_implementation="sdpa"
        ).to(self.device)
        model_name = get_model_list()[0]
        self.fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=self.device)
        self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=self.device)
        self.engagement_supported = getattr(self.fer, "classifier_weights", None) is not None and self.fer.classifier_weights.shape[1] == 2560

        self.camid = camid
        self.cap = cv2.VideoCapture(self.camid)
        # flip the frame upside down as the camera is mounted upside down

        

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.video: Optional[np.ndarray] = None
        self.i = 0
        self.action_label = ""
        self.last_confidence = 0.0
        self.all_frames: List[np.ndarray] = []
        self.emotion = "none"
        self.engagement = "none"

    def _ensure_video_buffer(self, frame: np.ndarray) -> None:
        if self.video is None or self.video.shape[1:] != frame.shape:
            self.video = np.empty((16, *frame.shape), dtype=frame.dtype)
            self.i = 0

    def predict_actions(self, frame: np.ndarray) -> Tuple[str, float]:
        self._ensure_video_buffer(frame)
        self.video[self.i] = frame.copy()
        self.i += 1
        if self.i == 16:
            self.i = 0
            inputs = self.image_processor(list(self.video), return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)

            predicted_label = logits.argmax(-1).item()
            action_label = self.model.config.id2label[predicted_label]
            confidence = float(probabilities[0, predicted_label].item())
            self.video = np.empty((16, *frame.shape), dtype=frame.dtype)

            if confidence > self.confidence_threshold:
                self.action_label = action_label
                self.last_confidence = confidence
            else:
                self.action_label = ""
                self.last_confidence = confidence

        return self.action_label, self.last_confidence

    def recognize_faces(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        bounding_boxes, probs = self.mtcnn.detect(frame, landmarks=False)
        if bounding_boxes is None or probs is None:
            return [], []

        bounding_boxes = bounding_boxes[probs > 0.9]
        facial_images = []
        for bbox in bounding_boxes:
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            facial_images.append(frame[y1:y2, x1:x2, :])
        return facial_images, bounding_boxes

    def predict_engagement(self, frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
        display_emotion = self.emotion
        display_engagement = self.engagement

        facial_images, bboxes = self.recognize_faces(frame)
        if facial_images:
            emotions, _ = self.fer.predict_emotions(facial_images, logits=True)
            if emotions:
                self.emotion = emotions[0]
                display_emotion = self.emotion

            if self.engagement_supported:
                self.all_frames.extend(facial_images)
                if len(self.all_frames) >= 10:
                    engagements, _ = self.fer.predict_engagement(
                        self.all_frames, sliding_window_width=10
                    )
                    if engagements:
                        self.engagement = engagements[0]
                    self.all_frames = []
                display_engagement = self.engagement
            else:
                self.engagement = "none"
                display_engagement = self.engagement

            for bbox in bboxes:
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            display_emotion = "none"

        cv2.putText(
            frame,
            f"{display_emotion}, {display_engagement}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        return frame, display_emotion, display_engagement

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class ActionsRecogniserNode(Node):
    def __init__(self) -> None:
        super().__init__("actions_recogniser")

        self.declare_parameter("camera", "0")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("model_path", "/home/hguda/colab_dress_ws/src/checkpoint-120")
        self.declare_parameter("confidence_threshold", 0.9)
        self.declare_parameter("timer_period", 0.05)
        self.declare_parameter("topic", "/actions_recognised")
        self.declare_parameter("emotions_topic", "/engagement/emotions")
        self.declare_parameter("engagement_topic", "/engagement/status")

        camera = self.get_parameter("camera").value
        if isinstance(camera, str):
            camera_str = camera.strip()
            if camera_str.isdigit():
                camera = int(camera_str)
            else:
                camera = camera_str
        device = self.get_parameter("device").value
        model_path = self.get_parameter("model_path").value
        confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        timer_period = float(self.get_parameter("timer_period").value)
        topic = self.get_parameter("topic").value
        emotions_topic = self.get_parameter("emotions_topic").value
        engagement_topic = self.get_parameter("engagement_topic").value

        self.publisher_ = self.create_publisher(String, topic, 10)
        self.emotions_publisher = self.create_publisher(String, emotions_topic, 10)
        self.engagement_publisher = self.create_publisher(String, engagement_topic, 10)

        self.actions = ActionsPerf(
            device=device,
            camid=camera,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        if not self.actions.engagement_supported:
            self.get_logger().warning(
                "EmotiEffLib engagement inference is unavailable with the installed models; publishing 'none' on /engagement/status."
            )

        self.last_published: Optional[str] = None
        # self.window_name = "Action Recognition + Engagement"
        # cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Publishing actions on {topic}")
        self.get_logger().info(f"Publishing emotions on {emotions_topic}")
        self.get_logger().info(f"Publishing engagement on {engagement_topic}")
        self.get_logger().info(f"Using device: {device} (camera={camera}, type={type(camera).__name__})")

    def timer_callback(self) -> None:
        success, frame = self.actions.cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not success:
            self.get_logger().warning("Could not read frame from camera")
            return

        display_frame = frame.copy()

        action_label, confidence = self.actions.predict_actions(frame)
        if action_label:
            cv2.putText(
                display_frame,
                f"{action_label}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

        emotion = "none"
        engagement = "none"
        try:
            display_frame, emotion, engagement = self.actions.predict_engagement(display_frame)
        except Exception as exc:
            self.get_logger().warning(f"Engagement inference failed, showing raw frame: {exc}")

        cv2.imshow("Intention", display_frame)
        cv2.waitKey(1)
        if action_label:
            msg = String()
            msg.data = action_label
            self.publisher_.publish(msg)
            self.last_published = action_label
            self.get_logger().info(
                f"Published action: {action_label} (confidence={confidence:.2f})"
            )

        # self.emotions_publisher.publish(String(data=emotion))
        # self.engagement_publisher.publish(String(data=engagement))

    def destroy_node(self):
        if hasattr(self, "actions"):
            self.actions.close()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActionsRecogniserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
