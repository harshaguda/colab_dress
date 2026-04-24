from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String


class ActionsPerf:
    def __init__(
        self,
        device: str = "cpu",
        camid: Union[int, str] = 0,
        model_path: str = "/home/hguda/colab_dress_ws/src/checkpoint-120",
        confidence_threshold: float = 0.9,
        frame_rotation: int = 0,
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
        self.engagement_supported = hasattr(self.fer, "predict_engagement")

        self.camid = camid
        self.frame_rotation = frame_rotation if frame_rotation in (0, 90, 180, 270) else 0
        self.cap = cv2.VideoCapture(self.camid)
        
        # Force deterministic orientation from OpenCV/camera backend.
        # Disable backend auto-rotation so only `frame_rotation` controls orientation.
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

        cv2.waitKey(1000)  # Allow OpenCV to initialize the camera
        success, frame = self.cap.read()
        cv2.waitKey(1000)  # Allow camera to warm up and provide valid frames
        success, frame = self.cap.read()

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.video: Optional[np.ndarray] = None
        self.i = 0
        self.action_label = ""
        self.last_confidence = 0.0
        self.face_frame_buffer: Deque[np.ndarray] = deque(maxlen=10)
        self.attention_history: Deque[bool] = deque(maxlen=5)
        self.emotion = "none"
        self.engagement = "none"
        self.non_adaptive = False

    def normalize_frame_orientation(self, frame: np.ndarray) -> np.ndarray:
        
        if self.frame_rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.frame_rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.frame_rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

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

    def recognize_faces(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        bounding_boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
        if bounding_boxes is None or probs is None or landmarks is None:
            return [], [], []

        keep = probs > 0.7
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        facial_images = []
        valid_bboxes = []
        valid_landmarks = []
        for bbox, lms in zip(bounding_boxes, landmarks):
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            facial_images.append(frame[y1:y2, x1:x2, :])
            valid_bboxes.append(bbox)
            valid_landmarks.append(lms)
        return facial_images, valid_bboxes, valid_landmarks

    def _infer_attention_from_face_box(self, bbox: np.ndarray, frame_shape: Tuple[int, int, int]) -> str:
        x1, y1, x2, y2 = bbox.astype(int)[0:4]
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        face_area_ratio = (face_w * face_h) / float(frame_shape[0] * frame_shape[1])
        size_ok = face_area_ratio >= 0.03

        return "paying_attention" if size_ok else "not_paying_attention"

    def _normalize_engagement_label(self, label: str) -> Optional[str]:
        normalized = label.lower().strip().replace("-", "_").replace(" ", "_")
        if normalized in {"paying_attention", "engaged", "engagement", "attentive", "focused"}:
            return "paying_attention"
        if normalized in {
            "not_paying_attention",
            "disengaged",
            "distracted",
            "not_engaged",
            "unfocused",
            "inattentive",
        }:
            return "not_paying_attention"
        return None

    def _infer_attention_from_landmarks(self, landmarks: np.ndarray) -> str:
        # MTCNN: [left_eye, right_eye, nose, mouth_left, mouth_right]
        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist < 8.0:
            return "not_paying_attention"

        eye_mid = (left_eye + right_eye) / 2.0
        mouth_mid = (mouth_left + mouth_right) / 2.0

        nose_center_offset = abs(float(nose[0] - eye_mid[0])) / eye_dist
        mouth_center_offset = abs(float(mouth_mid[0] - eye_mid[0])) / eye_dist
        eye_roll = abs(float(left_eye[1] - right_eye[1])) / eye_dist

        eye_to_mouth = float(mouth_mid[1] - eye_mid[1])
        if eye_to_mouth <= 1.0:
            return "not_paying_attention"
        nose_vertical_ratio = float((nose[1] - eye_mid[1]) / eye_to_mouth)

        looking = (
            nose_center_offset < 0.20
            and mouth_center_offset < 0.22
            and eye_roll < 0.16
            and 0.25 <= nose_vertical_ratio <= 0.82
        )
        return "paying_attention" if looking else "not_paying_attention"

    def predict_engagement(self, frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
        display_emotion = self.emotion
        display_engagement = self.engagement

        if self.non_adaptive:
            display_emotion = "none"
            display_engagement = "paying_attention"
            return frame, display_emotion, display_engagement

        facial_images, bboxes, landmarks = self.recognize_faces(frame)
        if facial_images:
            # Keep only the primary (largest) face for stable temporal engagement inference.
            if len(bboxes) > 1:
                areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
                best_idx = int(np.argmax(areas))
            else:
                best_idx = 0

            facial_images = [facial_images[best_idx]]
            bboxes = np.array([bboxes[best_idx]])
            landmarks = np.array([landmarks[best_idx]])

            emotions, _ = self.fer.predict_emotions(facial_images, logits=True)
            if emotions:
                self.emotion = emotions[0]
                display_emotion = self.emotion

            if self.engagement_supported:
                self.face_frame_buffer.append(facial_images[0])
                model_engagement = None
                if len(self.face_frame_buffer) == self.face_frame_buffer.maxlen:
                    try:
                        engagements, _ = self.fer.predict_engagement(
                            list(self.face_frame_buffer),
                            sliding_window_width=self.face_frame_buffer.maxlen,
                        )
                        if engagements:
                            model_engagement = self._normalize_engagement_label(
                                str(engagements[0])
                            )
                    except Exception:
                        model_engagement = None

                if model_engagement is not None:
                    self.engagement = model_engagement
                else:
                    landmark_attention = self._infer_attention_from_landmarks(landmarks[0])
                    # Camera-position-independent decision: based on facial geometry.
                    # Use face-size fallback only when landmarks indicate attention.
                    if landmark_attention == "paying_attention":
                        self.engagement = self._infer_attention_from_face_box(bboxes[0], frame.shape)
                    else:
                        self.engagement = "not_paying_attention"

                self.attention_history.append(self.engagement == "paying_attention")
                if len(self.attention_history) >= 3:
                    votes = sum(self.attention_history)
                    self.engagement = (
                        "paying_attention"
                        if votes >= (len(self.attention_history) // 2 + 1)
                        else "not_paying_attention"
                    )

                display_engagement = self.engagement
            else:
                landmark_attention = self._infer_attention_from_landmarks(landmarks[0])
                if landmark_attention == "paying_attention":
                    self.engagement = self._infer_attention_from_face_box(bboxes[0], frame.shape)
                else:
                    self.engagement = "not_paying_attention"

                self.attention_history.append(self.engagement == "paying_attention")
                if len(self.attention_history) >= 3:
                    votes = sum(self.attention_history)
                    self.engagement = (
                        "paying_attention"
                        if votes >= (len(self.attention_history) // 2 + 1)
                        else "not_paying_attention"
                    )
                display_engagement = self.engagement

            for bbox in bboxes:
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            display_emotion = "none"
            self.engagement = "not_paying_attention"
            self.attention_history.append(False)
            display_engagement = self.engagement

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
        self.declare_parameter("frame_rotation", 0)
        self.declare_parameter("timer_period", 0.05)
        self.declare_parameter("topic", "/actions_recognised")
        self.declare_parameter("emotions_topic", "/engagement/emotions")
        self.declare_parameter("engagement_topic", "/engagement/status")
        self.declare_parameter("non_adaptive_flag_topic", "/non_adaptive_flag")
        self.declare_parameter("display_image_topic", "/intention/display_frame/compressed")

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
        frame_rotation = int(self.get_parameter("frame_rotation").value)
        if frame_rotation not in (0, 90, 180, 270):
            self.get_logger().warning(
                f"Invalid frame_rotation={frame_rotation}; using 0. Allowed: 0, 90, 180, 270"
            )
            frame_rotation = 0
        timer_period = float(self.get_parameter("timer_period").value)
        topic = self.get_parameter("topic").value
        emotions_topic = self.get_parameter("emotions_topic").value
        engagement_topic = self.get_parameter("engagement_topic").value
        non_adaptive_flag_topic = self.get_parameter("non_adaptive_flag_topic").value
        display_image_topic = self.get_parameter("display_image_topic").value

        self.publisher_ = self.create_publisher(String, topic, 10)
        self.emotions_publisher = self.create_publisher(String, emotions_topic, 10)
        self.engagement_publisher = self.create_publisher(String, engagement_topic, 10)
        self.display_image_publisher = self.create_publisher(
            CompressedImage, display_image_topic, 10
        )

        self.actions = ActionsPerf(
            device=device,
            camid=camera,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            frame_rotation=frame_rotation,
        )
        self.non_adaptive_flag_sub = self.create_subscription(
            Bool,
            non_adaptive_flag_topic,
            self._non_adaptive_flag_callback,
            10,
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
        self.get_logger().info(f"Publishing display frames on {display_image_topic}")
        self.get_logger().info(f"Listening non-adaptive flag on {non_adaptive_flag_topic}")
        self.get_logger().info(f"Using device: {device} (camera={camera}, type={type(camera).__name__})")
        self.get_logger().info(f"Frame rotation set to {frame_rotation} degrees")

    def _non_adaptive_flag_callback(self, msg: Bool) -> None:
        non_adaptive = bool(msg.data)
        if self.actions.non_adaptive != non_adaptive:
            self.actions.non_adaptive = non_adaptive
            self.get_logger().info(
                f"Updated non_adaptive mode from topic: {self.actions.non_adaptive}"
            )

    def timer_callback(self) -> None:
        cv2.waitKey(1)  # Allow OpenCV to process window events
        success, frame = self.actions.cap.read()
        if not success:
            self.get_logger().warning("Could not read frame from camera")
            return
        frame = self.actions.normalize_frame_orientation(frame)

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
        ok, encoded_frame = cv2.imencode(".jpg", display_frame)
        if ok:
            image_msg = CompressedImage()
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.format = "jpeg"
            image_msg.data = encoded_frame.tobytes()
            self.display_image_publisher.publish(image_msg)
        cv2.waitKey(1)
        if action_label:
            msg = String()
            msg.data = action_label
            self.publisher_.publish(msg)
            self.last_published = action_label
            # self.get_logger().info(
            #     f"Published action: {action_label} (confidence={confidence:.2f})"
            # )

        self.emotions_publisher.publish(String(data=emotion))
        self.engagement_publisher.publish(String(data=engagement))

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
