import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from typing import List


class Emotions(object):
    def __init__(self, device='cpu', camid=0):
        self.device = device
        self.fer = None  # Placeholder for the FER model
        self.all_scores = None
        self.engage_flag = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = get_model_list()[0]

        self.fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=self.device)

        self.all_frames = []
        self.all_scores = None
        self.camid = camid

        self.cap = cv2.VideoCapture(self.camid)

        engage_flag = False
        self.engagement = "none"
        self.emotion = "none"

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
    
    

    def predict_emotions(self):
        # Placeholder for emotion prediction logic
        success, image = self.cap.read()
        if not success:
            return None

        # image = cv2.cvtColor(frame)
        # image = cv2.flip(frame, 0)  # Flip the image horizontally
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
        
        return image, self.emotion, self.engagement

    def predict_engagement(self, all_frames):
        # Placeholder for engagement prediction logic
        return [], []  # Return empty lists for now
    

if __name__ == "__main__":
    emotions = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=4)
    while True:
        
        image, _, _ = emotions.predict_emotions()
        
        # Display the frame with detected faces and emotions
        if image is not None:
            cv2.imshow('Emotion Detection', image)
        else:
            print("No frame captured.")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    emotions.cap.release()
    cv2.destroyAllWindows()