
import torch
import numpy as np

from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download
import time
import cv2

np.random.seed(0)


class ActionsPerf(object):
    def __init__(self, device='cpu', camid=0):
        self.device = device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # self.model = VideoMAEForVideoClassification.from_pretrained("checkpoint-120", attn_implementation="sdpa").to(self.device)
        self.model = VideoMAEForVideoClassification.from_pretrained("/home/hguda/colab_dress_ws/src/checkpoint-120", attn_implementation="sdpa").to(self.device)
        

        self.camid = camid
        self.cap = cv2.VideoCapture(self.camid)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # ret, frame = self.cap.read()
        self.video = np.empty((16, 480, 640, 3))
        # self.video[:-1] = self.video[1:]
        # self.video[-1] = frame
        # self.video = 
        # self.video = frame.copy()
        self.i = 0
        self.action_label = ""

    def predict_actions(self, frame):

        # self.video[:-1] = self.video[1:]
        # self.video[-1] = frame    
        # print( self.video.shape)
        # self.video = np.stack([self.video, frame])
        self.video[self.i] = frame.copy()
        self.i += 1
        if self.i == 16:
            self.i = 0
            inputs = self.image_processor(list(self.video), return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                p = torch.nn.functional.softmax(logits, dim=1)

            # model predicts one of the 400 Kinetics-400 classes
            predicted_label = logits.argmax(-1).item()
            action_label = self.model.config.id2label[predicted_label]
            self.video = np.empty((16, *frame.shape))
            if p[0, predicted_label] > 0.9:
                self.action_label = action_label
            else:
                self.action_label = ""

        cv2.putText(frame, f"{self.action_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return self.action_label, frame


if __name__ == "__main__":
    actions = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid="/dev/v4l/by-id/usb-Creative_Technology_Ltd._Creative_Senz3D_VF0780_K8VF0780404001001T-video-index0")
    while True:
        start_time = time.time()
        success, frame = actions.cap.read()
        if not success:
            break
        action_label, image = actions.predict_actions(frame)
        
        # Display the frame with detected faces and emotions
        if image is not None:
            cv2.putText(image, f"{action_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Action Recognition', image)
        total_time = time.time() - start_time
        # print(f"Inference time: {total_time:.3f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break