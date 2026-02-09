#!/usr/bin/env python3
import argparse
import time

import cv2
import mediapipe as mp
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe pose detection with OpenCV VideoCapture")
    parser.add_argument("--camera", type=str, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    use_solutions = hasattr(mp, "solutions")
    if use_solutions:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        pose_ctx = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    else:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parents[1] / "resource" / "pose_landmarker_full.task"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if use_solutions:
                results = pose_ctx.process(image_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    for landmarks in result.pose_landmarks:
                        for lm in landmarks:
                            x_px = int(lm.x * frame.shape[1])
                            y_px = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x_px, y_px), 2, (0, 255, 0), -1)

            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if use_solutions:
            pose_ctx.close()
        else:
            landmarker.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
