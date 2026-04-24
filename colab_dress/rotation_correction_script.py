# Source - https://stackoverflow.com/a/55747773
# Posted by Rafay Khan, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-09, License - CC BY-SA 4.0

import cv2
import argparse
import os
import re
import shutil
import subprocess

import ffmpeg


def degrees_to_rotate_code(rotation_degrees):
    rotation_degrees = rotation_degrees % 360
    if rotation_degrees == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if rotation_degrees == 180:
        return cv2.ROTATE_180
    if rotation_degrees == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None

def check_rotation(path_video_file):
    # Live camera devices (e.g. /dev/v4l/*) typically have no rotation metadata.
    # Also, ffmpeg-python requires the external binary `ffprobe` to be installed.
    if shutil.which("ffprobe") is None:
        print("ffprobe not found in PATH; skipping metadata-based rotation detection.")
        return None

    try:
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)
    except (ffmpeg.Error, FileNotFoundError):
        print("Could not read rotation metadata; continuing without auto-rotation.")
        return None

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    try:
        rotation = int(meta_dict['streams'][0]['tags']['rotate'])
    except (KeyError, IndexError, ValueError, TypeError):
        return None

    return degrees_to_rotate_code(rotation)


def get_v4l2_rotation(path_video_file):
    if shutil.which("v4l2-ctl") is None:
        return None

    # v4l2-ctl expects a real /dev/videoX device; resolve symlink if needed.
    device_path = os.path.realpath(path_video_file)
    if not device_path.startswith("/dev/video"):
        return None

    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--get-ctrl=rotate"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    # Expected format: "rotate: 180"
    match = re.search(r"rotate\s*:\s*(-?\d+)", result.stdout)
    if not match:
        return None

    try:
        rotation = int(match.group(1))
    except ValueError:
        return None

    return degrees_to_rotate_code(rotation)


def resolve_rotation(path_video_file, manual_rotation_deg=None):
    if manual_rotation_deg is not None:
        rotate_code = degrees_to_rotate_code(manual_rotation_deg)
        if rotate_code is None and manual_rotation_deg % 360 != 0:
            print("Manual rotation must be one of: 0, 90, 180, 270")
        else:
            print(f"Using manual rotation: {manual_rotation_deg % 360} degrees")
        return rotate_code

    rotate_code = check_rotation(path_video_file)
    if rotate_code is not None:
        print("Using rotation metadata from ffprobe")
        return rotate_code

    rotate_code = get_v4l2_rotation(path_video_file)
    if rotate_code is not None:
        print("Using rotation from V4L2 camera control")
        return rotate_code

    print("No rotation metadata/control found; using frame as-is.")
    return None

parser = argparse.ArgumentParser(description="Camera rotation correction helper")
parser.add_argument(
    "--video-path",
    default="/dev/v4l/by-id/usb-Creative_Technology_Ltd._Creative_Senz3D_VF0780_K8VF0780404001001T-video-index0",
    help="Path to video file or camera device",
)
parser.add_argument(
    "--rotation-deg",
    type=int,
    default=None,
    help="Manual rotation in degrees (0, 90, 180, 270). Overrides auto detection.",
)

args = parser.parse_args()
video_path = args.video_path

# open a pointer to the video file stream
vs = cv2.VideoCapture(video_path)
if not vs.isOpened():
    raise RuntimeError(f"Could not open video source: {video_path}")

# Disable backend auto-rotation so orientation is deterministic.
if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
    vs.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

# check if video requires rotation
rotateCode = resolve_rotation(video_path, args.rotation_deg)

# loop over frames from the video file stream
while True:
    # grab the frame from the file
    grabbed, frame = vs.read()

    # if frame not grabbed -> end of the video
    if not grabbed:
        break

    # check if the frame needs to be rotated
    if rotateCode is not None:
        frame = cv2.rotate(frame, rotateCode)

    cv2.imshow("rotation_correction", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()