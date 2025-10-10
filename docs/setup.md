# Setup Guide

Follow the steps below to prepare a development machine for COLAB Dress.

## 1. Base System

- **OS:** Ubuntu 22.04 LTS
- **ROS 2:** Humble Hawksbill (desktop variant recommended)
- **Hardware:** Intel RealSense D435/D435i, NVIDIA GPU optional (engagement detector benefits from CUDA)

Install ROS 2 (if not already):

```bash
sudo apt update && sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo rosdep init
rosdep update
```

## 2. Workspace Layout

Create a workspace if you do not already have one:

```bash
mkdir -p ~/repos/myrepos
cd ~/repos/myrepos
git clone https://github.com/harshaguda/colab_dress_ros2.git
```

Source ROS 2 before building:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 3. Python Dependencies

The nodes rely on a set of Python libraries that are not ROS packages. Install them with pip into your user site-packages (or a virtual environment if preferred):

```bash
python3 -m pip install --user \
  mediapipe==0.10.21 \
  opencv-contrib-python==4.11.0.86 \
  pyrealsense2 \
  facenet-pytorch \
  emotiefflib \
  scipy \
  mkdocs mkdocs-material
```

> **Tip:** If pip reports permission issues, add `--break-system-packages` (Ubuntu 24.04) or use a virtual environment.

## 4. Build Order

The ROS messages/services live in a separate package. Build in this order:

```bash
cd ~/repos/myrepos/colab_dress_ros2
colcon build --packages-select colab_dress_interfaces
colcon build --packages-select colab_dress
source install/setup.bash
```

Re-source `install/setup.bash` after every build or add it to your shell startup.

## 5. RealSense Setup

Install the librealsense udev rules (if not already available):

```bash
sudo apt install ros-humble-realsense2-camera librealsense2-utils
sudo apt install ros-humble-realsense2-description
```

Verify the camera stream:

```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
```

You should see `/camera/camera/color/image_raw` and `/camera/camera/depth/image_rect_raw` topics streaming.

## 6. Optional GPU Accelerations

- **TensorRT / CUDA:** Not required, but the engagement detector and MediaPipe may run faster with CUDA-enabled builds.
- **OpenCL:** MediaPipe CPU path performs well without GPU acceleration.

## 7. Environment Checks

Useful sanity commands:

```bash
ros2 doctor
ros2 pkg executables colab_dress
ros2 topic list | grep pose_estimator
```

If any dependencies are missing, run `rosdep install --from-paths src -i -y` to pull in missing system packages.
