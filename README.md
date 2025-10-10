# COLAB Dress – ROS 2 Perception & Manipulation Suite

[![Deploy MkDocs](https://github.com/harshaguda/colab_dress/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/harshaguda/colab_dress/actions/workflows/gh-pages.yml)


COLAB Dress brings together perception-driven ROS 2 nodes for the collaborative dressing project. The stack fuses RealSense sensing, MediaPipe pose estimation, ArUco-based calibration, MoveIt trajectory execution, and engagement inference so that a robot can perceive a person and react to their posture in real time.

## Highlights

- **Fully ROS 2 Humble native**: Python nodes packaged with `ament_python`, launchable together or independently.
- **Depth-informed human pose**: MediaPipe-based 2D landmarks with an on-demand 3D lookup service using RealSense depth data.
- **Marker-driven calibration**: ArUco detector and transform publisher produce the base→camera transform used by the rest of the stack.
- **MoveIt trajectory executor**: Accepts live edits to end-effector pose sequences for adaptive dressing motions.
- **Engagement inference**: Emotion and engagement classifier powered by EmotiEff for responsive HRI.
- **Batteries-included launch**: Single launch file spins up RealSense, perception, calibration, and engagement nodes.

## Repository Layout

```
colab_dress_ros2/
├── README.md                  # You are here
├── mkdocs.yml                 # MkDocs configuration for the documentation site
├── docs/                      # Project documentation (rendered with MkDocs)
├── ARUCO_README.md            # Legacy ArUco usage notes (also merged into docs)
├── src/
│   └── colab_dress/           # Main ROS 2 package (nodes, launch files)
│       ├── colab_dress/       # Python nodes
│       ├── launch/            # Launch descriptions
│       ├── srv/               # Custom service definitions
│       └── setup.py, package.xml, …
└── translation_matrix.npy     # Example camera calibration output
```

## Quickstart

### Prerequisites

- Ubuntu 22.04 with ROS 2 Humble (desktop or equivalent install)
- Intel RealSense D435/D435i with librealsense drivers
- Python dependencies (install once in your ROS environment):

```bash
python3 -m pip install --user \
  mediapipe==0.10.21 opencv-contrib-python==4.11.0.86 \
  pyrealsense2 facenet-pytorch emotiefflib scipy
```

> ℹ️  MediaPipe requires protobuf ≥4.25.3. If you hit `GetMessageClass` errors, upgrade with `python3 -m pip install --user 'protobuf>=4.25.3'`.

### Build the workspace

```bash
cd ~/repos/myrepos/colab_dress_ros2
colcon build --packages-select colab_dress_interfaces
colcon build --packages-select colab_dress
source install/setup.bash
```

### Launch everything

```bash
ros2 launch colab_dress colab_dress.launch.py
```

This launch file starts the RealSense driver (with depth alignment), publishes the base→camera transform, spins up pose estimation, the 2D→3D service, and the engagement detector. Topic names and parameters can be tweaked in the launch file.

### Run nodes individually

```bash
# Pose estimation with 3D translation enabled
ros2 run colab_dress pose_estimator

# ArUco calibration workflow
ros2 run colab_dress aruco_detect
ros2 run colab_dress aruco_marker_listener

# Base→camera transform broadcaster (uses translation_matrix.npy)
ros2 run colab_dress camera_transform_publisher

# End-effector trajectory executor (MoveIt required)
ros2 run colab_dress end_effector_trajectory_executor
```

## Core Components

| Component | Purpose | Key Interfaces |
|-----------|---------|----------------|
| `pose_estimator.py` | MediaPipe pose landmarks, optional 3D lookup | Publishes `Pose2DArray` (`/pose_estimator/pose_2d`), `PoseArray` (`/pose_estimator/pose_3d`), consumes RealSense color stream |
| `get_3d_point_service.py` | Converts pixel coordinates to 3D points using aligned depth | Service `get_3d_point (Get3DPoint)` |
| `aruco_detector.py` | Detects ArUco markers, computes extrinsics, saves `translation_matrix.npy` | Publishes `ArucoMarker` on `/aruco_markers` |
| `camera_transform_publisher.py` | Broadcasts transform using saved 4×4 matrix | TF: `base_link → external_camera_link` |
| `end_effector_trajectory_executor.py` | Executes or appends pose trajectories via MoveIt | Subscribes `/end_effector_trajectory`, `/end_effector_trajectory_append`, publishes `/end_effector_trajectory_status` |
| `engagement_detector.py` | Emotion + engagement inference | Publishes `/engagement/emotions`, `/engagement/status` |

More detailed API notes, topics, and parameters live in the [documentation site](./docs/index.md).

## Calibration Workflow (ArUco → Transform)

1. Place a known-size ArUco marker in the robot frame.
2. Launch the RealSense driver and run `ros2 run colab_dress aruco_detect`.
3. When the marker is detected, press **s** in the visualization window to save `translation_matrix.npy`.
4. `camera_transform_publisher` loads this matrix and augments it with the color→depth rotation used by the pose estimator.

See [`docs/calibration.md`](./docs/calibration.md) for step-by-step screenshots and troubleshooting tips.

## Development Workflow

- **Linting:**

  ```bash
  colcon test --packages-select colab_dress --ctest-args -R lint
  ```

- **Unit / integration tests:** (placeholder – add suites under `test/`).
- **Formatting:** Follow PEP 8; `ament_flake8` runs during tests.
- **Documentation:** Update `docs/` and rebuild with MkDocs (see below).

## Documentation Site

This repository ships with a MkDocs site so the docs look great on GitHub Pages.

```bash
python3 -m pip install --user mkdocs mkdocs-material
mkdocs serve  # live preview at http://127.0.0.1:8000/
mkdocs build  # outputs static site in site/
```

To publish on GitHub Pages, enable Pages in the repository settings and deploy the `site/` directory (e.g., via GitHub Actions). A sample workflow is provided in [`docs/github-pages.md`](./docs/github-pages.md).

## Support & Contributions

Pull requests are welcome! If you add nodes or integrations, extend the documentation and tests accordingly. For questions, open an issue and include logs plus your environment details (`ros2 doctor` helps).

## License

Apache-2.0 © 2025 Harsha Guda.
