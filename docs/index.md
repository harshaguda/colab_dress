# COLAB Dress Documentation

Welcome to the COLAB Dress ROS 2 stack documentation. This site explains how the perception and manipulation pipeline works, how to set it up, and how to extend it for your own experiments.

## What is COLAB Dress?

COLAB Dress is a collaborative dressing research platform that connects human pose/engagement perception with a robot arm driven by MoveIt. The stack:

- streams color + depth from an Intel RealSense camera;
- calibrates the camera extrinsics via ArUco markers and publishes a TF transform;
- detects upper-body pose landmarks with MediaPipe and can lift them into 3D;
- provides ArUco marker detection utilities for calibration and marker tracking;
- executes end-effector trajectories that can be modified at runtime; and
- infers engagement/emotion cues for more responsive interactions.

## Key Features

- ROS 2 Humble native, using `ament_python`
- Modular nodes that can run independently or together via launch files
- Real-time MediaPipe pose estimation with 2D→3D translation service
- ArUco-driven calibration workflow producing reusable transforms
- End-effector trajectory executor that accepts live edits
- Engagement detection built on EmotiEff emotion recognition

## Architecture at a Glance

```mermaid
graph TD
    RS[RealSense Camera] -->|Color + Depth| PoseEstimator
    RS -->|Color| ArucoDetector
    RS -->|Depth| Get3DPoint
    ArucoDetector -->|translation_matrix.npy| CameraTransform
    CameraTransform -->|TF base→camera| PoseEstimator
    PoseEstimator -->|Pose2DArray| Planner[Downstream Consumers]
    PoseEstimator -->|PoseArray (optional)| Planner
    Planner -->|PoseArray| TrajectoryExecutor
    TrajectoryExecutor -->|MoveGroup goals| MoveIt
    PoseEstimator -->|Pixels| Get3DPoint
    Get3DPoint -->|3D point| PoseEstimator
    RS -->|Color| EngagementDetector
    EngagementDetector -->|Status| UX[UI / HRI]
```

## Where to Start

1. **Read the [setup guide](setup.md)** for prerequisites and build instructions.
2. **Calibrate the camera** using the [ArUco workflow](calibration.md).
3. **Explore the nodes** and their interfaces in the [node reference](nodes.md).
4. **Run the full pipeline** with the launch instructions in [launches.md](launches.md).
5. **Troubleshoot common issues** via the [FAQ](troubleshooting.md).

## Staying Up to Date

- The latest source lives on GitHub: `https://github.com/harshaguda/colab_dress`
- Issues & discussions are the preferred channel for support
- Documentation contributions are welcome—edit these Markdown files and open a PR

## Next Steps

- Dive into the [pipeline walkthrough](pipeline.md) for end-to-end context.
- Learn how to keep this documentation in sync with GitHub Pages in [github-pages.md](github-pages.md).
