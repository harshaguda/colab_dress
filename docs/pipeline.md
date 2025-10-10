# Pipeline Walkthrough

This document follows the data flow through the COLAB Dress stack to highlight how perception, calibration, and motion planning connect.

## 1. Sensor Ingestion

- The RealSense driver (`realsense2_camera`) publishes synchronized color and depth streams.
- `rs_launch.py` is included in our main launch file with depth alignment enabled so that pixel coordinates translate directly between color and depth frames.

## 2. Calibration with ArUco

1. Run `ros2 run colab_dress aruco_detect` while viewing the marker.
2. Once the marker is detected the node displays the annotated image. Press **S** to store `translation_matrix.npy`.
3. The matrix contains the homogeneous transform from the robot base frame to the camera. It is stored in the workspace root for reuse.

See [calibration.md](calibration.md) for detailed instructions.

## 3. Publishing the Transform

- `camera_transform_publisher` loads the saved matrix, applies a fixed color→depth rotation correction, and publishes `base_link → external_camera_link` as a TF transform.
- Any downstream node querying TF now understands where the camera is relative to the robot.

## 4. Pose Estimation

- `pose_estimator` subscribes to the RealSense color stream.
- MediaPipe’s BlazePose model runs on each frame, emitting landmarks for right wrist, elbow, and shoulder (extendable for full body).
- 2D landmarks are published on `/pose_estimator/pose_2d` as a `Pose2DArray` message.

### Optional 3D Projection

If `translate=True` (default in the launch file), the node:

1. Calls the `get_3d_point` service for each landmark pixel.
2. Receives metric coordinates in the camera frame.
3. Publishes them as a `PoseArray` on `/pose_estimator/pose_3d`.

This translation relies on the aligned depth stream and the calibration described above.

## 5. 3D Point Service

- `get_3d_point_service` maintains the latest aligned depth image along with color/depth intrinsics.
- When queried it converts the pixel to depth-space, reads the depth value, and deprojects the point into 3D space using `rs2_deproject_pixel_to_point`.
- The resulting coordinates are returned in meters.

## 6. Engagement Detection

- `engagement_detector` consumes the same color stream.
- Faces are detected via `facenet_pytorch.MTCNN`, then passed to EmotiEff for emotion and engagement classification.
- Results are published on `/engagement/emotions` and `/engagement/status`, enabling UI feedback or behavior switching.

## 7. Motion Execution

- External planners (or higher-level controllers) can transform the 3D pose data into MoveIt waypoints.
- `end_effector_trajectory_executor` listens for `PoseArray` messages on `/end_effector_trajectory` or `/end_effector_trajectory_append`.
- It serially executes each waypoint via the MoveIt `MoveGroup` action, with optional cartesian interpolation.
- Status updates are broadcast on `/end_effector_trajectory_status`.

## 8. Launch Integration

`colab_dress.launch.py` wires everything together:

1. Starts `rs_launch.py` with depth alignment.
2. Runs `camera_transform_publisher` (requires prior calibration file).
3. Launches `pose_estimator`, `get_3d_point_service`, and `engagement_detector`.

A single command brings up the complete perception→engagement pipeline.

```bash
ros2 launch colab_dress colab_dress.launch.py
```

## 9. Optional Add-ons

- **Trajectory demos:** `set_end_effector_pose_demo.launch.py` showcases publishing test trajectories.
- **Marker listeners:** `aruco_marker_listener` prints marker detections for debugging.
- **Custom consumers:** Subscribe to `Pose2DArray`/`PoseArray` for analytics, or extend the pipeline with your own nodes.

## 10. Extending the Pipeline

- Add more landmarks by modifying `required_landmarks` in `pose_estimator.py`.
- Integrate skeleton tracking or additional sensors by reusing the TF transform.
- Replace the engagement model with your preferred HRI signal—ensure the topics remain consistent for downstream tools.
