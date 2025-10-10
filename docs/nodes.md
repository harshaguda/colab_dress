# Node Reference

This page lists the primary ROS 2 nodes provided by the `colab_dress` package and summarizes the interfaces they expose.

## Pose Estimator (`pose_estimator.py`)

- **Purpose:** Detect upper-body pose landmarks using MediaPipe and optionally publish 3D points via the Get3DPoint service.
- **Executable:** `ros2 run colab_dress pose_estimator`

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `color_image_topic` | string | `/camera/camera/color/image_raw` | RealSense color topic to subscribe to |
| `debug` | bool | `False` | Show OpenCV visualization window with landmarks |
| `translate` | bool | `False` | If `True`, request 3D points from the service and publish `PoseArray` |

### Published Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/pose_estimator/pose_2d` | `colab_dress_interfaces/msg/Pose2DArray` | Landmark pixels (right wrist, elbow, shoulder) |
| `/pose_estimator/pose_3d` | `geometry_msgs/msg/PoseArray` | Optional when `translate=True`, 3D positions |

### Subscribed Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | Incoming RGB frames |

### Services/Clients

| Name | Type | Role |
|------|------|------|
| `get_3d_point` | `colab_dress_interfaces/srv/Get3DPoint` | Client, converts pixel coordinates to metric coordinates |

---

## Get 3D Point Service (`get_3d_point_service.py`)

- **Purpose:** Convert color pixel coordinates to 3D coordinates using aligned depth.
- **Executable:** `ros2 run colab_dress get_3d_point_service`

### Published Topics

_None_

### Subscribed Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Depth image (in mm) |
| `/camera/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | Depth intrinsics |
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | Color image (for visualization) |
| `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Color intrinsics |

### Service

| Name | Type | Description |
|------|------|-------------|
| `get_3d_point` | `colab_dress_interfaces/srv/Get3DPoint` | Request `(px, py)` and receive `(rx, ry, rz)` in meters |

---

## ArUco Detector (`aruco_detector.py`)

- **Purpose:** Detect ArUco markers, publish their pixels, and help compute calibration transforms.
- **Executable:** `ros2 run colab_dress aruco_detect`

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `color_image_topic` | string | `/camera/camera/color/image_raw` | RGB input |
| `color_info_topic` | string | `/camera/camera/color/camera_info` | Camera intrinsics |
| `aruco_dict_type` | string | `DICT_5X5_50` | OpenCV dictionary identifier |

### Published Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/aruco_markers` | `colab_dress_interfaces/msg/ArucoMarker` | One message per detected marker |

### Subscribed Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | RGB frames |
| `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Intrinsics |

### Utilities

Press **S** in the OpenCV window to save `translation_matrix.npy` (base→camera) using the detected marker pose.

---

## Camera Transform Publisher (`camera_transform_publisher.py`)

- **Purpose:** Broadcast the `base_link → external_camera_link` transform computed from calibration.
- **Executable:** `ros2 run colab_dress camera_transform_publisher`

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `matrix_path` | string | `translation_matrix.npy` | Path to 4×4 homogeneous transform |
| `parent_frame` | string | `base_link` | Parent frame ID |
| `child_frame` | string | `external_camera_link` | Child frame ID |
| `publish_rate` | double | `10.0` | Frequency (Hz) |

### Published TF

Broadcasts TF frames via `/tf` and `/tf_static` matching the configured parent/child frames.

---

## End Effector Trajectory Executor (`end_effector_trajectory_executor.py`)

- **Purpose:** Execute pose trajectories with MoveIt, allowing replacements or appends mid-execution.
- **Executable:** `ros2 run colab_dress end_effector_trajectory_executor`

### Parameters (subset)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `group_name` | string | `manipulator` | MoveIt planning group |
| `reference_frame` | string | `base_link` | Planning frame |
| `end_effector_link` | string | `end_effector_link` | Link to constrain |
| `position_tolerance` | double | `0.01` | Spherical tolerance (meters) |
| `orientation_tolerance` | double | `0.02` | Orientation tolerance (radians) |
| `execution_mode` | string | `pose` | `pose` or `cartesian` |
| `cartesian_steps` | integer | `10` | Sub-steps when `execution_mode=cartesian` |

### Topics

| Topic | Type | Direction | Notes |
|-------|------|-----------|-------|
| `/end_effector_trajectory` | `geometry_msgs/msg/PoseArray` | Subscribe | Replace trajectory |
| `/end_effector_trajectory_append` | `geometry_msgs/msg/PoseArray` | Subscribe | Append poses |
| `/end_effector_trajectory_status` | `std_msgs/msg/String` | Publish | Status updates |

### Action Client

| Action | Notes |
|--------|-------|
| `/move_action (moveit_msgs/action/MoveGroup)` | Sends waypoint goals sequentially |

---

## Engagement Detector (`engagement_detector.py`)

- **Purpose:** Detect faces, classify emotion & engagement using EmotiEff models.
- **Executable:** `ros2 run colab_dress engagement_detector`

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `device` | string | `auto` | Chooses `cuda` if available, otherwise `cpu` |
| `color_image_topic` | string | `/camera/camera/color/image_raw` | RGB stream |

### Topics

| Topic | Type | Direction | Notes |
|-------|------|-----------|-------|
| `/engagement/emotions` | `std_msgs/msg/String` | Publish | Dominant emotion label |
| `/engagement/status` | `std_msgs/msg/String` | Publish | Engagement label from EmotiEff |

### Dependencies

Requires `facenet-pytorch` (MTCNN) and `emotiefflib` models; the first run downloads ONNX weights.

---

## Utility Nodes

The package also includes sample publishers/subscribers in `colab_dress/publisher_member_function.py` and `subscriber_member_function.py` which serve as templates for new nodes.

For a full list of executables:

```bash
ros2 pkg executables colab_dress
```
