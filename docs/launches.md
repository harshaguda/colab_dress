# Launch Files

COLAB Dress ships with several launch descriptions that combine the nodes in different ways.

## `colab_dress.launch.py`

The primary launch file used day-to-day. It:

1. Includes `realsense2_camera/rs_launch.py` with depth alignment and pointcloud support enabled.
2. Starts `camera_transform_publisher` to broadcast the base→camera transform (requires prior `translation_matrix.npy`).
3. Launches `pose_estimator` (debug=false, translate=true by default when run standalone).
4. launches `get_3d_point_service` for 2D→3D lookups.
5. Launches `engagement_detector` to publish emotion + engagement status.

```bash
ros2 launch colab_dress colab_dress.launch.py
```

### Customizing

- Override launch arguments by editing the file or wrapping it in your own launch.
- To disable engagement detection, comment out the node entry or create a variant launch file.
- To change `pose_estimator` arguments, use the `parameters` list in the node description.

## `camera_transfrom.launch.py`

Standalone launch to broadcast the camera transform without starting other nodes.

```bash
ros2 launch colab_dress camera_transfrom.launch.py
```

### Use Case

- Run during calibration validation to verify TF alignment without starting perception nodes.

## `set_end_effector_pose_demo.launch.py`

Demo launch that publishes example trajectories to the end-effector executor.

```bash
ros2 launch colab_dress set_end_effector_pose_demo.launch.py
```

### Contents

- Starts `end_effector_trajectory_executor`
- Publishes a sample `PoseArray` to `/end_effector_trajectory`
- Useful to validate MoveIt connectivity without human input

## Tips

- Run launches from a sourced workspace: `source install/setup.bash`
- Use `--show-args` to list declared arguments: `ros2 launch colab_dress colab_dress.launch.py --show-args`
- Combine with `ros2 topic echo` or `rviz2` to visualize outputs in real time
