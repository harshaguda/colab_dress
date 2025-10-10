# Troubleshooting

A collection of common issues and suggested fixes when working with COLAB Dress.

## Pose Estimator Errors

### `module 'google.protobuf.message_factory' has no attribute 'GetMessageClass'`

- Cause: MediaPipe ≥0.10 expects `protobuf>=4.25.3`.
- Fix: `python3 -m pip install --user 'protobuf>=4.25.3'`
- If other packages require protobuf 3.x (e.g., `tf2onnx`), install them in a virtual environment to avoid version pin conflicts.

### Slow FPS or High CPU

- Disable `debug` to stop the OpenCV window.
- Ensure you are using release builds of OpenCV/MediaPipe (pip wheels include SIMD optimizations).
- Use a machine with AVX2 capable CPU; otherwise expect lower throughput.

## RealSense Issues

### `xioctl(VIDIOC_QBUF) failed … No such device`

- Usually indicates a USB reset or insufficient power.
- Reseat the USB cable and ensure you are using a USB 3 port.
- Disable power management: `sudo echo on | sudo tee /sys/bus/usb/devices/<port>/power/control`
- Check `dmesg` for kernel logs; upgrade librealsense if firmware errors persist.

### No Depth Data in Service

- Confirm `align_depth.enable:=true` when launching `rs_launch.py`.
- Ensure the service has received both depth and color `CameraInfo` messages; check logs for “Camera calibration data received”.
- Use `ros2 topic hz /camera/camera/depth/image_rect_raw` to verify publishing.

## Calibration Problems

### Translation Matrix Not Saved

- The node only saves once **S** is pressed after a successful detection.
- Verify the marker ID matches the expected dictionary; switch dictionary via `aruco_dict_type` parameter if needed.
- The matrix is saved relative to the **current working directory**; confirm you have write permissions.

### Transform Looks Wrong in RViz

- Confirm your robot base frame matches the `parent_frame` parameter (defaults to `base_link`).
- Adjust the hard-coded offset in `aruco_detector.py` (`Trans[0, 3] += 0.15`) to fit your rig.
- Re-run calibration with the marker in a precise, measured location.

## MoveIt Execution Failures

- Ensure the MoveGroup action server `/move_action` is running (via your robot or simulation).
- Check tolerances—set `position_tolerance` to a larger value for coarse demos.
- Inspect `/end_effector_trajectory_status` logs for specific error codes from MoveIt.

## Engagement Detector Issues

### Missing Models or Slow Startup

- The first run downloads ONNX models; provide an active internet connection.
- For offline usage, pre-download by running the node once where internet is available, then cache the models in `~/.cache/emotiefflib`.

### CUDA Not Found

- The node automatically falls back to CPU. Ensure CUDA drivers are installed if GPU acceleration is desired.

## General Debug Tips

- Use `ros2 run --prefix 'gdb -ex run --args' …` for deep debugging.
- `ros2 doctor` reports missing dependencies.
- `tf2_tools view_frames` visualizes TF trees—great for verifying camera transforms.
- Enable `export RCLPY_LOG_LEVEL=DEBUG` to print verbose logs from nodes.
