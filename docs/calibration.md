# Camera Calibration with ArUco

Use this guide to create or refresh the `translation_matrix.npy` file that encodes the transform from the robot base frame to the external camera.

## Requirements

- Printed ArUco marker with known physical size (default code assumes **5 cm** square)
- RealSense camera running with aligned depth (`align_depth.enable:=true`)
- `colab_dress` package built and sourced

## Step-by-Step

1. **Launch the RealSense driver**
   ```bash
   ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
   ```
2. **Start the ArUco detector**
   ```bash
   ros2 run colab_dress aruco_detect
   ```
3. **Position the marker** in front of the camera. Ensure it is flat, fills a reasonable portion of the frame, and is not heavily blurred.
4. **Verify intrinsics** – the node waits for `CameraInfo` before logging “Camera calibration data received”.
5. **Press `S`** in the OpenCV visualization window once the marker is detected. The node will:
   - solve for the marker pose with `solvePnP`
   - convert rotation vector to a matrix and compute the homogeneous transform
   - save the matrix to `translation_matrix.npy` in the current working directory

## Verifying the Matrix

Inspect the saved matrix:

```python
import numpy as np
m = np.load("translation_matrix.npy")
print(m)
```

The last row should be `[0, 0, 0, 1]`. The first three columns encode the rotation; the last column is the translation (meters).

## Using the Matrix

- `camera_transform_publisher` loads this file on startup (`matrix_path` parameter) and broadcasts the TF transform.
- If you place the camera in a new location, rerun the steps above to regenerate the matrix.

## Tips & Troubleshooting

- **Inconsistent results:** verify the physical marker size matches the hard-coded value (`marker_size = 0.05`). Adjust as needed and rebuild.
- **Blurry detections:** increase lighting, or move the marker closer until edges are crisp.
- **Multiple markers:** the node saves the most recent detection. Remove extra markers to avoid ambiguity.
- **Coordinate offsets:** the script adds a `+0.15` m adjustment along x to account for mechanical offsets. Modify in `aruco_detector.py` if your rig differs.
