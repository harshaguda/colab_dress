## Compute Camera-to-Base Transformation

You need to chain transformations to map points from the Camera frame to the Robot Base frame using the ArUco marker as a bridge.

The chain is: $P_{Base} = T_{Marker \to Base} \times T_{Camera \to Marker} \times P_{Camera}$

- **$T_{Camera \to Marker}$**: You already have this saved in `translation_matrix.npy`.
- **$T_{Marker \to Base}$**: You need to construct this from your "known position" of the marker.

### Steps
1.  **Define $T_{Marker \to Base}$**: Create a `4x4` matrix representing the marker's known rotation and translation relative to the robot base.
2.  **Load $T_{Camera \to Marker}$**: Load the saved `translation_matrix.npy` file.
3.  **Compute $T_{Camera \to Base}$**: Multiply the matrices: `T_base_camera = T_marker_base @ T_camera_marker`.
4.  **Transform Points**: Multiply your unknown camera points (as `[x, y, z, 1]`) by `T_base_camera`.

### Further Considerations
1.  **Marker Orientation**: Ensure the "Known Position" accounts for the ArUco marker's specific axis orientation (Z-axis is perpendicular to the tag face).
2.  **Verification**: Test with `(0,0,0)` in camera frame to see the camera's position in base frame.
