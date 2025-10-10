# Development Guide

Guidelines for contributing to COLAB Dress.

## Coding Standards

- Python nodes follow PEP 8 style (checked via `ament_flake8`).
- Use descriptive log messages with `self.get_logger()`.
- Keep ROS parameters configurable; avoid hard-coding topics where possible.
- Prefer dependency injection for testability.

## Branch Workflow

1. Fork or branch from `main`.
2. Create a descriptive branch name (e.g., `feature/aruco-refactor`).
3. Commit with meaningful messages (`git commit -m "Add ArUco reprojection tests"`).
4. Open a Pull Request and link to relevant issues or docs.

## Testing & Linting

```bash
colcon test --packages-select colab_dress
```

- Add tests under `src/colab_dress/test/` (pytest or launch tests).
- Use `ros2 run colab_dress ...` for manual smoke tests.

## Documentation

- Update `README.md` for high-level changes.
- Add detailed notes or tutorials in `docs/`.
- Run `mkdocs serve` before pushing to ensure links render correctly.

## Release Checklist

- Regenerate `translation_matrix.npy` if the physical setup changed.
- Verify launch files with real hardware (`ros2 launch colab_dress colab_dress.launch.py`).
- Update version numbers in `package.xml` and `setup.py` when publishing releases.

## Useful Commands

```bash
# List executables
ros2 pkg executables colab_dress

# Inspect TF frames
ros2 run tf2_tools view_frames

# Echo pose estimator output
ros2 topic echo /pose_estimator/pose_2d
```

## Support

Open issues on GitHub with:

- ROS 2 version (`ros2 doctor` output)
- Hardware description (camera model, manipulator)
- Steps to reproduce
- Relevant logs or backtraces

We strive for collaborative, well-documented contributions—thank you!
