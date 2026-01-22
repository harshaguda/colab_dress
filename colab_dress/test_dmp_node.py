from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped
except Exception as exc:  # pragma: no cover - ROS 2 not available in all contexts
    raise RuntimeError("rclpy is required to run this test script") from exc

from colab_dress.dmp import DMPNode


def _make_pose_array(points, frame_id: str = "") -> PoseArray:
    msg = PoseArray()
    msg.header.frame_id = frame_id
    msg.poses = []
    for pt in points:
        pose = Pose()
        pose.position.x = float(pt[0])
        pose.position.y = float(pt[1])
        pose.position.z = float(pt[2])
        pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


class DMPNodeTester(Node):
    def __init__(self, dmp_node: DMPNode):
        super().__init__("dmp_node_tester")
        self.dmp_node = dmp_node
        self.pose_pub = self.create_publisher(PoseArray, "dmp/arm_poses", 10)
        self.shoulder_pub = self.create_publisher(PointStamped, "dmp/shoulder_position", 10)
        self.rollout_sub = self.create_subscription(
            PoseArray, "dmp/dmp_rollout", self._rollout_cb, 10
        )
        self.current_pose_pub = self.create_publisher(
            PoseStamped, "/NS_1/franka_robot_state_broadcaster/current_pose", 10
        )
        self.delta_pose_sub = self.create_subscription(
            PoseStamped, "/delta_pose", self._delta_pose_cb, 10
        )

        self.stage = 0
        self.demo_goal = np.array([0.5, 0.0, 0.2], dtype=float)
        self.shoulder_init = np.array([0.1, 0.0, 0.0], dtype=float)
        self.shoulder_update = np.array([0.2, -0.1, 0.05], dtype=float)
        self.goal_offset = self.demo_goal - self.shoulder_init
        self.last_rollout_end: Optional[np.ndarray] = None
        self.initial_rollout_end: Optional[np.ndarray] = None

        self.current_pose = np.zeros(3, dtype=float)
        self.expected_points: Optional[np.ndarray] = None
        self.delta_index = 0
        self.settle_repeats_left = 0
        self.test_failed = False

        self.timeout_timer = self.create_timer(20.0, self._timeout)
        self.timer = self.create_timer(0.1, self._tick)

    def _publish_shoulder(self, pos: np.ndarray, frame_id: str = "") -> None:
        msg = PointStamped()
        msg.header.frame_id = frame_id
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])
        self.shoulder_pub.publish(msg)

    def _tick(self) -> None:
        self._publish_current_pose()
        if self.stage == 0:
            self.get_logger().info("Publishing shoulder init and demo trajectory.")
            self._publish_shoulder(self.shoulder_init, frame_id="base")
            demo_points = np.array([
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.1],
                self.demo_goal,
            ])
            msg = _make_pose_array(demo_points, frame_id="base")
            self.pose_pub.publish(msg)
            self.stage = 1

    def _rollout_cb(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        rollout_end = np.array(
            [msg.poses[-1].position.x, msg.poses[-1].position.y, msg.poses[-1].position.z],
            dtype=float,
        )
        self.last_rollout_end = rollout_end

        if self.stage == 1:
            self.initial_rollout_end = rollout_end
            self.get_logger().info("Initial rollout received. Updating shoulder.")
            self._publish_shoulder(self.shoulder_update, frame_id=msg.header.frame_id)
            self.stage = 2
            return

        if self.stage == 2:
            if self.initial_rollout_end is None:
                self.get_logger().error("Missing initial rollout. Test failed.")
                rclpy.shutdown()
                return
            expected_delta = self.shoulder_update - self.shoulder_init
            actual_delta = rollout_end - self.initial_rollout_end
            if np.linalg.norm(actual_delta - expected_delta) < 0.2:
                self.get_logger().info("Updated rollout shifted with shoulder update. Test passed.")
            else:
                self.get_logger().error("Updated rollout did not shift as expected. Test failed.")
            rclpy.shutdown()

    def _publish_current_pose(self) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "base"
        msg.pose.position.x = float(self.current_pose[0])
        msg.pose.position.y = float(self.current_pose[1])
        msg.pose.position.z = float(self.current_pose[2])
        msg.pose.orientation.w = 1.0
        self.current_pose_pub.publish(msg)

        if self.settle_repeats_left > 0:
            self.settle_repeats_left -= 1

    def _delta_pose_cb(self, msg: PoseStamped) -> None:
        if self.test_failed:
            return

        delta = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        target = self.current_pose + delta

        if self.expected_points is None:
            y_rollout, _, _ = self.dmp_node.dmp.rollout()
            expected = y_rollout[::10]
            if expected.shape[0] > 15:
                expected = expected[:15]
            self.expected_points = expected

        if self.expected_points is None or self.delta_index >= self.expected_points.shape[0]:
            self.get_logger().error("Received too many delta_pose messages.")
            self.test_failed = True
            rclpy.shutdown()
            return

        expected_target = self.expected_points[self.delta_index]
        if np.linalg.norm(target - expected_target) > 1e-2:
            self.get_logger().error(
                f"Delta pose mismatch at index {self.delta_index}: expected "
                f"{expected_target} got {target}"
            )
            self.test_failed = True
            rclpy.shutdown()
            return

        self.delta_index += 1
        self.current_pose = target
        self.settle_repeats_left = 6

        if self.expected_points is not None and self.delta_index >= self.expected_points.shape[0]:
            self.get_logger().info("Delta pose publishing test passed.")
            rclpy.shutdown()

    def _timeout(self) -> None:
        if self.test_failed:
            return
        self.get_logger().error("Test timed out waiting for delta_pose messages.")
        self.test_failed = True
        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    dmp_node = DMPNode()
    dmp_node.delta_reference = "current"
    tester = DMPNodeTester(dmp_node)
    executor = MultiThreadedExecutor()
    executor.add_node(dmp_node)
    executor.add_node(tester)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        dmp_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
