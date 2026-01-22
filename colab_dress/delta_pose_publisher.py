from typing import List

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, PoseArray
except Exception as exc:  # pragma: no cover - ROS 2 not available in all contexts
    raise RuntimeError("rclpy is required to run this script") from exc


class DeltaPosePublisher(Node):
    def __init__(self) -> None:
        super().__init__("delta_pose_publisher")

        self.declare_parameter("topic", "/delta_pose")
        self.declare_parameter("publish_period", 1.0)
        self.declare_parameter("frame_id", "base")

        self.topic = str(self.get_parameter("topic").value)
        self.publish_period = float(self.get_parameter("publish_period").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.publisher = self.create_publisher(PoseStamped, self.topic, 10)
        # self.rollout_sub = self.create_subscription(
        #     PoseArray, "/dmp/dmp_rollout", self._rollout_cb, 10
        # )

        self.points = [[0.424, -0.025, 0.286],
                        [0.424, -0.025, 0.24032321844813684],
                        [0.424, -0.025, 0.21110690485015132],
                        [0.424, -0.025, 0.2034098110336771],
                        [0.424, -0.025, 0.18120995871724715],
                        [0.424, -0.025, 0.14720430844839827],
                        [0.424, -0.025, 0.1208236525042143],
                        [0.424, -0.025, 0.10526486302093382],
                        [0.424, -0.025, 0.09655636364517096],
                        [0.424, -0.025, 0.09095696306084796],
                        [0.424, -0.025, 0.086]]
        self.index = 0

        self.timer = self.create_timer(self.publish_period, self._publish_next)
        self.get_logger().info(
            f"Publishing delta_pose points to '{self.topic}' every "
            f"{self.publish_period:.2f}s"
        )
        # self.get_logger().info("Subscribed to /dmp/dmp_rollout")

    def _rollout_cb(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        self.points = [
            [pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses
        ]
        self.index = 0
        self.frame_id = msg.header.frame_id or self.frame_id
        self.get_logger().info(f"Loaded {len(self.points)} points from /dmp/dmp_rollout")

    def _publish_next(self) -> None:
        if not self.points:
            return
        
        point = self.points[self.index]
        self.index = (self.index + 1) #% len(self.points)
        if self.index >= 8:
            rclpy.shutdown()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(point[0])
        msg.pose.position.y = float(point[1])
        msg.pose.position.z = float(point[2])
        msg.pose.orientation.w = 1.0
        self.publisher.publish(msg)
        self.get_logger().info(f"Published delta_pose: {point}")


def main() -> None:
    rclpy.init()
    node = DeltaPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
