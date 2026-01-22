try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
except Exception as exc:  # pragma: no cover - ROS 2 not available in all contexts
    raise RuntimeError("rclpy is required to run this script") from exc


class CurrentPosePublisher(Node):
    def __init__(self) -> None:
        super().__init__("test_send_current_pose")
        self.declare_parameter("topic", "/NS_1/franka_robot_state_broadcaster/current_pose")
        self.declare_parameter("frame_id", "base")
        self.declare_parameter("publish_period", 0.1)
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", 0.10)

        self.topic = str(self.get_parameter("topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_period = float(self.get_parameter("publish_period").value)
        self.position = [
            float(self.get_parameter("x").value),
            float(self.get_parameter("y").value),
            float(self.get_parameter("z").value),
        ]

        self.publisher = self.create_publisher(PoseStamped, self.topic, 10)
        self.timer = self.create_timer(self.publish_period, self._publish_current_pose)
        self.get_logger().info(
            f"Publishing current_pose to '{self.topic}' every {self.publish_period:.2f}s"
        )

    def _publish_current_pose(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(self.position[0])
        msg.pose.position.y = float(self.position[1])
        msg.pose.position.z = float(self.position[2])
        msg.pose.orientation.w = 1.0
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = CurrentPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
