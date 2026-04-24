from collections import deque
from enum import Enum
import time
from typing import Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header, String, Bool
from geometry_msgs.msg import PoseArray, Pose, PointStamped

import random

class DressState(Enum):
    WAIT_FOR_ACTION = "wait_for_action"
    WAIT_FOR_ENGAGEMENT = "wait_for_engagement"
    CAPTURE_POSE = "capture_pose"
    DRESSING = "dressing"


class DressNode(Node):
    """Finite state machine to coordinate dressing based on actions, engagement, and pose."""

    def __init__(self):
        super().__init__("dress_node")

        self.declare_parameter("actions_topic", "/actions_recognised")
        self.declare_parameter("emotions_topic", "/engagement/emotions")
        self.declare_parameter("engagement_topic", "/engagement/status")
        self.declare_parameter("pose_topic", "/pose_estimator/pose_3d")
        self.declare_parameter("trajectory_status_topic", "/dmp/trajectory_status")
        self.declare_parameter("shoulder_update_flag_topic", "/dmp/shoulder_update_enabled")
        self.declare_parameter("non_adaptive_flag_topic", "/non_adaptive_flag")

        self.declare_parameter("approach_labels", ["approach"])
        self.declare_parameter("extend_labels", ["extendarm"])
        self.declare_parameter("recede_labels", ["recede"])

        self.declare_parameter("required_emotions", ["happy", "neutral"])
        self.declare_parameter("required_engagement", ["1"])

        self.declare_parameter("action_timeout", 2.0)
        self.declare_parameter("emotion_timeout", 2.0)
        self.declare_parameter("engagement_timeout", 2.0)
        self.declare_parameter("pose_timeout", 1.0)

        self.declare_parameter("pose_buffer_len", 20)
        self.declare_parameter("capture_pose_samples", 20)
        self.declare_parameter("capture_pose_window_sec", 2.0)
        self.declare_parameter("min_pose_samples", 20)
        self.declare_parameter("max_pose_std", 0.1)
        self.declare_parameter("shoulder_index", 2)
        self.declare_parameter("shoulder_update_threshold", 0.1)
        self.declare_parameter("min_shoulder_update_period", 0.2)
        self.declare_parameter("frame_id_default", "base")

        actions_topic = self.get_parameter("actions_topic").value
        emotions_topic = self.get_parameter("emotions_topic").value
        engagement_topic = self.get_parameter("engagement_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        trajectory_status_topic = self.get_parameter("trajectory_status_topic").value
        shoulder_update_flag_topic = self.get_parameter("shoulder_update_flag_topic").value
        non_adaptive_flag_topic = self.get_parameter("non_adaptive_flag_topic").value

        self._approach_labels = self._normalize_labels(
            self.get_parameter("approach_labels").value
        )
        self._extend_labels = self._normalize_labels(
            self.get_parameter("extend_labels").value
        )
        self._recede_labels = self._normalize_labels(
            self.get_parameter("recede_labels").value
        )
        self._required_emotions = self._normalize_labels(
            self.get_parameter("required_emotions").value
        )
        self._required_engagement = self._normalize_labels(
            self.get_parameter("required_engagement").value
        )

        self._action_timeout = float(self.get_parameter("action_timeout").value)
        self._emotion_timeout = float(self.get_parameter("emotion_timeout").value)
        self._engagement_timeout = float(self.get_parameter("engagement_timeout").value)
        self._pose_timeout = float(self.get_parameter("pose_timeout").value)

        self._pose_buffer_len = int(self.get_parameter("pose_buffer_len").value)
        self._capture_pose_samples = int(self.get_parameter("capture_pose_samples").value)
        self._capture_pose_window_sec = float(
            self.get_parameter("capture_pose_window_sec").value
        )
        self._min_pose_samples = int(self.get_parameter("min_pose_samples").value)
        self._max_pose_std = float(self.get_parameter("max_pose_std").value)
        self._shoulder_index = int(self.get_parameter("shoulder_index").value)
        self._shoulder_update_threshold = float(
            self.get_parameter("shoulder_update_threshold").value
        )
        self._min_shoulder_update_period = float(
            self.get_parameter("min_shoulder_update_period").value
        )
        self._frame_id_default = self.get_parameter("frame_id_default").value

        self.action_sub = self.create_subscription(
            String, actions_topic, self._action_callback, 10
        )
        self.emotion_sub = self.create_subscription(
            String, emotions_topic, self._emotion_callback, 10
        )
        self.engagement_sub = self.create_subscription(
            String, engagement_topic, self._engagement_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseArray, pose_topic, self._pose_callback, qos_profile_sensor_data
        )
        self.trajectory_status_sub = self.create_subscription(
            String, trajectory_status_topic, self._trajectory_status_callback, 10
        )

        self.dmp_pub = self.create_publisher(PoseArray, "dmp/arm_poses", 10)
        self.shoulder_pub = self.create_publisher(PointStamped, "dmp/shoulder_position", 10)
        self.shoulder_flag_pub = self.create_publisher(Bool, shoulder_update_flag_topic, 10)
        self.non_adaptive_flag_pub = self.create_publisher(Bool, non_adaptive_flag_topic, 10)

        self._state = DressState.WAIT_FOR_ACTION
        self._pose_buffer = deque(maxlen=max(self._pose_buffer_len, self._capture_pose_samples))
        self._latest_pose: Optional[np.ndarray] = None
        self._latest_pose_time: Optional[float] = None
        self._latest_frame_id: str = ""

        self._last_action_label: Optional[str] = None
        self._last_action_time: Optional[float] = None
        self._last_approach_time: Optional[float] = None
        self._last_extend_time: Optional[float] = None
        self._recede_triggered: bool = False

        self._last_emotion: Optional[str] = None
        self._last_emotion_time: Optional[float] = None
        self._last_engagement: Optional[str] = None
        self._last_engagement_time: Optional[float] = None

        self._last_published_shoulder: Optional[np.ndarray] = None
        self._last_shoulder_publish_time: Optional[float] = None
        self._trajectory_active: bool = False
        self._capture_start_time: Optional[float] = None
        self._shoulder_update_enabled: Optional[bool] = None
        self._non_adaptive_flag_value: Optional[bool] = None
        self._capture_paused_for_engagement: bool = False
        self._dressing_paused_for_engagement: bool = False

        self.dress_flag = True #random.random() < 0.5

        self.timer = self.create_timer(0.1, self._tick)
        self._publish_shoulder_update_flag(True)
        self._publish_non_adaptive_flag(True)
        self.get_logger().info("Dress FSM node ready")

    def _publish_shoulder_update_flag(self, enabled: bool) -> None:
        if self._shoulder_update_enabled == enabled:
            return
        msg = Bool()
        msg.data = bool(enabled)
        self.shoulder_flag_pub.publish(msg)
        self._shoulder_update_enabled = enabled

    def _publish_non_adaptive_flag(self, enabled: bool) -> None:
        if self._non_adaptive_flag_value == enabled:
            return
        msg = Bool()
        msg.data = bool(enabled)
        self.non_adaptive_flag_pub.publish(msg)
        self._non_adaptive_flag_value = enabled

    @staticmethod
    def _normalize_labels(labels) -> Set[str]:
        if labels is None:
            return set()
        if isinstance(labels, str):
            labels = [labels]
        return {str(label).strip().lower() for label in labels if str(label).strip()}

    def _action_callback(self, msg: String) -> None:
        label = msg.data.strip().lower()
        if not label:
            return
        now = time.time()
        self._last_action_label = label
        self._last_action_time = now

        if label in self._recede_labels:
            self._recede_triggered = True
            self.get_logger().info("Recede action detected; stopping dressing")
            return

        if label in self._approach_labels:
            self._last_approach_time = now
        if label in self._extend_labels:
            self._last_extend_time = now

    def _emotion_callback(self, msg: String) -> None:
        label = msg.data.strip().lower()
        if not label:
            return
        self._last_emotion = label
        self._last_emotion_time = time.time()

    def _engagement_callback(self, msg: String) -> None:
        label = msg.data.strip().lower()
        if label in {"true", "false"}:
            label = "1" if label == "true" else "0"
        if not label:
            return
        self._last_engagement = label
        self._last_engagement_time = time.time()

    def _trajectory_status_callback(self, msg: String) -> None:
        status = msg.data.strip().lower()
        if status in {"active", "updated", "running"}:
            self._trajectory_active = True
        elif status in {"completed", "idle", "done"}:
            self._trajectory_active = False

    def _pose_callback(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        curr_poses = np.array(
            [[p.position.x, p.position.y, p.position.z] for p in msg.poses], dtype=float
        )
        self._latest_pose = curr_poses
        self._latest_pose_time = time.time()
        if msg.header.frame_id:
            self._latest_frame_id = msg.header.frame_id

        if self._state == DressState.CAPTURE_POSE and len(self._pose_buffer) < self._capture_pose_samples:
            self._pose_buffer.append(curr_poses)

        # if self._state == DressState.DRESSING:
        #     self._maybe_publish_shoulder_update(curr_poses, msg.header)

    def _tick(self) -> None:
        # if self._recede_triggered:
        #     self._reset("recede action")
        #     return

        if not self.dress_flag:
            self.get_logger().info("Dress flag is false; skipping dressing process")
            exit(0)

        action_ok = self._action_ok()
        engagement_ok = self._engagement_ok()

        if self._state == DressState.WAIT_FOR_ACTION:
            if self._trajectory_active:
                return
            if action_ok:
                self._transition(DressState.WAIT_FOR_ENGAGEMENT) # skipping wait for engagement since we want to start dressing as soon as action is detected
            return

        if self._state == DressState.WAIT_FOR_ENGAGEMENT:
            if self._trajectory_active:
                return
            if not action_ok:
                self._transition(DressState.WAIT_FOR_ACTION)
                return
            if engagement_ok:
                self._pose_buffer.clear()
                self._capture_start_time = time.time()
                self.get_logger().info(
                    f"Dressing initiated: collecting {self._capture_pose_samples} poses within {self._capture_pose_window_sec:.2f}s"
                )
                self._transition(DressState.CAPTURE_POSE)
            return

        if self._state == DressState.CAPTURE_POSE:
            if self._trajectory_active:
                return
            if self._capture_start_time is None:
                self._capture_start_time = time.time()

            if not engagement_ok:
                if not self._capture_paused_for_engagement:
                    self.get_logger().warning(
                        "Pose capture paused: waiting for desired emotion/engagement"
                    )
                self._capture_paused_for_engagement = True
                self._capture_start_time = time.time()
                return

            if self._capture_paused_for_engagement:
                self._capture_paused_for_engagement = False
                self._capture_start_time = time.time()
                self.get_logger().info(
                    "Desired emotion/engagement restored; resuming pose capture"
                )

            if (
                time.time() - self._capture_start_time > self._capture_pose_window_sec
                and len(self._pose_buffer) < self._capture_pose_samples
            ):
                self.get_logger().warning(
                    f"Only {len(self._pose_buffer)}/{self._capture_pose_samples} poses received in "
                    f"{self._capture_pose_window_sec:.2f}s; recollecting"
                )
                self._pose_buffer.clear()
                self._capture_start_time = time.time()
                return

            # if not action_ok or not engagement_ok:
            #     self._transition(DressState.WAIT_FOR_ACTION)
            #     return
            if len(self._pose_buffer) < self._capture_pose_samples:
                return
            ready, pose_median = self._pose_reliable(required_samples=self._capture_pose_samples)
            self.get_logger().info(f"Ready: {ready}, Pose Median: {pose_median}")
            if ready and pose_median is not None:
                self.get_logger().info("Captured reliable pose; starting dressing")
                self._transition(DressState.DRESSING)
                self._publish_arm_poses(pose_median)
            else:
                self.get_logger().warning(
                    f"Pose reliability failed; recollecting {self._capture_pose_samples} new poses"
                )
                self._pose_buffer.clear()
                self._capture_start_time = time.time()
            return

        if self._state == DressState.DRESSING:
            if not engagement_ok:
                if not self._dressing_paused_for_engagement:
                    self._dressing_paused_for_engagement = True
                    self._publish_shoulder_update_flag(False)
                    self.get_logger().warning(
                        "Dressing paused: waiting for desired emotion/engagement"
                    )
                return

            if self._dressing_paused_for_engagement:
                self._dressing_paused_for_engagement = False
                self._publish_shoulder_update_flag(True)
                self.get_logger().info(
                    "Desired emotion/engagement restored; resuming dressing"
                )

            if not self._trajectory_active:
                self.get_logger().info("Dressing cycle completed. Exiting process.")
                self._publish_shoulder_update_flag(False)
                exit(0)

    def _transition(self, new_state: DressState) -> None:
        if new_state == self._state:
            return
        self.get_logger().info(f"State: {self._state.value} -> {new_state.value}")
        self._state = new_state
        self._publish_shoulder_update_flag(new_state == DressState.DRESSING)
        if new_state != DressState.CAPTURE_POSE:
            self._capture_start_time = None
            self._capture_paused_for_engagement = False
        if new_state != DressState.DRESSING:
            self._dressing_paused_for_engagement = False
        if new_state == DressState.WAIT_FOR_ACTION:
            self._pose_buffer.clear()

    def _reset(self, reason: str) -> None:
        self.get_logger().info(f"Resetting FSM: {reason}")
        self._recede_triggered = False
        self._pose_buffer.clear()
        self._capture_paused_for_engagement = False
        self._dressing_paused_for_engagement = False
        self._state = DressState.WAIT_FOR_ACTION

    def _action_ok(self) -> bool:
        
        return self.dress_flag
        return random.random() < 0.5

    def _engagement_ok(self) -> bool:

        return self.dress_flag
        return random.random() < 0.5

    def _pose_fresh(self) -> bool:
        if self._latest_pose_time is None:
            return False
        return (time.time() - self._latest_pose_time) <= self._pose_timeout

    def _pose_reliable(self, required_samples: Optional[int] = None) -> Tuple[bool, Optional[np.ndarray]]:
        required = self._min_pose_samples if required_samples is None else int(required_samples)
        self.get_logger().info(f"Pose buffer size: {len(self._pose_buffer)}, Required: {required}")
        if len(self._pose_buffer) < required:
            return False, None
        stack = np.stack(self._pose_buffer, axis=0)
        pose_median = np.median(stack, axis=0)
        pose_std = np.std(stack, axis=0)
        self.get_logger().info(f"Pose std: {pose_std}, Max allowed: {self._max_pose_std}")
        if np.all(pose_std <= self._max_pose_std):
            wrist = pose_median[1]
            elbow = pose_median[2]
            v_n = elbow - wrist
            nrm = np.linalg.norm(v_n)
            if nrm < 1e-8:
                return False, None
            v_n /= nrm
            ext_wrist = -v_n * 0.2 + wrist
            new_pose_median = np.array([pose_median[0], ext_wrist, pose_median[1], pose_median[2], pose_median[3]])
            self.get_logger().info(f"{new_pose_median}")

            return True, new_pose_median
        return False, None

    def _publish_arm_poses(self, pose_points: np.ndarray) -> None:
        header = Header()
        header.frame_id = self._latest_frame_id or self._frame_id_default
        header.stamp = self.get_clock().now().to_msg()

        msg = PoseArray()
        msg.header = header
        msg.poses = []
        for pt in pose_points:
            pose = Pose()
            pose.position.x = float(pt[0])
            pose.position.y = float(pt[1])
            pose.position.z = float(pt[2])
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.dmp_pub.publish(msg)
        self._trajectory_active = True
        self._publish_shoulder(pose_points, header)
        self.get_logger().info("Published initial DMP arm poses")

    def _publish_shoulder(self, pose_points: np.ndarray, header: Header) -> None:
        if pose_points.shape[0] <= self._shoulder_index:
            self.get_logger().warning("Shoulder index out of range; cannot publish shoulder")
            return
        shoulder = pose_points[self._shoulder_index]
        msg = PointStamped()
        msg.header = header
        msg.point.x = float(shoulder[0])
        msg.point.y = float(shoulder[1])
        msg.point.z = float(shoulder[2])
        self.shoulder_pub.publish(msg)
        self._last_published_shoulder = shoulder
        self._last_shoulder_publish_time = time.time()

    def _maybe_publish_shoulder_update(self, pose_points: np.ndarray, header: Header) -> None:
        if pose_points.shape[0] <= self._shoulder_index:
            return
        shoulder = pose_points[self._shoulder_index]
        if self._last_published_shoulder is None:
            self._publish_shoulder(pose_points, header)
            return
        delta = np.linalg.norm(shoulder - self._last_published_shoulder)
        if delta < self._shoulder_update_threshold:
            return
        now = time.time()
        if (
            self._last_shoulder_publish_time is not None
            and now - self._last_shoulder_publish_time < self._min_shoulder_update_period
        ):
            return
        self._publish_shoulder(pose_points, header)
        self.get_logger().info("Published updated shoulder position")


def main(args=None):
    rclpy.init(args=args)
    dress_node = DressNode()
    rclpy.spin(dress_node)
    dress_node.destroy_node()
    rclpy.shutdown()