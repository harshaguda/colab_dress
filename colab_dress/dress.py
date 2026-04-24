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
        self.declare_parameter("attention_gate_topic", "/engagement/attention_gate")
        self.declare_parameter("pose_topic", "/pose_estimator/pose_3d")
        self.declare_parameter("trajectory_status_topic", "/dmp/trajectory_status")
        self.declare_parameter("shoulder_update_flag_topic", "/dmp/shoulder_update_enabled")
        self.declare_parameter("non_adaptive_flag_topic", "/non_adaptive_flag")
        self.declare_parameter("enable_shoulder_updates_on_start", False)

        self.declare_parameter("approach_labels", ["approach"])
        self.declare_parameter("extend_labels", ["extendarm"])
        self.declare_parameter("recede_labels", ["recede"])

        self.declare_parameter("required_emotions", ["happy", "neutral"])
        self.declare_parameter("required_engagement", ["paying_attention"])

        self.declare_parameter("action_timeout", 2.0)
        self.declare_parameter("emotion_timeout", 2.0)
        self.declare_parameter("engagement_timeout", 2.0)
        self.declare_parameter("attention_pause_grace_sec", 0.4)
        self.declare_parameter("pose_timeout", 1.0)

        self.declare_parameter("pose_buffer_len", 10)
        self.declare_parameter("capture_pose_samples", 10)
        self.declare_parameter("capture_pose_window_sec", 2.0)
        self.declare_parameter("min_pose_samples", 10)
        self.declare_parameter("max_pose_std", 0.1)
        self.declare_parameter("shoulder_index", 2)
        self.declare_parameter("shoulder_update_threshold", 0.1)
        self.declare_parameter("min_shoulder_update_period", 0.2)
        self.declare_parameter("frame_id_default", "base")
        self.declare_parameter("arm_pose_z_offset", 0.05)

        actions_topic = self.get_parameter("actions_topic").value
        emotions_topic = self.get_parameter("emotions_topic").value
        engagement_topic = self.get_parameter("engagement_topic").value
        attention_gate_topic = self.get_parameter("attention_gate_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        trajectory_status_topic = self.get_parameter("trajectory_status_topic").value
        shoulder_update_flag_topic = self.get_parameter("shoulder_update_flag_topic").value
        non_adaptive_flag_topic = self.get_parameter("non_adaptive_flag_topic").value
        enable_shoulder_updates_on_start = bool(
            self.get_parameter("enable_shoulder_updates_on_start").value
        )

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
        self._attention_pause_grace_sec = float(
            self.get_parameter("attention_pause_grace_sec").value
        )
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
        self._arm_pose_z_offset = float(self.get_parameter("arm_pose_z_offset").value)

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
        self.attention_gate_pub = self.create_publisher(String, attention_gate_topic, 10)

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
        self._attention_lost_since: Optional[float] = None
        self._attention_gate_state: str = "1"

        self._last_published_shoulder: Optional[np.ndarray] = None
        self._last_shoulder_publish_time: Optional[float] = None
        self._trajectory_active: bool = False
        self._capture_start_time: Optional[float] = None
        self._shoulder_update_enabled: Optional[bool] = None
        self._non_adaptive_flag_value: Optional[bool] = None
        self._dressing_paused_for_status: bool = False

        self.timer = self.create_timer(0.1, self._tick)
        self._publish_attention_gate("1")
        self._publish_shoulder_update_flag(enable_shoulder_updates_on_start)
        self._publish_non_adaptive_flag(False)
        self.get_logger().info(
            f"Startup shoulder updates enabled: {enable_shoulder_updates_on_start}"
        )
        self.get_logger().info("Dress FSM node ready")

    def _publish_shoulder_update_flag(self, enabled: bool) -> None:
        if self._shoulder_update_enabled == enabled:
            return
        msg = Bool()
        msg.data = bool(enabled)
        self.shoulder_flag_pub.publish(msg)
        self._shoulder_update_enabled = enabled
        self.get_logger().info(f"Published shoulder update flag: {self._shoulder_update_enabled}")

    def _publish_non_adaptive_flag(self, enabled: bool) -> None:
        if self._non_adaptive_flag_value == enabled:
            return
        msg = Bool()
        msg.data = bool(enabled)
        self.non_adaptive_flag_pub.publish(msg)
        self._non_adaptive_flag_value = enabled
        self.get_logger().info(f"Published /non_adaptive_flag: {self._non_adaptive_flag_value}")

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
        self._update_attention_gate_state()

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

        action_ok = self._action_ok()
        self.get_logger().info(f"Action OK: {action_ok}")
        engagement_ok = self._engagement_ok()
        self._update_attention_gate_state()

        if self._state == DressState.WAIT_FOR_ACTION:
            if self._trajectory_active:
                return
            if action_ok:
                self._transition(DressState.WAIT_FOR_ENGAGEMENT) # skipping wait for engagement since we want to start dressing as soon as action is detected
            return

        if self._state == DressState.WAIT_FOR_ENGAGEMENT:
            if self._trajectory_active:
                return
            # if not action_ok:
            #     self._transition(DressState.WAIT_FOR_ACTION)
            #     return
            if engagement_ok:
                self._pose_buffer.clear()
                self._capture_start_time = time.time()
                time.sleep(5)
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
            if not self._attention_gate_is_ok():
                if not self._dressing_paused_for_status:
                    self._dressing_paused_for_status = True
                    self._publish_shoulder_update_flag(False)
                    self.get_logger().warning(
                        "Dressing paused: waiting for /engagement/attention_gate == '1'"
                    )
                return

            if self._dressing_paused_for_status:
                self._dressing_paused_for_status = False
                self._publish_shoulder_update_flag(True)
                self.get_logger().info(
                    "Dressing resumed: /engagement/attention_gate is '1'"
                )

            if not self._trajectory_active:
                self.get_logger().info("Dressing cycle completed. Exiting process.")
                self._publish_shoulder_update_flag(False)
                # wait 15 seconds to allow any final messages to be sent before exiting
                time.sleep(15)
                exit(0)

    def _transition(self, new_state: DressState) -> None:
        if new_state == self._state:
            return
        self.get_logger().info(f"State: {self._state.value} -> {new_state.value}")
        self._state = new_state
        self._publish_shoulder_update_flag(new_state == DressState.DRESSING)
        if new_state != DressState.CAPTURE_POSE:
            self._capture_start_time = None
        if new_state != DressState.DRESSING:
            self._dressing_paused_for_status = False
        if new_state == DressState.WAIT_FOR_ACTION:
            self._pose_buffer.clear()

    def _reset(self, reason: str) -> None:
        self.get_logger().info(f"Resetting FSM: {reason}")
        self._recede_triggered = False
        self._pose_buffer.clear()
        self._dressing_paused_for_status = False
        self._state = DressState.WAIT_FOR_ACTION

    # def _action_ok(self) -> bool:
        
    #     return True

    #     now = time.time()
    #     if self._last_approach_time is None or self._last_extend_time is None:
    #         return False
    #     if now - self._last_approach_time > self._action_timeout:
    #         return False
    #     if now - self._last_extend_time > self._action_timeout:
    #         return False
    #     return True

    def _action_ok(self) -> bool:
        

        now = time.time()
        if self._last_approach_time is None:
            return False
        if now - self._last_approach_time > self._action_timeout:
            return False
        # if now - self._last_extend_time > self._action_timeout:
        #     return False
        return True

    def _engagement_ok(self) -> bool:
        # return True
        now = time.time()
        if self._last_emotion is None or self._last_emotion_time is None:
            return False
        if now - self._last_emotion_time > self._emotion_timeout:
            return False
        if self._required_emotions and self._last_emotion not in self._required_emotions:
            return False

        if not self._required_engagement:
            return True
        if self._last_engagement is None or self._last_engagement_time is None:
            return False
        if now - self._last_engagement_time > self._engagement_timeout:
            return False
        if self._last_engagement not in self._required_engagement:
            return False
        # self.get_logger().info(f"Engagement OK: Emotion={self._last_emotion}, Engagement={self._last_engagement}")
        return True

    def _engagement_status_is_one(self) -> bool:
        if self._last_engagement is None or self._last_engagement_time is None:
            return False
        if time.time() - self._last_engagement_time > self._engagement_timeout:
            return False
        return self._last_engagement == "paying_attention" or self._last_engagement == "1"

    def _publish_attention_gate(self, value: str) -> None:
        if value == self._attention_gate_state:
            return
        msg = String()
        msg.data = value
        self.attention_gate_pub.publish(msg)
        self._attention_gate_state = value
        if value == "1":
            self.get_logger().info("Published /engagement/attention_gate='1' (attentive)")
        else:
            self.get_logger().warning("Published /engagement/attention_gate='0' (not attentive)")

    def _attention_gate_is_ok(self) -> bool:
        return self._attention_gate_state == "1"

    def _update_attention_gate_state(self) -> None:
        now = time.time()
        currently_attentive = self._engagement_status_is_one()

        if currently_attentive:
            self._attention_lost_since = None
            self._publish_attention_gate("1")
            return

        if self._attention_lost_since is None:
            self._attention_lost_since = now

        if now - self._attention_lost_since >= self._attention_pause_grace_sec:
            self._publish_attention_gate("0")
        else:
            self._publish_attention_gate("1")

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
            pose.position.z = float(pt[2] + self._arm_pose_z_offset)
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.dmp_pub.publish(msg)
        self._trajectory_active = True
        self._publish_shoulder(pose_points, header)
        self.get_logger().info(
            f"Published initial DMP arm poses (z offset: {self._arm_pose_z_offset:.3f} m)"
        )

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
        if self._shoulder_update_enabled:
            self._publish_shoulder(pose_points, header)
            self.get_logger().info("Published updated shoulder position")


def main(args=None):
    rclpy.init(args=args)
    dress_node = DressNode()
    rclpy.spin(dress_node)
    dress_node.destroy_node()
    rclpy.shutdown()