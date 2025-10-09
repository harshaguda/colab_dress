#!/usr/bin/env python3
"""Execute an end-effector trajectory and allow modifications during execution.

This node accepts a trajectory as a `geometry_msgs/PoseArray` on
`/end_effector_trajectory` (replaces current trajectory) and
`/end_effector_trajectory_append` (appends to the current trajectory).

When a new trajectory is received the node cancels the currently executing
MoveGroup goal (if any) and replans+executes from the current robot state.
Each waypoint is sent as an individual MoveGroup goal (serial execution).
This keeps the implementation simple and portable. It can be extended to
compute a single combined trajectory if desired.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import List

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from std_msgs.msg import Header, String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive


class EndEffectorTrajectoryExecutor(Node):
    """Node that executes a sequence of end-effector poses and supports live edits.

    Contract:
    - Inputs: PoseArray messages on `/end_effector_trajectory` (replace) and
      `/end_effector_trajectory_append` (append).
    - Outputs: status messages on `/end_effector_trajectory_status` (String).
    - Behavior: Cancel currently executing goal and start executing the new
      trajectory when modifications are received.
    Error modes: planning/execution failures are published to status topic.
    """

    def __init__(self) -> None:
        super().__init__("end_effector_trajectory_executor")

        self.declare_parameter("group_name", "manipulator")
        self.declare_parameter("reference_frame", "base_link")
        self.declare_parameter("end_effector_link", "end_effector_link")
        self.declare_parameter("position_tolerance", 0.01)
        self.declare_parameter("orientation_tolerance", 0.02)
        self.declare_parameter("max_velocity_scaling", 0.2)
        self.declare_parameter("max_acceleration_scaling", 0.2)
        self.declare_parameter("planning_time", 5.0)

        self._group_name = self.get_parameter("group_name").get_parameter_value().string_value
        self._reference_frame = self.get_parameter("reference_frame").get_parameter_value().string_value
        self._end_effector_link = self.get_parameter("end_effector_link").get_parameter_value().string_value
        # Use parameter accessor methods to be robust to Unknown/None values
        self._position_tolerance = float(
            self.get_parameter("position_tolerance").get_parameter_value().double_value
        )
        self._orientation_tolerance = float(
            self.get_parameter("orientation_tolerance").get_parameter_value().double_value
        )
        self._velocity_scaling = float(
            self.get_parameter("max_velocity_scaling").get_parameter_value().double_value
        )
        self._accel_scaling = float(
            self.get_parameter("max_acceleration_scaling").get_parameter_value().double_value
        )
        self._planning_time = float(self.get_parameter("planning_time").get_parameter_value().double_value)

        self._trajectory_lock = threading.Lock()
        self._trajectory: List[PoseStamped] = []

        # Event that signals new trajectory has arrived and current goal should be cancelled
        self._new_trajectory_event = threading.Event()

        # Current move action goal handle (if any)
        self._current_goal_handle = None

        # Action client to MoveGroup
        self._action_client = ActionClient(self, MoveGroup, "/move_action")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server not available")

        # Subscribers to set or append trajectory
        self.create_subscription(PoseArray, "/end_effector_trajectory", self._trajectory_callback, 10)
        self.create_subscription(PoseArray, "/end_effector_trajectory_append", self._append_callback, 10)

        # Status publisher
        self._status_pub = self.create_publisher(String, "/end_effector_trajectory_status", 10)
        # Worker thread runs the execution loop. It uses a dedicated executor so
        # that waiting on action futures does not interfere with any external
        # rclpy.spin calls running in other threads.
        from rclpy.executors import SingleThreadedExecutor

        self._worker_executor = SingleThreadedExecutor()
        # Add this node to the executor so callbacks/futures are handled
        self._worker_executor.add_node(self)

        self._shutdown = False
        self._worker_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._worker_thread.start()

        self.get_logger().info("EndEffectorTrajectoryExecutor ready")

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)
        self.get_logger().info(text)

    def _trajectory_callback(self, msg: PoseArray) -> None:
        """Replace current trajectory with the received PoseArray.

        Cancels any currently-executing goal and signals worker thread to restart.
        """
        with self._trajectory_lock:
            poses = []
            for p in msg.poses:
                stamped = PoseStamped()
                stamped.header = msg.header if isinstance(msg.header, Header) else Header()
                stamped.header.frame_id = msg.header.frame_id
                stamped.pose = copy.deepcopy(p)
                poses.append(stamped)
            self._trajectory = poses

        # Signal worker to cancel current goal and use new trajectory
        self._new_trajectory_event.set()
        self._publish_status("Trajectory replaced; will replan and execute")

    def _append_callback(self, msg: PoseArray) -> None:
        """Append poses to the current trajectory.

        If no trajectory is running, this acts like a replace.
        """
        with self._trajectory_lock:
            if not self._trajectory:
                base_header = msg.header
                for p in msg.poses:
                    stamped = PoseStamped()
                    stamped.header = base_header
                    stamped.pose = copy.deepcopy(p)
                    self._trajectory.append(stamped)
            else:
                for p in msg.poses:
                    stamped = PoseStamped()
                    stamped.header = msg.header
                    stamped.pose = copy.deepcopy(p)
                    self._trajectory.append(stamped)

        self._new_trajectory_event.set()
        self._publish_status("Trajectory appended; will replan/continue")

    def _build_movegroup_goal_for_pose(self, pose_stamped: PoseStamped) -> MoveGroup.Goal:
        request = MotionPlanRequest()
        request.workspace_parameters.header.frame_id = self._reference_frame
        # Provide a reasonable workspace box
        request.workspace_parameters.min_corner.x = -2.0
        request.workspace_parameters.min_corner.y = -2.0
        request.workspace_parameters.min_corner.z = -2.0
        request.workspace_parameters.max_corner.x = 2.0
        request.workspace_parameters.max_corner.y = 2.0
        request.workspace_parameters.max_corner.z = 2.0

        request.start_state.is_diff = True

        request.goal_constraints = [Constraints()]
        constraint = request.goal_constraints[0]
        constraint.name = "end_effector_pose"

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self._reference_frame
        pos_constraint.link_name = self._end_effector_link
        pos_constraint.constraint_region.primitives = [SolidPrimitive()]
        pos_constraint.constraint_region.primitives[0].type = SolidPrimitive.SPHERE
        pos_constraint.constraint_region.primitives[0].dimensions = [self._position_tolerance]
        pos_constraint.constraint_region.primitive_poses = [Pose()]
        pos_constraint.constraint_region.primitive_poses[0].position.x = pose_stamped.pose.position.x
        pos_constraint.constraint_region.primitive_poses[0].position.y = pose_stamped.pose.position.y
        pos_constraint.constraint_region.primitive_poses[0].position.z = pose_stamped.pose.position.z
        pos_constraint.constraint_region.primitive_poses[0].orientation = pose_stamped.pose.orientation
        constraint.position_constraints = [pos_constraint]

        # Orientation constraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self._reference_frame
        ori_constraint.link_name = self._end_effector_link
        ori_constraint.orientation = pose_stamped.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = self._orientation_tolerance
        ori_constraint.absolute_y_axis_tolerance = self._orientation_tolerance
        ori_constraint.absolute_z_axis_tolerance = self._orientation_tolerance
        constraint.orientation_constraints = [ori_constraint]

        request.planner_id = ""
        request.group_name = self._group_name
        request.num_planning_attempts = 6
        request.allowed_planning_time = self._planning_time
        request.max_velocity_scaling_factor = self._velocity_scaling
        request.max_acceleration_scaling_factor = self._accel_scaling

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.planning_scene_diff.is_diff = True
        return goal_msg

    def _send_goal_and_wait(self, goal_msg: MoveGroup.Goal) -> bool:
        """Send MoveGroup goal and wait for completion. Returns True on success."""
        send_goal_future = self._action_client.send_goal_async(goal_msg)

        # Wait for the send_goal_future using the worker executor
        while not send_goal_future.done():
            # If a new trajectory arrived, abort waiting
            if self._new_trajectory_event.is_set():
                return False
            # Process executor callbacks
            self._worker_executor.spin_once(timeout_sec=0.1)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self._publish_status("MoveGroup action did not return a goal handle")
            return False
        if not goal_handle.accepted:
            self._publish_status("MoveGroup goal was rejected")
            return False

        # Save current goal handle so it can be cancelled by new trajectory requests
        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()

        # Wait for the result while processing callbacks on the worker executor
        while not result_future.done():
            if self._new_trajectory_event.is_set():
                try:
                    cancel_future = goal_handle.cancel_goal_async()
                    # wait briefly for cancel to be processed
                    wait_start = time.time()
                    while not cancel_future.done() and (time.time() - wait_start) < 1.0:
                        self._worker_executor.spin_once(timeout_sec=0.05)
                except Exception:
                    pass
                self._publish_status("Current goal cancelled due to trajectory update")
                return False
            self._worker_executor.spin_once(timeout_sec=0.1)

        result_handle = result_future.result()
        self._current_goal_handle = None
        if result_handle is None:
            self._publish_status("MoveGroup action returned no result")
            return False

        result = result_handle.result
        if result.error_code.val != 1:  # MoveItErrorCodes::SUCCESS
            self._publish_status(f"Execution failed with error code {result.error_code.val}")
            return False

        return True

    def _execution_loop(self) -> None:
        """Worker loop that serially executes waypoints from the current trajectory."""
        while rclpy.ok() and not self._shutdown:
            # Wait until we have a trajectory
            with self._trajectory_lock:
                has_traj = bool(self._trajectory)
            if not has_traj:
                time.sleep(0.1)
                continue

            # Clear the new-trajectory flag before starting execution
            self._new_trajectory_event.clear()

            # Copy the current trajectory to execute
            with self._trajectory_lock:
                work_list = list(self._trajectory)

            self._publish_status(f"Starting execution of {len(work_list)} waypoint(s)")

            executed_all = True
            for idx, pose_stamped in enumerate(work_list):
                # If a new trajectory has been requested, break and restart
                if self._new_trajectory_event.is_set():
                    executed_all = False
                    break

                goal_msg = self._build_movegroup_goal_for_pose(pose_stamped)
                self._publish_status(f"Executing waypoint {idx + 1}/{len(work_list)}")
                success = self._send_goal_and_wait(goal_msg)
                if not success:
                    # If cancelled due to new trajectory we break to handle it
                    if self._new_trajectory_event.is_set():
                        executed_all = False
                        break
                    self._publish_status("Stopping execution due to failure")
                    # Stop executing further waypoints on error
                    executed_all = False
                    break

            # If we finished executing the copied trajectory fully and it was not
            # interrupted or failed, clear the stored trajectory so we don't loop.
            if executed_all and not self._new_trajectory_event.is_set():
                with self._trajectory_lock:
                    # Only clear if the stored trajectory matches what we executed
                    if self._trajectory == work_list:
                        self._trajectory = []
                self._publish_status("Trajectory execution completed")

            # Small delay before checking for updated trajectory
            time.sleep(0.05)

        self.get_logger().info("Execution loop exiting")

    def destroy_node(self) -> None:  # type: ignore[override]
        self._shutdown = True
        # Signal worker thread to stop
        self._new_trajectory_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = EndEffectorTrajectoryExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
