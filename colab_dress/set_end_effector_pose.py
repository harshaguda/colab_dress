#!/usr/bin/env python3
"""MoveIt example that drives the Kinova Gen3 end effector to a pose using the MoveIt action interface."""

from __future__ import annotations

import math
import sys
from typing import Iterable, List, Sequence

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Sequence[float]:
    """Return (x, y, z, w) quaternion from roll/pitch/yaw."""
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


class MoveItPoseSetter(Node):
    """Node that plans (and optionally executes) a Cartesian end-effector pose using MoveIt action."""

    def __init__(self) -> None:
        super().__init__("kinova_moveit_set_pose")

        self.declare_parameter("group_name", "manipulator")
        self.declare_parameter("reference_frame", "base_link")
        self.declare_parameter("end_effector_link", "end_effector_link")
        self.declare_parameter("target_position", "[0.4, 0.0, 0.4]")
        self.declare_parameter("target_orientation_rpy", "[3.14, 0.0, 0.0]")
        self.declare_parameter("target_orientation_quaternion", "")
        self.declare_parameter("plan_only", False)
        self.declare_parameter("position_tolerance", 0.01)
        self.declare_parameter("orientation_tolerance", 0.02)
        self.declare_parameter("max_velocity_scaling", 0.2)
        self.declare_parameter("max_acceleration_scaling", 0.2)
        self.declare_parameter("planning_time", 5.0)

        group_name = self.get_parameter("group_name").get_parameter_value().string_value
        reference_frame = self.get_parameter("reference_frame").get_parameter_value().string_value
        end_effector_link = self.get_parameter("end_effector_link").get_parameter_value().string_value

        self._target_position = self._get_float_list("target_position", expected_length=3)
        quaternion_override = self._get_float_list(
            "target_orientation_quaternion", expected_length=4, allow_empty=True
        )
        if quaternion_override:
            self._target_orientation = quaternion_override
        else:
            rpy = self._get_float_list("target_orientation_rpy", expected_length=3)
            self._target_orientation = list(_quaternion_from_euler(*rpy))

        self._plan_only = bool(self.get_parameter("plan_only").get_parameter_value().bool_value)
        self._position_tolerance = self._get_float_param("position_tolerance")
        self._orientation_tolerance = self._get_float_param("orientation_tolerance")
        self._velocity_scaling = max(0.0, min(1.0, self._get_float_param("max_velocity_scaling")))
        self._accel_scaling = max(0.0, min(1.0, self._get_float_param("max_acceleration_scaling")))
        self._planning_time = self._get_float_param("planning_time")

        self.get_logger().info(
            f"Using MoveIt group '{group_name}' with reference frame '{reference_frame}' and end effector '{end_effector_link}'"
        )

        # Create the MoveGroup action client
        self._action_client = ActionClient(self, MoveGroup, "/move_action")

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server not available")

        self._group_name = group_name
        self._reference_frame = reference_frame
        self._end_effector_link = end_effector_link

    def _get_float_param(self, name: str) -> float:
        value = self.get_parameter(name).value
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str) and value:
            return float(value)
        raise ValueError(f"Parameter '{name}' must be a numeric value")

    def _get_float_list(
        self, name: str, *, expected_length: int, allow_empty: bool = False
    ) -> List[float]:
        raw = self.get_parameter(name).value
        values: Iterable[float]
        if isinstance(raw, (list, tuple)):
            values = [float(v) for v in raw]
        elif isinstance(raw, str):
            cleaned = raw.replace("[", "").replace("]", "").replace(";", ",")
            if not cleaned.strip():
                values = []
            else:
                values = [float(part) for part in cleaned.split(",") if part]
        else:
            if allow_empty and raw in (None, ""):
                values = []
            else:
                raise ValueError(f"Parameter '{name}' must be a list of numbers")

        result = list(values)
        if not result and allow_empty:
            return []
        if len(result) != expected_length:
            raise ValueError(
                f"Parameter '{name}' must contain exactly {expected_length} values, got {len(result)}"
            )
        return result

    def execute(self) -> bool:
        # Build the MotionPlanRequest
        request = MotionPlanRequest()
        request.workspace_parameters.header = Header()
        request.workspace_parameters.header.frame_id = self._reference_frame
        # Set a default workspace; adjust as needed
        request.workspace_parameters.min_corner.x = -1.0
        request.workspace_parameters.min_corner.y = -1.0
        request.workspace_parameters.min_corner.z = -1.0
        request.workspace_parameters.max_corner.x = 1.0
        request.workspace_parameters.max_corner.y = 1.0
        request.workspace_parameters.max_corner.z = 1.0

        request.start_state.is_diff = True  # Use current state as start

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
        # pos_constraint.constraint_region.primitive_poses[0].frame_id = self._reference_frame
        pos_constraint.constraint_region.primitive_poses[0].position.x = self._target_position[0]
        pos_constraint.constraint_region.primitive_poses[0].position.y = self._target_position[1]
        pos_constraint.constraint_region.primitive_poses[0].position.z = self._target_position[2]
        pos_constraint.constraint_region.primitive_poses[0].orientation.x = self._target_orientation[0]
        pos_constraint.constraint_region.primitive_poses[0].orientation.y = self._target_orientation[1]
        pos_constraint.constraint_region.primitive_poses[0].orientation.z = self._target_orientation[2]
        pos_constraint.constraint_region.primitive_poses[0].orientation.w = self._target_orientation[3]
        constraint.position_constraints = [pos_constraint]

        # Orientation constraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self._reference_frame
        ori_constraint.link_name = self._end_effector_link
        ori_constraint.orientation.x = self._target_orientation[0]
        ori_constraint.orientation.y = self._target_orientation[1]
        ori_constraint.orientation.z = self._target_orientation[2]
        ori_constraint.orientation.w = self._target_orientation[3]
        ori_constraint.absolute_x_axis_tolerance = self._orientation_tolerance
        ori_constraint.absolute_y_axis_tolerance = self._orientation_tolerance
        ori_constraint.absolute_z_axis_tolerance = self._orientation_tolerance
        constraint.orientation_constraints = [ori_constraint]

        request.planner_id = ""  # Use default planner
        request.group_name = self._group_name
        request.num_planning_attempts = 10
        request.allowed_planning_time = self._planning_time
        request.max_velocity_scaling_factor = self._velocity_scaling
        request.max_acceleration_scaling_factor = self._accel_scaling

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options.plan_only = self._plan_only
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.planning_scene_diff.is_diff = True

        self.get_logger().info(
            "Planning to pose (%.3f, %.3f, %.3f) with quaternion (%.3f, %.3f, %.3f, %.3f)"
            % (
                self._target_position[0],
                self._target_position[1],
                self._target_position[2],
                self._target_orientation[0],
                self._target_orientation[1],
                self._target_orientation[2],
                self._target_orientation[3],
            )
        )

        send_goal_future = self._action_client.send_goal_async(goal_msg)

        # Wait for the result
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.get_logger().error("MoveGroup action did not return a goal handle")
            return False
        if not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result_handle = result_future.result()
        if result_handle is None:
            self.get_logger().error("MoveGroup action returned no result")
            return False
        result = result_handle.result

        if result.error_code.val != 1:  # MoveItErrorCodes::SUCCESS
            self.get_logger().error(f"Planning failed with error code {result.error_code.val}")
            return False

        if self._plan_only:
            self.get_logger().info("Planning succeeded")
        else:
            self.get_logger().info("Motion execution succeeded")
        return True

def main(argv: List[str] | None = None) -> None:
    argv = sys.argv if argv is None else argv

    rclpy.init(args=argv)

    node = MoveItPoseSetter()
    try:
        success = node.execute()
        if not success:
            node.get_logger().warn("End-effector did not reach the requested pose")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
