#!/usr/bin/env python3
"""Launch MoveIt for the Kinova Gen3 and drive the end effector to a pose."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    delay_value = float(LaunchConfiguration("delay_before_command").perform(context))
    plan_only_raw = LaunchConfiguration("plan_only").perform(context).lower()
    plan_only_value = plan_only_raw in ("true", "1", "yes", "on")

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("kinova_gen3_7dof_robotiq_2f_85_moveit_config"),
                    "launch",
                    "move_group.launch.py",
                ]
            )
        )
    )

    parameters = {
        "target_position": LaunchConfiguration("target_position").perform(context),
        "target_orientation_rpy": LaunchConfiguration("target_orientation_rpy").perform(context),
        "plan_only": plan_only_value,
        "group_name": LaunchConfiguration("group_name").perform(context),
    }

    command_node = Node(
        package="kinova_gen3_7dof_robotiq_2f_85_moveit_config",
        executable="set_end_effector_pose.py",
        name="moveit_set_pose",
        output="screen",
        parameters=[parameters],
    )

    delayed_command = TimerAction(period=delay_value, actions=[command_node])

    return [move_group_launch, delayed_command]


def generate_launch_description() -> LaunchDescription:
    target_position_arg = DeclareLaunchArgument(
        "target_position",
        default_value="[0.4, 0.0, 0.4]",
        description="Goal position for the end effector in the base frame (meters).",
    )
    target_orientation_rpy_arg = DeclareLaunchArgument(
        "target_orientation_rpy",
        default_value="[3.1415, 0.0, 0.0]",
        description="Roll, pitch, yaw target for the end effector (radians).",
    )
    plan_only_arg = DeclareLaunchArgument(
        "plan_only",
        default_value="false",
        description="Set to true to only plan without execution.",
    )
    delay_seconds_arg = DeclareLaunchArgument(
        "delay_before_command",
        default_value="5.0",
        description="Seconds to wait before sending the MoveIt goal.",
    )
    group_name_arg = DeclareLaunchArgument(
        "group_name",
        default_value="manipulator",
        description="MoveIt planning group to command.",
    )

    return LaunchDescription(
        [
            target_position_arg,
            target_orientation_rpy_arg,
            plan_only_arg,
            delay_seconds_arg,
            group_name_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
