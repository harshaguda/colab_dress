#!/usr/bin/env python3

"""Launch file for colab_dress: RealSense camera, camera transform publisher, pose estimator, and engagement detector."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

    # ros2 topic pub --once /cartesian_trajectory geometry_msgs/msg/PoseArray "{header: {frame_id: 'fr3_link0'}, poses: [{position: {x: 0.2, y: -0.3, z: 0.3}, orientation: {x: -0.7071, y: 0.7071, z: 0.0, w: 0.0}}]}"

    publish_trajectory = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '--once',
            '/cartesian_trajectory',
            'geometry_msgs/msg/PoseArray',
            "{header: {frame_id: 'fr3_link0'}, poses: [{position: {x: 0.176, y: -0.360, z: 0.3}, orientation: {x: -0.7071, y: 0.7071, z: 0.0, w: 0.0}}]}",
        ],      
        output='log',
    )

    publish_engage = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '-r',
            '10',
            '/engagement/status',
            'std_msgs/msg/String',
            '{data: paying_attention}',
        ],
        output='log',
    )

    publish_non_adaptive_flag = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '-r',
            '10',
            '/non_adaptive_flag',
            'std_msgs/msg/Bool',
            '{data: true}',
        ],
        output='log',
    )


    return LaunchDescription([
        publish_trajectory,
        publish_engage,
        publish_non_adaptive_flag,
    ])