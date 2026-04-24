#!/usr/bin/env python3

"""Launch file for colab_dress: RealSense camera, camera transform publisher, pose estimator, and engagement detector."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    baseline = Node(
        package='colab_dress',
        executable='dress_no_adapt_node',
        name='dress_no_adapt_node',
        output='screen',
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


    return LaunchDescription([
        baseline,
        publish_engage,
    ])