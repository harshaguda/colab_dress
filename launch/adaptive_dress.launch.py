#!/usr/bin/env python3

"""Launch file for colab_dress: RealSense camera, camera transform publisher, pose estimator, and engagement detector."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    frame_rotation = LaunchConfiguration('frame_rotation')

    actions_recogniser_node = Node(
        package='colab_dress',
        executable='actions_recogniser',
        name='actions_recogniser',
        output='screen',
        parameters=[
            {
                'camera': '/dev/v4l/by-id/usb-Creative_Technology_Ltd._Creative_Senz3D_VF0780_K8VF0780404001001T-video-index0',
                'frame_rotation': frame_rotation,
            }
        ],
    )

    # adaptive_dress = Node(
    #     package='colab_dress',
    #     executable='dress_node',
    #     name='dress_adapt_node',
    #     output='screen',
    # )


    return LaunchDescription([
        DeclareLaunchArgument(
            'frame_rotation',
            default_value='0',
            description='Rotation to apply to camera frames: one of 0, 90, 180, 270',
        ),
        actions_recogniser_node,
        # Wait 15 seconds before starting the dress node to allow pose estimator to initialize and start publishing poses
        ExecuteProcess(
            cmd=[
                'sleep', '15', '&&',
                'ros2', 'run', 'colab_dress', 'dress_node',
            ],
            shell=True,
            output='log',
        ),
        # adaptive_dress,
    ])