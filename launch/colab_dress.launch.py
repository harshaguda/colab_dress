#!/usr/bin/env python3

"""Launch file for colab_dress: RealSense camera, camera transform publisher, pose estimator, and engagement detector."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Include RealSense camera launch with specified parameters
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
        }.items()
    )

    # Camera transform publisher node
    camera_transform_publisher_node = Node(
        package='colab_dress',
        executable='camera_transform_publisher',
        name='camera_transform_publisher',
        output='screen',
    )

    # Pose estimator node
    pose_estimator_node = Node(
        package='colab_dress',
        executable='simple_pose_estimator',
        name='simple_pose_estimator',
        output='screen',
    )

    dmp_node = Node(
        package='colab_dress',
        executable='dmp_node',
        name='dmp_node',
        output='screen',
    )

    # Dress node
    # dress_node = Node(
    #     package='colab_dress',
    #     executable='dress_node',
    #     name='dress_node',
    #     output='screen',
    # )

    return LaunchDescription([
        realsense_launch,
        camera_transform_publisher_node,
        pose_estimator_node,
        dmp_node,
        # dress_node,
    ])