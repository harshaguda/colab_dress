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
        executable='pose_estimator',
        name='pose_estimator',
        output='screen',
    )

    # 2D to 3D point service service node
    get_3d_point_service_node = Node(
        package='colab_dress',
        executable='get_3d_point_service',
        name='get_3d_point_service',
        output='screen',
    )

    # Engagement detector node
    engagement_detector_node = Node(
        package='colab_dress',
        executable='engagement_detector',
        name='engagement_detector',
        output='screen',
    )

    return LaunchDescription([
        realsense_launch,
        camera_transform_publisher_node,
        pose_estimator_node,
        get_3d_point_service_node,
        engagement_detector_node,
    ])