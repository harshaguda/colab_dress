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
    # 2D to 3D point service service node
    get_3d_point_service_node = Node(
        package='colab_dress',
        executable='get_3d_point_service',
        name='get_3d_point_service',
        output='screen',
    )

    # Aruco marker publisher node
    aruco_marker_publisher_node = Node(
        package='colab_dress',
        executable='aruco_detect',
        name='aruco_detect',
        output='screen',
    )


    return LaunchDescription([
        realsense_launch,
        get_3d_point_service_node,
        aruco_marker_publisher_node,
    ])