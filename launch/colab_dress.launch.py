#!/usr/bin/env python3

"""Launch file for colab_dress: RealSense camera, camera transform publisher, pose estimator, and combined action/engagement detector."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    frame_rotation = LaunchConfiguration('frame_rotation')

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

    # Dress node
    # dress_node = Node(
    #     package='colab_dress',
    #     executable='dress_node',
    #     name='dress_node',
    #     output='screen',
    # )

    return LaunchDescription([
        DeclareLaunchArgument(
            'frame_rotation',
            default_value='180',
            description='Rotation to apply to camera frames: one of 0, 90, 180, 270',
        ),
        realsense_launch,
        camera_transform_publisher_node,
        pose_estimator_node,
        dmp_node,
        actions_recogniser_node,
    ])