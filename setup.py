from setuptools import find_packages, setup

package_name = 'colab_dress'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hguda',
    maintainer_email='harshavardhan.guda@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'talker = colab_dress.publisher_member_function:main',
                'listener = colab_dress.subscriber_member_function:main',
                "publish_image = colab_dress.publish_camera:main",
                "subscribe_image = colab_dress.subscribe_camera:main",
                "aruco_detect = colab_dress.aruco_detector:main",
                "aruco_marker_listener = colab_dress.aruco_marker_listener:main",
                "get_3d_point_service = colab_dress.get_3d_point_service:main",
                "get_3d_point_client = colab_dress.get_3d_point_client:main",
                "pose_estimator = colab_dress.pose_estimator:main",
                "engagement_detector = colab_dress.engagement_detector:main",
                "set_end_effector_pose = colab_dress.set_end_effector_pose:main",
        ],
    },
)
