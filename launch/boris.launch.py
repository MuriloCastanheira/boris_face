from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'boris_face'

    urdf_file_path = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        'boris_face.urdf'
    )

    rviz_config_file = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'boris.rviz'
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': open(urdf_file_path).read()}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
    ])