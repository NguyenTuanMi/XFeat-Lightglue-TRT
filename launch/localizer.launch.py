from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('xfeat_lightglue_trt')

    return LaunchDescription([
        DeclareLaunchArgument('camera_topic',
            default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic',
            default_value='/camera/camera_info'),
        DeclareLaunchArgument('pose_topic',
            default_value='/auv/board_pose'),

        Node(
            package='xfeat_lightglue_trt',
            executable='localizer_main',
            name='localizer_node',
            output='screen',
            parameters=[
                os.path.join(pkg, 'config', 'localizer.param.yaml'),
                {
                    # Absolute paths resolved at launch time
                    'xfeat_engine':
                        os.path.join(pkg, 'weights', 'xfeat_1_480_640.engine'),
                    'lightglue_engine':
                        os.path.join(pkg, 'weights', 'lightglue_L6_1_480_640.engine'),
                    'reference_image':
                        os.path.join(pkg, 'reference', 'board_reference.png'),
                    'config_path': 
                        os.path.join(pkg, 'config', 'xfeat_lightglue.yaml'),
                    # Topic overrides from launch args
                    'camera_topic':      LaunchConfiguration('camera_topic'),
                    'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                    'pose_topic':        LaunchConfiguration('pose_topic'),
                }
            ]
        ),
    ])