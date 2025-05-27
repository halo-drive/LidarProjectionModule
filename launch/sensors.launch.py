#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package path
    pkg_path = get_package_share_directory('lidar_projection_lane')
    
    # Launch arguments
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Base frame for the system'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # VLP-16 Puck configuration
    vlp16_puck_config = PathJoinSubstitution([
        FindPackageShare('lidar_projection_lane'),
        'config', 'lidar_params', 'vlp16_0.yaml'
    ])
    
    # VLP-16 High-res configuration  
    vlp16_highres_config = PathJoinSubstitution([
        FindPackageShare('lidar_projection_lane'),
        'config', 'lidar_params', 'vlp16_1.yaml'
    ])
    
    # VLP-16 Puck driver node
    vlp16_puck_driver = Node(
        package='velodyne_driver',
        executable='velodyne_driver_node',
        name='velodyne_puck_driver',
        namespace='velodyne_puck',
        parameters=[vlp16_puck_config, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'device_ip': '192.168.1.201',  # Adjust IP as needed
            'port': 2368,
            'frame_id': 'velodyne_puck'
        }],
        remappings=[
            ('velodyne_packets', '/velodyne_puck/velodyne_packets')
        ]
    )
    
    # VLP-16 Puck pointcloud conversion
    vlp16_puck_convert = Node(
        package='velodyne_pointcloud',
        executable='velodyne_convert_node',
        name='velodyne_puck_convert',
        namespace='velodyne_puck',
        parameters=[vlp16_puck_config, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'calibration': PathJoinSubstitution([
                FindPackageShare('velodyne_pointcloud'),
                'params', 'VLP16db.yaml'
            ])
        }],
        remappings=[
            ('velodyne_packets', '/velodyne_puck/velodyne_packets'),
            ('velodyne_points', '/velodyne_puck/points')
        ]
    )
    
    # VLP-16 High-res driver node
    vlp16_highres_driver = Node(
        package='velodyne_driver',
        executable='velodyne_driver_node', 
        name='velodyne_highres_driver',
        namespace='velodyne_highres',
        parameters=[vlp16_highres_config, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'device_ip': '192.168.1.202',  # Adjust IP as needed
            'port': 2369,  # Different port
            'frame_id': 'velodyne_highres'
        }],
        remappings=[
            ('velodyne_packets', '/velodyne_highres/velodyne_packets')
        ]
    )
    
    # VLP-16 High-res pointcloud conversion
    vlp16_highres_convert = Node(
        package='velodyne_pointcloud',
        executable='velodyne_convert_node',
        name='velodyne_highres_convert', 
        namespace='velodyne_highres',
        parameters=[vlp16_highres_config, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'calibration': PathJoinSubstitution([
                FindPackageShare('velodyne_pointcloud'),
                'params', 'VLP16db.yaml'
            ])
        }],
        remappings=[
            ('velodyne_packets', '/velodyne_highres/velodyne_packets'),
            ('velodyne_points', '/velodyne_highres/points')
        ]
    )
    
    # Ground extraction node
    ground_extraction_node = Node(
        package='lidar_projection_lane',
        executable='lidar_ground_extraction_node',
        name='ground_extraction',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'base_frame': LaunchConfiguration('base_frame'),
            'vlp16_puck_frame': 'velodyne_puck',
            'vlp16_highres_frame': 'velodyne_highres',
            'vlp16_puck_topic': '/velodyne_puck/points',
            'vlp16_highres_topic': '/velodyne_highres/points',
            'publish_merged_cloud': True,
            'publish_ground_visualization': True,
            'processing_frequency': 10.0,
            
            # Filter parameters
            'filter.voxel_size': 0.1,
            'filter.statistical_k': 50,
            'filter.statistical_std_mul': 1.0,
            'filter.min_range': 0.9,
            'filter.max_range': 100.0,
            'filter.min_height': -3.0,
            'filter.max_height': 5.0,
            
            # Ground extraction parameters
            'extraction.distance_threshold': 0.05,
            'extraction.max_iterations': 1000,
            'extraction.probability': 0.99,
            'extraction.max_slope': 0.3,
            'extraction.use_cuda': True
        }],
        output='screen'
    )
    
    # Static transform publishers for sensor positions
    # These should be calibrated for your specific setup
    puck_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='velodyne_puck_tf',
        arguments=[
            '0.0', '0.5', '1.5',    # x, y, z translation
            '0.0', '0.0', '0.0', '1.0',  # quaternion rotation
            LaunchConfiguration('base_frame'),
            'velodyne_puck'
        ]
    )
    
    highres_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='velodyne_highres_tf',
        arguments=[
            '0.0', '-0.5', '1.5',   # x, y, z translation  
            '0.0', '0.0', '0.0', '1.0',  # quaternion rotation
            LaunchConfiguration('base_frame'),
            'velodyne_highres'
        ]
    )
    
    # RViz for visualization
    rviz_config = PathJoinSubstitution([
        FindPackageShare('lidar_projection_lane'),
        'config', 'ground_extraction.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    return LaunchDescription([
        base_frame_arg,
        use_sim_time_arg,
        
        # Sensor drivers
        vlp16_puck_driver,
        vlp16_puck_convert,
        vlp16_highres_driver,
        vlp16_highres_convert,
        
        # Transform publishers
        puck_tf,
        highres_tf,
        
        # Processing node
        ground_extraction_node,
        
        # Visualization
        rviz_node
    ])