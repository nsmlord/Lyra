                      
                                                                    
                                                                         
                                                                               

import os
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

LYRA_DIR = '/home/pi/Lyra'                     

def generate_launch_description():
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([
            FindPackageShare("pupper_v3_description"),
            "description", "pupper_v3.urdf.xacro"
        ]),
    ])
    robot_description = {"robot_description": robot_description_content}

                                                                        
                                                                        
    lab3_yaml = os.path.join(LYRA_DIR, 'lab_3.yaml')
    lab2_yaml = os.path.join(LYRA_DIR, 'lab_2.yaml')

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
                                                          
        parameters=[robot_description, lab3_yaml, lab2_yaml],
        output="both",
    )
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

                                                                             
    joint_state_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )
    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )
                                                             
    forward_command_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["forward_command_controller",
                   "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )
                                                                      
    forward_kp_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["forward_kp_controller",
                   "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )
    forward_kd_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["forward_kd_controller",
                   "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )

                                                                             
                                                                       
    delay_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[
                forward_command_controller_spawner,
                forward_kp_controller_spawner,
                forward_kd_controller_spawner,
            ],
        )
    )

    return LaunchDescription([
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        delay_after_jsb,
    ])
