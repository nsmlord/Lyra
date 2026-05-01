# pupper_art.launch.py
# Brings up ONLY the ROS 2 control stack — no drawing node.
# Run pupper_art.py or sample_standing_pose.py manually in a separate terminal.

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

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, os.path.join(LYRA_DIR, 'lab_3.yaml')],
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
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )
    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["imu_sensor_broadcaster", "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )
    forward_command_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["forward_command_controller", "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "30"],
    )

    # forward_command_controller starts after joint_state_broadcaster is ready
    delay_fcc = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[forward_command_controller_spawner],
        )
    )

    return LaunchDescription([
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        delay_fcc,
        # ── NO pupper_art node here — run it manually ──
    ])
