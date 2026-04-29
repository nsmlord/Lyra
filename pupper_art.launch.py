# pupper_art.launch.py
# Launch the Pupper V3 controllers + the art-drawing node.

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, ThisLaunchFileDir

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Robot description ──────────────────────────────────────────────────
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("pupper_v3_description"),
                    "description",
                    "pupper_v3.urdf.xacro",
                ]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    robot_controllers = PathJoinSubstitution(
        [
            ThisLaunchFileDir(),
            "lab_3.yaml",          # reuse existing YAML (no changes needed)
        ]
    )

    # ── ROS 2 control infrastructure ──────────────────────────────────────
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
        ],
    )

    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
        ],
    )

    forward_command_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "forward_command_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
        ],
    )

    # ── Art-drawing node (starts after forward_command_controller) ─────────
    pupper_art_node = Node(
        package="pupper_art",           # adjust to your actual ROS 2 package name
        executable="pupper_art",
        output="both",
    )

    # Delay forward_command_controller until joint_state_broadcaster is up
    delay_fcc_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[forward_command_controller_spawner],
        )
    )

    # Delay art node until forward_command_controller is up
    delay_art_after_fcc = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=forward_command_controller_spawner,
            on_exit=[pupper_art_node],
        )
    )

    nodes = [
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        delay_fcc_after_jsb,
        delay_art_after_fcc,
    ]

    return LaunchDescription(nodes)
