# pupper_art_sim.launch.py
# Simulation-only launch: fake plant + art state machine + RViz.
# No real robot, no controller_manager, no hardware interface needed.

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Robot description (same URDF, used only for RViz visualisation) ───
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

    # ── robot_state_publisher (converts joint_states → TF for RViz) ───────
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # ── Simulation node (fake plant + full PupperArt state machine) ────────
    # This publishes /joint_states and /marker so RViz sees everything.
    # Replace 'pupper_art' with your actual ROS 2 package name if needed.
    pupper_art_sim_node = Node(
        package="pupper_art",
        executable="pupper_art_sim",    # entry point defined in setup.py
        output="both",
    )

    # ── RViz with your existing lab_2 config ───────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", PathJoinSubstitution(
            [FindPackageShare("pupper_art"), "config", "lab_2.rviz"]
        )],
        output="screen",
    )

    return LaunchDescription([
        robot_state_pub_node,
        pupper_art_sim_node,
        rviz_node,
    ])
