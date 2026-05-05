                          
                                                                
                                                                     

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

                                                                            
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

                                                                            
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

                                                                             
                                                                       
                                                                         
    pupper_art_sim_node = Node(
        package="pupper_art",
        executable="pupper_art_sim",                                     
        output="both",
    )

                                                                             
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
