"""
sample_standing_pose.py
=======================
Run this WHILE the ROS stack is already up (controllers active).
Physically hold the Pupper in its desired standing pose, then press Enter.
Prints the exact joint angles to copy into pupper_art.py.

Usage:
  # Terminal 1 (already running):
  ros2 launch pupper_art.launch.py

  # Terminal 2:
  python3 sample_standing_pose.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

JOINTS = [
    'leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3',
    'leg_front_l_1', 'leg_front_l_2', 'leg_front_l_3',
    'leg_back_r_1',  'leg_back_r_2',  'leg_back_r_3',
    'leg_back_l_1',  'leg_back_l_2',  'leg_back_l_3',
]

class Sampler(Node):
    def __init__(self):
        super().__init__('pose_sampler')
        self.positions = None
        self.sub = self.create_subscription(
            JointState, '/joint_states', self._cb, 10)

    def _cb(self, msg):
        try:
            self.positions = np.array(
                [msg.position[msg.name.index(j)] for j in JOINTS])
        except ValueError:
            pass

def main():
    rclpy.init()
    node = Sampler()

    print("Spinning — waiting for /joint_states...")
    while node.positions is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("Got joint states. \n")

    print("=" * 60)
    print("Physically place the Pupper in its standing pose now.")
    print("Make sure all 4 feet are flat and evenly loaded.")
    print("=" * 60)
    input("\nPress ENTER to sample the current joint positions...")

    # Flush a few callbacks to get a fresh reading
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    pos = node.positions
    print("\n── Sampled joint positions ──────────────────────────────")
    for name, val in zip(JOINTS, pos):
        print(f"  {name:<22} {val:+.6f} rad  ({np.degrees(val):+.2f} deg)")

    rf = pos[0:3]
    lf = pos[3:6]
    rb = pos[6:9]
    lb = pos[9:12]

    print("\n── Drop these into pupper_art.py ────────────────────────")
    print(f"STAND_ANGLES_RF = np.array([{rf[0]:+.5f}, {rf[1]:+.5f}, {rf[2]:+.5f}])")
    print(f"STAND_ANGLES_LF = np.array([{lf[0]:+.5f}, {lf[1]:+.5f}, {lf[2]:+.5f}])")
    print(f"STAND_ANGLES_RB = np.array([{rb[0]:+.5f}, {rb[1]:+.5f}, {rb[2]:+.5f}])")
    print(f"STAND_ANGLES_LB = np.array([{lb[0]:+.5f}, {lb[1]:+.5f}, {lb[2]:+.5f}])")

    print("\n── Also sample a few times for stability check ──────────")
    readings = []
    for i in range(5):
        input(f"  Press ENTER for sample {i+1}/5...")
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
        readings.append(node.positions.copy())
        print(f"  LF: {node.positions[3:6]}  RB: {node.positions[6:9]}  LB: {node.positions[9:12]}")

    readings = np.array(readings)
    print(f"\n  Std dev across samples (should be < 0.01 if robot is stable):")
    print(f"  {readings.std(axis=0)}")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
