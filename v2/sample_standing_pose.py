"""
sample_standing_pose.py
=======================
Two modes — pick when prompted:

  [r] RELAX  — publishes zero commands so servos go limp.
               Physically arrange the 3 standing legs, then press Enter to sample.

  [s] SAMPLE — just samples current joint positions without relaxing first.
               Use this if the robot is already in the right pose.

Requires the ROS stack to be running:
  Terminal 1:  ros2 launch pupper_art.launch.py
  Terminal 2:  python3 sample_standing_pose.py

Output: exact numpy lines to paste into pupper_art.py
"""

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

JOINTS = [
    'leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3',
    'leg_front_l_1', 'leg_front_l_2', 'leg_front_l_3',
    'leg_back_r_1',  'leg_back_r_2',  'leg_back_r_3',
    'leg_back_l_1',  'leg_back_l_2',  'leg_back_l_3',
]

class PoseSampler(Node):
    def __init__(self):
        super().__init__('pose_sampler')
        self.positions = None
        self.sub = self.create_subscription(
            JointState, '/joint_states', self._cb, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/forward_command_controller/commands', 10)

    def _cb(self, msg):
        try:
            self.positions = np.array(
                [msg.position[msg.name.index(j)] for j in JOINTS])
        except ValueError:
            pass

    def relax(self):
        """Send zeros — servos go limp so you can move the legs by hand."""
        msg = Float64MultiArray()
        msg.data = [0.0] * 12
        # Publish several times to make sure it lands
        for _ in range(50):
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info('Relaxed — all joints limp. Arrange the legs now.')

    def flush(self, n=20):
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.05)

    def sample(self):
        self.flush(30)
        return self.positions.copy()


def print_results(pos):
    rf = pos[0:3];  lf = pos[3:6]
    rb = pos[6:9];  lb = pos[9:12]

    print("\n── Sampled joint positions ──────────────────────────────")
    for name, val in zip(JOINTS, pos):
        print(f"  {name:<22} {val:+.6f} rad  ({np.degrees(val):+.2f} deg)")

    print("\n── Paste these into pupper_art.py (replace the STAND_ANGLES lines) ──")
    print(f"    STAND_ANGLES_LF = np.array([{lf[0]:+.5f}, {lf[1]:+.5f}, {lf[2]:+.5f}])")
    print(f"    STAND_ANGLES_RB = np.array([{rb[0]:+.5f}, {rb[1]:+.5f}, {rb[2]:+.5f}])")
    print(f"    STAND_ANGLES_LB = np.array([{lb[0]:+.5f}, {lb[1]:+.5f}, {lb[2]:+.5f}])")
    print(f"\n    # RF at home (for reference only — RF moves during drawing):")
    print(f"    # STAND_ANGLES_RF = np.array([{rf[0]:+.5f}, {rf[1]:+.5f}, {rf[2]:+.5f}])")


def main():
    rclpy.init()
    node = PoseSampler()

    print("Waiting for /joint_states...")
    while node.positions is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("Got joint states.\n")

    print("=" * 60)
    print("  [r]  RELAX legs first, then sample")
    print("       (use this when robot is stiff after drawing)")
    print()
    print("  [s]  SAMPLE current pose directly")
    print("       (use this when robot is already in standing pose)")
    print("=" * 60)
    choice = input("\nEnter r or s: ").strip().lower()

    if choice == 'r':
        print("\nRelaxing all joints — legs will go limp...")
        node.relax()
        print("\nArrange LF, RB, LB legs into a stable standing position.")
        print("Keep the robot supported so it doesn't fall.")
        input("Press ENTER when legs are in position to sample...")
        pos = node.sample()
        print_results(pos)

        # Optional: take 3 more samples to check stability
        print("\n── Stability check (3 more samples) ────────────────────")
        readings = [pos]
        for i in range(3):
            input(f"  Press ENTER for sample {i+2}/4 (hold still)...")
            readings.append(node.sample())
            print(f"  LF: {readings[-1][3:6]}  RB: {readings[-1][6:9]}  LB: {readings[-1][9:12]}")
        std = np.array(readings).std(axis=0)
        print(f"\n  Std dev (want < 0.01 rad for reliable standing angles):")
        print(f"  LF: {std[3:6]}  RB: {std[6:9]}  LB: {std[9:12]}")
        if std.max() > 0.01:
            print("  ⚠  High variance — robot may have shifted. Try again.")
        else:
            print("  ✓  Stable — safe to use these values.")

    elif choice == 's':
        input("\nPress ENTER to sample current joint positions...")
        pos = node.sample()
        print_results(pos)

    else:
        print("Unknown choice, exiting.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
