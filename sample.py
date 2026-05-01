"""
sample_standing_pose.py
=======================
Requires the updated pupper_art.launch.py (which spawns kp + kd controllers).

  [r] RELAX  — sets kp=0, kd=0 → true zero-torque limp mode
               arrange the 3 standing legs by hand, then press Enter to sample

  [s] SAMPLE — reads current joint positions without changing anything

  [t] TIGHTEN — restores reasonable kp/kd so the robot holds position again
                (useful after relaxing if you want to test the standing pose
                 before running pupper_art.py)

Usage:
  Terminal 1:  ros2 launch pupper_art.launch.py
  Terminal 2:  python3 sample_standing_pose.py
"""

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

# Reasonable standing gains to restore after relaxing
# (same values the Pupper uses in normal operation — adjust if needed)
DEFAULT_KP = 5.0
DEFAULT_KD = 0.1

class PoseSampler(Node):
    def __init__(self):
        super().__init__('pose_sampler')
        self.positions = None
        self.sub = self.create_subscription(
            JointState, '/joint_states', self._cb, 10)
        self.kp_pub = self.create_publisher(
            Float64MultiArray, '/forward_kp_controller/commands', 10)
        self.kd_pub = self.create_publisher(
            Float64MultiArray, '/forward_kd_controller/commands', 10)

    def _cb(self, msg):
        try:
            self.positions = np.array(
                [msg.position[msg.name.index(j)] for j in JOINTS])
        except ValueError:
            pass

    def _publish_gains(self, kp_val, kd_val, duration_s=1.0):
        kp_msg = Float64MultiArray(data=[kp_val] * 12)
        kd_msg = Float64MultiArray(data=[kd_val] * 12)
        steps = int(duration_s / 0.02)
        for _ in range(steps):
            self.kp_pub.publish(kp_msg)
            self.kd_pub.publish(kd_msg)
            rclpy.spin_once(self, timeout_sec=0.02)

    def relax(self):
        print("Setting kp=0, kd=0 — joints going limp...")
        self._publish_gains(0.0, 0.0, duration_s=1.0)
        print("Done. Legs are free to move by hand.")

    def tighten(self):
        print(f"Restoring kp={DEFAULT_KP}, kd={DEFAULT_KD}...")
        self._publish_gains(DEFAULT_KP, DEFAULT_KD, duration_s=1.0)
        print("Done. Joints holding position again.")

    def flush(self, n=30):
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.05)

    def sample(self):
        self.flush(30)
        return self.positions.copy()


def print_results(pos):
    lf = pos[3:6]
    rb = pos[6:9]
    lb = pos[9:12]
    rf = pos[0:3]

    print("\n── Sampled joint positions ──────────────────────────────────────")
    for name, val in zip(JOINTS, pos):
        print(f"  {name:<22} {val:+.6f} rad  ({np.degrees(val):+.2f} deg)")

    print("\n── Paste these 3 lines into pupper_art.py ───────────────────────")
    print(f"    STAND_ANGLES_LF = np.array([{lf[0]:+.5f}, {lf[1]:+.5f}, {lf[2]:+.5f}])")
    print(f"    STAND_ANGLES_RB = np.array([{rb[0]:+.5f}, {rb[1]:+.5f}, {rb[2]:+.5f}])")
    print(f"    STAND_ANGLES_LB = np.array([{lb[0]:+.5f}, {lb[1]:+.5f}, {lb[2]:+.5f}])")
    print(f"\n    # RF home (reference only):")
    print(f"    # STAND_ANGLES_RF = np.array([{rf[0]:+.5f}, {rf[1]:+.5f}, {rf[2]:+.5f}])")


def main():
    rclpy.init()
    node = PoseSampler()

    print("Waiting for /joint_states...")
    while node.positions is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("Got joint states.\n")

    while True:
        print("=" * 60)
        print("  [r]  RELAX    — kp=kd=0, legs go limp, then sample")
        print("  [s]  SAMPLE   — read current pose without changing gains")
        print("  [t]  TIGHTEN  — restore gains so robot holds position")
        print("  [q]  QUIT")
        print("=" * 60)
        choice = input("\nEnter r / s / t / q: ").strip().lower()

        if choice == 'r':
            node.relax()
            print("\nArrange LF, RB, LB legs into a stable standing position.")
            print("Keep the robot supported so it doesn't fall.")
            input("Press ENTER when ready to sample...")
            pos = node.sample()
            print_results(pos)

            # Stability check
            print("\n── Stability check (3 more samples) ────────────────────")
            readings = [pos]
            for i in range(3):
                input(f"  Press ENTER for sample {i+2}/4 (hold perfectly still)...")
                readings.append(node.sample())
                p = readings[-1]
                print(f"  LF: {np.round(p[3:6],4)}  "
                      f"RB: {np.round(p[6:9],4)}  "
                      f"LB: {np.round(p[9:12],4)}")
            std = np.array(readings).std(axis=0)
            print(f"\n  Max std dev: {std.max():.4f} rad "
                  f"({'✓ stable' if std.max() < 0.01 else '⚠ high — robot shifted, try again'})")

        elif choice == 's':
            input("Press ENTER to sample current joint positions...")
            pos = node.sample()
            print_results(pos)

        elif choice == 't':
            node.tighten()

        elif choice == 'q':
            break
        else:
            print("Unknown choice.\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
