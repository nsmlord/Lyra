#!/usr/bin/env python3
"""
set_gains.py
------------
Interactive gain controller for Pupper. Runs indefinitely in one terminal.

    r  →  relax   (kp=0,   kd=0  )  full limp, zero torque
    s  →  stiffen (kp=5,   kd=0.1)  light standing stiffness
    q  →  quit

Run AFTER pupper_art.launch.py is up.
"""

import sys
import threading
import termios
import tty
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

JOINTS = [
    'leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3',
    'leg_front_l_1', 'leg_front_l_2', 'leg_front_l_3',
    'leg_back_r_1',  'leg_back_r_2',  'leg_back_r_3',
    'leg_back_l_1',  'leg_back_l_2',  'leg_back_l_3',
]
N = len(JOINTS)

PRESETS = {
    'relax':   {'kp': 0.0, 'kd': 0.0},
    'stiffen': {'kp': 5.0, 'kd': 0.1},
}


def getch():
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class GainPublisher(Node):
    def __init__(self):
        super().__init__('set_gains')

        self.kp_pub = self.create_publisher(
            Float64MultiArray, '/forward_kp_controller/commands', 10)
        self.kd_pub = self.create_publisher(
            Float64MultiArray, '/forward_kd_controller/commands', 10)

        self._kp = 0.0
        self._kd = 0.0
        self._lock = threading.Lock()

        self.create_timer(0.02, self._publish)  # 50 Hz

    def set_gains(self, kp: float, kd: float):
        with self._lock:
            self._kp = kp
            self._kd = kd

    def _publish(self):
        with self._lock:
            kp, kd = self._kp, self._kd
        self.kp_pub.publish(Float64MultiArray(data=[kp] * N))
        self.kd_pub.publish(Float64MultiArray(data=[kd] * N))


def input_loop(node: GainPublisher):
    print('\nControls:  r = relax   s = stiffen   q = quit\n')
    while True:
        key = getch()
        if key == 'r':
            p = PRESETS['relax']
            node.set_gains(p['kp'], p['kd'])
            print(f'\r[RELAX]   kp={p["kp"]}, kd={p["kd"]}          ', flush=True)
        elif key == 's':
            p = PRESETS['stiffen']
            node.set_gains(p['kp'], p['kd'])
            print(f'\r[STIFFEN] kp={p["kp"]}, kd={p["kd"]}          ', flush=True)
        elif key in ('q', '\x03'):  # q or Ctrl+C
            print('\rQuitting...                                    ')
            rclpy.shutdown()
            break


def main():
    rclpy.init()
    node = GainPublisher()

    # Keypress loop runs on main thread; ROS spins on a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        input_loop(node)
    except KeyboardInterrupt:
        print('\rInterrupted.')
        rclpy.shutdown()
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
