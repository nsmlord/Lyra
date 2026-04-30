#!/usr/bin/env python3
"""
set_gains.py
------------
Interactive gain controller for Pupper. Runs indefinitely in one terminal.

    r      →  relax   (kp=0, kd=0)    go limp so you can manually pose the legs
    s      →  stiffen (kp=5, kd=0.1)  lock joints at wherever they currently are
    Enter  →  log current joint positions to stdout
    q      →  quit

Workflow: press r to go limp, physically move the legs into a pose,
press s to stiffen and hold that position, press Enter to log it.

Run AFTER pupper_art.launch.py is up.
"""

import sys
import threading
import termios
import tty
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

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

        # Latest joint state received from the broadcaster
        self._joint_positions: dict[str, float] = {}
        self._js_lock = threading.Lock()

        self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self.create_timer(0.02, self._publish)  # 50 Hz

    # ── gains ──────────────────────────────────────────────────────────────

    def set_gains(self, kp: float, kd: float):
        with self._lock:
            self._kp = kp
            self._kd = kd

    def _publish(self):
        with self._lock:
            kp, kd = self._kp, self._kd
        self.kp_pub.publish(Float64MultiArray(data=[kp] * N))
        self.kd_pub.publish(Float64MultiArray(data=[kd] * N))

    # ── joint states ───────────────────────────────────────────────────────

    def _js_callback(self, msg: JointState):
        with self._js_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = pos

    def log_joint_states(self):
        with self._js_lock:
            snapshot = dict(self._joint_positions)

        if not snapshot:
            print('\r[LOG] No joint state data yet — is joint_state_broadcaster running?')
            return

        print('\r\n--- joint positions ---')
        for joint in JOINTS:
            val = snapshot.get(joint, float('nan'))
            print(f'  {joint:<22} {val:+.4f} rad')
        print('-----------------------\n', flush=True)


def input_loop(node: GainPublisher):
    print('\nControls:  r = relax   s = stiffen (hold pose)   Enter = log joints   q = quit\n')
    while True:
        key = getch()

        if key == 'r':
            node.set_gains(**PRESETS['relax'])
            print('\r[RELAX]   kp=0.0, kd=0.0  — move legs freely          ', flush=True)

        elif key == 's':
            node.set_gains(**PRESETS['stiffen'])
            print('\r[STIFFEN] kp=5.0, kd=0.1  — holding current position  ', flush=True)

        elif key in ('\r', '\n'):   # Enter
            node.log_joint_states()

        elif key in ('q', '\x03'):  # q or Ctrl+C
            print('\rQuitting...                                            ')
            rclpy.shutdown()
            break


def main():
    rclpy.init()
    node = GainPublisher()

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
