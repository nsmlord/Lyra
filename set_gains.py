#!/usr/bin/env python3
"""
set_gains.py
------------
Interactive gain controller for Pupper. Runs indefinitely in one terminal.

    r      →  relax   (kp=0, kd=0)    go limp so you can manually pose the legs
    s      →  stiffen                  read current joint positions, latch them
                                       as the position command, THEN raise gains
                                       so the robot holds exactly where you left it
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

KP_STIFF = 5.0
KD_STIFF = 0.1


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
        self.pos_pub = self.create_publisher(
            Float64MultiArray, '/forward_command_controller/commands', 10)

        self._kp = 0.0
        self._kd = 0.0
        self._gain_lock = threading.Lock()

        self._joint_positions: dict[str, float] = {}
        self._js_lock = threading.Lock()

        self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self.create_timer(0.02, self._publish)  # 50 Hz

    # ── joint state subscription ───────────────────────────────────────────

    def _js_callback(self, msg: JointState):
        with self._js_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = pos

    def _current_positions(self) -> list[float] | None:
        with self._js_lock:
            snapshot = dict(self._joint_positions)
        if not all(j in snapshot for j in JOINTS):
            return None
        return [snapshot[j] for j in JOINTS]

    # ── gain control ───────────────────────────────────────────────────────

    def relax(self):
        with self._gain_lock:
            self._kp = 0.0
            self._kd = 0.0

    def stiffen(self) -> bool:
        """
        Latch current measured positions as the position setpoint,
        then raise gains. Returns False if joint states aren't available yet.
        """
        positions = self._current_positions()
        if positions is None:
            return False

        # 1. Send the position command FIRST while gains are still zero
        self.pos_pub.publish(Float64MultiArray(data=positions))

        # 2. Small sleep so the position command lands before gains go up
        import time; time.sleep(0.05)

        # 3. Now raise gains — controller targets the pose we just sent
        with self._gain_lock:
            self._kp = KP_STIFF
            self._kd = KD_STIFF

        return True

    def _publish(self):
        with self._gain_lock:
            kp, kd = self._kp, self._kd
        self.kp_pub.publish(Float64MultiArray(data=[kp] * N))
        self.kd_pub.publish(Float64MultiArray(data=[kd] * N))

    # ── logging ────────────────────────────────────────────────────────────

    def log_joint_states(self):
        positions = self._current_positions()
        if positions is None:
            print('\r[LOG] No joint state data yet — is joint_state_broadcaster running?')
            return
        print('\r\n--- joint positions ---')
        for joint, val in zip(JOINTS, positions):
            print(f'  {joint:<22} {val:+.4f} rad')
        print('-----------------------\n', flush=True)


def input_loop(node: GainPublisher):
    print('\nControls:  r = relax   s = stiffen (hold pose)   Enter = log joints   q = quit\n')
    while True:
        key = getch()

        if key == 'r':
            node.relax()
            print('\r[RELAX]   kp=0.0, kd=0.0  — move legs freely               ', flush=True)

        elif key == 's':
            ok = node.stiffen()
            if ok:
                print(f'\r[STIFFEN] kp={KP_STIFF}, kd={KD_STIFF}  — holding current position  ', flush=True)
            else:
                print('\r[STIFFEN] failed — no joint state data yet, try again     ', flush=True)

        elif key in ('\r', '\n'):
            node.log_joint_states()

        elif key in ('q', '\x03'):
            print('\rQuitting...                                                  ')
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
