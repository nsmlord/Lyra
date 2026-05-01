"""
pupper_art.py — Pupper V3 floor art drawing node
=================================================
Uses Inverse Kinematics + Cascaded PID (position → angular-velocity → torque)
to move the RIGHT FRONT leg (pen tip) along pre-planned shapes on the floor.
Left front and both rear legs remain in a fixed standing stance.

Architecture
------------
  desired_ee_position
       │
       ▼
  [Position PID]  → desired joint angular velocity
       │
       ▼
  [Angular-velocity PID]  → desired torque increment
       │
       ▼
  [Torque PID / integrator]  → final position command
       │
       ▼
  /forward_command_controller/commands

Shapes supported: circle, star (5-pointed)
The node draws the circle first, pauses, then draws the star.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import scipy.optimize

np.set_printoptions(precision=4, suppress=True)

# ──────────────────────────────────────────────
# Homogeneous transform helpers
# ──────────────────────────────────────────────

def rotation_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def rotation_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def translation(x, y, z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]], dtype=float)


# ──────────────────────────────────────────────
# Shape generators
# ──────────────────────────────────────────────

def generate_circle(center_x, center_y, z_floor, radius=0.04, n_points=120):
    """Generate (x, y, z) waypoints for a circle in the floor plane."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = []
    for a in angles:
        pts.append(np.array([
            center_x + radius * np.cos(a),
            center_y + radius * np.sin(a),
            z_floor
        ]))
    pts.append(pts[0])          # close the shape
    return pts


def generate_star(center_x, center_y, z_floor, r_outer=0.045, r_inner=0.018, n_points=5):
    """Generate (x, y, z) waypoints for a 5-pointed star in the floor plane."""
    pts = []
    for i in range(2 * n_points + 1):
        r = r_outer if (i % 2 == 0) else r_inner
        angle = np.pi / 2 + i * np.pi / n_points   # start at top
        pts.append(np.array([
            center_x + r * np.cos(angle),
            center_y + r * np.sin(angle),
            z_floor
        ]))
    return pts


# ──────────────────────────────────────────────
# Cascaded PID controller (per-joint, 3 joints)
# ──────────────────────────────────────────────

class CascadedPID:
    """
    Three-stage cascaded PID for a single joint:
      Stage 1: position error       → desired angular velocity
      Stage 2: angular-velocity error → desired torque increment
      Stage 3: torque error          → final torque / position command

    In practice the robot accepts POSITION commands, so the output of stage 3
    is added as a small delta to the current IK-computed joint angle.
    """

    def __init__(self,
                 kp_pos=1,  ki_pos=0.001, kd_pos=0.001,
                 kp_vel=1,  ki_vel=0.001, kd_vel=0.001,
                 kp_trq=0.0001,  ki_trq=0.00001, kd_trq=0.000001,
                 dt=0.005, max_vel=2.0, max_trq=1.0, max_delta=0.05):
        self.dt = dt
        self.kp_pos, self.ki_pos, self.kd_pos = kp_pos, ki_pos, kd_pos
        self.kp_vel, self.ki_vel, self.kd_vel = kp_vel, ki_vel, kd_vel
        self.kp_trq, self.ki_trq, self.kd_trq = kp_trq, ki_trq, kd_trq
        self.max_vel, self.max_trq, self.max_delta = max_vel, max_trq, max_delta
        self.reset()

    def reset(self):
        self._int_pos = 0.0
        self._int_vel = 0.0
        self._int_trq = 0.0
        self._prev_pos_err = 0.0
        self._prev_vel_err = 0.0
        self._prev_trq_err = 0.0

    def update(self, desired_angle: float, current_angle: float, current_velocity: float):
        """
        Returns a delta that should be ADDED to the IK-computed joint angle.
        """
        dt = self.dt

        # ── Stage 1: position PID → desired angular velocity ──────────────
        pos_err = desired_angle - current_angle
        self._int_pos += pos_err * dt
        d_pos_err = (pos_err - self._prev_pos_err) / dt
        self._prev_pos_err = pos_err

        desired_vel = (self.kp_pos * pos_err
                       + self.ki_pos * self._int_pos
                       + self.kd_pos * d_pos_err)
        desired_vel = np.clip(desired_vel, -self.max_vel, self.max_vel)

        # ── Stage 2: velocity PID → desired torque ─────────────────────────
        vel_err = desired_vel - current_velocity
        self._int_vel += vel_err * dt
        d_vel_err = (vel_err - self._prev_vel_err) / dt
        self._prev_vel_err = vel_err

        desired_trq = (self.kp_vel * vel_err
                       + self.ki_vel * self._int_vel
                       + self.kd_vel * d_vel_err)
        desired_trq = np.clip(desired_trq, -self.max_trq, self.max_trq)

        # ── Stage 3: torque PID → position delta ──────────────────────────
        # We treat "torque error" as (desired_trq − 0) since we want to achieve
        # the desired torque from rest (robot is position-controlled).
        trq_err = desired_trq
        self._int_trq += trq_err * dt
        d_trq_err = (trq_err - self._prev_trq_err) / dt
        self._prev_trq_err = trq_err

        delta = (self.kp_trq * trq_err
                 + self.ki_trq * self._int_trq
                 + self.kd_trq * d_trq_err)
        delta = np.clip(delta, -self.max_delta, self.max_delta)

        return delta


# ──────────────────────────────────────────────
# Main ROS 2 node
# ──────────────────────────────────────────────

class PupperArt(Node):

    # ── Standing angles for stationary legs (LF, RB, LB) ─────────────────
    # These keep the body level while the RF leg draws.
    # Obtained by running IK for each leg at its nominal stand position.
    STAND_ANGLES_LF = np.array([ 0.00,  0.65, -1.30])
    STAND_ANGLES_RB = np.array([ 0.00,  0.65, -1.30])
    STAND_ANGLES_LB = np.array([ 0.00,  0.65, -1.30])

    # RF pen tip height while drawing (metres, in body frame, negative = down)
    PEN_Z = -0.14

    # RF "pen-up" height for transitions
    PEN_UP_Z = -0.10

    # RF leg centre position (body frame x-y)
    RF_CENTER_X =  0.06
    RF_CENTER_Y = -0.09

    # Drawing parameters
    CIRCLE_RADIUS   = 0.035   # m
    STAR_R_OUTER    = 0.040   # m
    STAR_R_INNER    = 0.016   # m
    N_CIRCLE_PTS    = 150
    DRAW_FREQ       = 50.0    # Hz  (waypoint advance rate)
    CTRL_FREQ       = 200.0   # Hz  (PD / publish rate)

    def __init__(self):
        super().__init__('pupper_art')

        # ── ROS interfaces ─────────────────────────────────────────────────
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self._joint_cb, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/forward_command_controller/commands', 10)

        # ── State ──────────────────────────────────────────────────────────
        self.joint_positions  = None
        self.joint_velocities = None

        # 12-element command array: [RF(3), LF(3), RB(3), LB(3)]
        self.cmd = np.zeros(12)

        # IK-computed target for RF leg
        self.target_rf = np.array(self.STAND_ANGLES_LF)  # init to stand

        # Cascaded PIDs — one per RF joint
        dt_pid = 1.0 / self.CTRL_FREQ
        self.pids = [CascadedPID(dt=dt_pid) for _ in range(3)]

        # ── Drawing sequence ───────────────────────────────────────────────
        cx, cy = self.RF_CENTER_X, self.RF_CENTER_Y
        self.shapes = [
            generate_circle(cx, cy, self.PEN_Z,
                            radius=self.CIRCLE_RADIUS,
                            n_points=self.N_CIRCLE_PTS),
            generate_star(cx, cy, self.PEN_Z,
                          r_outer=self.STAR_R_OUTER,
                          r_inner=self.STAR_R_INNER),
        ]
        self.shape_names = ['circle', 'star']

        self.current_shape_idx = 0
        self.current_wp_idx    = 0
        self.phase             = 'stand'   # stand → pen_down → draw → pen_up → done
        self.phase_counter     = 0
        self.STAND_TICKS       = int(2.0 * self.CTRL_FREQ)   # 2 s stand-still
        self.PEN_DOWN_TICKS    = int(1.0 * self.CTRL_FREQ)   # 1 s lower pen

        self.draw_wp_counter   = 0         # counts ctrl ticks per waypoint hold

        # ── Timers ─────────────────────────────────────────────────────────
        self.ctrl_timer = self.create_timer(1.0 / self.CTRL_FREQ, self._ctrl_cb)

        self.get_logger().info('PupperArt node started — waiting for joint states …')

    # ──────────────────────────────────────────────────────────────────────
    # Forward kinematics (RF leg only)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rf_fk(theta):
        T01 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T12 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T23 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T3e = translation(0.06231, -0.06216, 0.01800)
        return (T01 @ T12 @ T23 @ T3e)[:3, 3]

    # ──────────────────────────────────────────────────────────────────────
    # Inverse kinematics (RF leg, numerical)
    # ──────────────────────────────────────────────────────────────────────

    def _rf_ik(self, target_ee: np.ndarray, initial_guess=None) -> np.ndarray:
        if initial_guess is None:
            initial_guess = [0.0, 0.65, -1.30]

        def cost(theta):
            return self._rf_fk(theta) - target_ee

        result = scipy.optimize.least_squares(
            cost,
            x0=initial_guess,
            method='lm',
            ftol=1e-7, xtol=1e-7, gtol=1e-7,
            max_nfev=200
        )
        return result.x

    # ──────────────────────────────────────────────────────────────────────
    # Joint state callback
    # ──────────────────────────────────────────────────────────────────────

    JOINT_ORDER = [
        'leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3',
        'leg_front_l_1', 'leg_front_l_2', 'leg_front_l_3',
        'leg_back_r_1',  'leg_back_r_2',  'leg_back_r_3',
        'leg_back_l_1',  'leg_back_l_2',  'leg_back_l_3',
    ]

    def _joint_cb(self, msg):
        try:
            idxs = [msg.name.index(j) for j in self.JOINT_ORDER]
            self.joint_positions  = np.array([msg.position[i] for i in idxs])
            self.joint_velocities = np.array([msg.velocity[i] for i in idxs])
        except ValueError as e:
            self.get_logger().warn(f'Joint name not found: {e}')

    # ──────────────────────────────────────────────────────────────────────
    # State-machine helpers
    # ──────────────────────────────────────────────────────────────────────

    def _current_waypoint(self):
        shape = self.shapes[self.current_shape_idx]
        return shape[self.current_wp_idx]

    def _advance_waypoint(self):
        shape = self.shapes[self.current_shape_idx]
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(shape):
            # Finished this shape
            self.get_logger().info(
                f'Finished drawing {self.shape_names[self.current_shape_idx]}!')
            self.current_shape_idx += 1
            self.current_wp_idx = 0
            if self.current_shape_idx >= len(self.shapes):
                self.phase = 'done'
            else:
                self.phase = 'pen_up'
                self.phase_counter = 0
                self.get_logger().info(
                    f'Starting {self.shape_names[self.current_shape_idx]}…')

    # ──────────────────────────────────────────────────────────────────────
    # Control callback (200 Hz)
    # ──────────────────────────────────────────────────────────────────────

    def _ctrl_cb(self):
        if self.joint_positions is None:
            return

        pos = self.joint_positions
        vel = self.joint_velocities

        rf_pos = pos[0:3]
        rf_vel = vel[0:3]

        # ── State machine ──────────────────────────────────────────────────
        CTRL_PER_WP = max(1, int(self.CTRL_FREQ / self.DRAW_FREQ))

        if self.phase == 'stand':
            # Hold standing pose for STAND_TICKS
            target_rf_ee = np.array([self.RF_CENTER_X,
                                     self.RF_CENTER_Y,
                                     self.PEN_UP_Z])
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.STAND_TICKS:
                self.phase = 'pen_down'
                self.phase_counter = 0
                self.get_logger().info('Standing stable — lowering pen…')

        elif self.phase == 'pen_down':
            # Smoothly lower pen to drawing height
            alpha = min(1.0, self.phase_counter / self.PEN_DOWN_TICKS)
            target_z = self.PEN_UP_Z + alpha * (self.PEN_Z - self.PEN_UP_Z)
            wp = self._current_waypoint()
            target_rf_ee = np.array([wp[0], wp[1], target_z])
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.PEN_DOWN_TICKS:
                self.phase = 'draw'
                self.draw_wp_counter = 0
                self.get_logger().info(
                    f'Drawing {self.shape_names[self.current_shape_idx]}…')

        elif self.phase == 'draw':
            wp = self._current_waypoint()
            self.target_rf = self._rf_ik(wp, initial_guess=list(rf_pos))
            self.draw_wp_counter += 1
            if self.draw_wp_counter >= CTRL_PER_WP:
                self.draw_wp_counter = 0
                self._advance_waypoint()

        elif self.phase == 'pen_up':
            # Lift pen, then go to start of next shape
            alpha = min(1.0, self.phase_counter / self.PEN_DOWN_TICKS)
            target_z = self.PEN_Z + alpha * (self.PEN_UP_Z - self.PEN_Z)
            next_wp = self._current_waypoint()
            target_rf_ee = np.array([next_wp[0], next_wp[1], target_z])
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.PEN_DOWN_TICKS:
                self.phase = 'pen_down'
                self.phase_counter = 0

        elif self.phase == 'done':
            # Raise pen, return to stand centre
            target_rf_ee = np.array([self.RF_CENTER_X,
                                     self.RF_CENTER_Y,
                                     self.PEN_UP_Z])
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))

        # ── Cascaded PID correction on RF joints ──────────────────────────
        rf_cmd = np.zeros(3)
        for j in range(3):
            delta = self.pids[j].update(
                desired_angle=self.target_rf[j],
                current_angle=rf_pos[j],
                current_velocity=rf_vel[j]
            )
            rf_cmd[j] = self.target_rf[j] + delta

        # ── Stationary legs: pure IK hold (no cascaded PID needed) ────────
        lf_cmd = self.STAND_ANGLES_LF
        rb_cmd = self.STAND_ANGLES_RB
        lb_cmd = self.STAND_ANGLES_LB

        # ── Pack and publish [RF, LF, RB, LB] ─────────────────────────────
        self.cmd = np.concatenate([rf_cmd, lf_cmd, rb_cmd, lb_cmd])
        msg = Float64MultiArray()
        msg.data = self.cmd.tolist()
        self.cmd_pub.publish(msg)

        # ── Logging (throttled) ───────────────────────────────────────────
        if self.phase == 'draw':
            shape_name  = self.shape_names[self.current_shape_idx]
            n_wps       = len(self.shapes[self.current_shape_idx])
            self.get_logger().info(
                f'[{shape_name}] wp {self.current_wp_idx}/{n_wps} | '
                f'RF target: {self.target_rf} | '
                f'RF actual: {rf_pos} | '
                f'EE target: {self._rf_fk(self.target_rf)} | '
                f'EE actual: {self._rf_fk(rf_pos)}',
                throttle_duration_sec=0.5
            )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    rclpy.init()
    node = PupperArt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted — sending zero command…')
    finally:
        zero = Float64MultiArray()
        zero.data = [0.0] * 12
        node.cmd_pub.publish(zero)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
