"""
pupper_art_sim.py
=================
Pre-deployment simulation node — no URDF, no robot_state_publisher, no hardware needed.

What this does
--------------
  1. Publishes a static TF transform world → base_link so RViz has a valid frame.
  2. Runs the full PupperArt state machine (IK + cascaded PID) against a fake plant.
  3. Publishes /marker as a LINE_STRIP so you see the EE trail drawn live in RViz.
  4. Slows drawing down 5x vs real deployment so you can actually watch it in RViz.
  5. Replays (clears and redraws) automatically so you can keep watching.

Usage
-----
  Terminal 1:   python3 pupper_art_sim.py
  Terminal 2:   ros2 run rviz2 rviz2

  In RViz:
    Global Options → Fixed Frame: world
    Add → By topic → /marker → Marker
    (optionally add Axes display to confirm the TF frame)

Tweak these at the top of the file
------------------------------------
  DRAW_FREQ   lower = slower/more visible  (default 10 Hz, real robot uses 50 Hz)
  SERVO_LAG   0 = perfect servo, 1 = never moves
  NOISE_STD   Gaussian noise on fake joint positions (rad)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import StaticTransformBroadcaster
import numpy as np
import scipy.optimize

np.set_printoptions(precision=4, suppress=True)

# ─────────────────────────────────────────────────────────────
# Copy of transform helpers (kept self-contained so you can run
# this file independently of pupper_art.py being on the path)
# ─────────────────────────────────────────────────────────────

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

def rf_fk(theta):
    T01 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    T12 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T23 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
    T3e = translation(0.06231, -0.06216, 0.01800)
    return (T01 @ T12 @ T23 @ T3e)[:3, 3]

def rf_ik(target_ee, initial_guess=None):
    if initial_guess is None:
        initial_guess = [0.0, 0.65, -1.30]
    result = scipy.optimize.least_squares(
        lambda theta: rf_fk(theta) - target_ee,
        x0=initial_guess, method='lm',
        ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=200
    )
    return result.x


# ─────────────────────────────────────────────────────────────
# Shape generators (identical to pupper_art.py)
# ─────────────────────────────────────────────────────────────

def generate_circle(cx, cy, z, radius=0.035, n=150):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = [np.array([cx + radius*np.cos(a), cy + radius*np.sin(a), z]) for a in angles]
    pts.append(pts[0])
    return pts

def generate_star(cx, cy, z, r_outer=0.05, r_inner=0.02, n_points=5, n_total=120):
    """
    Generate a dense star path by interpolating along each edge between
    alternating outer (tip) and inner (valley) vertices.
    n_total: approximate total number of points (distributed evenly across edges).
    """
    # Build the sparse corner vertices (outer tip, inner valley, ...)
    corners = []
    for i in range(2 * n_points):
        r = r_outer if (i % 2 == 0) else r_inner
        a = np.pi / 2 + i * np.pi / n_points
        corners.append(np.array([cx + r * np.cos(a), cy + r * np.sin(a), z]))
 
    n_edges = len(corners)  # = 2 * n_points = 10
    pts_per_edge = max(2, n_total // n_edges)
 
    pts = []
    for i in range(n_edges):
        start = corners[i]
        end   = corners[(i + 1) % n_edges]
        # linspace from start→end, endpoint=False avoids duplicating the corner
        for t in np.linspace(0, 1, pts_per_edge, endpoint=False):
            pts.append(start + t * (end - start))
 
    pts.append(pts[0])  # close the path
    return pts



# ─────────────────────────────────────────────────────────────
# Cascaded PID (identical to pupper_art.py)
# ─────────────────────────────────────────────────────────────

class CascadedPID:
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
        self._int_pos = self._int_vel = self._int_trq = 0.0
        self._prev_pos_err = self._prev_vel_err = self._prev_trq_err = 0.0

    def update(self, desired_angle, current_angle, current_velocity):
        dt = self.dt
        pos_err = desired_angle - current_angle
        self._int_pos += pos_err * dt
        d_pos = (pos_err - self._prev_pos_err) / dt
        self._prev_pos_err = pos_err
        desired_vel = np.clip(
            self.kp_pos*pos_err + self.ki_pos*self._int_pos + self.kd_pos*d_pos,
            -self.max_vel, self.max_vel)

        vel_err = desired_vel - current_velocity
        self._int_vel += vel_err * dt
        d_vel = (vel_err - self._prev_vel_err) / dt
        self._prev_vel_err = vel_err
        desired_trq = np.clip(
            self.kp_vel*vel_err + self.ki_vel*self._int_vel + self.kd_vel*d_vel,
            -self.max_trq, self.max_trq)

        trq_err = desired_trq
        self._int_trq += trq_err * dt
        d_trq = (trq_err - self._prev_trq_err) / dt
        self._prev_trq_err = trq_err
        delta = np.clip(
            self.kp_trq*trq_err + self.ki_trq*self._int_trq + self.kd_trq*d_trq,
            -self.max_delta, self.max_delta)
        return delta


# ─────────────────────────────────────────────────────────────
# Fake plant (replaces the real robot hardware)
# ─────────────────────────────────────────────────────────────

class FakePlant:
    """
    One-step lag servo model.
    cmd_positions are what the controller sends;
    actual_positions trail by SERVO_LAG per step.
    """
    SERVO_LAG  = 0.7    # ∈ [0, 1]: 0 = perfect, 1 = never moves
    NOISE_STD  = 0.001  # rad — mimics encoder noise

    # Initial standing pose [RF, LF, RB, LB] × 3 joints
    STAND_INIT = np.array([0.0, 0.65, -1.30] * 4, dtype=float)

    def __init__(self):
        self.positions  = self.STAND_INIT.copy()
        self.velocities = np.zeros(12)
        self._prev_pos  = self.positions.copy()

    def step(self, cmd: np.ndarray, dt: float):
        """Integrate one timestep given a 12-element position command."""
        lag = self.SERVO_LAG
        new_pos = lag * self.positions + (1 - lag) * cmd
        noise   = np.random.normal(0, self.NOISE_STD, 12)
        self.velocities = (new_pos - self._prev_pos) / dt
        self._prev_pos  = self.positions.copy()
        self.positions  = new_pos + noise


# ─────────────────────────────────────────────────────────────
# Simulator node
# ─────────────────────────────────────────────────────────────

class PupperArtSim(Node):

    # ── Same constants as PupperArt ────────────────────────────────────────
    STAND_ANGLES_LF = np.array([ 0.00,  0.65, -1.30])
    STAND_ANGLES_RB = np.array([ 0.00,  0.65, -1.30])
    STAND_ANGLES_LB = np.array([ 0.00,  0.65, -1.30])

    PEN_Z        = -0.14
    PEN_UP_Z     = -0.10
    RF_CENTER_X  =  0.06
    RF_CENTER_Y  = -0.09

    CIRCLE_RADIUS = 0.035
    STAR_R_OUTER  = 0.040
    STAR_R_INNER  = 0.016
    N_CIRCLE_PTS  = 150

    # Slowed way down vs real robot so you can watch in RViz.
    # Crank DRAW_FREQ back up to 50.0 when deploying on hardware.
    DRAW_FREQ  = 10.0    # Hz  (real = 50 Hz)
    CTRL_FREQ  = 200.0   # Hz
    SIM_DT     = 1.0 / CTRL_FREQ

    def __init__(self):
        super().__init__('pupper_art_sim')

        # ── Static TF: world → base_link ───────────────────────────────────
        # Without this RViz has no valid frame and silently drops all markers.
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id  = 'base_link'
        tf_msg.transform.rotation.w = 1.0   # identity rotation
        self._tf_broadcaster.sendTransform(tf_msg)

        # ── Publishers ─────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Float64MultiArray, '/forward_command_controller/commands', 10)
        self.marker_pub = self.create_publisher(Marker,            '/marker', 10)

        # ── Fake plant ─────────────────────────────────────────────────────
        self.plant = FakePlant()

        # ── PIDs ───────────────────────────────────────────────────────────
        self.pids = [CascadedPID(dt=self.SIM_DT) for _ in range(3)]

        # ── Drawing sequence ───────────────────────────────────────────────
        cx, cy = self.RF_CENTER_X, self.RF_CENTER_Y
        self.shapes = [
            generate_circle(cx, cy, self.PEN_Z,
                            radius=self.CIRCLE_RADIUS,
                            n=self.N_CIRCLE_PTS),
            generate_star(cx, cy, self.PEN_Z,
                          r_outer=self.STAR_R_OUTER,
                          r_inner=self.STAR_R_INNER),
        ]
        self.shape_names = ['circle', 'star']

        # ── State machine ──────────────────────────────────────────────────
        self.phase             = 'stand'
        self.phase_counter     = 0
        self.STAND_TICKS       = int(2.0 * self.CTRL_FREQ)
        self.PEN_DOWN_TICKS    = int(1.0 * self.CTRL_FREQ)
        self.current_shape_idx = 0
        self.current_wp_idx    = 0
        self.draw_wp_counter   = 0
        self.target_rf         = np.array(self.STAND_ANGLES_LF)

        # ── RViz EE trail (two separate trails, one per shape) ────────────
        self.trail_circle: list[Point] = []
        self.trail_star:   list[Point] = []

        # ── Main control loop ──────────────────────────────────────────────
        self.ctrl_timer = self.create_timer(self.SIM_DT, self._step)

        self.get_logger().info(
            '\n========================================\n'
            '  PupperArt SIMULATOR started\n'
            '  In RViz:\n'
            '    Global Options -> Fixed Frame: world\n'
            '    Add -> By topic -> /marker -> Marker\n'
            '  DRAW_FREQ slowed to 10 Hz so you can watch.\n'
            '  Will replay automatically when done.\n'
            '========================================'
        )

    # ── FK / IK helpers (same as pupper_art.py) ────────────────────────────

    @staticmethod
    def _fk(theta): return rf_fk(theta)

    def _ik(self, target, initial_guess=None):
        return rf_ik(target, initial_guess=initial_guess)

    # ── State machine helpers ───────────────────────────────────────────────

    def _current_waypoint(self):
        return self.shapes[self.current_shape_idx][self.current_wp_idx]

    def _advance_waypoint(self):
        shape = self.shapes[self.current_shape_idx]
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(shape):
            self.get_logger().info(
                f'[SIM] Finished drawing {self.shape_names[self.current_shape_idx]}!')
            self.current_shape_idx += 1
            self.current_wp_idx = 0
            if self.current_shape_idx >= len(self.shapes):
                self.phase = 'done'
                self.get_logger().info('[SIM] All shapes complete. Replaying in 3 s…')
            else:
                self.phase = 'pen_up'
                self.phase_counter = 0
                self.get_logger().info(
                    f'[SIM] Moving to {self.shape_names[self.current_shape_idx]}…')

    # ── Main simulation step (200 Hz) ──────────────────────────────────────

    def _step(self):
        pos = self.plant.positions
        vel = self.plant.velocities
        rf_pos, rf_vel = pos[0:3], vel[0:3]

        CTRL_PER_WP = max(1, int(self.CTRL_FREQ / self.DRAW_FREQ))

        # ── Exact copy of PupperArt._ctrl_cb state machine ─────────────────
        if self.phase == 'stand':
            target_rf_ee = np.array([self.RF_CENTER_X, self.RF_CENTER_Y, self.PEN_UP_Z])
            self.target_rf = self._ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.STAND_TICKS:
                self.phase = 'pen_down'
                self.phase_counter = 0
                self.get_logger().info('[SIM] Standing stable — lowering pen…')

        elif self.phase == 'pen_down':
            alpha = min(1.0, self.phase_counter / self.PEN_DOWN_TICKS)
            target_z = self.PEN_UP_Z + alpha * (self.PEN_Z - self.PEN_UP_Z)
            wp = self._current_waypoint()
            target_rf_ee = np.array([wp[0], wp[1], target_z])
            self.target_rf = self._ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.PEN_DOWN_TICKS:
                self.phase = 'draw'
                self.draw_wp_counter = 0
                self.get_logger().info(
                    f'[SIM] Drawing {self.shape_names[self.current_shape_idx]}…')

        elif self.phase == 'draw':
            wp = self._current_waypoint()
            self.target_rf = self._ik(wp, initial_guess=list(rf_pos))
            self.draw_wp_counter += 1
            if self.draw_wp_counter >= CTRL_PER_WP:
                self.draw_wp_counter = 0
                self._advance_waypoint()

        elif self.phase == 'pen_up':
            alpha = min(1.0, self.phase_counter / self.PEN_DOWN_TICKS)
            target_z = self.PEN_Z + alpha * (self.PEN_UP_Z - self.PEN_Z)
            next_wp = self._current_waypoint()
            target_rf_ee = np.array([next_wp[0], next_wp[1], target_z])
            self.target_rf = self._ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.PEN_DOWN_TICKS:
                self.phase = 'pen_down'
                self.phase_counter = 0

        elif self.phase == 'done':
            target_rf_ee = np.array([self.RF_CENTER_X, self.RF_CENTER_Y, self.PEN_UP_Z])
            self.target_rf = self._ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            # After 3 seconds, clear trails and replay from the start
            if self.phase_counter >= int(3.0 * self.CTRL_FREQ):
                self.trail_circle.clear()
                self.trail_star.clear()
                self._clear_marker()
                self.plant = FakePlant()
                for pid in self.pids:
                    pid.reset()
                self.phase             = 'stand'
                self.phase_counter     = 0
                self.current_shape_idx = 0
                self.current_wp_idx    = 0
                self.draw_wp_counter   = 0
                self.target_rf         = np.array(self.STAND_ANGLES_LF)
                self.get_logger().info('[SIM] Replaying…')

        # ── Cascaded PID ───────────────────────────────────────────────────
        rf_cmd = np.zeros(3)
        for j in range(3):
            delta = self.pids[j].update(
                desired_angle=self.target_rf[j],
                current_angle=rf_pos[j],
                current_velocity=rf_vel[j]
            )
            rf_cmd[j] = self.target_rf[j] + delta

        cmd = np.concatenate([rf_cmd,
                              self.STAND_ANGLES_LF,
                              self.STAND_ANGLES_RB,
                              self.STAND_ANGLES_LB])

        # ── Step the fake plant ────────────────────────────────────────────
        self.plant.step(cmd, self.SIM_DT)

        # ── Publish /forward_command_controller/commands ───────────────────
        fc = Float64MultiArray()
        fc.data = cmd.tolist()
        self.cmd_pub.publish(fc)

        # ── Accumulate and publish EE trail during draw phase ─────────────
        if self.phase == 'draw':
            ee = self._fk(self.plant.positions[0:3])
            p = Point(x=float(ee[0]), y=float(ee[1]), z=float(ee[2]))
            if self.current_shape_idx == 0:
                self.trail_circle.append(p)
            else:
                self.trail_star.append(p)
            self._publish_trails()

    def _make_line_marker(self, marker_id, points, r, g, b):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'   # matches the static TF we published
        m.ns              = 'ee_trail'
        m.id              = marker_id
        m.type            = Marker.LINE_STRIP
        m.action          = Marker.ADD
        m.scale.x         = 0.004     # line width in metres
        m.color.r         = float(r)
        m.color.g         = float(g)
        m.color.b         = float(b)
        m.color.a         = 1.0
        m.points          = points
        return m

    def _publish_trails(self):
        if self.trail_circle:
            self.marker_pub.publish(
                self._make_line_marker(0, self.trail_circle, 0.2, 0.7, 1.0))  # blue
        if self.trail_star:
            self.marker_pub.publish(
                self._make_line_marker(1, self.trail_star,   1.0, 0.6, 0.1))  # orange

    def _clear_marker(self):
        """Send DELETE_ALL so RViz wipes the old trails before replay."""
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'
        m.action          = Marker.DELETEALL
        self.marker_pub.publish(m)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = PupperArtSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[SIM] Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
