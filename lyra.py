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

import sys
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

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

def generate_circle(center_x, center_y, z_floor, radius=0.01, n_points=120, loop_factor=1.5):
    """Generate (x, y, z) waypoints for a circle in the floor plane, with loop_factor loops."""
    angles = np.linspace(0, loop_factor * 2 * np.pi, int(loop_factor * n_points), endpoint=False)
    pts = []
    for a in angles:
        pts.append(np.array([
            center_x + radius * np.cos(a),
            center_y + radius * np.sin(a),
            z_floor
        ]))
    return pts


def generate_star(cx, cy, z, r_outer=0.02, r_inner=0.005, n_points=5, n_total=120, loop_factor=1.5):
    """
    Generate a dense star path by interpolating along each edge between
    alternating outer (tip) and inner (valley) vertices.
    n_total: approximate total number of points (distributed evenly across edges).
    loop_factor: number of times to loop through the star.
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
    for loop in range(int(loop_factor) + 1):  # for 1.5, do 1 full + half
        for i in range(n_edges):
            if loop == int(loop_factor) and i >= n_edges // 2:  # for half loop, stop at half
                break
            start = corners[i]
            end   = corners[(i + 1) % n_edges]
            # linspace from start→end, endpoint=False avoids duplicating the corner
            for t in np.linspace(0, 1, pts_per_edge, endpoint=False):
                pts.append(start + t * (end - start))
    return pts


# ──────────────────────────────────────────────
# Custom path drawing window
# ──────────────────────────────────────────────

def capture_custom_path(center_x, center_y, z_floor,
                        workspace_radius=0.05, n_interp=200):
    """
    Opens an interactive matplotlib window for drawing a custom path.

    The canvas is centred on (center_x, center_y) and the grey circle shows
    the approximate reachable workspace of the RF leg.

    Controls:
      Left-click  — add a waypoint
      Right-click — finish and confirm the path
      Middle-click / 'u' key — undo the last point
      Close window without right-clicking — cancels (returns None)

    Returns a list of np.array([x, y, z]) waypoints interpolated to n_interp
    points, or None if cancelled.
    """
    # Shared state between callbacks
    state = {'points': [], 'confirmed': False, 'cancelled': False}

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title('Lyra — Draw Custom Path')

    # Workspace boundary
    boundary = plt.Circle((center_x, center_y), workspace_radius,
                           color='#cccccc', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(boundary)
    ax.plot(center_x, center_y, '+', color='#999999', markersize=12)

    ax.set_xlim(center_x - workspace_radius * 1.3, center_x + workspace_radius * 1.3)
    ax.set_ylim(center_y - workspace_radius * 1.3, center_y + workspace_radius * 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, robot body frame)')
    ax.set_ylabel('Y (m, robot body frame)')
    ax.set_title(
        'Left-click: add point   Right-click: confirm   U: undo\n'
        'Draw a single connected path — no gaps or jumps.',
        fontsize=9
    )

    # Live plot objects
    path_line, = ax.plot([], [], 'b-o', linewidth=2, markersize=5, zorder=3)
    start_dot, = ax.plot([], [], 'go', markersize=10, zorder=4, label='start')
    end_dot,   = ax.plot([], [], 'rs', markersize=10, zorder=4, label='end')
    status_text = ax.text(
        0.01, 0.01, 'No points yet.',
        transform=ax.transAxes, fontsize=8, color='#444444',
        verticalalignment='bottom'
    )

    def _redraw():
        pts = state['points']
        if len(pts) == 0:
            path_line.set_data([], [])
            start_dot.set_data([], [])
            end_dot.set_data([], [])
            status_text.set_text('No points yet.')
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            path_line.set_data(xs, ys)
            start_dot.set_data([xs[0]], [ys[0]])
            end_dot.set_data([xs[-1]], [ys[-1]])
            status_text.set_text(f'{len(pts)} point(s).  Right-click to confirm.')
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:   # left-click: add point
            x, y = event.xdata, event.ydata
            # Clamp to workspace circle
            dx, dy = x - center_x, y - center_y
            dist = np.hypot(dx, dy)
            if dist > workspace_radius:
                scale = workspace_radius / dist
                x = center_x + dx * scale
                y = center_y + dy * scale
            state['points'].append((x, y))
            _redraw()
        elif event.button == 3:  # right-click: confirm
            if len(state['points']) >= 2:
                state['confirmed'] = True
                plt.close(fig)
            else:
                status_text.set_text('Need at least 2 points!')
                fig.canvas.draw_idle()
        elif event.button == 2:  # middle-click: undo
            if state['points']:
                state['points'].pop()
                _redraw()

    def _on_key(event):
        if event.key in ('u', 'U', 'backspace'):
            if state['points']:
                state['points'].pop()
                _redraw()

    def _on_close(event):
        if not state['confirmed']:
            state['cancelled'] = True

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    fig.canvas.mpl_connect('close_event', _on_close)

    plt.tight_layout()
    plt.show(block=True)   # blocks until the window is closed

    if state['cancelled'] or not state['confirmed'] or len(state['points']) < 2:
        print('[custom path] Cancelled or too few points — aborting.', flush=True)
        return None

    # ── Interpolate the clicked points to n_interp evenly-spaced waypoints ──
    raw = np.array(state['points'])   # shape (N, 2)

    # Compute cumulative arc length along the raw path
    diffs = np.diff(raw, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        print('[custom path] Path is too short — aborting.', flush=True)
        return None

    # Uniform re-parameterisation
    t_uniform = np.linspace(0, total_len, n_interp)
    x_interp = np.interp(t_uniform, cum_len, raw[:, 0])
    y_interp = np.interp(t_uniform, cum_len, raw[:, 1])

    waypoints = [np.array([x_interp[i], y_interp[i], z_floor])
                 for i in range(n_interp)]

    print(f'[custom path] {len(waypoints)} waypoints generated '
          f'(total path length ≈ {total_len*100:.1f} cm).', flush=True)
    return waypoints


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

    # ── Standing angles for all four legs ─────────────────────────────────


    STAND_ANGLES_LF = np.array([-1.72766, -0.44671, +2.51156])
    STAND_ANGLES_RB = np.array([+1.66891, +1.72504, -2.31167])
    STAND_ANGLES_LB = np.array([-1.60368, -1.68193, +2.64584])

    #STAND_ANGLES_LF = np.array([-1.84630, -0.45434, +2.65576])
    #STAND_ANGLES_RB = np.array([+1.62009, +1.65790, -2.35630])
    #STAND_ANGLES_LB = np.array([-0.31697, -0.01030, +0.42069])


    # RF home:
    STAND_ANGLES_RF_HOME = np.array([+1.04291, -0.39978, +0.69436])


    # Servo stiffness — ramped from 0 → these values during stand_up phase
    SERVO_KP      = 5.0   # position gain — enough to stand, not so much it snaps
    SERVO_KD      = 0.2   # damping gain
    RAMP_UP_SECS  = 3.0   # seconds to ramp from limp to full stiffness

    # RF pen tip height while drawing (metres, in body frame, negative = down)
    PEN_Z = -0.14

    # RF "pen-up" height for transitions
    PEN_UP_Z = -0.10

    # RF leg centre position (body frame x-y)
    RF_CENTER_X =  0.06
    RF_CENTER_Y = -0.09

    # Drawing parameters
    CIRCLE_RADIUS   = 0.01   # m
    STAR_R_OUTER    = 0.03   # m
    STAR_R_INNER    = 0.01   # m
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
        self.kp_pub  = self.create_publisher(
            Float64MultiArray, '/forward_kp_controller/commands', 10)
        self.kd_pub  = self.create_publisher(
            Float64MultiArray, '/forward_kd_controller/commands', 10)

        # ── State ──────────────────────────────────────────────────────────
        self.joint_positions  = None
        self.joint_velocities = None

        # 12-element command array: [RF(3), LF(3), RB(3), LB(3)]
        self.cmd = np.zeros(12)

        # IK-computed target for RF leg — use RF home, not LF angles
        self.target_rf = np.array(self.STAND_ANGLES_RF_HOME)

        # Cascaded PIDs — one per RF joint
        dt_pid = 1.0 / self.CTRL_FREQ
        self.pids = [CascadedPID(dt=dt_pid) for _ in range(3)]

        # ── Drawing sequence ───────────────────────────────────────────────
        self.current_shape = None
        self.shape_name = None
        self.current_wp_idx = 0
        self.desired_positions = []
        self.actual_positions = []

        # stand_up: ramp all 4 legs to standing with increasing kp/kd
        # stand:    hold for 2 s, confirm stability
        # pen_down → draw → pen_up → done → standing_hold (until q)
        self.phase             = 'idle'   # waits for 's' keypress
        self.phase_counter     = 0
        self.RAMP_TICKS        = int(self.RAMP_UP_SECS * self.CTRL_FREQ)
        self.STAND_TICKS       = int(2.0 * self.CTRL_FREQ)
        self.PEN_DOWN_TICKS    = int(1.0 * self.CTRL_FREQ)
        self.TRANSITION_TICKS  = int(1.0 * self.CTRL_FREQ)  # make transitions slow and smooth

        self.draw_wp_counter   = 0

        # ── Timers ─────────────────────────────────────────────────────────
        self.ctrl_timer = self.create_timer(1.0 / self.CTRL_FREQ, self._ctrl_cb)

        # ── Keypress thread ────────────────────────────────────────────────
        self._key_thread = threading.Thread(target=self._key_loop, daemon=True)
        self._key_thread.start()

        self.get_logger().info(
            'PupperArt ready.\n'
            '  s = stand up\n'
            '  c = draw circle\n'
            '  p = draw star\n'
            '  d = draw custom path (opens interactive window)\n'
            '  q = relax and quit\n'
        )

    def _start_drawing(self, shape_name, custom_waypoints=None):
        cx, cy = self.RF_CENTER_X, self.RF_CENTER_Y
        if shape_name == 'circle':
            self.current_shape = generate_circle(cx, cy, self.PEN_Z,
                                                 radius=self.CIRCLE_RADIUS,
                                                 n_points=self.N_CIRCLE_PTS)
        elif shape_name == 'star':
            self.current_shape = generate_star(cx, cy, self.PEN_Z,
                                               r_outer=self.STAR_R_OUTER,
                                               r_inner=self.STAR_R_INNER)
        elif shape_name == 'custom':
            if custom_waypoints is None or len(custom_waypoints) < 2:
                print('[custom] No valid waypoints — aborting.', flush=True)
                return
            self.current_shape = custom_waypoints
        self.shape_name = shape_name
        self.current_wp_idx = 0
        self.desired_positions = []
        self.actual_positions = []
        self.phase = 'move_to_start'
        self.phase_counter = 0
        print(f'[DRAWING {shape_name.upper()}]', flush=True)

    def _key_loop(self):
        print('\nControls:  s = stand   c = circle   p = star   d = draw custom   q = relax + quit\n', flush=True)
        while rclpy.ok():
            key = input().strip().lower()
            if key == 's':
                if self.phase == 'idle':
                    self.phase = 'stand_up'
                    self.phase_counter = 0
                    print('[STANDING UP]', flush=True)
                else:
                    print('[already standing or drawing]', flush=True)
            elif key == 'c':
                if self.phase in ['standing_hold', 'done']:
                    self._start_drawing('circle')
                elif self.phase == 'idle':
                    print('[press s to stand first]', flush=True)
                else:
                    print(f'[not ready — phase: {self.phase}]', flush=True)
            elif key == 'p':
                if self.phase in ['standing_hold', 'done']:
                    self._start_drawing('star')
                elif self.phase == 'idle':
                    print('[press s to stand first]', flush=True)
                else:
                    print(f'[not ready — phase: {self.phase}]', flush=True)
            elif key == 'd':
                if self.phase in ['standing_hold', 'done']:
                    print(
                        '[CUSTOM PATH] Opening drawing window…\n'
                        '  Left-click to add points, right-click to confirm, U to undo.\n'
                        '  Draw a single connected path — no gaps or jumps.\n'
                        '  The dashed circle shows the RF leg workspace boundary.',
                        flush=True
                    )
                    waypoints = capture_custom_path(
                        center_x=self.RF_CENTER_X,
                        center_y=self.RF_CENTER_Y,
                        z_floor=self.PEN_Z,
                        workspace_radius=max(self.STAR_R_OUTER * 1.5, 0.04),
                        n_interp=200,
                    )
                    if waypoints is not None:
                        self._start_drawing('custom', custom_waypoints=waypoints)
                    else:
                        print('[custom path cancelled — still in standing_hold]', flush=True)
                elif self.phase == 'idle':
                    print('[press s to stand first]', flush=True)
                else:
                    print(f'[not ready — phase: {self.phase}]', flush=True)
            elif key == 'q':
                print('[RELAXING AND QUITTING]', flush=True)
                self.phase = 'relax_quit'

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
        return self.current_shape[self.current_wp_idx]

    def _advance_waypoint(self):
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(self.current_shape):
            # Finished this shape
            self.get_logger().info(f'Finished drawing {self.shape_name}!')
            self.phase = 'pen_up'
            self.phase_counter = 0

    def _plot_trajectory(self):
        if not self.desired_positions or not self.actual_positions:
            return
        desired = np.array(self.desired_positions)
        actual = np.array(self.actual_positions)
        
        # RMSE
        rmse = np.sqrt(np.mean((desired - actual)**2, axis=0))
        print(f'RMSE for {self.shape_name}: X={rmse[0]:.4f}, Y={rmse[1]:.4f}, Z={rmse[2]:.4f}')
        
        # Plot desired vs actual
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(desired[:, 0], desired[:, 1], 'b-', label='Desired', linewidth=2)
        plt.plot(actual[:, 0], actual[:, 1], 'r--', label='Actual', linewidth=1)
        plt.title(f'{self.shape_name.capitalize()} Trajectory')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(1, 2, 2)
        plt.plot(desired[:, 2], 'b-', label='Desired Z')
        plt.plot(actual[:, 2], 'r--', label='Actual Z')
        plt.title(f'{self.shape_name.capitalize()} Z Position')
        plt.xlabel('Time step')
        plt.ylabel('Z')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.shape_name}_trajectory.png')
        # plt.show()  # Removed to avoid blocking
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

        if self.phase == 'idle':
            return   # waiting for 's'

        elif self.phase == 'stand_up':
            # Ramp kp/kd linearly from 0 → SERVO_KP/KD over RAMP_TICKS.
            # All 4 legs are commanded to their standing targets the whole time,
            # so as stiffness increases they pull themselves to the right pose.
            alpha  = min(1.0, self.phase_counter / self.RAMP_TICKS)
            kp_now = alpha * self.SERVO_KP
            kd_now = alpha * self.SERVO_KD
            self.kp_pub.publish(Float64MultiArray(data=[kp_now] * 12))
            self.kd_pub.publish(Float64MultiArray(data=[kd_now] * 12))

            stand_cmd = np.concatenate([
                self.STAND_ANGLES_RF_HOME,
                self.STAND_ANGLES_LF,
                self.STAND_ANGLES_RB,
                self.STAND_ANGLES_LB,
            ])
            self.cmd_pub.publish(Float64MultiArray(data=stand_cmd.tolist()))

            self.phase_counter += 1
            if self.phase_counter >= self.RAMP_TICKS:
                self.phase = 'stand'
                self.phase_counter = 0
                self.get_logger().info('All legs standing — holding 2 s before drawing…')
            return   # skip PID / publish below during ramp

        elif self.phase == 'stand':
            # Hold all four legs at standing targets for STAND_TICKS.
            # kp/kd are already at full value from stand_up phase.
            self.target_rf = np.array(self.STAND_ANGLES_RF_HOME)
            stand_cmd = np.concatenate([
                self.STAND_ANGLES_RF_HOME,
                self.STAND_ANGLES_LF,
                self.STAND_ANGLES_RB,
                self.STAND_ANGLES_LB,
            ])
            self.cmd_pub.publish(Float64MultiArray(data=stand_cmd.tolist()))
            self.phase_counter += 1
            if self.phase_counter >= self.STAND_TICKS:
                self.phase = 'standing_hold'
                self.phase_counter = 0
                self.get_logger().info('Standing stable — press c for circle, p for star, or d for custom.')
            return   # skip PID / publish below during hold

        elif self.phase == 'standing_hold':
            # Hold standing pose indefinitely; waiting for 'c', 'p' or 'q'
            stand_cmd = np.concatenate([
                self.STAND_ANGLES_RF_HOME,
                self.STAND_ANGLES_LF,
                self.STAND_ANGLES_RB,
                self.STAND_ANGLES_LB,
            ])
            self.cmd_pub.publish(Float64MultiArray(data=stand_cmd.tolist()))
            return

        elif self.phase == 'move_to_start':
            # Smoothly move the pen to the shape start position at pen-up height
            first_wp = self._current_waypoint()
            start_target = np.array([first_wp[0], first_wp[1], self.PEN_UP_Z])
            current_ee = self._rf_fk(rf_pos)
            alpha = min(1.0, self.phase_counter / self.TRANSITION_TICKS)
            target_rf_ee = current_ee + alpha * (start_target - current_ee)
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.TRANSITION_TICKS:
                self.phase = 'pen_down'
                self.phase_counter = 0
                self.get_logger().info('Lowering pen to start drawing…')

        elif self.phase == 'pen_down':
            # Smoothly lower pen from pen-up height to drawing height while holding XY at start
            first_wp = self._current_waypoint()
            alpha = min(1.0, self.phase_counter / self.PEN_DOWN_TICKS)
            target_z = self.PEN_UP_Z + alpha * (self.PEN_Z - self.PEN_UP_Z)
            target_rf_ee = np.array([first_wp[0], first_wp[1], target_z])
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.PEN_DOWN_TICKS:
                self.phase = 'draw'
                self.draw_wp_counter = 0
                self.get_logger().info(
                    f'Drawing {self.shape_name}…')

        elif self.phase == 'draw':
            wp = self._current_waypoint()
            self.target_rf = self._rf_ik(wp, initial_guess=list(rf_pos))
            self.desired_positions.append(wp)
            self.actual_positions.append(self._rf_fk(rf_pos))
            self.draw_wp_counter += 1
            if self.draw_wp_counter >= CTRL_PER_WP:
                self.draw_wp_counter = 0
                self._advance_waypoint()

        elif self.phase == 'pen_up':
            # Smoothly lift the pen vertically at the final drawing XY position
            current_ee = self._rf_fk(rf_pos)
            target_lift = np.array([current_ee[0], current_ee[1], self.PEN_UP_Z])
            alpha = min(1.0, self.phase_counter / self.TRANSITION_TICKS)
            target_rf_ee = current_ee + alpha * (target_lift - current_ee)
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.TRANSITION_TICKS:
                self.phase = 'return_home'
                self.phase_counter = 0
                self.get_logger().info('Pen lifted — returning smoothly to home position…')

        elif self.phase == 'return_home':
            # Smoothly move the pen home at the raised pen-up height
            current_ee = self._rf_fk(rf_pos)
            target_center = np.array([self.RF_CENTER_X, self.RF_CENTER_Y, self.PEN_UP_Z])
            alpha = min(1.0, self.phase_counter / self.TRANSITION_TICKS)
            target_rf_ee = current_ee + alpha * (target_center - current_ee)
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= self.TRANSITION_TICKS:
                self.phase = 'standing_hold'
                self.phase_counter = 0
                self._plot_trajectory()
                self.get_logger().info('Ready for next command — press c for circle, p for star, or d for custom.')

        elif self.phase == 'done':
            # Raise pen back to home, hold briefly, then return to standing_hold
            target_rf_ee = np.array([self.RF_CENTER_X,
                                     self.RF_CENTER_Y,
                                     self.PEN_UP_Z])
            self.target_rf = self._rf_ik(target_rf_ee, initial_guess=list(rf_pos))
            self.phase_counter += 1
            if self.phase_counter >= int(1.0 * self.CTRL_FREQ):
                self.get_logger().info('Drawing complete — standing by. Press q to quit.')
                self.phase = 'standing_hold'
                self.phase_counter = 0

        elif self.phase == 'relax_quit':
            zero = Float64MultiArray(data=[0.0] * 12)
            self.cmd_pub.publish(zero)
            self.kp_pub.publish(zero)
            self.kd_pub.publish(zero)
            sys.exit(0)

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
            n_wps = len(self.current_shape)
            self.get_logger().info(
                f'[{self.shape_name}] wp {self.current_wp_idx}/{n_wps} | '
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
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()