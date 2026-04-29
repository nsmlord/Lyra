"""
pupper_art_playground.py
========================
Offline verification — no ROS required.

Generates:
  planned_paths.png          — circle & star floor-plane paths
  ik_accuracy.png            — planned vs IK-achieved EE for both shapes
  pid_simulation_circle.png  — cascaded PID tracking, all 3 joints, circle
  pid_simulation_star.png    — cascaded PID tracking, all 3 joints, star
  pid_ee_circle.png          — EE path: IK target vs PID output, circle
  pid_ee_star.png            — EE path: IK target vs PID output, star
  full_sequence_ee.png       — full circle-then-star PID EE path

Run:  python3 pupper_art_playground.py
"""

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

np.set_printoptions(precision=4, suppress=True)


# ──────────────────────────────────────────────
# Transforms
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
# RF leg FK / IK
# ──────────────────────────────────────────────

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
        x0=initial_guess,
        method='lm',
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
        max_nfev=300
    )
    return result.x


# ──────────────────────────────────────────────
# Shape generators
# ──────────────────────────────────────────────

def generate_circle(cx, cy, z, radius=0.035, n=120):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = [np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z]) for a in angles]
    pts.append(pts[0])
    return pts

def generate_star(cx, cy, z, r_outer=0.05, r_inner=0.02, n_points=5):
    pts = []
    for i in range(2 * n_points + 1):
        r = r_outer if (i % 2 == 0) else r_inner
        a = np.pi / 2 + i * np.pi / n_points
        pts.append(np.array([cx + r * np.cos(a), cy + r * np.sin(a), z]))
    return pts


# ──────────────────────────────────────────────
# Cascaded PID  (identical to pupper_art.py)
# ──────────────────────────────────────────────

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

        # Stage 1: position error → desired angular velocity
        pos_err = desired_angle - current_angle
        self._int_pos += pos_err * dt
        d_pos = (pos_err - self._prev_pos_err) / dt
        self._prev_pos_err = pos_err
        desired_vel = np.clip(
            self.kp_pos * pos_err + self.ki_pos * self._int_pos + self.kd_pos * d_pos,
            -self.max_vel, self.max_vel)

        # Stage 2: velocity error → desired torque
        vel_err = desired_vel - current_velocity
        self._int_vel += vel_err * dt
        d_vel = (vel_err - self._prev_vel_err) / dt
        self._prev_vel_err = vel_err
        desired_trq = np.clip(
            self.kp_vel * vel_err + self.ki_vel * self._int_vel + self.kd_vel * d_vel,
            -self.max_trq, self.max_trq)

        # Stage 3: torque → position delta command
        trq_err = desired_trq
        self._int_trq += trq_err * dt
        d_trq = (trq_err - self._prev_trq_err) / dt
        self._prev_trq_err = trq_err
        delta = np.clip(
            self.kp_trq * trq_err + self.ki_trq * self._int_trq + self.kd_trq * d_trq,
            -self.max_delta, self.max_delta)
        return delta


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

def solve_path_ik(pts, initial_guess=None):
    guess = list(initial_guess) if initial_guess is not None else [0.0, 0.65, -1.30]
    thetas, ees, errors = [], [], []
    for wp in pts:
        theta = rf_ik(wp, initial_guess=guess)
        ee    = rf_fk(theta)
        errors.append(np.linalg.norm(ee - wp))
        thetas.append(theta)
        ees.append(ee)
        guess = list(theta)
    return np.array(thetas), np.array(ees), np.array(errors)


def simulate_pid_all_joints(desired_thetas, dt=0.005):
    """
    Open-loop cascaded PID simulation for all 3 RF joints.
    Plant model: position-controlled, 1-step lag.
    """
    pids = [CascadedPID(dt=dt) for _ in range(3)]
    n    = len(desired_thetas)
    sim  = np.zeros((n, 3))
    cur  = desired_thetas[0].copy()
    vel  = np.zeros(3)
    sim[0] = cur
    for k in range(1, n):
        cmd = np.zeros(3)
        for j in range(3):
            delta  = pids[j].update(desired_thetas[k, j], cur[j], vel[j])
            cmd[j] = desired_thetas[k, j] + delta
        vel = (cmd - cur) / dt
        cur = cmd.copy()
        sim[k] = cur
    return sim


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    CX, CY, Z = 0.06, -0.09, -0.14

    circle_pts = generate_circle(CX, CY, Z)
    star_pts   = generate_star(CX, CY, Z)

    # ══════════════════════════════════════════════════════════════════════
    # 1. Planned floor-plane paths
    # ══════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pts, name in zip(axes1, [circle_pts, star_pts], ['Circle', 'Star']):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, 'b-o', markersize=3, label='Planned path')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'Planned {name} — RF end-effector (floor plane)')
        ax.set_aspect('equal'); ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig('planned_paths.png', dpi=150)
    print('Saved planned_paths.png')

    # ══════════════════════════════════════════════════════════════════════
    # 2. IK accuracy — circle AND star
    # ══════════════════════════════════════════════════════════════════════
    print('\n── IK accuracy ──')
    results = {}
    for pts, name in [(circle_pts, 'circle'), (star_pts, 'star')]:
        thetas, ees, errs = solve_path_ik(pts)
        results[name] = (thetas, ees, errs)
        print(f'  {name:6s}  max={max(errs)*1000:.4f} mm  mean={np.mean(errs)*1000:.4f} mm')

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes2, ['circle', 'star']):
        thetas, ees, _ = results[name]
        pts = circle_pts if name == 'circle' else star_pts
        plan_x = [p[0] for p in pts]; plan_y = [p[1] for p in pts]
        ax.plot(plan_x, plan_y, 'b--', lw=2,   label='Planned')
        ax.plot(ees[:, 0], ees[:, 1], 'r-', lw=1.5, label='IK achieved')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'{name.capitalize()} — planned vs IK-achieved EE')
        ax.set_aspect('equal'); ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig('ik_accuracy.png', dpi=150)
    print('Saved ik_accuracy.png')

    # ══════════════════════════════════════════════════════════════════════
    # 3. Cascaded PID simulation — ALL 3 JOINTS, BOTH SHAPES
    # ══════════════════════════════════════════════════════════════════════
    DT           = 0.005
    joint_labels = ['Joint 1 — abduction', 'Joint 2 — shoulder pitch', 'Joint 3 — knee']
    colors       = ['tab:blue', 'tab:orange', 'tab:green']

    print('\n── Cascaded PID simulation ──')
    for name in ['circle', 'star']:
        thetas, ees_ik, _ = results[name]
        sim_thetas = simulate_pid_all_joints(thetas, dt=DT)
        t_axis     = np.arange(len(thetas)) * DT

        max_err = np.max(np.abs(thetas - sim_thetas))
        print(f'  {name}: max joint tracking error = {max_err:.5f} rad '
              f'({max_err * 180/np.pi:.4f} deg)')

        # per-joint angle plots
        fig3, axes3 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fig3.suptitle(f'Cascaded PID simulation — RF leg, {name} path', fontsize=13)
        for j, (ax, label, color) in enumerate(zip(axes3, joint_labels, colors)):
            ax.plot(t_axis, thetas[:, j],     'k--', lw=1.5, label='IK target')
            ax.plot(t_axis, sim_thetas[:, j], color=color, lw=1.5, label='PID output')
            err = thetas[:, j] - sim_thetas[:, j]
            ax.fill_between(t_axis, err, alpha=0.2, color='red', label='Tracking error')
            ax.set_ylabel('Angle (rad)')
            ax.set_title(label, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True)
        axes3[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        fname = f'pid_simulation_{name}.png'
        plt.savefig(fname, dpi=150)
        print(f'  Saved {fname}')

        # EE path reconstructed from PID-simulated joint angles
        pid_ees = np.array([rf_fk(th) for th in sim_thetas])
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.plot(ees_ik[:, 0],  ees_ik[:, 1],  'b--', lw=2,   label='IK target EE')
        ax4.plot(pid_ees[:, 0], pid_ees[:, 1], 'r-',  lw=1.5, label='PID output EE')
        ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
        ax4.set_title(f'{name.capitalize()} — EE path: IK target vs PID output')
        ax4.set_aspect('equal'); ax4.grid(True); ax4.legend()
        plt.tight_layout()
        fname2 = f'pid_ee_{name}.png'
        plt.savefig(fname2, dpi=150)
        print(f'  Saved {fname2}')

    # ══════════════════════════════════════════════════════════════════════
    # 4. Full drawing sequence — circle then star, stitched together
    # ══════════════════════════════════════════════════════════════════════
    all_pts          = circle_pts + star_pts
    all_thetas, _, _ = solve_path_ik(all_pts)
    all_sim          = simulate_pid_all_joints(all_thetas, dt=DT)
    all_pid_ee       = np.array([rf_fk(th) for th in all_sim])
    n_c              = len(circle_pts)

    fig5, ax5 = plt.subplots(figsize=(7, 7))
    ax5.plot(all_pid_ee[:n_c, 0], all_pid_ee[:n_c, 1], 'b-', lw=2, label='Circle (PID EE)')
    ax5.plot(all_pid_ee[n_c:, 0], all_pid_ee[n_c:, 1], 'r-', lw=2, label='Star (PID EE)')
    ax5.set_xlabel('X (m)'); ax5.set_ylabel('Y (m)')
    ax5.set_title('Full drawing sequence — PID-simulated EE path (floor plane)')
    ax5.set_aspect('equal'); ax5.grid(True); ax5.legend()
    plt.tight_layout()
    plt.savefig('full_sequence_ee.png', dpi=150)
    print('Saved full_sequence_ee.png')

    plt.show()
    print('\nDone.')


if __name__ == '__main__':
    main()
