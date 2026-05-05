   

import sys
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import scipy.optimize
import matplotlib
import importlib
import os

                                                
                              
                                                
                    
                                                             
                                                                     
                                                                     
                                                                
 
                                                                 
                                                                        
                                                                       
                                                                    
                                                                     
                                                                    
                                                  

def _try_backend(name):
                                                                                
    try:
        importlib.import_module(f"matplotlib.backends.backend_{name.lower()}")
        matplotlib.use(name)
        print(f"[INFO] Matplotlib backend selected: {name}", flush=True)
        return True
    except ImportError as e:
        print(f"[DEBUG] Backend {name} unavailable: {e}", flush=True)
        return False
    except Exception as e:
        print(f"[DEBUG] Backend {name} error: {e}", flush=True)
        return False

def _select_backend():
    forced = os.environ.get("MPLBACKEND", "")
    if forced:
        matplotlib.use(forced)
        print(f"[INFO] MPLBACKEND override: {forced}", flush=True)
        return forced
                                                                 
                                                   
    if _try_backend("TkAgg"):
        return "TkAgg"
                                                                       
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "eglfs")
        print("[INFO] No $DISPLAY — setting QT_QPA_PLATFORM=eglfs", flush=True)
    if _try_backend("Qt5Agg"):
        return "Qt5Agg"
                                  
    matplotlib.use("Agg")
    print(
        "[WARNING] No interactive matplotlib backend available.\n"
        "  The drawing window will not appear.\n"
        "  Fix:  sudo apt install python3-tk   (then restart)", flush=True
    )
    return "Agg"

_active_backend = _select_backend()

                                                                          
                                                                            
                                                               
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

                                                
                                   
                                                
                                                

        
CIRCLE_RADIUS      = 0.01                               
CIRCLE_N_POINTS    = 150                        
CIRCLE_LOOP_FACTOR = 1.5                     

                  
STAR_R_OUTER       = 0.03                             
STAR_R_INNER       = 0.01                                
STAR_N_POINTS      = 5                           
STAR_N_TOTAL       = 120                                           
STAR_LOOP_FACTOR   = 1.5                     

                            
CUSTOM_WORKSPACE_RADIUS = max(STAR_R_OUTER * 1.5, 0.04)                       
CUSTOM_N_INTERP    = 200                                            


                                                
                               
                                                

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


                                                
                  
                                                

def generate_circle(center_x, center_y, z_floor, radius=None, n_points=None, loop_factor=None):
                                                                                               
    if radius      is None: radius      = CIRCLE_RADIUS
    if n_points    is None: n_points    = CIRCLE_N_POINTS
    if loop_factor is None: loop_factor = CIRCLE_LOOP_FACTOR
    angles = np.linspace(0, loop_factor * 2 * np.pi, int(loop_factor * n_points), endpoint=False)
    pts = []
    for a in angles:
        pts.append(np.array([
            center_x + radius * np.cos(a),
            center_y + radius * np.sin(a),
            z_floor
        ]))
    return pts


def generate_star(cx, cy, z, r_outer=None, r_inner=None, n_points=None, n_total=None, loop_factor=None):
           
    if r_outer     is None: r_outer     = STAR_R_OUTER
    if r_inner     is None: r_inner     = STAR_R_INNER
    if n_points    is None: n_points    = STAR_N_POINTS
    if n_total     is None: n_total     = STAR_N_TOTAL
    if loop_factor is None: loop_factor = STAR_LOOP_FACTOR
                                                                     
    corners = []
    for i in range(2 * n_points):
        r = r_outer if (i % 2 == 0) else r_inner
        a = np.pi / 2 + i * np.pi / n_points
        corners.append(np.array([cx + r * np.cos(a), cy + r * np.sin(a), z]))
 
    n_edges = len(corners)                       
    pts_per_edge = max(2, n_total // n_edges)
 
    pts = []
    for loop in range(int(loop_factor) + 1):                             
        for i in range(n_edges):
            if loop == int(loop_factor) and i >= n_edges // 2:                               
                break
            start = corners[i]
            end   = corners[(i + 1) % n_edges]
                                                                                   
            for t in np.linspace(0, 1, pts_per_edge, endpoint=False):
                pts.append(start + t * (end - start))
    return pts


                                                
                            
                                                

def capture_custom_path(center_x, center_y, z_floor,
                        workspace_radius=0.05, n_interp=200):
           
                                    
    state = {'points': [], 'confirmed': False, 'cancelled': False}

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title('Lyra — Draw Custom Path')

                        
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
        if event.button == 1:                          
            x, y = event.xdata, event.ydata
                                       
            dx, dy = x - center_x, y - center_y
            dist = np.hypot(dx, dy)
            if dist > workspace_radius:
                scale = workspace_radius / dist
                x = center_x + dx * scale
                y = center_y + dy * scale
            state['points'].append((x, y))
            _redraw()
        elif event.button == 3:                        
            if len(state['points']) >= 2:
                state['confirmed'] = True
                plt.close(fig)
            else:
                status_text.set_text('Need at least 2 points!')
                fig.canvas.draw_idle()
        elif event.button == 2:                      
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
    plt.show(block=True)                                      

    if state['cancelled'] or not state['confirmed'] or len(state['points']) < 2:
        print('[custom path] Cancelled or too few points — aborting.', flush=True)
        return None

                                                                              
    raw = np.array(state['points'])                 

                                                      
    diffs = np.diff(raw, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        print('[custom path] Path is too short — aborting.', flush=True)
        return None

                                 
    t_uniform = np.linspace(0, total_len, n_interp)
    x_interp = np.interp(t_uniform, cum_len, raw[:, 0])
    y_interp = np.interp(t_uniform, cum_len, raw[:, 1])

    waypoints = [np.array([x_interp[i], y_interp[i], z_floor])
                 for i in range(n_interp)]

    print(f'[custom path] {len(waypoints)} waypoints generated '
          f'(total path length ≈ {total_len*100:.1f} cm).', flush=True)
    return waypoints


                                                
                                               
                                                

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
        self._int_pos = 0.0
        self._int_vel = 0.0
        self._int_trq = 0.0
        self._prev_pos_err = 0.0
        self._prev_vel_err = 0.0
        self._prev_trq_err = 0.0

    def update(self, desired_angle: float, current_angle: float, current_velocity: float):
                   
        dt = self.dt

                                                                            
        pos_err = desired_angle - current_angle
        self._int_pos += pos_err * dt
        d_pos_err = (pos_err - self._prev_pos_err) / dt
        self._prev_pos_err = pos_err

        desired_vel = (self.kp_pos * pos_err
                       + self.ki_pos * self._int_pos
                       + self.kd_pos * d_pos_err)
        desired_vel = np.clip(desired_vel, -self.max_vel, self.max_vel)

                                                                             
        vel_err = desired_vel - current_velocity
        self._int_vel += vel_err * dt
        d_vel_err = (vel_err - self._prev_vel_err) / dt
        self._prev_vel_err = vel_err

        desired_trq = (self.kp_vel * vel_err
                       + self.ki_vel * self._int_vel
                       + self.kd_vel * d_vel_err)
        desired_trq = np.clip(desired_trq, -self.max_trq, self.max_trq)

                                                                            
                                                                               
                                                                      
        trq_err = desired_trq
        self._int_trq += trq_err * dt
        d_trq_err = (trq_err - self._prev_trq_err) / dt
        self._prev_trq_err = trq_err

        delta = (self.kp_trq * trq_err
                 + self.ki_trq * self._int_trq
                 + self.kd_trq * d_trq_err)
        delta = np.clip(delta, -self.max_delta, self.max_delta)

        return delta


                                                
                 
                                                

class PupperArt(Node):

                                                                            


    STAND_ANGLES_LF = np.array([-1.72766, -0.44671, +2.51156])
    STAND_ANGLES_RB = np.array([+1.66891, +1.72504, -2.31167])
    STAND_ANGLES_LB = np.array([-1.60368, -1.68193, +2.64584])

                                                               
                                                               
                                                               


              
    STAND_ANGLES_RF_HOME = np.array([+1.04291, -0.39978, +0.69436])


                                                                          
    SERVO_KP      = 5.0                                                          
    SERVO_KD      = 0.2                 
    RAMP_UP_SECS  = 3.0                                                

                                                                              
    PEN_Z = -0.14

                                        
    PEN_UP_Z = -0.10

                                             
    RF_CENTER_X =  0.06
    RF_CENTER_Y = -0.09

                                  
    DRAW_FREQ       = 50.0                                 
    CTRL_FREQ       = 200.0                            

    def __init__(self):
        super().__init__('pupper_art')

                                                                             
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self._joint_cb, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/forward_command_controller/commands', 10)
        self.kp_pub  = self.create_publisher(
            Float64MultiArray, '/forward_kp_controller/commands', 10)
        self.kd_pub  = self.create_publisher(
            Float64MultiArray, '/forward_kd_controller/commands', 10)

                                                                             
        self.joint_positions  = None
        self.joint_velocities = None

                                                                
        self.cmd = np.zeros(12)

                                                                    
        self.target_rf = np.array(self.STAND_ANGLES_RF_HOME)

                                          
        dt_pid = 1.0 / self.CTRL_FREQ
        self.pids = [CascadedPID(dt=dt_pid) for _ in range(3)]

                                                                             
        self.current_shape = None
        self.shape_name = None
        self.current_wp_idx = 0
        self.desired_positions = []
        self.actual_positions = []

                                                                     
                                                   
                                                                   
        self.phase             = 'idle'                           
        self.phase_counter     = 0
        self.RAMP_TICKS        = int(self.RAMP_UP_SECS * self.CTRL_FREQ)
        self.STAND_TICKS       = int(2.0 * self.CTRL_FREQ)
        self.PEN_DOWN_TICKS    = int(1.0 * self.CTRL_FREQ)
        self.TRANSITION_TICKS  = int(1.0 * self.CTRL_FREQ)                                    

        self.draw_wp_counter   = 0

                                                                               
        self._custom_path_request = threading.Event()

                                                                             
        self.ctrl_timer = self.create_timer(1.0 / self.CTRL_FREQ, self._ctrl_cb)

                                                                             
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
            self.current_shape = generate_circle(cx, cy, self.PEN_Z)
        elif shape_name == 'star':
            self.current_shape = generate_star(cx, cy, self.PEN_Z)
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
                                                                
                                                                                
                                                                                     
                    self._custom_path_request.set()
                elif self.phase == 'idle':
                    print('[press s to stand first]', flush=True)
                else:
                    print(f'[not ready — phase: {self.phase}]', flush=True)
            elif key == 'q':
                print('[RELAXING AND QUITTING]', flush=True)
                self.phase = 'relax_quit'

                                                                            
                                      
                                                                            

    @staticmethod
    def _rf_fk(theta):
        T01 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T12 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T23 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T3e = translation(0.06231, -0.06216, 0.01800)
        return (T01 @ T12 @ T23 @ T3e)[:3, 3]

                                                                            
                                            
                                                                            

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

                                                                            
                           
                                                                            

    def _current_waypoint(self):
        return self.current_shape[self.current_wp_idx]

    def _advance_waypoint(self):
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(self.current_shape):
                                 
            self.get_logger().info(f'Finished drawing {self.shape_name}!')
            self.phase = 'pen_up'
            self.phase_counter = 0

    def _plot_trajectory(self):
        if not self.desired_positions or not self.actual_positions:
            return
        desired = np.array(self.desired_positions)
        actual = np.array(self.actual_positions)
        
              
        rmse = np.sqrt(np.mean((desired - actual)**2, axis=0))
        print(f'RMSE for {self.shape_name}: X={rmse[0]:.4f}, Y={rmse[1]:.4f}, Z={rmse[2]:.4f}')
        
                                
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
                                                 
                                                                            

    def _ctrl_cb(self):
        if self.joint_positions is None:
            return

        pos = self.joint_positions
        vel = self.joint_velocities

        rf_pos = pos[0:3]
        rf_vel = vel[0:3]

                                                                             
        CTRL_PER_WP = max(1, int(self.CTRL_FREQ / self.DRAW_FREQ))

        if self.phase == 'idle':
            return                    

        elif self.phase == 'stand_up':
                                                                       
                                                                                
                                                                               
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
            return                                         

        elif self.phase == 'stand':
                                                                     
                                                                  
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
            return                                         

        elif self.phase == 'standing_hold':
                                                                          
            stand_cmd = np.concatenate([
                self.STAND_ANGLES_RF_HOME,
                self.STAND_ANGLES_LF,
                self.STAND_ANGLES_RB,
                self.STAND_ANGLES_LB,
            ])
            self.cmd_pub.publish(Float64MultiArray(data=stand_cmd.tolist()))
            return

        elif self.phase == 'move_to_start':
                                                                                
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

                                                                            
        rf_cmd = np.zeros(3)
        for j in range(3):
            delta = self.pids[j].update(
                desired_angle=self.target_rf[j],
                current_angle=rf_pos[j],
                current_velocity=rf_vel[j]
            )
            rf_cmd[j] = self.target_rf[j] + delta

                                                                            
        lf_cmd = self.STAND_ANGLES_LF
        rb_cmd = self.STAND_ANGLES_RB
        lb_cmd = self.STAND_ANGLES_LB

                                                                            
        self.cmd = np.concatenate([rf_cmd, lf_cmd, rb_cmd, lb_cmd])
        msg = Float64MultiArray()
        msg.data = self.cmd.tolist()
        self.cmd_pub.publish(msg)

                                                                            
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


                                                
             
                                                

def main():
    rclpy.init()
    node = PupperArt()

    if matplotlib.get_backend() == 'Agg':
        print(
            '[WARNING] Non-interactive backend — drawing window will not appear.\n'
            '  Fix:  sudo apt install python3-tk   (then restart)', flush=True
        )

    try:
                                                                            
                                                                                
                                                                          
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
            if node._custom_path_request.is_set():
                node._custom_path_request.clear()
                waypoints = capture_custom_path(
                    center_x=node.RF_CENTER_X,
                    center_y=node.RF_CENTER_Y,
                    z_floor=node.PEN_Z,
                    workspace_radius=CUSTOM_WORKSPACE_RADIUS,
                    n_interp=CUSTOM_N_INTERP,
                )
                if waypoints is not None:
                    node._start_drawing('custom', custom_waypoints=waypoints)
                else:
                    print('[custom path cancelled — still in standing_hold]', flush=True)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()