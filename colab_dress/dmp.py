from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.interpolate import make_smoothing_spline
except Exception:  # pragma: no cover - older SciPy fallback
    make_smoothing_spline = None
from scipy.interpolate import UnivariateSpline
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Bool
    from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker
except Exception:  # pragma: no cover - ROS 2 not available in all contexts
    rclpy = None
    Node = object
    class _RosMsgStub:
        pass

    PoseArray = Pose = PointStamped = PoseStamped = Path = Marker = String = Bool = _RosMsgStub

class CanonicalSystem(object):
    """
    A class to represent a canonical system with a state vector and a time step.
    The state vector is initialized to zero and can be updated with a new state.
    """

    def __init__(self, ax=1.0, dt=0.01, pattern="exp", tau_y=0.6, tau=1.0, T=1.5, av=4.0, v_max=2.0):
        """
        Initializes the CanonicalSystem with parameters for the system dynamics.
        :param ax: Decay rate for the state.
        :param dt: Time step for the simulation.
        :param pattern: Type of step function, either "exp" for exponential or "discrete" for discrete.
        :param tau_y: Time constant for the state update.
        :param tau: Time constant for the system dynamics.
        :param T: Total time for the simulation.
        :param av: Coefficient for velocity dynamics.
        :param v_max: Maximum velocity.
        """
        self.av = av
        self.v_max = v_max
        self.s = 1.0
        self.ax = ax
        self.time_step = 0.0
        self.dt = dt
        self.pattern = pattern
        self.tau_y = tau_y
        self.tau = tau
        self.timesteps = int(T / dt)
        self.run_time = T
        self.v = 1.0

        self.dv = 0.0
        self.ds = 0.0
        self.v_max = 2.0
        if self.pattern == "exp":
            self.step = self.step_exp
        else:
            self.step = self.step_discrete

    def reset(self):
        if self.pattern == "exp":
            self.s = 0.0
            self.v = 1.0
        else:
            self.s = 1.0
            self.v = 1.0

    def step_exp(self):
        """
        Make a step in the system.
        """
        self.s += self.ds * self.dt * self.tau

        if self.s < 1.0:
            self.ds = 1 / self.tau_y
        else:
            self.ds = 0.0
        return self.s
    def step_v(self):
        self.v += self.dv * self.dt

        self.dv = - self.av * self.v * ( 1 - self.v / self.v_max)
        return self.v


    def step_discrete(self):
        self.s += (-self.ax * self.s) * self.tau * self.dt
        return self.s

    def rollout(self):
        self.s_track = np.zeros(self.timesteps)
        self.reset()

        for t in range(self.timesteps):
            self.s_track[t] = self.s
            self.step()
        return self.s_track.reshape(-1,1)
    
    def rollout_v(self):
        self.v_track = np.zeros(self.timesteps)
        self.reset()

        for t in range(self.timesteps):
            self.v_track[t] = self.step_v()
        return self.v_track.reshape(-1,1)

    def __test(self):
        """
        A simple test to visualize the system's behavior.
        """
        s_track = self.rollout()
        v_track = self.rollout_v()
        plt.plot(np.arange(self.timesteps) * self.dt, s_track)
        plt.plot(np.arange(self.timesteps) * self.dt, v_track)
        plt.legend(['s(t)', 'v(t)'])
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.title('Canonical System State Over Time')
        plt.grid()
        plt.show()
# CanonicalSystem()._CanonicalSystem__test()  # Run the test method to visualize the system's behavior.
# CanonicalSystem(pattern="discrete")._CanonicalSystem__test()

class DMPDG(object):
    """
    A class to represent a DMPDG (Dynamic Multi-Point Decision Graph) object.
    This class is a placeholder for future implementation.
    """

    def __init__(self, 
                 n_dmps=1, 
                 n_bfs=20, 
                 dt=0.01, 
                 ay=25.0, 
                 ax=1, 
                 tau_y=0.6, 
                 tau=1.0,
                 v_max=2.0, 
                 ag=6.0, 
                 av=4.0,
                 y0=1.0,
                 g=0.5,
                 pattern="exp",
                 dmp_type="vanilla",
                 T=1.5,
                 dim=None,
                 ):
        """
        Initialize the DMPDG object with optional arguments.
        """
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        self.ax = ax
        self.ay = ay
        self.by = self.ay / 4
        self.tau_y = tau_y
        self.tau = tau
        self.pattern = pattern
        self.cs = CanonicalSystem(ax=self.ax, 
                                  dt=self.dt, 
                                  tau_y=self.tau_y, 
                                  tau=self.tau, 
                                  pattern=self.pattern,
                                  T=T)

        # self.set_c()
        self.imitate = False
        self.gen_centers()
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax
        # self.set_h()

        self.v_max = v_max
        self.ag = ag
        self.av = av
        self.y0 = y0
        self.goal = g
        self.f = np.zeros(self.cs.timesteps)
        self.dmp_type = dmp_type

        if self.dmp_type == "vanilla":
            self.step = self.step_vanilla
        elif self.dmp_type == "delayed":
            self.step = self.step_dg
        self.dim = dim if dim is not None else n_dmps

    def __str__(self):
        """
        Return a string representation of the DMPDG object.
        """
        return "DMPDG Object"
    
    def __repr__(self):
        """
        Return a detailed string representation of the DMPDG object.
        """
        return f"DMPDG(n_dmps={self.n_dmps}, n_bfs={self.n_bfs}, dt={self.dt})"
    
    def set_h(self):
        self.h = np.zeros(self.n_bfs)
        for i in range(self.n_bfs - 1):
            self.h[i] = 1 / (self.c[i] - self.c[i + 1]) ** 2
    
    def set_c(self):
        self.c = np.zeros(self.n_bfs)
        for i in range(1, self.n_bfs + 1):
            self.c[i - 1] = np.exp(-self.ax *((i -1)/ (self.n_bfs -1)))
    
    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]"""

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            c[n] = np.exp(-self.cs.ax * des_c[n])

        self.c = c.copy()


    def psi(self, s):
        return np.exp(-self.h * ((s - self.c) ** 2))
    
    def get_f_target(self, y_des: np.ndarray, dy_des: np.ndarray, ddy_des: np.ndarray) -> np.ndarray:
        """
        Calculate the target force based on the desired trajectory.
        """
        # y_des, dy_des, ddy_des = self.imitate_trajectory(y_des)

        # f_target = np.zeros((self.n_dmps, self.n_bfs))
        if self.dmp_type == "delayed":
            v = self.cs.rollout_v()
            goal_d = self.rollout_goal_d()
            f_target = ddy_des - self.ay * (self.by * (goal_d - y_des) - dy_des) #/ v
            f_target = f_target / v  # Remove problematic reshape
        elif self.dmp_type == "vanilla":
            f_target = ddy_des - self.ay*(self.by*(self.goal - y_des) - dy_des)
        return f_target

    def imitate_trajectory(self, y_des: np.ndarray):
        self.imitate = True
    
        self.y0 = y_des[0].copy()
        print(f"y0: {self.y0}")
        self.goal = self.goal_d= y_des[-1].copy()

        x = np.linspace([0], self.cs.run_time, y_des.shape[0])
        x_new = np.linspace(0, self.cs.run_time, self.cs.timesteps)

        y_des_smooth = np.empty((self.cs.timesteps, y_des.shape[1]))
        dy_des_smooth = np.empty((self.cs.timesteps, y_des.shape[1]))
        ddy_des_smooth = np.empty((self.cs.timesteps, y_des.shape[1]))

        for i in range(y_des.shape[1]):
            y_des_smooth[:, i] = _smooth_trajectory(x.flatten(), y_des[:, i], x_new)
            dy_des_smooth[:, i] = np.gradient(y_des_smooth[:, i]) / self.dt
            ddy_des_smooth[:, i] = np.gradient(dy_des_smooth[:, i]) / self.dt

        self.f_target = self.get_f_target(y_des_smooth, dy_des_smooth, ddy_des_smooth)
        print(f"f_target shape: {self.f_target.shape}")
        self.w = self.gen_weights(self.f_target)
        self.f = self.forcing_term()
        self.reset()
        self.cs.reset()
        return y_des_smooth, dy_des_smooth, ddy_des_smooth
    
    # def gen_weights(self, f_target):
    #     s = self.cs.rollout()
    #     Psi = self.psi(s)
    #     print(Psi.shape, f_target.shape)
    #     weights =  (np.linalg.pinv(Psi) @ f_target) / Psi.sum(axis=0, keepdims=True).T
    #     return weights
    
    def get_phi_inv(self):
        # Normalize basis functions and multiply by s (phase)
        s = self.cs.rollout().reshape(-1, 1)  # [K, 1]
        psi_track = self.psi(s)
        basis_matrix = psi_track / np.sum(psi_track, axis=1, keepdims=True)  # normalized
        Phi = basis_matrix * s # [K, N]
        return np.linalg.pinv(Phi)

    # def gen_weights(self, f_target):
    #     # Normalize basis functions and multiply by s (phase)
    #     s = self.cs.rollout().reshape(-1, 1)  # [K, 1]
    #     psi_track = self.psi(s)
    #     basis_matrix = psi_track / np.sum(psi_track, axis=1, keepdims=True)  # normalized
    #     Phi = basis_matrix * s # [K, N]
    #     # weights = np.linalg.pinv(Phi) @ dmp.f_target  # [N, 1]
    #     weights = np.linalg.pinv(Phi) @ f_target  # [N, 1]
    #     return weights
    
    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.psi(x_track)
        # efficiently calculate BF weights using weighted linear regression
        weights = np.zeros((self.n_bfs, self.dim))
        # self.w_ =  torch.tensor((np.linalg.inv(psi_track.T @ psi_track) @ psi_track.T )@ f_target)
        # spatial scaling term
        k = self.goal - self.y0
        for i in range(self.dim):
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, None, b] * f_target[:, None, i])
                denom = np.sum(x_track ** 2 * psi_track[:, None, b])
                weights[b, i] = numer / denom
                if abs(k[i]) > 1e-5:
                    # print(i)
                    weights[b, i] /= k[i]
        
        return weights

    def reset(self):
        self.z = np.zeros(self.dim)  # reset z to zero
        if isinstance(self.y0, np.ndarray):
            self.y = self.y0.copy()
        else: 
            self.y = self.y0
        self.dy = np.zeros(self.dim)  # reset dy to zero
        self.dz = np.zeros(self.dim)  # reset dz to zero
        self.ddy = np.zeros(self.dim)  # reset ddy to zero
        if isinstance(self.goal, np.ndarray):
            self.goal_d = self.y0.copy()
        else:
            self.goal_d = self.y0
        self.dgoal_d = np.zeros(self.dim)  # reset dgoal_d to zero
        self.v = np.ones(self.dim)  # reset v to ones
        self.dv = np.zeros(self.dim)  # reset dv to zero
        self.cs.reset()
        
    def step_vanilla(self, i=0):
        if self.imitate:
            x = self.cs.step()
            psi = self.psi(x)
            f = np.dot(psi, self.w) * x * (self.goal - self.y0) / (np.sum(psi))
            # print(f)
            # f = f[0]
            self.f_[i] = f
        else:
            f = 0
        self.dz = (self.ay * (self.by * (self.goal - self.y) - self.z) + f) / self.tau_y

        self.dy = self.z / self.tau_y

        self.z = self.z + self.dz * self.dt
        self.y = self.y + self.dy * self.dt
        
        return self.y, self.dy, self.dz / self.tau_y
    
    def set_goal(self, g):
        self.goal = g

    def step_dg(self, i=0):
        if self.imitate:
            x = self.cs.step()
            psi = self.psi(x)
            f = np.dot(psi, self.w) * x * (self.goal - self.y0)
            sum_psi = np.sum(psi)
            if np.abs(sum_psi) > 1e-6:
                f /= sum_psi
            # print(f)
            # f = f[0]
            # print(i, x, f)
            self.f_[i] = f
        else:
            f = 0

        self.v += self.dv * self.dt
        self.goal_d = self.goal_d + self.dgoal_d * self.dt
        self.z = self.z + self.dz * self.dt
        self.y = self.y + self.dy * self.dt
        
        self.dz = (self.ay * (self.by * (self.goal_d - self.y) - self.z) + self.v * f) / self.tau_y

        self.dy = self.z / self.tau_y
        self.dgoal_d = self.ag * (self.goal - self.goal_d) / self.tau_y
        self.dv = -self.av * self.v * (1 - (self.v/ self.v_max))
        
        
        return self.y, self.dy, self.dz / self.tau_y

    def step_kinova(self,):
        y, _, _ = self.step()
        return y - self.y0
    
    def forcing_term(self):
        Psi = self.psi(self.cs.rollout())
        sum_psi = (Psi.sum(axis=1, keepdims=True))
        sum_psi[np.where(abs(sum_psi) < 1e-6)[0]] = 1.0
        return ((Psi @ self.w) * self.cs.rollout()) #/ sum_psi

    def rollout_goal_d(self):
        goal_d = self.y0.copy()
        goal_d_rollout = np.zeros((self.cs.timesteps, self.dim))
        dgoal_d = np.zeros(self.dim)
        for i in range(self.cs.timesteps):
            goal_d  += dgoal_d * self.dt
            goal_d_rollout[i] = goal_d
            dgoal_d = self.ag * (self.goal - goal_d)
        return goal_d_rollout
    
    def rollout(self):
        """
        Generate a rollout of the DMPDG system based on a desired trajectory.
        Args:
            y_des (np.ndarray): Desired trajectory to follow.   
        """
        y_rollout = np.zeros((self.cs.timesteps, self.dim))
        dy_rollout = np.zeros((self.cs.timesteps, self.dim))
        ddy_rollout = np.zeros((self.cs.timesteps, self.dim))
        self.f_ = np.empty_like(self.f)
        self.reset()
        if self.dmp_type == "delayed":
            self.goal_d_rollout = np.zeros((self.cs.timesteps, self.dim))
            self.d_goal_d_rollout = np.zeros((self.cs.timesteps, self.dim))
        for t in range(self.cs.timesteps):
            y, dy, ddy = self.step(i=t)
            y_rollout[t] = y
            dy_rollout[t] = dy
            ddy_rollout[t] = ddy
            if self.dmp_type == "delayed":
                # print(t, self.goal_d)
                # print(self.goal_d_rollout[t], self.goal_d)

                self.goal_d_rollout[t] = self.goal_d
                self.d_goal_d_rollout[t] = self.dgoal_d
        self.cs.reset()
        return y_rollout, dy_rollout, ddy_rollout
    
    def __test(self):
        """
        A simple test to visualize the DMPDG system's behavior.
        """
        y, dy, ddy = self.rollout()
        fig , ax = plt.subplots(1, 3, figsize=(15, 5))
        t = np.arange(len(y)) * self.dt
        ax[0].plot(np.arange(len(y)) * self.dt, y, label='y')
        ax[0].set_title('y')
        ax[1].plot(np.arange(len(dy)) * self.dt, dy, label='dy', color='orange')
        ax[1].set_title('dy')
        ax[2].plot(np.arange(len(ddy)) * self.dt, ddy, label='ddy', color='green')
        ax[2].set_title('ddy')
        if self.dmp_type == "delayed":
            ax[0].plot(t, self.goal_d_rollout, '--')
            ax[1].plot(t, self.d_goal_d_rollout, '--')
        for a in ax:
            a.grid()
            a.legend()
            a.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t, self.psi(self.cs.rollout()), c="blue", linewidth=0.5)
        ax.plot(t, self.cs.rollout(), label='s(t)', linestyle='--')
        ax.plot(t, self.cs.rollout_v(), label='v(t)', linestyle=':')
        ax.set_title('Basis Functions')
        ax.set_xlabel('s')
        ax.set_ylabel('Activation')
        ax.grid()
        ax.legend()
        plt.tight_layout()
            
# dmp = DMPDG(n_dmps=1, n_bfs=20, dt=0.01, tau_y=.60, y0=1.0, g=0.5)
# DMPDG(pattern="discrete", dmp_type="delayed", T=1.0, dt=0.01, tau_y=.60, tau=1.0)._DMPDG__test()  # Run the test method to visualize the DMPDG system's behavior.


class DMPNode(Node):
    """
    ROS 2 node that learns a DMP from arm pose demonstrations and
    re-rolls out trajectories when the shoulder position is updated.
    """

    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run DMPNode")
        super().__init__("dmp_node")

        self.declare_parameter("n_bfs", 20)
        self.declare_parameter("dt", 0.01)
        self.declare_parameter("T", 1.5)
        self.declare_parameter("dmp_type", "delayed")
        self.declare_parameter("ay", 25.0)
        self.declare_parameter("ax", 1.0)
        self.declare_parameter("tau_y", 0.6)
        self.declare_parameter("tau", 1.0)
        self.declare_parameter("v_max", 2.0)
        self.declare_parameter("ag", 6.0)
        self.declare_parameter("av", 4.0)
        self.declare_parameter("n_dmps", 3)
        self.declare_parameter("rollout_path_topic", "/dmp/dmp_rollout_path")
        self.declare_parameter("publish_period", 1.0)
        self.declare_parameter("rollout_stride", 10)
        self.declare_parameter("trajectory_status_topic", "/dmp/trajectory_status")
        self.declare_parameter("shoulder_update_flag_topic", "/dmp/shoulder_update_enabled")

        n_bfs = int(self.get_parameter("n_bfs").value)
        dt = float(self.get_parameter("dt").value)
        T = float(self.get_parameter("T").value)
        dmp_type = str(self.get_parameter("dmp_type").value)
        ay = float(self.get_parameter("ay").value)
        ax = float(self.get_parameter("ax").value)
        tau_y = float(self.get_parameter("tau_y").value)
        tau = float(self.get_parameter("tau").value)
        v_max = float(self.get_parameter("v_max").value)
        ag = float(self.get_parameter("ag").value)
        av = float(self.get_parameter("av").value)
        n_dmps = int(self.get_parameter("n_dmps").value)
        rollout_path_topic = str(self.get_parameter("rollout_path_topic").value)
        publish_period = float(self.get_parameter("publish_period").value)
        rollout_stride = int(self.get_parameter("rollout_stride").value)
        trajectory_status_topic = str(self.get_parameter("trajectory_status_topic").value)
        shoulder_update_flag_topic = str(
            self.get_parameter("shoulder_update_flag_topic").value
        )

        self.dmp = DMPDG(
            n_dmps=n_dmps,
            n_bfs=n_bfs,
            dt=dt,
            ay=ay,
            ax=ax,
            tau_y=tau_y,
            tau=tau,
            v_max=v_max,
            ag=ag,
            av=av,
            y0=np.zeros(n_dmps),
            g=np.zeros(n_dmps),
            dmp_type=dmp_type,
            T=T,
            dim=n_dmps,
        )

        self.last_shoulder: Optional[np.ndarray] = None
        self.goal_offset: Optional[np.ndarray] = None
        self.last_frame_id: str = ""
        self.rollout_path_topic = rollout_path_topic
        self.rollout_stride = max(1, rollout_stride)
        self.shoulder_update_enabled: bool = True
        self.active_rollout: Optional[np.ndarray] = None
        self.active_frame_id: str = ""
        self.rollout_index: int = 0

        self.arm_sub = self.create_subscription(
            PoseArray, "dmp/arm_poses", self._arm_poses_cb, 10
        )
        self.shoulder_sub = self.create_subscription(
            PointStamped, "dmp/shoulder_position", self._shoulder_cb, 10
        )
        self.shoulder_flag_sub = self.create_subscription(
            Bool, shoulder_update_flag_topic, self._shoulder_flag_cb, 10
        )

        self.rollout_pub = self.create_publisher(PoseArray, "/cartesian_trajectory", 10)
        self.rollout_path_pub = self.create_publisher(Path, self.rollout_path_topic, 10)
        self.start_pub = self.create_publisher(Marker, "/dmp/start_point", 10)
        self.goal_pub = self.create_publisher(Marker, "/dmp/goal_point", 10)
        self.current_pub = self.create_publisher(Marker, "/dmp/current_waypoint", 10)
        self.status_pub = self.create_publisher(String, trajectory_status_topic, 10)
        timer_period = publish_period if publish_period > 0.0 else 0.5
        self.publish_timer = self.create_timer(timer_period, self._publish_next_rollout_point)
        self._publish_status("idle")
        self.get_logger().info(
            "DMP node ready: waiting for arm_poses and shoulder_position."
        )

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _shoulder_flag_cb(self, msg) -> None:
        prev_enabled = self.shoulder_update_enabled
        self.shoulder_update_enabled = bool(msg.data)
        state = "enabled" if self.shoulder_update_enabled else "disabled"
        self.get_logger().info(f"Shoulder updates {state}.")
        if self.active_rollout is not None and prev_enabled != self.shoulder_update_enabled:
            self._publish_status("active" if self.shoulder_update_enabled else "paused")

    def _arm_poses_cb(self, msg) -> None:
        if not msg.poses:
            self.get_logger().warning("Received empty arm_poses.")
            return

        self.last_frame_id = msg.header.frame_id
        y_des = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
        if y_des.shape[0] < 2:
            self.get_logger().warning("Need at least 2 poses to learn a trajectory.")
            return

        self.dmp.imitate_trajectory(y_des)

        if self.last_shoulder is not None:
            self.goal_offset = self.dmp.goal - self.last_shoulder
        else:
            self.goal_offset = None

        y_rollout, _, _ = self.dmp.rollout()
        self._set_active_rollout(y_rollout, self.last_frame_id, preserve_progress=False)

    def _shoulder_cb(self, msg) -> None:
        self.last_shoulder = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        if not self.shoulder_update_enabled:
            return
        if not self.dmp.imitate:
            return

        # If shoulder arrives after the demo was learned, initialize the offset
        # reference first; subsequent shoulder updates will trigger replanning.
        if self.goal_offset is None:
            self.goal_offset = self.dmp.goal - self.last_shoulder
            self.get_logger().info("Initialized shoulder-goal offset reference.")
            return

        new_goal = self.last_shoulder + self.goal_offset
        goal_shift = np.linalg.norm(new_goal - self.dmp.goal)
        if goal_shift < 1e-6:
            return

        self.dmp.set_goal(new_goal)
        y_rollout, _, _ = self.dmp.rollout()
        frame_id = msg.header.frame_id or self.last_frame_id
        self.get_logger().info(
            f"Shoulder update received; replanning rollout (goal shift: {goal_shift:.4f} m)"
        )
        self._set_active_rollout(y_rollout, frame_id, preserve_progress=True)

    def _set_active_rollout(
        self, y_rollout: np.ndarray, frame_id: str, preserve_progress: bool
    ) -> None:
        if y_rollout.size == 0:
            self.get_logger().warning("Received empty rollout; nothing to stream.")
            return

        y_rollout = y_rollout[:: self.rollout_stride]
        if y_rollout.size == 0:
            self.get_logger().warning("Downsampled rollout is empty; nothing to stream.")
            return

        prev_index = self.rollout_index
        self.active_rollout = y_rollout
        self.active_frame_id = frame_id
        if preserve_progress:
            self.rollout_index = min(prev_index, max(len(y_rollout) - 1, 0))
            self.get_logger().info(
                f"Shoulder moved: updated trajectory from waypoint {self.rollout_index + 1}/{len(y_rollout)}"
            )
            self._publish_status("updated")
        else:
            self.rollout_index = 0
            self.get_logger().info(
                f"Loaded rollout with {len(y_rollout)} points; streaming started."
            )
            self._publish_status("active")

        self._publish_rollout_visualization(y_rollout, frame_id, self.rollout_index)

    def _publish_rollout_visualization(
        self, y_rollout: np.ndarray, frame_id: str, start_index: int = 0
    ) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        path_msg = Path()
        path_msg.header = msg.header
        path_msg.poses = []

        start_index = int(max(0, min(start_index, len(y_rollout) - 1)))
        viz_rollout = y_rollout[start_index:]

        # start / goal markers
        start_marker = Marker()
        start_marker.header = msg.header
        start_marker.header.frame_id = "fr3_link0"  ## Header needs to be this for RViz visualization to work , do not change.
        start_marker.ns = "dmp_start"
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = float(viz_rollout[0, 0])
        start_marker.pose.position.y = float(viz_rollout[0, 1])
        start_marker.pose.position.z = float(viz_rollout[0, 2])
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.05
        start_marker.scale.y = 0.05
        start_marker.scale.z = 0.05
        start_marker.color.a = 1.0
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        self.start_pub.publish(start_marker)

        goal_marker = Marker()
        goal_marker.header = msg.header
        goal_marker.ns = "dmp_goal"
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(viz_rollout[-1, 0])
        goal_marker.pose.position.y = float(viz_rollout[-1, 1])
        goal_marker.pose.position.z = float(viz_rollout[-1, 2])
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.05
        goal_marker.scale.y = 0.05
        goal_marker.scale.z = 0.05
        goal_marker.color.a = 1.0
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        self.goal_pub.publish(goal_marker)

        current_marker = Marker()
        current_marker.header = msg.header
        current_marker.header.frame_id = "fr3_link0"  ## Header needs to be this for RViz visualization to work , do not change.
        current_marker.ns = "dmp_current"
        current_marker.id = 2
        current_marker.type = Marker.SPHERE
        current_marker.action = Marker.ADD
        current_marker.pose.position.x = float(viz_rollout[0, 0])
        current_marker.pose.position.y = float(viz_rollout[0, 1])
        current_marker.pose.position.z = float(viz_rollout[0, 2])
        current_marker.pose.orientation.w = 1.0
        current_marker.scale.x = 0.04
        current_marker.scale.y = 0.04
        current_marker.scale.z = 0.04
        current_marker.color.a = 1.0
        current_marker.color.r = 0.0
        current_marker.color.g = 0.4
        current_marker.color.b = 1.0
        self.current_pub.publish(current_marker)

        for pt in viz_rollout:
            pose = Pose()
            pose.position.x = float(pt[0])
            pose.position.y = float(pt[1])
            pose.position.z = float(pt[2])
            pose.orientation.x = -0.7071
            pose.orientation.y = 0.7071
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = pose
            path_msg.poses.append(pose_stamped)

        self.rollout_path_pub.publish(path_msg)

    def _publish_next_rollout_point(self) -> None:
        if self.active_rollout is None:
            return
        if not self.shoulder_update_enabled:
            return
        if self.rollout_index >= len(self.active_rollout):
            self.get_logger().info("Finished streaming rollout trajectory.")
            self.active_rollout = None
            self.rollout_index = 0
            self._publish_status("completed")
            return

        pt = self.active_rollout[self.rollout_index]
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = self.active_frame_id
        msg.header.frame_id = "fr3_link0" ## Header needs to be this for RViz visualization to work , do not change.
        pose = Pose()
        pose.position.x = float(pt[0])
        pose.position.y = float(pt[1])
        pose.position.z = float(pt[2])
        pose.orientation.x = -0.7071
        pose.orientation.y = 0.7071
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0
        msg.poses = [pose]
        self.rollout_pub.publish(msg)
        current_marker = Marker()
        current_marker.header = msg.header
        current_marker.ns = "dmp_current"
        current_marker.id = 2
        current_marker.type = Marker.SPHERE
        current_marker.action = Marker.ADD
        current_marker.pose.position.x = float(pt[0])
        current_marker.pose.position.y = float(pt[1])
        current_marker.pose.position.z = float(pt[2])
        current_marker.pose.orientation.w = 1.0
        current_marker.scale.x = 0.04
        current_marker.scale.y = 0.04
        current_marker.scale.z = 0.04
        current_marker.color.a = 1.0
        current_marker.color.r = 0.0
        current_marker.color.g = 0.4
        current_marker.color.b = 1.0
        self.current_pub.publish(current_marker)
        self.rollout_index += 1
        if self.active_rollout is not None:
            self._publish_rollout_visualization(
                self.active_rollout, self.active_frame_id, self.rollout_index
            )


def main() -> None:
    if rclpy is None:
        raise RuntimeError("rclpy is required to run DMPNode")
    rclpy.init()
    node = DMPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def make_arm_trajectory(points):
    lins = []
    for i in range(points.shape[0] - 1):
        lins.append(np.linspace(points[i], points[i+1], 50))
    traj0 = np.vstack(lins)
    # traj0 = np.vstack([np.linspace(points[0], points[1], 50),
    #      np.linspace(points[1], points[2], 50)])
    new_traj0 = np.empty_like(traj0)
    x = np.linspace(0, 1, traj0.shape[0])
    for i in range(3):
          new_traj0[:, i] = _smooth_trajectory(x, traj0[:, i], np.linspace(0, 1, traj0.shape[0]))
    return new_traj0


def _smooth_trajectory(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    if make_smoothing_spline is not None:
        spl = make_smoothing_spline(x, y)
        return spl(x_new).copy()
    # Fallback for older SciPy versions
    if len(x) <= 3:
        return np.interp(x_new, x, y).copy()
    spl = UnivariateSpline(x, y, s=len(x) * 1e-4)
    return spl(x_new).copy()