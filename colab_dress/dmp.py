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
    from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped
    from nav_msgs.msg import Path
except Exception:  # pragma: no cover - ROS 2 not available in all contexts
    rclpy = None
    Node = object
    PoseArray = Pose = PointStamped = PoseStamped = Path = None

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
        self.declare_parameter("delta_pose_topic", "/target_pose_dmp")
        self.declare_parameter(
            "current_ee_topic",
            "/NS_1/franka_robot_state_broadcaster/current_pose",
        )
        self.declare_parameter("publish_stride", 10)
        self.declare_parameter("publish_chunk_size", 15)
        self.declare_parameter("settle_position_tolerance", 0.01)
        self.declare_parameter("settle_velocity_tolerance", 0.01)
        self.declare_parameter("settle_count_required", 5)
        self.declare_parameter("rollout_path_topic", "/dmp/dmp_rollout_path")
        self.declare_parameter("max_delta_norm", 0.03)
        self.declare_parameter("max_delta_axis", 0.02)
        self.declare_parameter("min_publish_period", 1.0)
        self.declare_parameter("delta_reference", "initial")
        self.declare_parameter("publish_timer_period", 0.1)

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
        delta_pose_topic = str(self.get_parameter("delta_pose_topic").value)
        current_ee_topic = str(self.get_parameter("current_ee_topic").value)
        publish_stride = int(self.get_parameter("publish_stride").value)
        publish_chunk_size = int(self.get_parameter("publish_chunk_size").value)
        settle_position_tolerance = float(
            self.get_parameter("settle_position_tolerance").value
        )
        settle_velocity_tolerance = float(
            self.get_parameter("settle_velocity_tolerance").value
        )
        settle_count_required = int(
            self.get_parameter("settle_count_required").value
        )
        rollout_path_topic = str(self.get_parameter("rollout_path_topic").value)
        max_delta_norm = float(self.get_parameter("max_delta_norm").value)
        max_delta_axis = float(self.get_parameter("max_delta_axis").value)
        min_publish_period = float(self.get_parameter("min_publish_period").value)
        publish_timer_period = float(self.get_parameter("publish_timer_period").value)
        delta_reference = str(self.get_parameter("delta_reference").value).strip().lower()
        if delta_reference not in {"current", "initial"}:
            self.get_logger().warning(
                "Invalid delta_reference '%s' (expected 'current' or 'initial'); defaulting to 'current'.",
                delta_reference,
            )
            delta_reference = "current"
        if delta_pose_topic == "/delta_pose" and delta_reference != "initial":
            self.get_logger().warning(
                "delta_reference is '%s' but controller expects /delta_pose as offset from initial pose. "
                "Set delta_reference:=initial to avoid large pose errors.",
                delta_reference,
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

        self.delta_pose_topic = delta_pose_topic
        self.current_ee_topic = current_ee_topic
        self.publish_stride = max(1, publish_stride)
        self.publish_chunk_size = max(1, publish_chunk_size)
        self.settle_position_tolerance = max(0.0, settle_position_tolerance)
        self.settle_velocity_tolerance = max(0.0, settle_velocity_tolerance)
        self.settle_count_required = max(1, settle_count_required)
        self.rollout_path_topic = rollout_path_topic
        self.max_delta_norm = max(0.0, max_delta_norm)
        self.max_delta_axis = max(0.0, max_delta_axis)
        self.min_publish_period = max(0.0, min_publish_period)
        self.delta_reference = delta_reference
        self.publish_timer_period = max(0.01, publish_timer_period)

        self.current_frame_id: str = ""

        self._pending_points: list[np.ndarray] = []
        self._next_point_index: int = 0
        self._waiting_for_settle: bool = False
        self._last_target: Optional[np.ndarray] = None
        self._settle_count: int = 0
        self._last_publish_time: Optional[float] = None
        self._initial_ee: Optional[np.ndarray] = None
        self._initial_frame_id: str = ""

        self.arm_sub = self.create_subscription(
            PoseArray, "dmp/arm_poses", self._arm_poses_cb, 10
        )
        self.shoulder_sub = self.create_subscription(
            PointStamped, "dmp/shoulder_position", self._shoulder_cb, 10
        )
        # self.current_pose_sub = self.create_subscription(
        #     PoseStamped, self.current_ee_topic, self._current_pose_cb, 10
        # )
        self.rollout_pub = self.create_publisher(PoseArray, "dmp/dmp_rollout", 10)
        self.rollout_path_pub = self.create_publisher(Path, self.rollout_path_topic, 10)
        self.delta_pose_pub = self.create_publisher(PoseStamped, self.delta_pose_topic, 10)
        self._publish_timer = self.create_timer(self.publish_timer_period, self._maybe_publish_next)

        self.get_logger().info(
            "DMP node ready: waiting for arm_poses and shoulder_position."
        )
        self.get_logger().info(
            f"Publishing delta poses to '{self.delta_pose_topic}' based on current EE "
            f"'{self.current_ee_topic}'."
        )
        self.get_logger().info(
            f"Delta reference mode: '{self.delta_reference}' "
            f"(controller must interpret /delta_pose accordingly)."
        )

    def _arm_poses_cb(self, msg: PoseArray) -> None:
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
        self._publish_rollout(y_rollout, self.last_frame_id)
        self._start_trajectory_publish(y_rollout, self.last_frame_id)

    def _shoulder_cb(self, msg: PointStamped) -> None:
        self.last_shoulder = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        if self.goal_offset is None or not self.dmp.imitate:
            return

        new_goal = self.last_shoulder + self.goal_offset
        self.dmp.set_goal(new_goal)
        y_rollout, _, _ = self.dmp.rollout()
        frame_id = msg.header.frame_id or self.last_frame_id
        self._publish_rollout(y_rollout, frame_id)
        self._start_trajectory_publish(y_rollout, frame_id)

    def _start_trajectory_publish(self, y_rollout: np.ndarray, frame_id: str) -> None:
        if y_rollout.size == 0:
            return

        sampled = y_rollout[:: self.publish_stride]
        if sampled.shape[0] > self.publish_chunk_size:
            sampled = sampled[: self.publish_chunk_size]

        self._pending_points = [pt.copy() for pt in sampled]
        self._next_point_index = 0
        self._waiting_for_settle = False
        self._last_target = None
        self._settle_count = 0
        self.last_frame_id = frame_id
        self._initial_ee = None
        self._initial_frame_id = frame_id

        self._maybe_publish_next()

    def _maybe_publish_next(self) -> None:
        if self._next_point_index >= len(self._pending_points):
            return

        if self._last_publish_time is not None:
            now = self.get_clock().now().seconds_nanoseconds()
            now_sec = float(now[0]) + float(now[1]) * 1e-9
            if (now_sec - self._last_publish_time) < self.min_publish_period:
                return
        target = self._pending_points[self._next_point_index]
        published_target = self._publish_delta_pose(target)
        if published_target is None:
            return
        self._last_target = published_target
        self._next_point_index += 1

    def _publish_delta_pose(self, target_point: np.ndarray) -> Optional[np.ndarray]:
        now = self.get_clock().now().seconds_nanoseconds()
        now_sec = float(now[0]) + float(now[1]) * 1e-9
        self._last_publish_time = now_sec
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.last_frame_id
        msg.pose.position.x = float(target_point[0])
        msg.pose.position.y = float(target_point[1])
        msg.pose.position.z = float(target_point[2])
        self.get_logger().info(
            f"Publishing delta pose: "
            f"index={self._next_point_index}, "
            f"x={msg.pose.position.x:.4f}, "
            f"y={msg.pose.position.y:.4f}, "
            f"z={msg.pose.position.z:.4f}"
        )
        msg.pose.orientation.w = 1.0
        self.delta_pose_pub.publish(msg)
        return target_point

    def _publish_rollout(self, y_rollout: np.ndarray, frame_id: str) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.poses = []
        path_msg = Path()
        path_msg.header = msg.header
        path_msg.poses = []
        for pt in y_rollout:
            pose = Pose()
            pose.position.x = float(pt[0])
            pose.position.y = float(pt[1])
            pose.position.z = float(pt[2])
            pose.orientation.w = 1.0
            msg.poses.append(pose)
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = pose
            path_msg.poses.append(pose_stamped)
        self.rollout_pub.publish(msg)
        self.rollout_path_pub.publish(path_msg)


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