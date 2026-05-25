import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .go2_robot_data import PinGo2Model
else:
    PinGo2Model = object

# --------------------------------------------------------------------------------
# Gait Setting
# --------------------------------------------------------------------------------

PHASE_OFFSET = np.array([0.5, 0.0, 0.0, 0.5]).reshape(4)    # trotting gait
HEIGHT_SWING = 0.1 # Height of the swing leg trajectory apex
LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}


class Gait():
    def __init__(self, frequency_hz, duty):
        self.gait_duty = duty
        self.gait_hz = frequency_hz
        self.terrain_provider = None
        self.forced_swing_legs = set()
        self.forced_stance_legs = set()

        self.gait_period = 1 / frequency_hz # Perioid
        self.stance_time = self.gait_duty * self.gait_period
        self.swing_time = (1-self.gait_duty) * self.gait_period

    def set_terrain_provider(self, terrain_provider):
        self.terrain_provider = terrain_provider

    def set_forced_leg_contacts(self, swing_legs=(), stance_legs=()):
        self.forced_swing_legs = {leg for leg in swing_legs if leg in LEG_INDEX}
        self.forced_stance_legs = {leg for leg in stance_legs if leg in LEG_INDEX}

    def _terrain_height_at_world(self, x_world: float, y_world: float, default: float = 0.0) -> float:
        if self.terrain_provider is None:
            return float(default)
        try:
            return float(self.terrain_provider.height_at_world(x_world, y_world, default=default))
        except Exception:
            return float(default)

    def _terrain_max_height_between_world(self, p0_world, p1_world, default: float = 0.0) -> float:
        if self.terrain_provider is None:
            return float(default)
        try:
            return float(self.terrain_provider.max_height_between_world(p0_world, p1_world, default=default))
        except Exception:
            return float(default)

    def _override_touchdown_world(self, leg: str, touchdown_world):
        if self.terrain_provider is None or not hasattr(self.terrain_provider, "override_touchdown_world"):
            return np.asarray(touchdown_world, dtype=float)
        try:
            return np.asarray(
                self.terrain_provider.override_touchdown_world(leg, touchdown_world),
                dtype=float,
            )
        except Exception:
            return np.asarray(touchdown_world, dtype=float)

    def _scalar_or_default(self, value, default: float = 0.0) -> float:
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 0 or not np.isfinite(arr[0]):
                return float(default)
            return float(arr[0])
        except Exception:
            return float(default)

    def _bounded_touchdown_z(self, raw_height: float, base_z: float) -> float:
        terrain_z = float(raw_height)
        if not np.isfinite(terrain_z):
            terrain_z = 0.0
        terrain_z = np.clip(terrain_z, -0.05, 0.50)
        return float(terrain_z + 0.02)

    def compute_current_mask(self, time):

        mask = self.compute_contact_table(time, 0, 1)
        return mask
    
    def compute_contact_table(self, t0: float, dt: float, N: int) -> np.ndarray:

        # times: (N,)
        t = t0 + np.arange(N) * dt
        t = t + dt/2

        # phases: (4,N)
        phases = np.mod(PHASE_OFFSET[:, None] + t[None, :] / self.gait_period, 1.0)

        # mask: (4,N) with 1=stance, 0=swing
        contact_table = (phases < self.gait_duty).astype(np.int32)
        for leg in self.forced_stance_legs:
            contact_table[LEG_INDEX[leg], :] = 1
        for leg in self.forced_swing_legs:
            contact_table[LEG_INDEX[leg], :] = 0
        return contact_table        
    

    def compute_touchdown_world_for_traj_purpose_only(self, go2:PinGo2Model, leg:str):
        base_pos = go2.current_config.base_pos
        base_vel = go2.current_config.base_vel
        R_z = go2.R_z
        yaw_rate = self._scalar_or_default(go2.yaw_rate_des_world)

        hip_offset = go2.get_hip_offset(leg)
        body_pos = np.array([base_pos[0], base_pos[1], 0])
        hip_pos_world = body_pos + R_z @ hip_offset

        t_swing = self.swing_time
        t_stance = self.stance_time

        # We are planning at takeoff
        T = t_swing + 0.5*t_stance
        pred_time = T / 2.0

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]
        pos_drift_term = [base_vel[0] * pred_time, base_vel[1] * pred_time, 0]

        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])  # or base_pos[0:2]
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
                                -dtheta * r_xy[1],
                                dtheta * r_xy[0],
                                0.0
                                ])
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(rotation_correction_term)
                                )
        raw_terrain_z = self._terrain_height_at_world(pos_touchdown_world[0], pos_touchdown_world[1])
        pos_touchdown_world[2] = self._bounded_touchdown_z(raw_terrain_z, base_pos[2])
        pos_touchdown_world = self._override_touchdown_world(leg, pos_touchdown_world)

        return pos_touchdown_world
    

    def compute_swing_traj_and_touchdown(self, go2:PinGo2Model, leg:str):

        # This function should only be called the moment the foot takes off
        base_pos = go2.current_config.base_pos
        pos_com_world = go2.pos_com_world

        vel_com_world = go2.vel_com_world
        R_z = go2.R_z
        yaw_rate = self._scalar_or_default(go2.yaw_rate_des_world)

        hip_offset = go2.get_hip_offset(leg)
        [foot_pos, foot_vel] = go2.get_single_foot_state_in_world(leg)
        body_pos = np.array([base_pos[0], base_pos[1], 0])


        hip_pos_world = body_pos + R_z @ hip_offset

        x_vel_des = self._scalar_or_default(go2.x_vel_des_world)
        y_vel_des = self._scalar_or_default(go2.y_vel_des_world)
        x_pos_des = self._scalar_or_default(go2.x_pos_des_world, pos_com_world[0])
        y_pos_des = self._scalar_or_default(go2.y_pos_des_world, pos_com_world[1])

        t_swing = self.swing_time
        t_stance = self.stance_time

        T = t_swing + 0.5*t_stance
        pred_time = T / 2.0

        # Forward (x) direction
        k_v_x = 0.4 * T          # ~0.2–0.3
        k_p_x = 0.1              # small

        # Lateral (y) direction – usually weaker
        k_v_y = 0.2 * T          # ~0.1
        k_p_y = 0.05

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]
        pos_drift_term = [x_vel_des * pred_time, y_vel_des * pred_time, 0]
        pos_correction_term = [k_p_x * (pos_com_world[0] - x_pos_des), k_p_y * (pos_com_world[1] - y_pos_des), 0]
        vel_correction_term = [k_v_x * (vel_com_world[0] - x_vel_des), k_v_y * (vel_com_world[1] - y_vel_des), 0]
        
        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])  # or base_pos[0:2]
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
                                -dtheta * r_xy[1],
                                dtheta * r_xy[0],
                                0.0
                                ])
    
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(pos_correction_term)
                                + np.array(vel_correction_term)
                                + np.array(rotation_correction_term)
                                )
        raw_terrain_z = self._terrain_height_at_world(pos_touchdown_world[0], pos_touchdown_world[1])
        pos_touchdown_world[2] = self._bounded_touchdown_z(raw_terrain_z, base_pos[2])
        pos_touchdown_world = self._override_touchdown_world(leg, pos_touchdown_world)
        
        terrain_peak = self._terrain_max_height_between_world(
            foot_pos, pos_touchdown_world, default=max(0.0, pos_touchdown_world[2] - 0.02)
        )
        terrain_peak = min(terrain_peak, 0.50)
        clearance_height = terrain_peak + 0.05
        h_sw = max(HEIGHT_SWING, 0.12, clearance_height - max(foot_pos[2], pos_touchdown_world[2]))
        h_sw = min(h_sw, 0.24)
        pos_foot_traj_eval_at_world = self.make_swing_trajectory(foot_pos, pos_touchdown_world, t_swing, h_sw=h_sw)
        return pos_foot_traj_eval_at_world, pos_touchdown_world


    def make_swing_trajectory(self, p0, pf, t_swing, h_sw):

        p0 = np.asarray(p0, dtype=float)
        pf = np.asarray(pf, dtype=float)
        T = float(t_swing)
        dp = pf - p0

        def eval_at(t):
            # phase s in [0,1]
            s = np.clip(t / T, 0.0, 1.0)

            # Minimum-jerk basis and its derivatives
            mj   = 10*s**3 - 15*s**4 + 6*s**5
            dmj  = 30*s**2 - 60*s**3 + 30*s**4           # d(mj)/ds
            d2mj = 60*s    - 180*s**2 + 120*s**3         # d^2(mj)/ds^2

            # Base (x,y,z) trajectory
            p = p0 + dp * mj
            v = (dp * dmj) / T
            a = (dp * d2mj) / (T**2)

            # Optional smooth z-bump: b(s)=64*s^3*(1-s)^3, with zero vel/acc at ends
            if h_sw != 0.0:
                b    = 64 * s**3 * (1 - s)**3
                db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)           # db/ds
                d2b  = 192 * ( 2*s*(1 - s)**2*(1 - 2*s)
                            - 2*s**2*(1 - s)*(1 - 2*s)
                            - 2*s**2*(1 - s)**2 )                  # d^2b/ds^2

                p[2] += h_sw * b
                v[2] += h_sw * db / T
                a[2] += h_sw * d2b / (T**2)

            return p, v, a

        return eval_at
    
