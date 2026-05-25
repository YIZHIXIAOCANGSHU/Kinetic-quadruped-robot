"""QP Whole-Body Controller (Weighted QP-WBC) for Unitree Go2.

Inspired by legged_control (qiayuanliao/legged_control).

Decision variable: x = [ddq(18), f_contact(12), tau(12)]^T  (42 dims)

Hard constraints:
  1. Floating-base dynamics: M*ddq - Jc^T*f - S^T*tau = -(C*dq + g)
  2. Friction cone (pyramid): mu*fz >= |fx|, mu*fz >= |fy|
  3. No-foot-slip (stance): J_i*ddq = -dJ_i*dq
  4. Torque limits, force box bounds

Weighted cost:
  W_force  * ||f_stance - f_mpc||^2
  W_base   * ||ddq_base - ddq_base_des||^2
  W_swing  * ||J_swing*ddq + dJ*dq - a_swing_des||^2
  W_tau    * ||tau||^2
"""

import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from .go2_robot_data import PinGo2Model
from .gait import Gait
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEGS = ["FL", "FR", "RL", "RR"]
LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}

N_DDQ   = 18
N_FORCE = 12
N_TAU   = 12
N_VARS  = 42

N_DYN  = 18
N_FRIC = 16
N_SLIP = 12
N_CON  = 46

SOLVER_OPTS = {
    'warm_start_primal': True,
    'warm_start_dual': True,
    "osqp": {
        "eps_abs": 1e-4,
        "eps_rel": 1e-4,
        "max_iter": 1000,
        "polish": False,
        "verbose": False,
        'adaptive_rho': True,
        "check_termination": 10,
        'adaptive_rho_interval': 25,
        "scaling": 5,
        "scaled_termination": True,
    },
}

KP_LIN = np.diag([400.0, 400.0, 800.0])
KD_LIN = np.diag([40.0,  40.0,  80.0])
KP_ANG = np.diag([600.0, 600.0, 400.0])
KD_ANG = np.diag([60.0,  60.0,  40.0])

KP_SWING = np.diag([400.0, 400.0, 400.0])
KD_SWING = np.diag([50.0,  50.0,  50.0])


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class WbcOutput:
    tau: np.ndarray                     # (12,) joint torque command
    ddq: np.ndarray                     # (18,) optimal generalized acceleration
    f_opt: np.ndarray                   # (12,) optimal contact forces
    solve_time_ms: float                # solver wall time in ms
    status: str                         # solver return status
    ddq_base_des: np.ndarray            # (6,) desired base acceleration
    a_swing_des: dict = field(default_factory=dict)

    @staticmethod
    def make_zero():
        return WbcOutput(
            tau=np.zeros(N_TAU),
            ddq=np.zeros(N_DDQ),
            f_opt=np.zeros(N_FORCE),
            solve_time_ms=0.0,
            status="not_solved",
            ddq_base_des=np.zeros(6),
            a_swing_des={leg: np.zeros(3) for leg in LEGS},
        )


# ---------------------------------------------------------------------------
# WBC Controller
# ---------------------------------------------------------------------------
class WbcController:
    def __init__(
        self,
        go2: PinGo2Model,
        *,
        W_force: float = 1.0,
        W_base_acc: float = 10.0,
        W_swing: float = 50.0,
        W_tau: float = 1e-3,
        mu: float = 0.8,
        fz_min: float = 10.0,
        tau_lim: np.ndarray | None = None,
        verbose: bool = False,
    ):
        self.W_force = W_force
        self.W_base_acc = W_base_acc
        self.W_swing = W_swing
        self.W_tau   = W_tau
        self.mu      = mu
        self.fz_min  = fz_min
        self.tau_lim = tau_lim if tau_lim is not None else np.full(N_TAU, 45.0)
        self.verbose = verbose

        # Warm-start storage
        self.x_prev     = None
        self.lam_x_prev = None
        self.lam_a_prev = None

        # Swing trajectory state (per leg)
        self._last_mask = np.array([2, 2, 2, 2])
        self._takeoff_time: dict[str, float] = {leg: 0.0 for leg in LEGS}
        self._swing_traj: dict[str, object]  = {leg: None for leg in LEGS}

        # Build sparsity and solver once
        self._build_sparsity(verbose)

    # =======================================================================
    # Public API
    # =======================================================================
    def solve(
        self,
        go2: PinGo2Model,
        gait: Gait,
        f_mpc: np.ndarray,          # (12,) MPC contact forces for this step
        current_time: float,
        x0_mpc: np.ndarray | None = None,  # (12,) MPC one-step-ahead state
    ) -> WbcOutput:
        t0 = time.perf_counter()

        # --- Swing trajectory management (takeoff detection) ---
        contact_mask = gait.compute_current_mask(current_time)
        self._update_swing_trajectories(go2, gait, contact_mask, current_time)

        # --- Desired base acceleration from MPC state ---
        ddq_base_des = self._compute_ddq_base_des(go2, x0_mpc)

        # --- Desired swing leg accelerations ---
        a_swing_des = {}
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            if contact_mask[idx] == 0:
                a_swing_des[leg] = self._compute_swing_accel_des(go2, leg, current_time)
            else:
                a_swing_des[leg] = np.zeros(3)

        # --- Build QP matrices ---
        H_dm, g_dm = self._build_H_and_g(go2, gait, f_mpc, contact_mask,
                                          ddq_base_des, a_swing_des, current_time)
        A_dm, lba, uba = self._build_A_and_bounds(go2, contact_mask)
        lbx, ubx = self._compute_box_bounds(contact_mask)

        t1 = time.perf_counter()

        # --- Solve QP ---
        qp_args = {'h': H_dm, 'g': g_dm, 'a': A_dm,
                    'lba': lba, 'uba': uba, 'lbx': lbx, 'ubx': ubx}

        if self.x_prev is not None:
            qp_args['x0']     = self.x_prev
            qp_args['lam_x0'] = self.lam_x_prev
            qp_args['lam_a0'] = self.lam_a_prev

        sol = self.solver(**qp_args)
        t2 = time.perf_counter()

        # --- Extract solution ---
        x_opt = sol["x"].full().flatten()
        ddq_opt = x_opt[0:N_DDQ]
        f_opt   = x_opt[N_DDQ:N_DDQ + N_FORCE]
        tau_opt = x_opt[N_DDQ + N_FORCE:]

        # --- Save warm start ---
        self.x_prev     = sol["x"]
        self.lam_x_prev = sol["lam_x"]
        self.lam_a_prev = sol["lam_a"]

        # --- Update mask memory ---
        self._last_mask = contact_mask.copy()

        # --- Timing ---
        t_build = (t1 - t0) * 1e3
        t_solve = (t2 - t1) * 1e3

        stats = self.solver.stats()
        status = stats.get('return_status', 'unknown')

        if self.verbose:
            print(f"[WBC] build={t_build:.2f}ms  solve={t_solve:.2f}ms  "
                  f"total={t_build + t_solve:.2f}ms  status={status}")

        return WbcOutput(
            tau=tau_opt,
            ddq=ddq_opt,
            f_opt=f_opt,
            solve_time_ms=t_build + t_solve,
            status=status,
            ddq_base_des=ddq_base_des,
            a_swing_des=a_swing_des,
        )

    # =======================================================================
    # Sparsity construction (one-time)
    # =======================================================================
    def _build_sparsity(self, verbose: bool) -> None:
        """Build fixed sparsity patterns and create the OSQP solver once."""
        rows_h, cols_h, vals_h = [], [], []
        rows_a, cols_a, vals_a = [], [], []

        # --- H matrix nonzeros (42x42) ---
        # ddq block (0:18, 0:18): fully dense (J^T*J couples everything)
        for i in range(N_DDQ):
            for j in range(N_DDQ):
                rows_h.append(i); cols_h.append(j); vals_h.append(1.0)

        # f block (18:30, 18:30): diagonal
        for i in range(N_FORCE):
            r = N_DDQ + i
            rows_h.append(r); cols_h.append(r); vals_h.append(1.0)

        # tau block (30:42, 30:42): diagonal
        for i in range(N_TAU):
            r = N_DDQ + N_FORCE + i
            rows_h.append(r); cols_h.append(r); vals_h.append(1.0)

        H_init = ca.DM.triplet(rows_h, cols_h, ca.DM(vals_h), N_VARS, N_VARS)
        self._H_sp = H_init.sparsity()
        self._H_rows = np.array(rows_h)
        self._H_cols = np.array(cols_h)

        # --- A matrix nonzeros (46x42) ---

        # Dynamics rows 0-17:
        #   M block (0:18, 0:18): fully dense
        for i in range(N_DDQ):
            for j in range(N_DDQ):
                rows_a.append(i); cols_a.append(j); vals_a.append(1.0)
        #   -Jc^T block (0:18, 18:30): fully dense
        for i in range(N_DDQ):
            for j in range(N_FORCE):
                rows_a.append(i); cols_a.append(N_DDQ + j); vals_a.append(1.0)
        #   -S^T block (0:18, 30:42): identity at rows 6-17
        for i in range(N_TAU):
            rows_a.append(6 + i); cols_a.append(N_DDQ + N_FORCE + i); vals_a.append(-1.0)

        # Friction rows 18-33 (16 rows):
        #   4 inequalities per leg, 2 nonzeros each
        r0 = N_DYN
        for leg_idx in range(4):
            fx = N_DDQ + 3 * leg_idx
            fy = fx + 1
            fz = fx + 2
            # fx - mu*fz <= 0
            rows_a.extend([r0, r0]); cols_a.extend([fx, fz]); vals_a.extend([1.0, -self.mu]); r0 += 1
            # -fx - mu*fz <= 0
            rows_a.extend([r0, r0]); cols_a.extend([fx, fz]); vals_a.extend([-1.0, -self.mu]); r0 += 1
            # fy - mu*fz <= 0
            rows_a.extend([r0, r0]); cols_a.extend([fy, fz]); vals_a.extend([1.0, -self.mu]); r0 += 1
            # -fy - mu*fz <= 0
            rows_a.extend([r0, r0]); cols_a.extend([fy, fz]); vals_a.extend([-1.0, -self.mu]); r0 += 1

        # No-slip rows 34-45 (12 rows):
        #   3 per leg, J_i block (34:46, 0:18): fully dense
        r0 = N_DYN + N_FRIC
        for i in range(N_SLIP):
            for j in range(N_DDQ):
                rows_a.append(r0 + i); cols_a.append(j); vals_a.append(1.0)

        A_init = ca.DM.triplet(rows_a, cols_a, ca.DM(vals_a), N_CON, N_VARS)
        self._A_sp = A_init.sparsity()
        self._A_rows = np.array(rows_a)
        self._A_cols = np.array(cols_a)

        # --- Create solver ---
        qp = {'h': self._H_sp, 'a': self._A_sp}
        self.solver = ca.conic('S', 'osqp', qp, SOLVER_OPTS)

        if verbose:
            nH_nnz = self._H_sp.nnz()
            nA_nnz = self._A_sp.nnz()
            print(f"\n[WBC Init] H: {N_VARS}x{N_VARS} nnz={nH_nnz}")
            print(f"[WBC Init] A: {N_CON}x{N_VARS} nnz={nA_nnz}")
            print("[WBC Init] OSQP solver created.\n")

    # =======================================================================
    # Swing trajectory management
    # =======================================================================
    def _update_swing_trajectories(
        self, go2: PinGo2Model, gait: Gait,
        contact_mask: np.ndarray, current_time: float
    ) -> None:
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            if self._last_mask[idx] != contact_mask[idx] and contact_mask[idx] == 0:
                # Takeoff detected
                self._takeoff_time[leg] = current_time
                traj, _ = gait.compute_swing_traj_and_touchdown(go2, leg)
                self._swing_traj[leg] = traj

    def _compute_swing_accel_des(
        self, go2: PinGo2Model, leg: str, current_time: float
    ) -> np.ndarray:
        traj = self._swing_traj[leg]
        if traj is None:
            return np.zeros(3)

        t_elapsed = current_time - self._takeoff_time[leg]
        foot_pos_des, foot_vel_des, foot_acc_des = traj(t_elapsed)
        foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

        pos_err = foot_pos_des - foot_pos_now
        vel_err = foot_vel_des - foot_vel_now

        a_des = foot_acc_des + KP_SWING @ pos_err + KD_SWING @ vel_err
        return a_des

    # =======================================================================
    # Desired base acceleration (PD on MPC one-step-ahead state)
    # =======================================================================
    def _compute_ddq_base_des(
        self, go2: PinGo2Model, x0_mpc: np.ndarray | None
    ) -> np.ndarray:
        if x0_mpc is None:
            return np.zeros(6)

        # x0_mpc = [px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        p_des     = x0_mpc[0:3]
        rpy_des   = x0_mpc[3:6]
        v_des     = x0_mpc[6:9]
        omega_des = x0_mpc[9:12]

        p_actual   = go2.current_config.base_pos
        rpy_actual = go2.current_config.compute_euler_angle_world()

        R = go2.R_body_to_world
        v_actual     = go2.vel_com_world
        omega_actual = R @ go2.current_config.base_ang_vel

        ori_err = rpy_des - rpy_actual
        ori_err = (ori_err + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        ddq_lin = KP_LIN @ (p_des - p_actual) + KD_LIN @ (v_des - v_actual)
        ddq_ang = KP_ANG @ ori_err + KD_ANG @ (omega_des - omega_actual)

        return np.concatenate([ddq_lin, ddq_ang])

    # =======================================================================
    # Build H and g (cost Hessian + linear term)
    # =======================================================================
    def _build_H_and_g(
        self, go2: PinGo2Model, gait: Gait, f_mpc: np.ndarray,
        contact_mask: np.ndarray, ddq_base_des: np.ndarray,
        a_swing_des: dict, current_time: float,
    ) -> tuple[ca.DM, ca.DM]:
        H = np.zeros((N_VARS, N_VARS))
        g = np.zeros((N_VARS, 1))

        # ---- Base acceleration tracking (ddq[0:6]) ----
        H[0:6, 0:6] += self.W_base_acc * np.eye(6)
        g[0:6, 0] = -2.0 * self.W_base_acc * ddq_base_des

        # ---- Swing leg Cartesian tracking ----
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            if contact_mask[idx] == 0:
                J_full = go2.compute_full_foot_Jacobian_world(leg)   # (3, 18)
                JtJ = J_full.T @ J_full                              # (18, 18)
                H[0:N_DDQ, 0:N_DDQ] += self.W_swing * JtJ

                Jdot_dq = go2.compute_Jdot_dq_world(leg)             # (3,)
                a_des = a_swing_des[leg]                             # (3,)
                g[0:N_DDQ, 0] += 2.0 * self.W_swing * (J_full.T @ (Jdot_dq - a_des))

        # ---- Contact force tracking (stance legs) ----
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            if contact_mask[idx] == 1:
                fi = N_DDQ + 3 * idx
                for j in range(3):
                    H[fi + j, fi + j] += self.W_force
                f_mpc_leg = f_mpc[3 * idx: 3 * idx + 3]
                g[fi:fi + 3, 0] = -2.0 * self.W_force * f_mpc_leg

        # ---- Torque regularization ----
        t0 = N_DDQ + N_FORCE
        H[t0:t0 + N_TAU, t0:t0 + N_TAU] += self.W_tau * np.eye(N_TAU)

        # Convert to CasADi DM via triplet (matches precomputed sparsity)
        h_vals = H[self._H_rows, self._H_cols]
        H_dm = ca.DM.triplet(self._H_rows.tolist(), self._H_cols.tolist(),
                              ca.DM(h_vals.ravel()), N_VARS, N_VARS)
        g_dm = ca.DM(g)
        return H_dm, g_dm

    # =======================================================================
    # Build A matrix and bounds for linear constraints
    # =======================================================================
    def _build_A_and_bounds(
        self, go2: PinGo2Model, contact_mask: np.ndarray
    ) -> tuple[ca.DM, ca.DM, ca.DM]:
        """Build A (46x42) and constraint bounds (lba, uba)."""
        g_vec, C, M = go2.compute_dynamcis_terms()
        dq = go2.current_config.get_dq()
        h = C @ dq + g_vec.flatten()   # bias (18,)

        A = np.zeros((N_CON, N_VARS))

        # ---- Dynamics rows 0-17 ----
        # M @ ddq
        A[0:N_DDQ, 0:N_DDQ] = M
        # -Jc^T @ f
        Jc = self._stack_contact_jacobians(go2)   # (12, 18)
        A[0:N_DDQ, N_DDQ:N_DDQ + N_FORCE] = -Jc.T
        # -S^T @ tau: joint DOFs (rows 6-17) get -I
        A[6:N_DDQ, N_DDQ + N_FORCE:] = -np.eye(N_TAU)

        lba = np.zeros(N_CON)
        uba = np.zeros(N_CON)
        lba[0:N_DYN] = -h
        uba[0:N_DYN] = -h

        # ---- Friction rows 18-33 ----
        r0 = N_DYN
        for leg_idx in range(4):
            fx = N_DDQ + 3 * leg_idx
            fy = fx + 1
            fz = fx + 2
            A[r0 + 0, fx] =  1.0; A[r0 + 0, fz] = -self.mu  #  fx - mu*fz <= 0
            A[r0 + 1, fx] = -1.0; A[r0 + 1, fz] = -self.mu  # -fx - mu*fz <= 0
            A[r0 + 2, fy] =  1.0; A[r0 + 2, fz] = -self.mu  #  fy - mu*fz <= 0
            A[r0 + 3, fy] = -1.0; A[r0 + 3, fz] = -self.mu  # -fy - mu*fz <= 0
            r0 += 4

        lba[N_DYN:N_DYN + N_FRIC] = -np.inf
        uba[N_DYN:N_DYN + N_FRIC] = 0.0

        # ---- No-slip rows 34-45 ----
        r0 = N_DYN + N_FRIC
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            J_full = go2.compute_full_foot_Jacobian_world(leg)  # (3, 18)
            Jdot_dq = go2.compute_Jdot_dq_world(leg)            # (3,)

            A[r0:r0 + 3, 0:N_DDQ] = J_full

            if contact_mask[idx] == 1:
                # Stance: enforce zero foot acceleration
                lba[r0:r0 + 3] = -Jdot_dq
                uba[r0:r0 + 3] = -Jdot_dq
            else:
                # Swing: deactivate
                lba[r0:r0 + 3] = -np.inf
                uba[r0:r0 + 3] =  np.inf

            r0 += 3

        a_vals = A[self._A_rows, self._A_cols]
        A_dm = ca.DM.triplet(self._A_rows.tolist(), self._A_cols.tolist(),
                              ca.DM(a_vals.ravel()), N_CON, N_VARS)
        lba_dm = ca.DM(lba)
        uba_dm = ca.DM(uba)
        return A_dm, lba_dm, uba_dm

    # =======================================================================
    # Box constraints on decision variables
    # =======================================================================
    def _compute_box_bounds(self, contact_mask: np.ndarray) -> tuple[ca.DM, ca.DM]:
        lbx = np.full((N_VARS, 1), -np.inf)
        ubx = np.full((N_VARS, 1),  np.inf)

        # Swing leg forces → zero
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            if contact_mask[idx] == 0:
                fi = N_DDQ + 3 * idx
                lbx[fi:fi + 3, 0] = 0.0
                ubx[fi:fi + 3, 0] = 0.0

        # Stance leg fz lower bound (prevent slipping)
        for leg in LEGS:
            idx = LEG_INDEX[leg]
            if contact_mask[idx] == 1:
                fz = N_DDQ + 3 * idx + 2
                lbx[fz, 0] = max(lbx[fz, 0], self.fz_min)

        # Torque limits
        t0 = N_DDQ + N_FORCE
        lbx[t0:t0 + N_TAU, 0] = -self.tau_lim
        ubx[t0:t0 + N_TAU, 0] =  self.tau_lim

        return ca.DM(lbx), ca.DM(ubx)

    # =======================================================================
    # Helpers
    # =======================================================================
    def _stack_contact_jacobians(self, go2: PinGo2Model) -> np.ndarray:
        """Stack all 4 foot Jacobians (world frame) into (12, 18)."""
        J_list = []
        for leg in LEGS:
            J_list.append(go2.compute_full_foot_Jacobian_world(leg))
        return np.vstack(J_list)
