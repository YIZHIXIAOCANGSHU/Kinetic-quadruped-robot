import sys
import os
import time
import site
import traceback
from dataclasses import dataclass

os.environ["MPLBACKEND"] = "TkAgg"


def _bootstrap_ros_pinocchio_numpy():
    """Load ROS Pinocchio and system numeric packages before user-site wheels."""
    user_site = site.getusersitepackages()
    user_site_paths = [
        user_site,
        os.path.join(user_site, "cmeel.prefix", "lib", "python3.10", "site-packages"),
        os.path.join(user_site, "rerun_sdk"),
    ]

    removed_paths = []
    for path in user_site_paths:
        while path in sys.path:
            sys.path.remove(path)
            removed_paths.append(path)

    import numpy as _np
    import scipy as _scipy
    import matplotlib as _matplotlib
    import pinocchio as _pinocchio

    if int(_np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(
            "ROS Humble Pinocchio was loaded with NumPy "
            f"{_np.__version__} from {_np.__file__}. Run with NumPy<2."
        )

    for path in reversed(removed_paths):
        sys.path.insert(0, path)

    return _np, _pinocchio


np, pinocchio = _bootstrap_ros_pinocchio_numpy()
import mujoco as mj

# Path setup for convex-mpc
curr_dir = os.path.dirname(os.path.abspath(__file__))
mpc_src_path = os.path.join(curr_dir, "go2-convex-mpc", "src")
sys.path.append(mpc_src_path)

try:
    from convex_mpc.go2_robot_data import PinGo2Model
    from convex_mpc.mujoco_model import MuJoCo_GO2_Model
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.centroidal_mpc import CentroidalMPC
    from convex_mpc.wbc_controller import WbcController
    from convex_mpc.gait import Gait
    import convex_mpc.gait as gait_module
except ImportError as e:
    print(f"Failed to import MPC modules: {e}")
    sys.exit(1)

# Optional Rerun bridge
try:
    from rerun_bridge import QuadrupedForceLogger
except ImportError:
    QuadrupedForceLogger = None

# --------------------------------------------------------------------------------
# Parameters & Settings
# --------------------------------------------------------------------------------

INITIAL_X_POS = 0.0
INITIAL_Y_POS = 0.0
INITIAL_Z_POS = 1.0  # 100cm height as requested
RENDER_HZ = 60.0 
RENDER_DT = 1.0 / RENDER_HZ
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200       
CTRL_DT = 1.0 / CTRL_HZ
GAIT_HZ = 3.0
GAIT_DUTY = 0.60
GAIT_T = 1.0 / GAIT_HZ
SPEED_RAMP_START_S = 2.4
SPEED_RAMP_END_S = 3.6
WALK_GAIT_START_X_VEL = 0.15
LEGS = ("FL", "FR", "RL", "RR")
LANDING_STABLE_TIME_S = 0.08
FOOT_CONTACT_MARGIN_M = 0.01
FOOT_FORCE_CONTACT_THRESHOLD_N = 3.0
MAX_CONSECUTIVE_SOLVER_FAILURES = 3
SOLVER_WARNING_INTERVAL_S = 0.25
POSTURE_KP = np.array([35.0, 45.0, 55.0] * 4)
POSTURE_KD = np.array([1.8, 2.2, 2.6] * 4)

@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float
    gait_duty: float = GAIT_DUTY
    gait_hz: float = GAIT_HZ
    swing_height: float = 0.08
    z_weight: float = 220.0
    vz_weight: float = 220.0

# Command Schedule: land, settle, then ramp into straight walking until stopped.
CMD_SCHEDULE = [
    BodyCmdPhase(0.0, 1.5, 0.0, 0.0, 0.30, 0.0, gait_duty=1.0, swing_height=0.06, z_weight=650.0, vz_weight=650.0),
    BodyCmdPhase(1.5, 2.4, 0.0, 0.0, 0.28, 0.0, gait_duty=0.75, swing_height=0.06, z_weight=350.0, vz_weight=350.0),
    BodyCmdPhase(2.4, float("inf"), 0.70, 0.0, 0.27, 0.0, gait_duty=GAIT_DUTY, swing_height=0.08, z_weight=220.0, vz_weight=220.0),
]

def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase
    # Default Stand
    return BodyCmdPhase(0.0, 100.0, 0.0, 0.0, 0.27, 0.0, gait_duty=1.0, z_weight=650.0, vz_weight=650.0)


def smoothstep(t_start: float, t_end: float, t: float) -> float:
    if t_end <= t_start:
        return 1.0 if t >= t_end else 0.0
    s = np.clip((t - t_start) / (t_end - t_start), 0.0, 1.0)
    return float(s * s * (3.0 - 2.0 * s))


def get_ramped_x_vel(cmd: BodyCmdPhase, t: float) -> float:
    if cmd.x_vel <= 0.0:
        return 0.0
    return cmd.x_vel * smoothstep(SPEED_RAMP_START_S, SPEED_RAMP_END_S, t)


def should_use_walk_gait(x_vel_cmd: float) -> bool:
    return abs(x_vel_cmd) >= WALK_GAIT_START_X_VEL


def apply_gait_params(gait: Gait, frequency_hz: float, duty: float) -> None:
    gait.gait_hz = frequency_hz
    gait.gait_duty = duty
    gait.gait_period = 1.0 / frequency_hz
    gait.stance_time = duty * gait.gait_period
    gait.swing_time = (1.0 - duty) * gait.gait_period


def lock_straight_reference(traj: ComTraj, go2: PinGo2Model) -> None:
    traj.pos_des_world[1] = INITIAL_Y_POS
    traj.pos_traj_world[1, :] = INITIAL_Y_POS
    traj.vel_traj_world[1, :] = 0.0
    traj.rpy_traj_world[2, :] = 0.0
    traj.omega_traj_world[2, :] = 0.0
    go2.y_pos_des_world = INITIAL_Y_POS
    go2.y_vel_des_world = 0.0
    go2.yaw_rate_des_world = 0.0


def _sensor_force_vector(model, data, sensor_id: int) -> np.ndarray:
    if sensor_id == -1:
        return np.zeros(3)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    values = np.asarray(data.sensordata[adr : adr + dim], dtype=float).reshape(-1)
    if dim >= 3:
        return values[:3]
    force = np.zeros(3)
    force[2] = values[0] if dim == 1 else 0.0
    return force


def _read_foot_contacts(model, data, floor_geom_id: int, foot_geom_ids: dict, force_sensor_ids: dict) -> tuple[dict, bool]:
    contacts = {leg: False for leg in LEGS}

    if floor_geom_id != -1:
        geom_to_leg = {gid: leg for leg, gid in foot_geom_ids.items() if gid != -1}
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist > FOOT_CONTACT_MARGIN_M:
                continue
            if contact.geom1 == floor_geom_id and contact.geom2 in geom_to_leg:
                contacts[geom_to_leg[contact.geom2]] = True
            elif contact.geom2 == floor_geom_id and contact.geom1 in geom_to_leg:
                contacts[geom_to_leg[contact.geom1]] = True

    for leg, sensor_id in force_sensor_ids.items():
        force = _sensor_force_vector(model, data, sensor_id)
        if np.linalg.norm(force) >= FOOT_FORCE_CONTACT_THRESHOLD_N:
            contacts[leg] = True

    return contacts, all(contacts.values())


def _posture_hold_torque(data, q_ref: np.ndarray, tau_lim: np.ndarray) -> np.ndarray:
    q_err = q_ref - np.asarray(data.qpos[7:19], dtype=float)
    dq = np.asarray(data.qvel[6:18], dtype=float)
    tau = POSTURE_KP * q_err - POSTURE_KD * dq
    return np.clip(tau, -tau_lim, tau_lim)


def _clear_solver_warm_start(*controllers) -> None:
    for controller in controllers:
        for attr in ("x_prev", "lam_x_prev", "lam_a_prev"):
            if hasattr(controller, attr):
                setattr(controller, attr, None)


def _reset_wbc_gait_memory(wbc_controller: WbcController) -> None:
    if hasattr(wbc_controller, "_last_mask"):
        wbc_controller._last_mask = np.ones(4, dtype=np.int32)
    if hasattr(wbc_controller, "_takeoff_time"):
        for leg in LEGS:
            wbc_controller._takeoff_time[leg] = 0.0
    if hasattr(wbc_controller, "_swing_traj"):
        for leg in LEGS:
            wbc_controller._swing_traj[leg] = None


def _hold_viewer_until_closed(viewer, reason: str) -> None:
    if not viewer.is_running():
        return
    print(reason)
    print("MuJoCo viewer is being kept open. Close the viewer or press Ctrl+C to exit.")
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(RENDER_DT)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")


def _pause_on_error_if_interactive() -> None:
    if not sys.stdin.isatty():
        return
    try:
        input("Press Enter to exit...")
    except EOFError:
        pass

# --------------------------------------------------------------------------------
# Main Simulation
# --------------------------------------------------------------------------------

def run_simulation():
    # Initialize models
    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model()
    gait = Gait(GAIT_HZ, GAIT_DUTY)
    initial_cmd = get_body_cmd(0.0)
    apply_gait_params(gait, initial_cmd.gait_hz, initial_cmd.gait_duty)
    
    # Initial configuration
    q_init = go2.current_config.get_q()
    q_init[0], q_init[1], q_init[2] = INITIAL_X_POS, INITIAL_Y_POS, INITIAL_Z_POS
    posture_q_ref = q_init[7:19].copy()
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    # MPC Tuning
    MPC_DT = GAIT_T / 16.0
    N_MPC = 16
    MPC_HZ = 1.0 / MPC_DT
    STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))
    
    # Global MPC Weight Override
    import convex_mpc.centroidal_mpc
    convex_mpc.centroidal_mpc.COST_MATRIX_Q = np.diag([
        5.0, 140.0, 220.0,    # Position (x, y, z)
        220.0, 260.0, 220.0,  # Orientation (roll, pitch, yaw)
        35.0, 80.0, 180.0,    # Velocity (vx, vy, vz)
        60.0, 80.0, 120.0     # Angular velocity (wx, wy, wz)
    ])

    traj = ComTraj(go2)
    traj.generate_traj(go2, gait, 0.0, initial_cmd.x_vel, 0.0, initial_cmd.z_pos, 0.0, time_step=MPC_DT, N=N_MPC)
    lock_straight_reference(traj, go2)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    # Torque Limits
    TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)

    # WBC Controller (replaces per-leg Jacobian mapping)
    wbc_controller = WbcController(
        go2, W_force=1.0, W_base_acc=10.0, W_swing=50.0, W_tau=1e-3,
        mu=0.8, fz_min=5.0, tau_lim=TAU_LIM, verbose=False,
    )
    X_opt = np.zeros((12, N_MPC), dtype=float)  # MPC state trajectory for WBC

    # Rerun Logging
    logger = QuadrupedForceLogger("Go2_Drop_Walk") if QuadrupedForceLogger else None
    f_sensor_names = ["FL_force", "FR_force", "RL_force", "RR_force"]
    t_sensor_names = ["FL_thigh_torque", "FR_thigh_torque", "RL_thigh_torque", "RR_thigh_torque"]
    
    f_sensor_ids = {n: mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_SENSOR, n) for n in f_sensor_names}
    t_sensor_ids = {n: mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_SENSOR, n) for n in t_sensor_names}
    floor_geom_id = mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_GEOM, "floor")
    foot_geom_ids = {leg: mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS}
    force_sensor_ids = {leg: f_sensor_ids.get(f"{leg}_force", -1) for leg in LEGS}

    print(f"Starting Go2 simulation: Drop from {INITIAL_Z_POS}m and Walk.")
    print("Waiting for four-foot ground contact before enabling MPC/WBC.")
    
    ctrl_i = 0
    tau_hold = np.zeros(12, dtype=float)
    next_render_t = 0.0
    ctrl_decim = SIM_HZ // CTRL_HZ
    landing_stable_s = 0.0
    wbc_enabled = False
    consecutive_solver_failures = 0
    last_solver_warning_t = -np.inf
    gait_phase_time = 0.0
    walk_gait_active = False
    k = 0

    import mujoco.viewer
    with mujoco.viewer.launch_passive(mujoco_go2.model, mujoco_go2.data) as viewer:
        try:
            while viewer.is_running():
                time_now = float(mujoco_go2.data.time)

                # Log to Rerun
                if logger and (k % 5 == 0):
                    forces = {n[:2]: mujoco_go2.data.sensordata[mujoco_go2.model.sensor_adr[f_sensor_ids[n]] + 2] 
                             if f_sensor_ids[n] != -1 else 0.0 for n in f_sensor_names}
                    torques = {n[:2]: mujoco_go2.data.sensordata[mujoco_go2.model.sensor_adr[t_sensor_ids[n]]] 
                              if t_sensor_ids[n] != -1 else 0.0 for n in t_sensor_names}
                    
                    logger.log_sensor_forces(time_now, forces)
                    logger.log_thigh_torques(time_now, torques)
                    logger.log_body_height(time_now, go2.current_config.base_pos[2])

                # Control Loop
                if (k % ctrl_decim) == 0:
                    cmd = get_body_cmd(time_now)
                    x_vel_cmd = get_ramped_x_vel(cmd, time_now)
                    use_walk_gait = should_use_walk_gait(x_vel_cmd)
                    effective_gait_duty = cmd.gait_duty if use_walk_gait else 1.0
                    
                    # Update Gait & MPC Parameters
                    apply_gait_params(gait, cmd.gait_hz, effective_gait_duty)
                    mpc.Q[2, 2] = cmd.z_weight
                    mpc.Q[8, 8] = cmd.vz_weight
                    
                    gait_module.HEIGHT_SWING = cmd.swing_height

                    if use_walk_gait:
                        if not walk_gait_active:
                            walk_gait_active = True
                            gait_phase_time = 0.0
                            _clear_solver_warm_start(mpc, wbc_controller)
                            _reset_wbc_gait_memory(wbc_controller)
                            print(f"\n[Gait] Starting walk gait at t={time_now:.3f}s (x_vel={x_vel_cmd:.3f} m/s).")
                        else:
                            gait_phase_time += CTRL_DT
                    else:
                        walk_gait_active = False
                        gait_phase_time = 0.0

                    controller_time = gait_phase_time if use_walk_gait else 0.0

                    mujoco_go2.update_pin_with_mujoco(go2)
                    foot_contacts, all_feet_contact = _read_foot_contacts(
                        mujoco_go2.model, mujoco_go2.data, floor_geom_id, foot_geom_ids, force_sensor_ids
                    )

                    if not wbc_enabled:
                        landing_stable_s = landing_stable_s + CTRL_DT if all_feet_contact else 0.0
                        tau_hold = _posture_hold_torque(mujoco_go2.data, posture_q_ref, TAU_LIM)

                        if landing_stable_s >= LANDING_STABLE_TIME_S:
                            wbc_enabled = True
                            consecutive_solver_failures = 0
                            ctrl_i = 0
                            _clear_solver_warm_start(mpc, wbc_controller)
                            _reset_wbc_gait_memory(wbc_controller)
                            print(f"[Landing] Four-foot contact stable for {landing_stable_s:.3f}s at t={time_now:.3f}s. Enabling MPC/WBC.")
                        else:
                            if k % max(ctrl_decim, int(0.25 * SIM_HZ)) == 0:
                                mask = "".join("1" if foot_contacts[leg] else "0" for leg in LEGS)
                                print(f"\r[Landing] t={time_now:.3f}s contacts {mask} stable={landing_stable_s:.3f}s", end="", flush=True)
                            ctrl_i += 1

                    if wbc_enabled:
                        try:
                            if (ctrl_i % STEPS_PER_MPC) == 0:
                                traj.generate_traj(go2, gait, controller_time, x_vel_cmd, 0.0, cmd.z_pos, 0.0, time_step=MPC_DT, N=N_MPC)

                                # Keep the reference as a straight line along world x.
                                lock_straight_reference(traj, go2)

                                sol = mpc.solve_QP(go2, traj, False)
                                w_opt = sol["x"].full().flatten()
                                N = traj.N
                                U_opt = w_opt[12 * N :].reshape((12, N), order="F")
                                X_opt = w_opt[:12 * N].reshape((12, N), order="F")

                            # WBC: single QP replaces per-leg Jacobian mapping
                            x0_mpc = X_opt[:, 1] if X_opt is not None else None
                            wbc_out = wbc_controller.solve(
                                go2, gait, U_opt[:, 0], controller_time, x0_mpc
                            )
                            tau_hold = np.clip(wbc_out.tau, -TAU_LIM, TAU_LIM)
                            ctrl_i += 1
                            consecutive_solver_failures = 0
                        except Exception as exc:
                            consecutive_solver_failures += 1
                            _clear_solver_warm_start(mpc, wbc_controller)
                            tau_hold = _posture_hold_torque(mujoco_go2.data, posture_q_ref, TAU_LIM)

                            if time_now - last_solver_warning_t >= SOLVER_WARNING_INTERVAL_S:
                                print(
                                    f"\n[Controller warning] t={time_now:.3f}s "
                                    f"MPC/WBC solve failed ({type(exc).__name__}: {exc}). "
                                    "Using posture hold torque."
                                )
                                last_solver_warning_t = time_now

                            if consecutive_solver_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES:
                                wbc_enabled = False
                                landing_stable_s = 0.0
                                ctrl_i = 0
                                walk_gait_active = False
                                gait_phase_time = 0.0
                                _reset_wbc_gait_memory(wbc_controller)
                                print(
                                    f"[Controller warning] {consecutive_solver_failures} consecutive solve failures. "
                                    "Waiting for stable four-foot contact again."
                                )

                mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
                mujoco_go2.set_joint_torque(tau_hold)
                mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
                
                if time_now >= next_render_t:
                    viewer.sync()
                    next_render_t += RENDER_DT
                
                # Slow down simulation to match real time
                time.sleep(SIM_DT * 0.5)
                k += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            return
        except Exception:
            print("\nUnexpected simulation error:")
            traceback.print_exc()
            _hold_viewer_until_closed(viewer, "Simulation stopped because of an unexpected error.")
            return

    print(f"Simulation ended.")

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception:
        print("\nFatal error before the MuJoCo viewer could be kept open:")
        traceback.print_exc()
        _pause_on_error_if_interactive()
