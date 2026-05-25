from __future__ import annotations

import sys
import os
import time
import site
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ["MPLBACKEND"] = "TkAgg"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

np = None
pinocchio = None
mj = None
PinGo2Model = None
MuJoCo_GO2_Model = None
ComTraj = None
CentroidalMPC = None
WbcController = None
Gait = None
gait_module = None
MujocoDepthCamera = None
MujocoLidarSimulator = None
Go2RerunLogger = None
DepthTerrainEstimator = None
DepthTerrainMemory = None
FlatTerrainProvider = None
StepObstacleDetector = None
StepClimbStateMachine = None
StepClimbTerrainProvider = None


def _load_runtime_dependencies() -> None:
    global np, pinocchio, mj
    global PinGo2Model, MuJoCo_GO2_Model, ComTraj, CentroidalMPC, WbcController, Gait
    global gait_module, MujocoDepthCamera, MujocoLidarSimulator, Go2RerunLogger
    global DepthTerrainEstimator, DepthTerrainMemory, FlatTerrainProvider
    global StepObstacleDetector, StepClimbStateMachine, StepClimbTerrainProvider
    global POSTURE_KP, POSTURE_KD

    if PinGo2Model is not None and mj is not None:
        return

    np, pinocchio = _bootstrap_ros_pinocchio_numpy()

    import mujoco as _mj

    try:
        from convex_mpc.go2_robot_data import PinGo2Model as _PinGo2Model
        from convex_mpc.mujoco_model import MuJoCo_GO2_Model as _MuJoCo_GO2_Model
        from convex_mpc.com_trajectory import ComTraj as _ComTraj
        from convex_mpc.centroidal_mpc import CentroidalMPC as _CentroidalMPC
        from convex_mpc.wbc_controller import WbcController as _WbcController
        from convex_mpc.gait import Gait as _Gait
        import convex_mpc.gait as _gait_module
    except ImportError as e:
        print(f"Failed to import MPC modules: {e}")
        sys.exit(1)

    try:
        from go2_simulation.sensors.depth_camera import MujocoDepthCamera as _MujocoDepthCamera
        from go2_simulation.sensors.lidar_simulator import (
            MujocoLidarSimulator as _MujocoLidarSimulator,
        )
        from go2_simulation.visualization.rerun_logger import Go2RerunLogger as _Go2RerunLogger
        from go2_simulation.perception.depth_terrain import (
            DepthTerrainEstimator as _DepthTerrainEstimator,
            DepthTerrainMemory as _DepthTerrainMemory,
            FlatTerrainProvider as _FlatTerrainProvider,
        )
        from go2_simulation.perception.step_detector import StepObstacleDetector as _StepObstacleDetector
        from go2_simulation.control.step_climb import StepClimbStateMachine as _StepClimbStateMachine
        from go2_simulation.control.step_terrain import StepClimbTerrainProvider as _StepClimbTerrainProvider
    except ImportError:
        _MujocoDepthCamera = None
        _MujocoLidarSimulator = None
        _Go2RerunLogger = None
        _DepthTerrainEstimator = None
        _DepthTerrainMemory = None
        _FlatTerrainProvider = None
        _StepObstacleDetector = None
        _StepClimbStateMachine = None
        _StepClimbTerrainProvider = None

    mj = _mj
    PinGo2Model = _PinGo2Model
    MuJoCo_GO2_Model = _MuJoCo_GO2_Model
    ComTraj = _ComTraj
    CentroidalMPC = _CentroidalMPC
    WbcController = _WbcController
    Gait = _Gait
    gait_module = _gait_module
    MujocoDepthCamera = _MujocoDepthCamera
    MujocoLidarSimulator = _MujocoLidarSimulator
    Go2RerunLogger = _Go2RerunLogger
    DepthTerrainEstimator = _DepthTerrainEstimator
    DepthTerrainMemory = _DepthTerrainMemory
    FlatTerrainProvider = _FlatTerrainProvider
    StepObstacleDetector = _StepObstacleDetector
    StepClimbStateMachine = _StepClimbStateMachine
    StepClimbTerrainProvider = _StepClimbTerrainProvider
    POSTURE_KP = np.array([35.0, 45.0, 55.0] * 4)
    POSTURE_KD = np.array([1.8, 2.2, 2.6] * 4)

# --------------------------------------------------------------------------------
# Parameters & Settings
# --------------------------------------------------------------------------------

INITIAL_X_POS = 0.0
INITIAL_Y_POS = 0.0
INITIAL_Z_POS = 1.0  # 100cm height as requested
RENDER_HZ = 60.0 
RENDER_DT = 1.0 / RENDER_HZ
DEPTH_LOG_HZ = 10.0
DEPTH_LOG_DT = 1.0 / DEPTH_LOG_HZ
DEPTH_CAMERA_WIDTH = 320
DEPTH_CAMERA_HEIGHT = 240
DEPTH_CAMERA_NAME = "front_camera"
LIDAR_LOG_HZ = 5.55
LIDAR_LOG_DT = 1.0 / LIDAR_LOG_HZ
LIDAR_SITE_NAME = "radar"
ENABLE_LIDAR = os.environ.get("GO2_ENABLE_LIDAR", "0").strip().lower() in ("1", "true", "yes", "on")
ENABLE_DEPTH_OBSTACLE = os.environ.get("GO2_DEPTH_OBSTACLE", "1").strip().lower() not in ("0", "false", "no", "off")
RERUN_RECORD_PATH = os.environ.get("GO2_RERUN_RECORD_PATH")
HEADLESS = os.environ.get("GO2_HEADLESS", "0").strip().lower() in ("1", "true", "yes", "on")
MAX_SIM_TIME_S = float(os.environ.get("GO2_MAX_SIM_TIME", "30.0"))
ENABLE_RERUN = os.environ.get("GO2_ENABLE_RERUN", "1").strip().lower() not in ("0", "false", "no", "off")
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200       
CTRL_DT = 1.0 / CTRL_HZ
GAIT_HZ = 3.0
GAIT_DUTY = 0.60
GAIT_T = 1.0 / GAIT_HZ
SPEED_RAMP_START_S = 2.4
SPEED_RAMP_END_S = 5.0
DEPTH_OBSTACLE_MAX_CRUISE_X_VEL = 0.35
WALK_GAIT_START_X_VEL = 0.08
LEGS = ("FL", "FR", "RL", "RR")
LANDING_STABLE_TIME_S = 0.08
FOOT_CONTACT_MARGIN_M = 0.01
FOOT_FORCE_CONTACT_THRESHOLD_N = 3.0
MAX_CONSECUTIVE_SOLVER_FAILURES = 3
SOLVER_WARNING_INTERVAL_S = 0.25
POSTURE_KP = None
POSTURE_KD = None

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
    s = min(max((t - t_start) / (t_end - t_start), 0.0), 1.0)
    return float(s * s * (3.0 - 2.0 * s))


def get_ramped_x_vel(cmd: BodyCmdPhase, t: float) -> float:
    if cmd.x_vel <= 0.0:
        return 0.0
    x_vel = cmd.x_vel * smoothstep(SPEED_RAMP_START_S, SPEED_RAMP_END_S, t)
    if ENABLE_DEPTH_OBSTACLE:
        x_vel = min(x_vel, DEPTH_OBSTACLE_MAX_CRUISE_X_VEL)
    return x_vel


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


def make_command_from_phase(cmd: BodyCmdPhase, x_vel_cmd: float):
    return SimpleNamespace(
        state="NORMAL",
        x_vel=float(x_vel_cmd),
        y_vel=float(cmd.y_vel),
        z_pos=float(cmd.z_pos),
        yaw_rate=float(cmd.yaw_rate),
        gait_duty=float(cmd.gait_duty),
        gait_hz=float(cmd.gait_hz),
        swing_height=float(cmd.swing_height),
        z_weight=float(cmd.z_weight),
        vz_weight=float(cmd.vz_weight),
        obstacle_distance_m=float("inf"),
        obstacle_height_m=0.0,
        support_height_m=0.0,
        memory_support_height_m=0.0,
        terrain_delta_m=0.0,
        terrain_confidence=0.0,
        support_source="none",
        active_step_height=0.0,
        active_leg_group="none",
        step_distance_m=float("inf"),
        step_confidence=0.0,
        force_gait=False,
        forced_swing_legs=(),
        forced_stance_legs=(),
        step_edge_world_x=float("nan"),
        raw_step_distance_m=float("inf"),
        remembered_step_distance_m=float("inf"),
        transition_reason="",
    )


def _current_base_yaw(go2: PinGo2Model) -> float:
    return float(go2.current_config.compute_euler_angle_world()[2])


def _read_foot_positions(go2: PinGo2Model) -> dict:
    positions = {}
    for leg in LEGS:
        foot_pos, _ = go2.get_single_foot_state_in_world(leg)
        positions[leg] = np.asarray(foot_pos, dtype=float).copy()
    return positions


def _predict_touchdown_positions(gait: Gait, go2: PinGo2Model) -> dict:
    return {
        leg: np.asarray(gait.compute_touchdown_world_for_traj_purpose_only(go2, leg), dtype=float)
        for leg in LEGS
    }


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

    geom_to_leg = {gid: leg for leg, gid in foot_geom_ids.items() if gid != -1}
    foot_geom_set = set(geom_to_leg)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > FOOT_CONTACT_MARGIN_M:
            continue
        if contact.geom1 in geom_to_leg and contact.geom2 not in foot_geom_set and model.geom_bodyid[contact.geom2] == 0:
            contacts[geom_to_leg[contact.geom1]] = True
        elif contact.geom2 in geom_to_leg and contact.geom1 not in foot_geom_set and model.geom_bodyid[contact.geom1] == 0:
            contacts[geom_to_leg[contact.geom2]] = True

    for leg, sensor_id in force_sensor_ids.items():
        force = _sensor_force_vector(model, data, sensor_id)
        if np.linalg.norm(force) >= FOOT_FORCE_CONTACT_THRESHOLD_N:
            contacts[leg] = True

    return contacts, all(contacts.values())


def _format_distance(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "inf"


def _format_contact_mask(foot_contacts: dict[str, bool]) -> str:
    return "".join("1" if foot_contacts.get(leg, False) else "0" for leg in LEGS)


def _format_leg_tuple(legs: tuple[str, ...]) -> str:
    return ",".join(legs) if legs else "none"


def _format_contact_table_preview(contact_table, max_cols: int = 4) -> str:
    table = np.asarray(contact_table, dtype=int)
    if table.ndim != 2 or table.shape[0] != len(LEGS):
        return "unavailable"
    cols = min(int(max_cols), table.shape[1])
    snapshots = []
    for col in range(cols):
        snapshots.append("".join(str(int(table[row, col])) for row in range(len(LEGS))))
    suffix = "..." if table.shape[1] > cols else ""
    return "|".join(snapshots) + suffix


def _simulation_success(max_base_x: float, min_base_z: float, max_consecutive_solver_failures_seen: int) -> bool:
    return (
        max_base_x >= 4.35
        and min_base_z >= 0.18
        and max_consecutive_solver_failures_seen < MAX_CONSECUTIVE_SOLVER_FAILURES
    )


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


class _HeadlessViewer:
    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        return None

# --------------------------------------------------------------------------------
# Main Simulation
# --------------------------------------------------------------------------------

def run_simulation():
    _load_runtime_dependencies()

    # Initialize models
    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model()
    gait = Gait(GAIT_HZ, GAIT_DUTY)
    flat_terrain = FlatTerrainProvider() if FlatTerrainProvider is not None else None
    if flat_terrain is not None and hasattr(gait, "set_terrain_provider"):
        gait.set_terrain_provider(flat_terrain)
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

    # Rerun Logging and depth perception
    logger = (
        Go2RerunLogger(
            "Go2_Drop_Walk",
            enable_lidar=ENABLE_LIDAR,
            enable_depth_diagnostics=ENABLE_DEPTH_OBSTACLE,
            record_path=RERUN_RECORD_PATH,
        )
        if Go2RerunLogger and ENABLE_RERUN
        else None
    )
    if HEADLESS and RERUN_RECORD_PATH is None:
        logger = None
    depth_camera = None
    lidar = None
    terrain_estimator = DepthTerrainEstimator() if ENABLE_DEPTH_OBSTACLE and DepthTerrainEstimator else None
    terrain_memory = DepthTerrainMemory() if terrain_estimator is not None and DepthTerrainMemory else None
    step_detector = StepObstacleDetector() if terrain_estimator is not None and StepObstacleDetector else None
    step_climber = (
        StepClimbStateMachine(
            nominal_body_height_m=CMD_SCHEDULE[-1].z_pos,
            approach_speed_mps=0.12,
            stop_distance_m=0.45,
            front_lift_duration_s=0.45,
            rear_lift_duration_s=0.45,
        )
        if terrain_estimator is not None and StepClimbStateMachine
        else None
    )
    step_terrain = StepClimbTerrainProvider() if StepClimbTerrainProvider else None
    latest_depth_frame = None
    latest_cloud = None
    latest_height_map = None
    latest_obstacle = None
    latest_local_obstacle = None
    latest_step_estimate = None
    latest_control_cmd = make_command_from_phase(initial_cmd, initial_cmd.x_vel)
    if step_terrain is not None:
        step_terrain.set_base_provider(flat_terrain)
        if hasattr(gait, "set_terrain_provider"):
            gait.set_terrain_provider(step_terrain)
    elif flat_terrain is not None and hasattr(gait, "set_terrain_provider"):
        gait.set_terrain_provider(flat_terrain)

    if (logger is not None or terrain_estimator is not None) and MujocoDepthCamera is not None:
        try:
            depth_camera = MujocoDepthCamera(
                mujoco_go2.model,
                width=DEPTH_CAMERA_WIDTH,
                height=DEPTH_CAMERA_HEIGHT,
                camera_name=DEPTH_CAMERA_NAME,
            )
        except Exception as exc:
            print(f"[Rerun warning] Depth camera disabled: {type(exc).__name__}: {exc}")
    if ENABLE_LIDAR and logger is not None and MujocoLidarSimulator is not None:
        try:
            lidar = MujocoLidarSimulator(
                mujoco_go2.model,
                site_name=LIDAR_SITE_NAME,
            )
        except Exception as exc:
            print(f"[Rerun warning] LiDAR disabled: {type(exc).__name__}: {exc}")
    elif not ENABLE_LIDAR:
        print("[Perception] LiDAR disabled by GO2_ENABLE_LIDAR=0.")

    if terrain_estimator is None:
        print("[Perception] Depth obstacle control disabled.")
    else:
        print("[Perception] Depth obstacle control enabled.")
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
    next_depth_log_t = 0.0
    next_lidar_log_t = 0.0
    ctrl_decim = SIM_HZ // CTRL_HZ
    landing_stable_s = 0.0
    wbc_enabled = False
    consecutive_solver_failures = 0
    last_solver_warning_t = -np.inf
    gait_phase_time = 0.0
    walk_gait_active = False
    last_printed_control_state = None
    last_control_state_print_t = -np.inf
    max_base_x = float(go2.current_config.base_pos[0])
    min_base_z = float(go2.current_config.base_pos[2])
    max_consecutive_solver_failures_seen = 0
    controller_ready_for_step = False
    k = 0

    viewer_context = nullcontext(_HeadlessViewer())
    if not HEADLESS:
        import mujoco.viewer
        viewer_context = mujoco.viewer.launch_passive(mujoco_go2.model, mujoco_go2.data)
    try:
        with viewer_context as viewer:
            try:
                while viewer.is_running():
                    time_now = float(mujoco_go2.data.time)
                    max_base_x = max(max_base_x, float(go2.current_config.base_pos[0]))
                    min_base_z = min(min_base_z, float(go2.current_config.base_pos[2]))
                    if HEADLESS and time_now >= MAX_SIM_TIME_S:
                        break

                    # Log to Rerun
                    if logger and (k % 5 == 0):
                        forces = {n[:2]: mujoco_go2.data.sensordata[mujoco_go2.model.sensor_adr[f_sensor_ids[n]] + 2]
                                 if f_sensor_ids[n] != -1 else 0.0 for n in f_sensor_names}
                        torques = {n[:2]: mujoco_go2.data.sensordata[mujoco_go2.model.sensor_adr[t_sensor_ids[n]]]
                                  if t_sensor_ids[n] != -1 else 0.0 for n in t_sensor_names}

                        logger.log_sensor_forces(time_now, forces)
                        logger.log_thigh_torques(time_now, torques)
                        logger.log_body_height(time_now, go2.current_config.base_pos[2])

                    if depth_camera and time_now >= next_depth_log_t:
                        latest_depth_frame = depth_camera.capture(mujoco_go2.data, time_now)
                        if logger:
                            logger.log_depth_frame(latest_depth_frame)

                        if terrain_estimator is not None:
                            mujoco_go2.update_pin_with_mujoco(go2)
                            latest_cloud, latest_height_map, latest_local_obstacle = terrain_estimator.update(
                                latest_depth_frame,
                                go2.current_config.base_pos,
                                _current_base_yaw(go2),
                            )
                            if terrain_memory is not None:
                                terrain_memory.update(
                                    latest_height_map,
                                    go2.current_config.base_pos,
                                    time_now,
                                )
                            if step_detector is not None:
                                latest_step_estimate = step_detector.update(latest_height_map, time_now)
                            if logger:
                                logger.log_depth_diagnostics(
                                    latest_cloud, latest_height_map, latest_local_obstacle, terrain_memory
                                )
                                if latest_step_estimate is not None and hasattr(logger, "log_step_state"):
                                    logger.log_step_state(time_now, latest_step_estimate, latest_control_cmd)
                        next_depth_log_t += DEPTH_LOG_DT

                    if logger and lidar and time_now >= next_lidar_log_t:
                        logger.log_lidar_frame(lidar.capture(mujoco_go2.data, time_now))
                        next_lidar_log_t += LIDAR_LOG_DT

                    # Control Loop
                    if (k % ctrl_decim) == 0:
                        cmd = get_body_cmd(time_now)
                        raw_x_vel_cmd = get_ramped_x_vel(cmd, time_now)

                        mujoco_go2.update_pin_with_mujoco(go2)
                        foot_contacts, all_feet_contact = _read_foot_contacts(
                            mujoco_go2.model, mujoco_go2.data, floor_geom_id, foot_geom_ids, force_sensor_ids
                        )
                        foot_positions = _read_foot_positions(go2)
                        if step_climber is not None:
                            step_cmd = step_climber.update(
                                time_now,
                                latest_step_estimate,
                                foot_contacts,
                                base_x=float(go2.current_config.base_pos[0]),
                                raw_x_vel=raw_x_vel_cmd,
                                raw_z_pos=float(cmd.z_pos),
                                controller_ready=controller_ready_for_step,
                            )
                            latest_control_cmd = SimpleNamespace(
                                state=step_cmd.state,
                                x_vel=step_cmd.x_vel,
                                y_vel=float(cmd.y_vel),
                                z_pos=step_cmd.z_pos,
                                yaw_rate=float(cmd.yaw_rate),
                                gait_duty=step_cmd.gait_duty,
                                gait_hz=step_cmd.gait_hz,
                                swing_height=step_cmd.swing_height,
                                z_weight=float(cmd.z_weight),
                                vz_weight=float(cmd.vz_weight),
                                obstacle_distance_m=step_cmd.step_distance_m,
                                obstacle_height_m=step_cmd.active_step_height,
                                support_height_m=0.0,
                                memory_support_height_m=0.0,
                                terrain_delta_m=step_cmd.active_step_height,
                                terrain_confidence=step_cmd.step_confidence,
                                support_source="step_state",
                                active_step_height=step_cmd.active_step_height,
                                active_leg_group=step_cmd.active_leg_group,
                                step_distance_m=step_cmd.step_distance_m,
                                step_confidence=step_cmd.step_confidence,
                                force_gait=step_cmd.force_gait,
                                forced_swing_legs=step_cmd.forced_swing_legs,
                                forced_stance_legs=step_cmd.forced_stance_legs,
                                step_edge_world_x=step_cmd.step_edge_world_x,
                                raw_step_distance_m=step_cmd.raw_step_distance_m,
                                remembered_step_distance_m=step_cmd.remembered_step_distance_m,
                                transition_reason=step_cmd.transition_reason,
                            )
                        else:
                            latest_control_cmd = make_command_from_phase(cmd, raw_x_vel_cmd)

                        x_vel_cmd = latest_control_cmd.x_vel
                        transition_reason = getattr(latest_control_cmd, "transition_reason", "")
                        if (
                            latest_control_cmd.state != last_printed_control_state
                            or time_now - last_control_state_print_t >= 0.5
                            or transition_reason
                        ):
                            distance = latest_control_cmd.obstacle_distance_m
                            edge_x = getattr(latest_control_cmd, "step_edge_world_x", float("nan"))
                            edge_text = f"{edge_x:.2f}" if np.isfinite(edge_x) else "nan"
                            contacts_text = _format_contact_mask(foot_contacts)
                            print(
                                f"\n[DepthCtrl] t={time_now:.3f}s state={latest_control_cmd.state} "
                                f"base_x={go2.current_config.base_pos[0]:.3f} base_z={go2.current_config.base_pos[2]:.3f} "
                                f"x_vel={latest_control_cmd.x_vel:.3f} z={latest_control_cmd.z_pos:.3f} "
                                f"swing={latest_control_cmd.swing_height:.3f} "
                                f"obs_d={_format_distance(distance)} "
                                f"raw_d={_format_distance(getattr(latest_control_cmd, 'raw_step_distance_m', float('inf')))} "
                                f"mem_d={_format_distance(getattr(latest_control_cmd, 'remembered_step_distance_m', float('inf')))} "
                                f"edge_x={edge_text} obs_h={latest_control_cmd.obstacle_height_m:.3f} "
                                f"group={latest_control_cmd.active_leg_group} "
                                f"contacts={contacts_text} "
                                f"conf={latest_control_cmd.terrain_confidence:.2f}"
                            )
                            if transition_reason:
                                print(f"[StepClimb] t={time_now:.3f}s {transition_reason}")
                            last_printed_control_state = latest_control_cmd.state
                            last_control_state_print_t = time_now

                        use_walk_gait = should_use_walk_gait(x_vel_cmd) or bool(getattr(latest_control_cmd, "force_gait", False))
                        effective_gait_duty = latest_control_cmd.gait_duty if use_walk_gait else 1.0

                        # Update Gait & MPC Parameters
                        apply_gait_params(gait, latest_control_cmd.gait_hz, effective_gait_duty)
                        if hasattr(gait, "set_forced_leg_contacts"):
                            gait.set_forced_leg_contacts(
                                swing_legs=getattr(latest_control_cmd, "forced_swing_legs", ()),
                                stance_legs=getattr(latest_control_cmd, "forced_stance_legs", ()),
                            )
                        if step_terrain is not None:
                            edge_x = getattr(latest_control_cmd, "step_edge_world_x", float("nan"))
                            try:
                                step_terrain.update(
                                    active_leg_group=getattr(latest_control_cmd, "active_leg_group", "none"),
                                    step_height_m=getattr(latest_control_cmd, "active_step_height", 0.0),
                                    touchdown_targets=None,
                                    step_edge_world_x=edge_x,
                                )
                                step_terrain.update(
                                    active_leg_group=getattr(latest_control_cmd, "active_leg_group", "none"),
                                    step_height_m=getattr(latest_control_cmd, "active_step_height", 0.0),
                                    touchdown_targets=_predict_touchdown_positions(gait, go2),
                                    step_edge_world_x=edge_x,
                                )
                            except Exception:
                                step_terrain.update(
                                    active_leg_group=getattr(latest_control_cmd, "active_leg_group", "none"),
                                    step_height_m=getattr(latest_control_cmd, "active_step_height", 0.0),
                                    touchdown_targets=None,
                                    step_edge_world_x=edge_x,
                                )
                        mpc.Q[2, 2] = latest_control_cmd.z_weight
                        mpc.Q[8, 8] = latest_control_cmd.vz_weight

                        gait_module.HEIGHT_SWING = latest_control_cmd.swing_height

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

                        if logger and (k % max(ctrl_decim, int(0.1 * SIM_HZ)) == 0):
                            logger.log_control_state(time_now, latest_control_cmd, go2.current_config.base_pos)
                            if latest_step_estimate is not None and hasattr(logger, "log_step_state"):
                                logger.log_step_state(time_now, latest_step_estimate, latest_control_cmd)
                            try:
                                logger.log_footholds(
                                    time_now,
                                    foot_positions,
                                    _predict_touchdown_positions(gait, go2),
                                    foot_contacts,
                                )
                            except Exception:
                                pass

                        if not wbc_enabled:
                            landing_stable_s = landing_stable_s + CTRL_DT if all_feet_contact else 0.0
                            tau_hold = _posture_hold_torque(mujoco_go2.data, posture_q_ref, TAU_LIM)

                            if landing_stable_s >= LANDING_STABLE_TIME_S:
                                wbc_enabled = True
                                consecutive_solver_failures = 0
                                ctrl_i = 0
                                U_opt = np.zeros_like(U_opt)
                                X_opt = np.zeros_like(X_opt)
                                _clear_solver_warm_start(mpc, wbc_controller)
                                _reset_wbc_gait_memory(wbc_controller)
                                print(f"[Landing] Four-foot contact stable for {landing_stable_s:.3f}s at t={time_now:.3f}s. Enabling MPC/WBC.")
                                controller_ready_for_step = True
                            else:
                                if k % max(ctrl_decim, int(0.25 * SIM_HZ)) == 0:
                                    mask = "".join("1" if foot_contacts[leg] else "0" for leg in LEGS)
                                    print(f"\r[Landing] t={time_now:.3f}s contacts {mask} stable={landing_stable_s:.3f}s", end="", flush=True)
                                ctrl_i += 1

                        if wbc_enabled:
                            solver_failure_layer = "controller"
                            try:
                                if (ctrl_i % STEPS_PER_MPC) == 0:
                                    traj.generate_traj(
                                        go2,
                                        gait,
                                        controller_time,
                                        x_vel_cmd,
                                        latest_control_cmd.y_vel,
                                        latest_control_cmd.z_pos,
                                        latest_control_cmd.yaw_rate,
                                        time_step=MPC_DT,
                                        N=N_MPC,
                                    )

                                    # Keep the reference as a straight line along world x.
                                    lock_straight_reference(traj, go2)

                                    solver_failure_layer = "MPC"
                                    sol = mpc.solve_QP(go2, traj, False)
                                    w_opt = sol["x"].full().flatten()
                                    N = traj.N
                                    U_opt = w_opt[12 * N :].reshape((12, N), order="F")
                                    X_opt = w_opt[:12 * N].reshape((12, N), order="F")

                                # WBC: single QP replaces per-leg Jacobian mapping
                                x0_mpc = X_opt[:, 1] if X_opt is not None else None
                                solver_failure_layer = "WBC"
                                wbc_out = wbc_controller.solve(
                                    go2, gait, U_opt[:, 0], controller_time, x0_mpc
                                )
                                tau_hold = np.clip(wbc_out.tau, -TAU_LIM, TAU_LIM)
                                ctrl_i += 1
                                consecutive_solver_failures = 0
                                controller_ready_for_step = True
                            except Exception as exc:
                                consecutive_solver_failures += 1
                                max_consecutive_solver_failures_seen = max(
                                    max_consecutive_solver_failures_seen,
                                    consecutive_solver_failures,
                                )
                                controller_ready_for_step = False
                                _clear_solver_warm_start(mpc, wbc_controller)
                                tau_hold = _posture_hold_torque(mujoco_go2.data, posture_q_ref, TAU_LIM)

                                if time_now - last_solver_warning_t >= SOLVER_WARNING_INTERVAL_S:
                                    contact_preview = (
                                        _format_contact_table_preview(traj.contact_table)
                                        if hasattr(traj, "contact_table")
                                        else "unavailable"
                                    )
                                    print(
                                        f"\n[Controller warning] t={time_now:.3f}s "
                                        f"{solver_failure_layer} solve failed ({type(exc).__name__}: {exc}). "
                                        f"state={latest_control_cmd.state} group={latest_control_cmd.active_leg_group} "
                                        f"contacts={_format_contact_mask(foot_contacts)} "
                                        f"forced_swing={_format_leg_tuple(getattr(latest_control_cmd, 'forced_swing_legs', ()))} "
                                        f"forced_stance={_format_leg_tuple(getattr(latest_control_cmd, 'forced_stance_legs', ()))} "
                                        f"contact_table={contact_preview}. Using posture hold torque."
                                    )
                                    last_solver_warning_t = time_now

                                if consecutive_solver_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES:
                                    wbc_enabled = False
                                    landing_stable_s = 0.0
                                    ctrl_i = 0
                                    U_opt = np.zeros_like(U_opt)
                                    X_opt = np.zeros_like(X_opt)
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

                    if not HEADLESS and time_now >= next_render_t:
                        viewer.sync()
                        next_render_t += RENDER_DT

                    if not HEADLESS:
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
    finally:
        if depth_camera is not None:
            depth_camera.close()

    success = _simulation_success(max_base_x, min_base_z, max_consecutive_solver_failures_seen)
    print(
        f"[Result] success={int(success)} max_base_x={max_base_x:.3f} "
        f"min_base_z={min_base_z:.3f} max_solver_failures={max_consecutive_solver_failures_seen} "
        f"sim_time={float(mujoco_go2.data.time):.3f}"
    )
    print(f"Simulation ended.")


def main() -> None:
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception:
        print("\nFatal error before the MuJoCo viewer could be kept open:")
        traceback.print_exc()
        _pause_on_error_if_interactive()


if __name__ == "__main__":
    main()
