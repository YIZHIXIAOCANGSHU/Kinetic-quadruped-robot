import sys
import os
import time
import mujoco as mj
import numpy as np
from dataclasses import dataclass

# Path setup for convex-mpc
curr_dir = os.path.dirname(os.path.abspath(__file__))
mpc_src_path = os.path.join(curr_dir, "go2-convex-mpc", "src")
sys.path.append(mpc_src_path)
os.environ["MPLBACKEND"] = "TkAgg"

try:
    from convex_mpc.go2_robot_data import PinGo2Model
    from convex_mpc.mujoco_model import MuJoCo_GO2_Model
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.centroidal_mpc import CentroidalMPC
    from convex_mpc.leg_controller import LegController
    from convex_mpc.gait import Gait
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

RUN_SIM_LENGTH_S = 8.0
RENDER_HZ = 60.0 
RENDER_DT = 1.0 / RENDER_HZ
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200       
CTRL_DT = 1.0 / CTRL_HZ

@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float
    gait_duty: float = 0.5
    gait_hz: float = 2.5
    swing_height: float = 0.1
    z_weight: float = 150.0
    vz_weight: float = 150.0

# Command Schedule: Landing then Natural Walking
CMD_SCHEDULE = [
    # 0.0~1.2s: Landing phase (Prep for impact, high stiffness)
    BodyCmdPhase(0.0, 1.2, 0.0, 0.0, 0.28, 0.0, gait_duty=1.0, z_weight=500.0, vz_weight=500.0),
    
    # 1.2~8.0s: Natural Walking phase (Trot gait)
    BodyCmdPhase(1.2, 8.0, 0.45, 0.0, 0.28, 0.0, gait_duty=0.5, gait_hz=2.5, swing_height=0.1),
]

def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase
    # Default Stand
    return BodyCmdPhase(0.0, 100.0, 0.0, 0.0, 0.28, 0.0, gait_duty=1.0)

# --------------------------------------------------------------------------------
# Main Simulation
# --------------------------------------------------------------------------------

def run_simulation():
    # Initialize models
    go2 = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model()
    leg_controller = LegController()
    gait = Gait(2.5, 0.5)
    
    # Initial configuration
    q_init = go2.current_config.get_q()
    q_init[0], q_init[1], q_init[2] = INITIAL_X_POS, INITIAL_Y_POS, INITIAL_Z_POS
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    # MPC Tuning
    MPC_DT = 0.025
    N_MPC = 16
    MPC_HZ = 1.0 / MPC_DT
    STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))
    
    # Global MPC Weight Override
    import convex_mpc.centroidal_mpc
    convex_mpc.centroidal_mpc.COST_MATRIX_Q = np.diag([
        20.0, 100.0, 150.0,   # Position (x, y, z)
        100.0, 100.0, 150.0,  # Orientation (roll, pitch, yaw)
        50.0, 50.0, 150.0,    # Velocity (vx, vy, vz)
        20.0, 20.0, 50.0      # Angular velocity (wx, wy, wz)
    ])

    traj = ComTraj(go2)
    traj.generate_traj(go2, gait, 0.0, 0.0, 0.0, 0.28, 0.0, time_step=MPC_DT, N=N_MPC)
    mpc = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    # Torque Limits
    TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)
    LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6), "RL": slice(6, 9), "RR": slice(9, 12)}

    # Rerun Logging
    logger = QuadrupedForceLogger("Go2_Drop_Walk") if QuadrupedForceLogger else None
    f_sensor_names = ["FL_force", "FR_force", "RL_force", "RR_force"]
    t_sensor_names = ["FL_thigh_torque", "FR_thigh_torque", "RL_thigh_torque", "RR_thigh_torque"]
    
    f_sensor_ids = {n: mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_SENSOR, n) for n in f_sensor_names}
    t_sensor_ids = {n: mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_SENSOR, n) for n in t_sensor_names}

    print(f"Starting Go2 simulation: Drop from {INITIAL_Z_POS}m and Walk.")
    
    ctrl_i = 0
    tau_hold = np.zeros(12, dtype=float)
    next_render_t = 0.0
    sim_steps = int(RUN_SIM_LENGTH_S * SIM_HZ)
    ctrl_decim = SIM_HZ // CTRL_HZ

    import mujoco.viewer
    with mujoco.viewer.launch_passive(mujoco_go2.model, mujoco_go2.data) as viewer:
        for k in range(sim_steps):
            if not viewer.is_running():
                break
                
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
                
                # Update Gait & MPC Parameters
                gait.gait_duty = cmd.gait_duty
                gait.gait_hz = cmd.gait_hz
                mpc.Q[2, 2] = cmd.z_weight
                mpc.Q[8, 8] = cmd.vz_weight
                
                # Inject swing height via library variable
                try:
                    import convex_mpc.gait as gait_module
                    gait_module.HEIGHT_SWING = cmd.swing_height
                except:
                    pass

                mujoco_go2.update_pin_with_mujoco(go2)

                if (ctrl_i % STEPS_PER_MPC) == 0:
                    traj.generate_traj(go2, gait, time_now, cmd.x_vel, cmd.y_vel, cmd.z_pos, cmd.yaw_rate, time_step=MPC_DT, N=N_MPC)
                    
                    # Correction for lateral drift/heading
                    traj.pos_traj_world[1, :] = 0.8 * traj.pos_traj_world[1, :] + 0.2 * INITIAL_Y_POS
                    traj.rpy_traj_world[2, :] = 0.8 * traj.rpy_traj_world[2, :] + 0.2 * 0.0

                    sol = mpc.solve_QP(go2, traj, False)
                    w_opt = sol["x"].full().flatten()
                    U_opt = w_opt[12 * traj.N :].reshape((12, traj.N), order="F")

                for leg_name in ["FL", "FR", "RL", "RR"]:
                    leg = leg_controller.compute_leg_torque(
                        leg_name, go2, gait, U_opt[LEG_SLICE[leg_name], 0], time_now
                    )
                    tau_hold[LEG_SLICE[leg_name]] = leg.tau
                
                tau_hold = np.clip(tau_hold, -TAU_LIM, TAU_LIM)
                ctrl_i += 1

            mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
            mujoco_go2.set_joint_torque(tau_hold)
            mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
            
            if time_now >= next_render_t:
                viewer.sync()
                next_render_t += RENDER_DT
            
            # Slow down simulation to match real time
            time.sleep(SIM_DT * 0.5)

    print(f"Simulation ended.")

if __name__ == "__main__":
    run_simulation()
