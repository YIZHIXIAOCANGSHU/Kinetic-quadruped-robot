import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from go2_simulation.control.step_climb import StepClimbStateMachine
from go2_simulation.control.step_terrain import StepClimbTerrainProvider
from go2_simulation.perception.depth_terrain import LocalHeightMap
from go2_simulation.perception.step_detector import StepObstacleDetector, StepEstimate


def _height_map(step_height: float | None = None) -> LocalHeightMap:
    x_edges = np.arange(0.15, 1.25, 0.05, dtype=np.float32)
    y_edges = np.arange(-0.35, 0.40, 0.05, dtype=np.float32)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    heights = np.zeros((nx, ny), dtype=np.float32)
    if step_height is not None:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        heights[x_centers >= 0.55, :] = step_height
    return LocalHeightMap(
        time_s=0.0,
        x_edges=x_edges,
        y_edges=y_edges,
        heights_m=heights,
        counts=np.full((nx, ny), 8, dtype=np.int32),
        confidence=np.ones((nx, ny), dtype=np.float32),
        origin_world=np.zeros(3, dtype=np.float32),
        yaw=0.0,
    )


def test_step_detector_ignores_flat_ground():
    detector = StepObstacleDetector(min_confirm_frames=3)

    estimate = None
    for t in (0.0, 0.1, 0.2, 0.3):
        estimate = detector.update(_height_map(None), time_s=t)

    assert estimate.valid is False


def test_step_detector_requires_consecutive_confirmed_frames():
    detector = StepObstacleDetector(min_confirm_frames=3)

    first = detector.update(_height_map(0.10), time_s=0.0)
    second = detector.update(_height_map(0.10), time_s=0.1)
    third = detector.update(_height_map(0.10), time_s=0.2)

    assert first.valid is False
    assert second.valid is False
    assert third.valid is True
    assert third.distance_m == pytest.approx(0.575, abs=0.03)
    assert third.height_m == pytest.approx(0.10, abs=0.02)


def test_step_detector_rejects_single_frame_noise():
    detector = StepObstacleDetector(min_confirm_frames=3)

    detector.update(_height_map(None), time_s=0.0)
    noisy = detector.update(_height_map(0.10), time_s=0.1)
    clean = detector.update(_height_map(None), time_s=0.2)

    assert noisy.valid is False
    assert clean.valid is False


def test_step_climb_state_machine_uses_min_dwell_and_ordered_phases():
    machine = StepClimbStateMachine(min_state_duration_s=0.3)
    estimate = StepEstimate(0.0, True, 0.28, 0.10, 1.0, 0.28)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}

    out0 = machine.update(0.0, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    out1 = machine.update(0.1, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    out2 = machine.update(0.31, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    out3 = machine.update(0.67, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    out4 = machine.update(1.03, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    out5 = machine.update(1.61, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)
    out6 = machine.update(1.97, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)
    out7 = machine.update(2.33, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)
    out8 = machine.update(2.85, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)

    assert out0.state == "STOP_BEFORE_STEP"
    assert out1.state == "STOP_BEFORE_STEP"
    assert out2.state == "LIFT_FL"
    assert out3.state == "LIFT_FR"
    assert out4.state == "SHIFT_FORWARD"
    assert out5.state == "LIFT_RL"
    assert out6.state == "LIFT_RR"
    assert out7.state == "RECOVER_WALK"
    assert out8.state in ("WALK_FLAT", "APPROACH_STEP")
    assert out2.active_leg_group == "FL"
    assert out3.active_leg_group == "FR"
    assert out5.active_leg_group == "RL"
    assert out6.active_leg_group == "RR"
    assert out2.force_gait is True
    assert out5.force_gait is True
    assert out2.forced_swing_legs == ("FL",)
    assert out2.forced_stance_legs == ("FR", "RL", "RR")
    assert out3.forced_swing_legs == ("FR",)
    assert out3.forced_stance_legs == ("FL", "RL", "RR")
    assert out5.forced_swing_legs == ("RL",)
    assert out5.forced_stance_legs == ("FL", "FR", "RR")
    assert out6.forced_swing_legs == ("RR",)
    assert out6.forced_stance_legs == ("FL", "FR", "RL")


def test_step_climb_stop_shift_and_recover_force_all_legs_to_stance():
    machine = StepClimbStateMachine(min_state_duration_s=0.3)
    estimate = StepEstimate(0.0, True, 0.28, 0.10, 1.0, 0.28)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}

    stop = machine.update(0.0, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    machine.update(0.31, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    machine.update(0.67, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    shift = machine.update(1.03, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    machine.update(1.61, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)
    machine.update(1.97, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)
    recover = machine.update(2.33, estimate, contacts, base_x=0.12, raw_x_vel=0.35, raw_z_pos=0.27)

    assert stop.state == "STOP_BEFORE_STEP"
    assert stop.forced_stance_legs == ("FL", "FR", "RL", "RR")
    assert shift.state == "SHIFT_FORWARD"
    assert shift.forced_stance_legs == ("FL", "FR", "RL", "RR")
    assert recover.state == "RECOVER_WALK"
    assert recover.forced_stance_legs == ("FL", "FR", "RL", "RR")


def test_gait_forced_leg_contacts_override_trot_mask():
    gait_path = SRC_DIR / "convex_mpc" / "gait.py"
    spec = importlib.util.spec_from_file_location("test_convex_mpc_gait", gait_path)
    gait_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gait_module)

    gait = gait_module.Gait(3.0, 0.60)
    gait.set_forced_leg_contacts(swing_legs=("FL", "FR"), stance_legs=("RL", "RR"))

    table = gait.compute_contact_table(0.0, 0.01, 5)

    assert np.all(table[0, :] == 0)
    assert np.all(table[1, :] == 0)
    assert np.all(table[2, :] == 1)
    assert np.all(table[3, :] == 1)


def test_step_climber_waits_for_controller_ready_before_lifting():
    machine = StepClimbStateMachine(min_state_duration_s=0.3)
    estimate = StepEstimate(0.0, True, 0.28, 0.10, 1.0, 0.28)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}

    stop = machine.update(0.0, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    waiting = machine.update(
        0.31,
        estimate,
        contacts,
        base_x=0.0,
        raw_x_vel=0.35,
        raw_z_pos=0.27,
        controller_ready=False,
    )
    lift = machine.update(
        0.62,
        estimate,
        contacts,
        base_x=0.0,
        raw_x_vel=0.35,
        raw_z_pos=0.27,
        controller_ready=True,
    )

    assert stop.state == "STOP_BEFORE_STEP"
    assert waiting.state == "STOP_BEFORE_STEP"
    assert waiting.forced_stance_legs == ("FL", "FR", "RL", "RR")
    assert lift.state == "LIFT_FL"


def test_step_climber_returns_to_stance_wait_if_controller_drops_out():
    machine = StepClimbStateMachine(min_state_duration_s=0.3)
    estimate = StepEstimate(0.0, True, 0.28, 0.10, 1.0, 0.28)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}

    machine.update(0.0, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    lift = machine.update(0.31, estimate, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    held = machine.update(
        0.36,
        estimate,
        contacts,
        base_x=0.0,
        raw_x_vel=0.35,
        raw_z_pos=0.27,
        controller_ready=False,
    )

    assert lift.state == "LIFT_FL"
    assert held.state == "STOP_BEFORE_STEP"
    assert held.forced_swing_legs == ()
    assert held.forced_stance_legs == ("FL", "FR", "RL", "RR")


def test_step_climber_remembers_edge_when_depth_temporarily_loses_step():
    machine = StepClimbStateMachine(min_state_duration_s=0.3, stop_distance_m=0.35)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}
    visible = StepEstimate(0.0, True, 0.60, 0.10, 1.0, 0.60)
    lost = StepEstimate(0.1, False, np.inf, 0.0, 0.0, np.inf)

    approach = machine.update(0.0, visible, contacts, base_x=0.0, raw_x_vel=0.35, raw_z_pos=0.27)
    stop = machine.update(0.4, lost, contacts, base_x=0.30, raw_x_vel=0.35, raw_z_pos=0.27)

    assert approach.state == "APPROACH_STEP"
    assert approach.step_edge_world_x == pytest.approx(0.60)
    assert stop.state == "STOP_BEFORE_STEP"
    assert stop.step_distance_m == pytest.approx(0.30)


def test_step_climber_keeps_locked_edge_when_farther_step_is_detected():
    machine = StepClimbStateMachine(min_state_duration_s=0.3, stop_distance_m=0.45)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}
    first_step = StepEstimate(0.0, True, 0.80, 0.10, 1.0, 0.80)
    farther_step = StepEstimate(0.1, True, 1.05, 0.10, 1.0, 1.05)

    approach = machine.update(0.0, first_step, contacts, base_x=1.00, raw_x_vel=0.35, raw_z_pos=0.27)
    still_approach = machine.update(0.1, farther_step, contacts, base_x=1.10, raw_x_vel=0.35, raw_z_pos=0.27)
    stop = machine.update(0.4, farther_step, contacts, base_x=1.36, raw_x_vel=0.35, raw_z_pos=0.27)

    assert approach.state == "APPROACH_STEP"
    assert approach.step_edge_world_x == pytest.approx(1.80)
    assert still_approach.step_edge_world_x == pytest.approx(1.80)
    assert still_approach.step_distance_m == pytest.approx(0.70)
    assert stop.state == "STOP_BEFORE_STEP"
    assert stop.step_distance_m == pytest.approx(0.44)


def test_step_climber_uses_locked_edge_when_detection_temporarily_drops():
    machine = StepClimbStateMachine(min_state_duration_s=0.3, stop_distance_m=0.45)
    contacts = {leg: True for leg in ("FL", "FR", "RL", "RR")}
    visible = StepEstimate(0.0, True, 0.80, 0.10, 1.0, 0.80)
    lost = StepEstimate(0.1, False, np.inf, 0.0, 0.0, np.inf)

    approach = machine.update(0.0, visible, contacts, base_x=1.00, raw_x_vel=0.35, raw_z_pos=0.27)
    stop = machine.update(0.4, lost, contacts, base_x=1.36, raw_x_vel=0.35, raw_z_pos=0.27)

    assert approach.state == "APPROACH_STEP"
    assert stop.state == "STOP_BEFORE_STEP"
    assert stop.step_edge_world_x == pytest.approx(1.80)
    assert stop.step_distance_m == pytest.approx(0.44)


def test_step_terrain_places_single_active_touchdown_inside_detected_edge():
    provider = StepClimbTerrainProvider()
    provider.update(
        active_leg_group="FL",
        step_height_m=0.10,
        touchdown_targets={"FL": np.asarray([0.40, 0.10, 0.02])},
        step_edge_world_x=0.50,
    )

    touchdown = provider.override_touchdown_world("FL", np.asarray([0.42, 0.10, 0.02]))
    other_front_touchdown = provider.override_touchdown_world("FR", np.asarray([0.42, -0.10, 0.02]))

    assert touchdown[0] == pytest.approx(0.44)
    assert touchdown[2] == pytest.approx(0.12)
    assert other_front_touchdown[0] == pytest.approx(0.42)
    assert other_front_touchdown[2] == pytest.approx(0.02)
