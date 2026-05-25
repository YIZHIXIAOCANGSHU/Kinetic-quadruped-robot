from types import SimpleNamespace

import pytest

from go2_simulation.control.obstacle_command import ObstacleAwareCommandAdapter
from go2_simulation.perception.depth_terrain import ObstacleEstimate


def _raw_cmd():
    return SimpleNamespace(
        x_vel=0.7,
        y_vel=0.0,
        z_pos=0.27,
        yaw_rate=0.0,
        gait_duty=0.60,
        gait_hz=3.0,
        swing_height=0.08,
        z_weight=220.0,
        vz_weight=220.0,
    )


def test_forward_obstacle_does_not_pre_raise_body_without_support():
    adapter = ObstacleAwareCommandAdapter(nominal_body_height_m=0.27)
    obstacle = ObstacleEstimate(
        time_s=0.0,
        state_hint="APPROACH_UP",
        distance_m=0.6,
        height_m=0.30,
        support_height_m=0.0,
        confidence=1.0,
        valid=True,
    )

    command = adapter.adapt(_raw_cmd(), ramped_x_vel=0.35, obstacle=obstacle, dt=0.005)

    assert command.state == "APPROACH_UP"
    assert command.x_vel <= 0.15
    assert command.swing_height >= 0.16
    assert command.z_pos == pytest.approx(0.27)


def test_confirmed_support_height_raises_body_smoothly():
    adapter = ObstacleAwareCommandAdapter(nominal_body_height_m=0.27, body_height_rate_limit_mps=0.15)
    obstacle = ObstacleEstimate(
        time_s=0.0,
        state_hint="ON_PLATFORM",
        distance_m=float("inf"),
        height_m=0.0,
        support_height_m=0.20,
        confidence=1.0,
        valid=True,
    )

    command = adapter.adapt(_raw_cmd(), ramped_x_vel=0.35, obstacle=obstacle, dt=0.10)

    assert command.state == "ON_PLATFORM"
    assert command.z_pos == pytest.approx(0.285)
    assert command.z_pos < 0.47


def test_climbing_speed_stays_above_walk_gait_threshold():
    adapter = ObstacleAwareCommandAdapter(nominal_body_height_m=0.27)
    obstacle = ObstacleEstimate(
        time_s=0.0,
        state_hint="CLIMBING",
        distance_m=0.30,
        height_m=0.20,
        support_height_m=0.0,
        confidence=1.0,
        valid=True,
    )

    command = adapter.adapt(_raw_cmd(), ramped_x_vel=0.35, obstacle=obstacle, dt=0.005)

    assert command.state == "CLIMBING"
    assert command.x_vel >= 0.08
