import numpy as np
import pytest

from go2_simulation.perception.depth_terrain import (
    DepthTerrainMemory,
    LocalHeightMap,
    ObstacleEstimate,
)


def _height_map(height: float = 0.20) -> LocalHeightMap:
    x_edges = np.asarray([0.0, 0.05, 0.10, 0.15], dtype=np.float32)
    y_edges = np.asarray([-0.05, 0.0, 0.05], dtype=np.float32)
    heights = np.full((3, 2), height, dtype=np.float32)
    counts = np.full((3, 2), 8, dtype=np.int32)
    confidence = np.ones((3, 2), dtype=np.float32)
    return LocalHeightMap(
        time_s=1.0,
        x_edges=x_edges,
        y_edges=y_edges,
        heights_m=heights,
        counts=counts,
        confidence=confidence,
        origin_world=np.zeros(3, dtype=np.float32),
        yaw=0.0,
    )


def test_memory_height_at_world_tracks_height_map():
    memory = DepthTerrainMemory()
    memory.update(_height_map(0.20), np.zeros(3), time_s=1.0)

    assert memory.height_at_world(0.075, 0.025) == pytest.approx(0.20)
    assert memory.max_height_between_world(
        np.asarray([0.0, 0.0, 0.0]), np.asarray([0.15, 0.0, 0.0])
    ) == pytest.approx(0.20)


def test_contact_feet_confirm_support_height():
    memory = DepthTerrainMemory()
    memory.update(_height_map(0.20), np.zeros(3), time_s=1.0)
    local_obstacle = ObstacleEstimate(1.0, "NORMAL", np.inf, 0.0, 0.0, 1.0, True)
    foot_positions = {
        "FL": np.asarray([0.075, 0.025, 0.22]),
        "FR": np.asarray([0.075, -0.025, 0.22]),
        "RL": np.asarray([0.025, 0.025, 0.22]),
        "RR": np.asarray([0.025, -0.025, 0.22]),
    }
    foot_contacts = {leg: True for leg in foot_positions}

    estimate = memory.estimate_obstacle(
        local_obstacle,
        _height_map(0.20),
        foot_positions,
        foot_contacts,
        np.zeros(3),
        time_s=1.0,
    )

    assert estimate.support_height_m == pytest.approx(0.20)
    assert estimate.memory_support_height_m == pytest.approx(0.20)
    assert estimate.support_source == "contact_memory"
