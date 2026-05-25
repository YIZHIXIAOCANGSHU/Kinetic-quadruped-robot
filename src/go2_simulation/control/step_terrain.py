from __future__ import annotations

import numpy as np


class StepClimbTerrainProvider:
    def __init__(self):
        self.base_provider = None
        self.active_leg_group = "none"
        self.step_height_m = 0.0
        self.step_edge_world_x = np.nan
        self.step_landing_margin_m = -0.06
        self._leg_targets: dict[str, np.ndarray] = {}
        self.match_radius_m = 0.22

    def set_base_provider(self, provider) -> None:
        self.base_provider = provider

    def update(
        self,
        *,
        active_leg_group: str,
        step_height_m: float,
        touchdown_targets: dict[str, np.ndarray] | None,
        step_edge_world_x: float = np.nan,
    ) -> None:
        self.active_leg_group = str(active_leg_group)
        self.step_height_m = float(max(0.0, step_height_m))
        self.step_edge_world_x = float(step_edge_world_x)
        self._leg_targets = {
            leg: np.asarray(pos, dtype=float).reshape(3)
            for leg, pos in (touchdown_targets or {}).items()
        }

    def height_at_world(self, x_world: float, y_world: float, default: float = 0.0) -> float:
        leg = self._matching_active_leg(x_world, y_world)
        if leg is not None:
            return self.step_height_m
        if self.base_provider is None:
            return float(default)
        try:
            return float(self.base_provider.height_at_world(x_world, y_world, default=default))
        except Exception:
            return float(default)

    def max_height_between_world(
        self,
        p0_world: np.ndarray,
        p1_world: np.ndarray,
        default: float = 0.0,
        samples: int = 9,
    ) -> float:
        if self.active_leg_group in ("FL", "FR", "RL", "RR"):
            return max(float(default), self.step_height_m)
        if self.base_provider is None:
            return float(default)
        try:
            return float(self.base_provider.max_height_between_world(p0_world, p1_world, default=default, samples=samples))
        except Exception:
            return float(default)

    def _matching_active_leg(self, x_world: float, y_world: float) -> str | None:
        if self.active_leg_group not in ("FL", "FR", "RL", "RR"):
            return None
        active = (self.active_leg_group,)
        point = np.asarray([x_world, y_world], dtype=float)
        for leg in active:
            target = self._leg_targets.get(leg)
            if target is None:
                continue
            if np.linalg.norm(point - target[:2]) <= self.match_radius_m:
                return leg
        return None

    def override_touchdown_world(self, leg: str, touchdown_world: np.ndarray) -> np.ndarray:
        if self.active_leg_group not in ("FL", "FR", "RL", "RR"):
            return np.asarray(touchdown_world, dtype=float)
        if leg != self.active_leg_group or not np.isfinite(self.step_edge_world_x):
            return np.asarray(touchdown_world, dtype=float)

        touchdown = np.asarray(touchdown_world, dtype=float).copy()
        touchdown[0] = self.step_edge_world_x + self.step_landing_margin_m
        touchdown[2] = self.step_height_m + 0.02
        return touchdown
