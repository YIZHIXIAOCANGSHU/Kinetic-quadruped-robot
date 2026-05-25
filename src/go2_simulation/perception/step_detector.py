from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepEstimate:
    time_s: float
    valid: bool
    distance_m: float
    height_m: float
    confidence: float
    edge_x_body: float


class StepObstacleDetector:
    def __init__(
        self,
        *,
        min_confirm_frames: int = 3,
        min_step_height_m: float = 0.06,
        max_step_height_m: float = 0.35,
        center_half_width_m: float = 0.25,
        min_distance_m: float = 0.20,
        max_distance_m: float = 1.10,
        edge_smoothing_alpha: float = 0.35,
    ):
        self.min_confirm_frames = int(min_confirm_frames)
        self.min_step_height_m = float(min_step_height_m)
        self.max_step_height_m = float(max_step_height_m)
        self.center_half_width_m = float(center_half_width_m)
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.edge_smoothing_alpha = float(edge_smoothing_alpha)
        self._confirm_count = 0
        self._last_raw: StepEstimate | None = None
        self._filtered: StepEstimate | None = None

    def update(self, height_map, time_s: float) -> StepEstimate:
        raw = self._detect(height_map, time_s)
        if not raw.valid:
            self._confirm_count = 0
            self._last_raw = raw
            self._filtered = None
            return StepEstimate(float(time_s), False, np.inf, 0.0, 0.0, np.inf)

        if self._last_raw is not None and self._last_raw.valid:
            close = abs(raw.distance_m - self._last_raw.distance_m) <= 0.15
            similar = abs(raw.height_m - self._last_raw.height_m) <= 0.08
            self._confirm_count = self._confirm_count + 1 if close and similar else 1
        else:
            self._confirm_count = 1
        self._last_raw = raw

        if self._confirm_count < self.min_confirm_frames:
            return StepEstimate(float(time_s), False, raw.distance_m, raw.height_m, raw.confidence, raw.edge_x_body)

        if self._filtered is None or not self._filtered.valid:
            filtered = raw
        else:
            a = self.edge_smoothing_alpha
            filtered = StepEstimate(
                time_s=float(time_s),
                valid=True,
                distance_m=float(a * raw.distance_m + (1.0 - a) * self._filtered.distance_m),
                height_m=float(a * raw.height_m + (1.0 - a) * self._filtered.height_m),
                confidence=float(max(raw.confidence, self._filtered.confidence * 0.95)),
                edge_x_body=float(a * raw.edge_x_body + (1.0 - a) * self._filtered.edge_x_body),
            )
        self._filtered = filtered
        return filtered

    def _detect(self, height_map, time_s: float) -> StepEstimate:
        heights = np.asarray(height_map.heights_m, dtype=np.float32)
        counts = np.asarray(height_map.counts)
        confidence = np.asarray(height_map.confidence, dtype=np.float32)
        valid = np.isfinite(heights) & (counts > 0)
        if not np.any(valid):
            return StepEstimate(float(time_s), False, np.inf, 0.0, 0.0, np.inf)

        y_centers = np.asarray(height_map.y_centers, dtype=np.float32)
        center_y = np.abs(y_centers) <= self.center_half_width_m
        if not np.any(valid[:, center_y]):
            return StepEstimate(float(time_s), False, np.inf, 0.0, 0.0, np.inf)

        per_x_height = np.full(heights.shape[0], np.nan, dtype=np.float32)
        per_x_confidence = np.zeros(heights.shape[0], dtype=np.float32)
        for ix in range(heights.shape[0]):
            row_valid = valid[ix, center_y]
            if not np.any(row_valid):
                continue
            row_heights = heights[ix, center_y][row_valid]
            per_x_height[ix] = float(np.nanpercentile(row_heights, 80.0))
            per_x_confidence[ix] = float(np.mean(confidence[ix, center_y][row_valid]))

        x_centers = np.asarray(height_map.x_centers, dtype=np.float32)
        near_mask = (x_centers >= 0.15) & (x_centers <= 0.45) & np.isfinite(per_x_height)
        if np.any(near_mask):
            ground_height = float(np.nanmedian(per_x_height[near_mask]))
        else:
            finite = np.isfinite(per_x_height)
            ground_height = float(np.nanmin(per_x_height[finite])) if np.any(finite) else 0.0

        lookahead = (
            (x_centers >= self.min_distance_m)
            & (x_centers <= self.max_distance_m)
            & np.isfinite(per_x_height)
        )
        deltas = per_x_height - ground_height
        step_mask = lookahead & (deltas >= self.min_step_height_m) & (deltas <= self.max_step_height_m)
        if not np.any(step_mask):
            return StepEstimate(float(time_s), False, np.inf, 0.0, 0.0, np.inf)

        edge_idx = int(np.where(step_mask)[0][0])
        edge_x = float(x_centers[edge_idx])
        plateau = step_mask & (x_centers >= edge_x) & (x_centers <= edge_x + 0.25)
        step_height = float(np.nanpercentile(deltas[plateau], 80.0)) if np.any(plateau) else float(deltas[edge_idx])
        step_height = float(np.clip(step_height, 0.0, self.max_step_height_m))
        conf = float(np.clip(np.nanmean(per_x_confidence[plateau]) if np.any(plateau) else per_x_confidence[edge_idx], 0.0, 1.0))
        return StepEstimate(float(time_s), True, edge_x, step_height, conf, edge_x)
