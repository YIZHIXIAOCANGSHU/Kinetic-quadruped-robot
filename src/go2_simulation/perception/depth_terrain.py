from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthPointCloudFrame:
    time_s: float
    points_world: np.ndarray
    points_body_yaw: np.ndarray
    camera_name: str = "front_camera"


@dataclass(frozen=True)
class LocalHeightMap:
    time_s: float
    x_edges: np.ndarray
    y_edges: np.ndarray
    heights_m: np.ndarray
    counts: np.ndarray
    confidence: np.ndarray
    origin_world: np.ndarray
    yaw: float

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    @property
    def global_confidence(self) -> float:
        return float(np.mean(self.confidence > 0.0)) if self.confidence.size else 0.0

    def height_at_body(self, x_body: float, y_body: float, default: float = 0.0) -> float:
        ix = int(np.searchsorted(self.x_edges, x_body, side="right") - 1)
        iy = int(np.searchsorted(self.y_edges, y_body, side="right") - 1)
        if ix < 0 or iy < 0 or ix >= self.heights_m.shape[0] or iy >= self.heights_m.shape[1]:
            return float(default)
        height = self.heights_m[ix, iy]
        if not np.isfinite(height) or self.counts[ix, iy] <= 0:
            return float(default)
        return float(height)

    def height_at_world(self, x_world: float, y_world: float, default: float = 0.0) -> float:
        c = np.cos(self.yaw)
        s = np.sin(self.yaw)
        delta = np.array([x_world - self.origin_world[0], y_world - self.origin_world[1]])
        x_body = c * delta[0] + s * delta[1]
        y_body = -s * delta[0] + c * delta[1]
        return self.height_at_body(float(x_body), float(y_body), default=default)

    def max_height_between_world(
        self,
        p0_world: np.ndarray,
        p1_world: np.ndarray,
        default: float = 0.0,
        samples: int = 9,
    ) -> float:
        p0 = np.asarray(p0_world, dtype=float)
        p1 = np.asarray(p1_world, dtype=float)
        heights = []
        for alpha in np.linspace(0.0, 1.0, max(2, samples)):
            p = p0 + alpha * (p1 - p0)
            heights.append(self.height_at_world(float(p[0]), float(p[1]), default=default))
        return float(np.max(heights)) if heights else float(default)

    def height_points_world(self) -> np.ndarray:
        ix, iy = np.where(np.isfinite(self.heights_m) & (self.counts > 0))
        if ix.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        xs = self.x_centers[ix]
        ys = self.y_centers[iy]
        zs = self.heights_m[ix, iy]
        c = np.cos(self.yaw)
        s = np.sin(self.yaw)
        world_x = self.origin_world[0] + c * xs - s * ys
        world_y = self.origin_world[1] + s * xs + c * ys
        return np.column_stack((world_x, world_y, zs)).astype(np.float32)

    def image(self, min_height: float = 0.0, max_height: float = 0.5) -> np.ndarray:
        img = np.nan_to_num(self.heights_m.T, nan=min_height)
        return np.clip(img, min_height, max_height).astype(np.float32)


@dataclass(frozen=True)
class ObstacleEstimate:
    time_s: float
    state_hint: str
    distance_m: float
    height_m: float
    support_height_m: float
    confidence: float
    valid: bool
    memory_support_height_m: float = 0.0
    terrain_delta_m: float = 0.0
    support_source: str = "none"


@dataclass
class _MemoryCell:
    height_m: float
    confidence: float
    time_s: float
    count: int


class DepthTerrainMemory:
    def __init__(
        self,
        grid_resolution_m: float = 0.05,
        max_age_s: float = 5.0,
        keep_radius_x_m: float = 3.0,
        keep_radius_y_m: float = 1.2,
        min_confidence: float = 0.05,
        terrain_change_threshold_m: float = 0.06,
        foot_height_offset_m: float = 0.02,
    ):
        self.grid_resolution_m = float(grid_resolution_m)
        self.max_age_s = float(max_age_s)
        self.keep_radius_x_m = float(keep_radius_x_m)
        self.keep_radius_y_m = float(keep_radius_y_m)
        self.min_confidence = float(min_confidence)
        self.terrain_change_threshold_m = float(terrain_change_threshold_m)
        self.foot_height_offset_m = float(foot_height_offset_m)
        self._cells: dict[tuple[int, int], _MemoryCell] = {}
        self._last_confirmed_support_height = 0.0
        self._last_confirmed_support_time = -np.inf
        self._last_support_source = "none"

    def _cell_key(self, x_world: float, y_world: float) -> tuple[int, int]:
        return (
            int(np.floor(float(x_world) / self.grid_resolution_m)),
            int(np.floor(float(y_world) / self.grid_resolution_m)),
        )

    def _cell_center(self, key: tuple[int, int]) -> tuple[float, float]:
        return (
            (key[0] + 0.5) * self.grid_resolution_m,
            (key[1] + 0.5) * self.grid_resolution_m,
        )

    def update(self, height_map: LocalHeightMap, base_pos_world: np.ndarray, time_s: float) -> None:
        heights = np.asarray(height_map.heights_m, dtype=np.float32)
        counts = np.asarray(height_map.counts)
        confidence = np.asarray(height_map.confidence, dtype=np.float32)
        valid_ix, valid_iy = np.where(
            np.isfinite(heights) & (counts > 0) & (confidence >= self.min_confidence)
        )
        if valid_ix.size:
            x_centers = height_map.x_centers[valid_ix]
            y_centers = height_map.y_centers[valid_iy]
            c = np.cos(height_map.yaw)
            s = np.sin(height_map.yaw)
            world_x = height_map.origin_world[0] + c * x_centers - s * y_centers
            world_y = height_map.origin_world[1] + s * x_centers + c * y_centers

            for xw, yw, height, conf, count in zip(
                world_x,
                world_y,
                heights[valid_ix, valid_iy],
                confidence[valid_ix, valid_iy],
                counts[valid_ix, valid_iy],
            ):
                height = float(np.clip(height, -0.05, 0.55))
                key = self._cell_key(float(xw), float(yw))
                previous = self._cells.get(key)
                if previous is None:
                    self._cells[key] = _MemoryCell(height, float(conf), float(time_s), int(count))
                    continue

                if float(conf) >= previous.confidence or time_s >= previous.time_s:
                    alpha = 0.65 if float(conf) >= previous.confidence else 0.35
                    blended = alpha * height + (1.0 - alpha) * previous.height_m
                    # Keep sharp upward edges visible for stair tops.
                    if height > previous.height_m + 0.04:
                        blended = height
                    previous.height_m = float(np.clip(blended, -0.05, 0.55))
                    previous.confidence = max(previous.confidence * 0.95, float(conf))
                    previous.time_s = float(time_s)
                    previous.count = max(previous.count, int(count))

        self.prune(base_pos_world, time_s)

    def prune(self, base_pos_world: np.ndarray, time_s: float) -> None:
        base = np.asarray(base_pos_world, dtype=float).reshape(3)
        stale_keys = []
        for key, cell in self._cells.items():
            xw, yw = self._cell_center(key)
            if (
                time_s - cell.time_s > self.max_age_s
                or abs(xw - base[0]) > self.keep_radius_x_m
                or abs(yw - base[1]) > self.keep_radius_y_m
            ):
                stale_keys.append(key)
        for key in stale_keys:
            self._cells.pop(key, None)

    def height_at_world(self, x_world: float, y_world: float, default: float = 0.0) -> float:
        result = self.height_and_confidence_at_world(x_world, y_world, default=default)
        return float(result[0])

    def height_and_confidence_at_world(
        self, x_world: float, y_world: float, default: float = 0.0
    ) -> tuple[float, float]:
        key = self._cell_key(x_world, y_world)
        best_cell = self._cells.get(key)
        if best_cell is not None:
            return float(best_cell.height_m), float(best_cell.confidence)

        best_distance = np.inf
        best_height = float(default)
        best_confidence = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_key = (key[0] + dx, key[1] + dy)
                cell = self._cells.get(neighbor_key)
                if cell is None:
                    continue
                cx, cy = self._cell_center(neighbor_key)
                distance = float(np.hypot(cx - x_world, cy - y_world))
                if distance < best_distance:
                    best_distance = distance
                    best_height = float(cell.height_m)
                    best_confidence = float(cell.confidence)
        return best_height, best_confidence

    def max_height_between_world(
        self,
        p0_world: np.ndarray,
        p1_world: np.ndarray,
        default: float = 0.0,
        samples: int = 13,
    ) -> float:
        p0 = np.asarray(p0_world, dtype=float).reshape(3)
        p1 = np.asarray(p1_world, dtype=float).reshape(3)
        values = []
        for alpha in np.linspace(0.0, 1.0, max(2, samples)):
            p = p0 + alpha * (p1 - p0)
            values.append(self.height_at_world(float(p[0]), float(p[1]), default=default))
        return float(np.max(values)) if values else float(default)

    def height_points_world(self) -> np.ndarray:
        if not self._cells:
            return np.empty((0, 3), dtype=np.float32)
        rows = []
        for key, cell in self._cells.items():
            xw, yw = self._cell_center(key)
            rows.append((xw, yw, cell.height_m))
        return np.asarray(rows, dtype=np.float32)

    def image_around(
        self,
        base_pos_world: np.ndarray,
        x_span_m: float = 3.0,
        y_span_m: float = 1.2,
        min_height: float = 0.0,
        max_height: float = 0.5,
    ) -> np.ndarray:
        base = np.asarray(base_pos_world, dtype=float).reshape(3)
        nx = max(2, int(np.ceil(x_span_m / self.grid_resolution_m)))
        ny = max(2, int(np.ceil(y_span_m / self.grid_resolution_m)))
        x0 = base[0] - 0.5 * x_span_m
        y0 = base[1] - 0.5 * y_span_m
        image = np.full((nx, ny), min_height, dtype=np.float32)
        for ix in range(nx):
            for iy in range(ny):
                x = x0 + (ix + 0.5) * self.grid_resolution_m
                y = y0 + (iy + 0.5) * self.grid_resolution_m
                image[ix, iy] = self.height_at_world(x, y, default=min_height)
        return np.clip(image.T, min_height, max_height).astype(np.float32)

    def estimate_obstacle(
        self,
        local_obstacle: ObstacleEstimate,
        height_map: LocalHeightMap,
        foot_positions: dict[str, np.ndarray],
        foot_contacts: dict[str, bool],
        base_pos_world: np.ndarray,
        time_s: float,
    ) -> ObstacleEstimate:
        support_height, memory_support, support_source, support_confidence = self._estimate_support_height(
            foot_positions, foot_contacts, time_s
        )
        distance, terrain_delta, front_confidence = self._estimate_front_delta(
            height_map, support_height
        )

        local_valid = bool(getattr(local_obstacle, "valid", False))
        local_distance = float(getattr(local_obstacle, "distance_m", np.inf))
        local_height = float(getattr(local_obstacle, "height_m", 0.0))
        local_confidence = float(getattr(local_obstacle, "confidence", 0.0))
        if (
            (not np.isfinite(distance) or abs(terrain_delta) < self.terrain_change_threshold_m)
            and local_valid
            and local_confidence > front_confidence
            and local_height > self.terrain_change_threshold_m
        ):
            distance = local_distance
            terrain_delta = local_height
            front_confidence = local_confidence

        if np.isfinite(distance) and terrain_delta < -self.terrain_change_threshold_m:
            state_hint = "DESCENDING" if distance < 0.35 else "APPROACH_DOWN"
        elif np.isfinite(distance) and terrain_delta > self.terrain_change_threshold_m:
            state_hint = "CLIMBING" if distance < 0.45 else "APPROACH_UP"
        elif support_height > 0.05:
            state_hint = "ON_PLATFORM"
        else:
            state_hint = "NORMAL"

        confidence = max(front_confidence, support_confidence, local_confidence if local_valid else 0.0)
        valid = bool(confidence > 0.0 or support_source != "none")
        return ObstacleEstimate(
            time_s=float(time_s),
            state_hint=state_hint,
            distance_m=float(distance),
            height_m=float(abs(terrain_delta)) if np.isfinite(distance) else 0.0,
            support_height_m=float(max(0.0, support_height)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            valid=valid,
            memory_support_height_m=float(max(0.0, memory_support)),
            terrain_delta_m=float(terrain_delta) if np.isfinite(distance) else 0.0,
            support_source=support_source,
        )

    def _estimate_support_height(
        self,
        foot_positions: dict[str, np.ndarray],
        foot_contacts: dict[str, bool],
        time_s: float,
    ) -> tuple[float, float, str, float]:
        heights = []
        memory_heights = []
        sources = []
        for leg, is_contact in foot_contacts.items():
            if not is_contact or leg not in foot_positions:
                continue
            foot = np.asarray(foot_positions[leg], dtype=float).reshape(3)
            memory_height, memory_conf = self.height_and_confidence_at_world(foot[0], foot[1], default=np.nan)
            foot_support_height = float(foot[2] - self.foot_height_offset_m)
            if np.isfinite(memory_height) and memory_conf > 0.0:
                candidate = max(memory_height, foot_support_height)
                memory_heights.append(memory_height)
                sources.append("contact_memory")
            else:
                candidate = foot_support_height
                sources.append("contact_foot")
            heights.append(float(np.clip(candidate, -0.03, 0.55)))

        if heights:
            support_height = float(np.percentile(heights, 60.0))
            support_height = max(0.0, support_height)
            memory_support = float(np.percentile(memory_heights, 60.0)) if memory_heights else support_height
            self._last_confirmed_support_height = support_height
            self._last_confirmed_support_time = float(time_s)
            self._last_support_source = "contact_memory" if "contact_memory" in sources else "contact_foot"
            return support_height, max(0.0, memory_support), self._last_support_source, 1.0

        if time_s - self._last_confirmed_support_time <= 0.35:
            return (
                float(self._last_confirmed_support_height),
                float(self._last_confirmed_support_height),
                "hold",
                0.5,
            )
        return 0.0, 0.0, "none", 0.0

    def _estimate_front_delta(
        self, height_map: LocalHeightMap, support_height: float
    ) -> tuple[float, float, float]:
        heights = np.asarray(height_map.heights_m, dtype=np.float32)
        valid = np.isfinite(heights) & (np.asarray(height_map.counts) > 0)
        y_centers = height_map.y_centers
        center_mask_y = np.abs(y_centers) <= 0.25
        if not np.any(valid[:, center_mask_y]):
            return np.inf, 0.0, 0.0

        per_x_height = np.full(valid.shape[0], np.nan, dtype=np.float32)
        per_x_conf = np.zeros(valid.shape[0], dtype=np.float32)
        for ix in range(valid.shape[0]):
            row_valid = valid[ix, center_mask_y]
            if not np.any(row_valid):
                continue
            row_heights = heights[ix, center_mask_y][row_valid]
            per_x_height[ix] = float(np.nanpercentile(row_heights, 80.0))
            per_x_conf[ix] = float(np.mean(height_map.confidence[ix, center_mask_y][row_valid]))

        x_centers = height_map.x_centers
        lookahead = (x_centers >= 0.25) & (x_centers <= 1.15) & np.isfinite(per_x_height)
        if not np.any(lookahead):
            return np.inf, 0.0, 0.0

        deltas = per_x_height - float(support_height)
        up_mask = lookahead & (deltas > self.terrain_change_threshold_m)
        down_mask = lookahead & (deltas < -self.terrain_change_threshold_m)
        first_up = int(np.where(up_mask)[0][0]) if np.any(up_mask) else None
        first_down = int(np.where(down_mask)[0][0]) if np.any(down_mask) else None

        if first_up is None and first_down is None:
            return np.inf, 0.0, float(np.nanmean(per_x_conf[lookahead]))
        if first_down is None or (first_up is not None and x_centers[first_up] <= x_centers[first_down]):
            idx = first_up
            nearby = lookahead & (x_centers >= x_centers[idx]) & (x_centers <= x_centers[idx] + 0.25)
            delta = float(np.nanmax(deltas[nearby]))
        else:
            idx = first_down
            nearby = lookahead & (x_centers >= x_centers[idx]) & (x_centers <= x_centers[idx] + 0.25)
            delta = float(np.nanmin(deltas[nearby]))
        return float(x_centers[idx]), delta, float(max(per_x_conf[idx], np.nanmean(per_x_conf[lookahead])))


class DepthTerrainEstimator:
    def __init__(
        self,
        x_range: tuple[float, float] = (0.15, 1.20),
        y_range: tuple[float, float] = (-0.35, 0.35),
        grid_resolution_m: float = 0.05,
        min_depth_m: float = 0.15,
        max_depth_m: float = 5.0,
        min_points_per_cell: int = 2,
        obstacle_height_threshold_m: float = 0.06,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.grid_resolution_m = float(grid_resolution_m)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_points_per_cell = int(min_points_per_cell)
        self.obstacle_height_threshold_m = float(obstacle_height_threshold_m)
        self.latest_height_map: LocalHeightMap | None = None
        self.latest_obstacle: ObstacleEstimate | None = None
        self.latest_cloud: DepthPointCloudFrame | None = None

    def update(self, frame, base_pos_world: np.ndarray, base_yaw: float):
        cloud = self.depth_to_point_cloud(frame, base_pos_world, base_yaw)
        height_map = self.make_height_map(cloud, frame.time_s, base_pos_world, base_yaw)
        obstacle = self.estimate_obstacle(height_map)

        self.latest_cloud = cloud
        self.latest_height_map = height_map
        self.latest_obstacle = obstacle
        return cloud, height_map, obstacle

    def depth_to_point_cloud(self, frame, base_pos_world: np.ndarray, base_yaw: float) -> DepthPointCloudFrame:
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        finite = np.isfinite(depth) & (depth >= self.min_depth_m) & (depth <= self.max_depth_m)
        if not np.any(finite):
            return DepthPointCloudFrame(
                time_s=frame.time_s,
                points_world=np.empty((0, 3), dtype=np.float32),
                points_body_yaw=np.empty((0, 3), dtype=np.float32),
                camera_name=frame.camera_name,
            )

        v, u = np.nonzero(finite)
        z = depth[v, u].astype(np.float64)
        h, w = depth.shape
        fovy = np.deg2rad(float(frame.fovy_deg))
        fy = 0.5 * h / np.tan(0.5 * fovy)
        fx = fy
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5

        x_cam = (u.astype(np.float64) - cx) * z / fx
        y_cam = (v.astype(np.float64) - cy) * z / fy
        cam_forward = z

        # MuJoCo camera convention: local -Z is optical forward, +X is image right, +Y is image up.
        points_camera = np.column_stack((x_cam, -y_cam, -cam_forward))
        camera_xmat = np.asarray(frame.camera_xmat, dtype=np.float64).reshape(3, 3)
        camera_pos = np.asarray(frame.camera_pos_world, dtype=np.float64).reshape(3)
        points_world = camera_pos[None, :] + points_camera @ camera_xmat.T

        base_pos = np.asarray(base_pos_world, dtype=np.float64).reshape(3)
        c = np.cos(base_yaw)
        s = np.sin(base_yaw)
        delta = points_world - base_pos[None, :]
        points_body = np.column_stack(
            (
                c * delta[:, 0] + s * delta[:, 1],
                -s * delta[:, 0] + c * delta[:, 1],
                points_world[:, 2],
            )
        )

        return DepthPointCloudFrame(
            time_s=frame.time_s,
            points_world=points_world.astype(np.float32),
            points_body_yaw=points_body.astype(np.float32),
            camera_name=frame.camera_name,
        )

    def make_height_map(
        self,
        cloud: DepthPointCloudFrame,
        time_s: float,
        base_pos_world: np.ndarray,
        base_yaw: float,
    ) -> LocalHeightMap:
        x_edges = np.arange(self.x_range[0], self.x_range[1] + self.grid_resolution_m, self.grid_resolution_m)
        y_edges = np.arange(self.y_range[0], self.y_range[1] + self.grid_resolution_m, self.grid_resolution_m)
        nx = len(x_edges) - 1
        ny = len(y_edges) - 1
        heights = np.full((nx, ny), np.nan, dtype=np.float32)
        counts = np.zeros((nx, ny), dtype=np.int32)
        confidence = np.zeros((nx, ny), dtype=np.float32)

        points = np.asarray(cloud.points_body_yaw, dtype=np.float32)
        if points.size:
            in_region = (
                (points[:, 0] >= self.x_range[0])
                & (points[:, 0] < self.x_range[1])
                & (points[:, 1] >= self.y_range[0])
                & (points[:, 1] < self.y_range[1])
                & np.isfinite(points[:, 2])
            )
            points = points[in_region]
            if points.size:
                ix = np.floor((points[:, 0] - self.x_range[0]) / self.grid_resolution_m).astype(int)
                iy = np.floor((points[:, 1] - self.y_range[0]) / self.grid_resolution_m).astype(int)
                for cell_x in range(nx):
                    for cell_y in range(ny):
                        mask = (ix == cell_x) & (iy == cell_y)
                        if not np.any(mask):
                            continue
                        z_values = points[mask, 2]
                        counts[cell_x, cell_y] = int(z_values.size)
                        if z_values.size >= self.min_points_per_cell:
                            heights[cell_x, cell_y] = float(np.percentile(z_values, 90.0))
                            confidence[cell_x, cell_y] = min(1.0, z_values.size / 12.0)

        return LocalHeightMap(
            time_s=float(time_s),
            x_edges=x_edges.astype(np.float32),
            y_edges=y_edges.astype(np.float32),
            heights_m=heights,
            counts=counts,
            confidence=confidence,
            origin_world=np.asarray(base_pos_world, dtype=np.float32).copy(),
            yaw=float(base_yaw),
        )

    def estimate_obstacle(self, height_map: LocalHeightMap) -> ObstacleEstimate:
        heights = height_map.heights_m
        valid = np.isfinite(heights) & (height_map.counts >= self.min_points_per_cell)
        if not np.any(valid):
            return ObstacleEstimate(height_map.time_s, "UNKNOWN", np.inf, 0.0, 0.0, 0.0, False)

        y_centers = height_map.y_centers
        center_mask_y = np.abs(y_centers) <= 0.25
        valid_center = valid[:, center_mask_y]
        heights_center = heights[:, center_mask_y]
        if not np.any(valid_center):
            return ObstacleEstimate(height_map.time_s, "UNKNOWN", np.inf, 0.0, 0.0, 0.0, False)

        per_x_height = np.full(valid_center.shape[0], np.nan, dtype=np.float32)
        per_x_conf = np.zeros(valid_center.shape[0], dtype=np.float32)
        for ix in range(valid_center.shape[0]):
            if np.any(valid_center[ix]):
                per_x_height[ix] = float(np.nanpercentile(heights_center[ix][valid_center[ix]], 90.0))
                per_x_conf[ix] = float(np.mean(height_map.confidence[ix, center_mask_y][valid_center[ix]]))

        support_region = (height_map.x_centers >= -0.05) & (height_map.x_centers <= 0.35)
        support_values = per_x_height[support_region & np.isfinite(per_x_height)]
        support_height = float(np.nanpercentile(support_values, 70.0)) if support_values.size else 0.0

        obstacle_mask = np.isfinite(per_x_height) & (
            per_x_height >= support_height + self.obstacle_height_threshold_m
        )
        if not np.any(obstacle_mask):
            confidence = float(np.nanmean(per_x_conf[np.isfinite(per_x_height)]))
            return ObstacleEstimate(height_map.time_s, "NORMAL", np.inf, 0.0, support_height, confidence, True)

        first_idx = int(np.where(obstacle_mask)[0][0])
        distance = float(height_map.x_centers[first_idx])
        obstacle_height = float(np.nanmax(per_x_height[obstacle_mask]) - support_height)
        confidence = float(np.nanmax(per_x_conf[obstacle_mask]))

        if distance > 0.65:
            state_hint = "APPROACH"
        elif obstacle_height > 0.05:
            state_hint = "STEP_UP"
        else:
            state_hint = "ON_OBSTACLE"

        return ObstacleEstimate(
            time_s=height_map.time_s,
            state_hint=state_hint,
            distance_m=distance,
            height_m=max(0.0, obstacle_height),
            support_height_m=max(0.0, support_height),
            confidence=confidence,
            valid=True,
        )


class FlatTerrainProvider:
    def height_at_world(self, x_world: float, y_world: float, default: float = 0.0) -> float:
        return float(default)

    def max_height_between_world(
        self,
        p0_world: np.ndarray,
        p1_world: np.ndarray,
        default: float = 0.0,
        samples: int = 9,
    ) -> float:
        return float(default)
