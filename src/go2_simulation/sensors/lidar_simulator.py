from dataclasses import dataclass

import mujoco as mj
import numpy as np


@dataclass(frozen=True)
class LidarFrame:
    time_s: float
    points_world: np.ndarray
    ranges_m: np.ndarray
    origin_world: np.ndarray
    sensor_name: str = "radar"


class MujocoLidarSimulator:
    """MuJoCo ray-cast LiDAR using Unitree Go2/L2 public parameters."""

    def __init__(
        self,
        model: mj.MjModel,
        site_name: str = "radar",
        horizontal_fov_deg: float = 360.0,
        vertical_fov_deg: float = 96.0,
        min_range_m: float = 0.05,
        max_range_m: float = 30.0,
        points_per_second: int = 64000,
        scan_hz: float = 5.55,
    ):
        self.model = model
        self.site_name = site_name
        self.site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"MuJoCo site '{site_name}' was not found in the model.")

        self.body_exclude = int(model.site_bodyid[self.site_id])
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.points_per_second = int(points_per_second)
        self.scan_hz = float(scan_hz)
        self.ray_count = int(round(self.points_per_second / self.scan_hz))
        self._ray_dirs_lidar = self._make_ray_directions()

    def _make_ray_directions(self) -> np.ndarray:
        if self.ray_count <= 0:
            raise ValueError("LiDAR ray count must be positive.")
        if not (0.0 < self.horizontal_fov_deg <= 360.0):
            raise ValueError("LiDAR horizontal FOV must be in (0, 360].")
        if not (0.0 < self.vertical_fov_deg < 180.0):
            raise ValueError("LiDAR vertical FOV must be in (0, 180).")

        indices = np.arange(self.ray_count, dtype=np.float64)
        golden_ratio_conjugate = 0.6180339887498949
        azimuth = np.deg2rad(self.horizontal_fov_deg) * (
            (indices * golden_ratio_conjugate) % 1.0
        )
        elevation = np.deg2rad(self.vertical_fov_deg) * (
            (indices + 0.5) / self.ray_count - 0.5
        )

        cos_el = np.cos(elevation)
        directions = np.column_stack(
            (
                cos_el * np.cos(azimuth),
                cos_el * np.sin(azimuth),
                np.sin(elevation),
            )
        )
        return np.ascontiguousarray(directions, dtype=np.float64)

    def capture(self, data: mj.MjData, time_s: float) -> LidarFrame:
        origin_world = np.asarray(data.site_xpos[self.site_id], dtype=np.float64).copy()
        site_xmat = np.asarray(data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3)
        ray_dirs_world = np.ascontiguousarray(self._ray_dirs_lidar @ site_xmat.T)

        geomid = np.full(self.ray_count, -1, dtype=np.int32)
        distances = np.full(self.ray_count, -1.0, dtype=np.float64)
        ray_dirs_flat = ray_dirs_world.reshape(-1)
        try:
            mj.mj_multiRay(
                self.model,
                data,
                origin_world,
                ray_dirs_flat,
                None,
                1,
                self.body_exclude,
                geomid,
                distances,
                self.ray_count,
                self.max_range_m,
            )
        except TypeError:
            mj.mj_multiRay(
                self.model,
                data,
                origin_world,
                ray_dirs_flat,
                None,
                1,
                self.body_exclude,
                geomid,
                distances,
                None,
                self.ray_count,
                self.max_range_m,
            )

        valid = (
            (geomid >= 0)
            & np.isfinite(distances)
            & (distances >= self.min_range_m)
            & (distances <= self.max_range_m)
        )
        ranges_m = distances[valid].astype(np.float32, copy=True)
        points_world = (
            origin_world[None, :] + ray_dirs_world[valid] * ranges_m[:, None]
        ).astype(np.float32, copy=False)

        return LidarFrame(
            time_s=float(time_s),
            points_world=points_world,
            ranges_m=ranges_m,
            origin_world=origin_world.astype(np.float32, copy=False),
            sensor_name=self.site_name,
        )
