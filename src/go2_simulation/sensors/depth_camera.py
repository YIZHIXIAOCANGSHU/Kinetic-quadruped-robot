from dataclasses import dataclass

import mujoco as mj
import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    time_s: float
    depth_m: np.ndarray
    rgb: np.ndarray | None = None
    camera_name: str = "front_camera"
    camera_pos_world: np.ndarray | None = None
    camera_xmat: np.ndarray | None = None
    width: int = 0
    height: int = 0
    fovy_deg: float = 70.0


class MujocoDepthCamera:
    def __init__(
        self,
        model: mj.MjModel,
        width: int = 320,
        height: int = 240,
        camera_name: str = "front_camera",
    ):
        self.model = model
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id == -1:
            raise ValueError(f"MuJoCo camera '{camera_name}' was not found in the model.")
        self.renderer = mj.Renderer(model, height=height, width=width)

    def capture(self, data: mj.MjData, time_s: float) -> DepthFrame:
        self.renderer.update_scene(data, camera=self.camera_name)

        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        return DepthFrame(
            time_s=float(time_s),
            depth_m=np.asarray(depth, dtype=np.float32),
            rgb=np.asarray(rgb),
            camera_name=self.camera_name,
            camera_pos_world=np.asarray(data.cam_xpos[self.camera_id], dtype=np.float32).copy(),
            camera_xmat=np.asarray(data.cam_xmat[self.camera_id], dtype=np.float32).reshape(3, 3).copy(),
            width=self.width,
            height=self.height,
            fovy_deg=float(self.model.cam_fovy[self.camera_id]),
        )

    def close(self) -> None:
        self.renderer.close()
