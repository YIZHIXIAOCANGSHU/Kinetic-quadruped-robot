from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


class Go2RerunLogger:
    def __init__(
        self,
        app_name: str = "Go2_Simulation",
        *,
        enable_lidar: bool = False,
        enable_depth_diagnostics: bool = True,
        record_path: str | None = None,
    ):
        self.enable_lidar = bool(enable_lidar)
        self.enable_depth_diagnostics = bool(enable_depth_diagnostics)
        rr.init(app_name, spawn=True)
        blueprint = self._make_blueprint()
        if record_path:
            try:
                rr.save(Path(record_path))
                print(f"[Rerun] Recording to {record_path}")
            except Exception as exc:
                print(f"[Rerun warning] Could not save recording: {type(exc).__name__}: {exc}")
        rr.send_blueprint(blueprint)
        print("[Rerun] Depth camera diagnostics and scalar views are ready.")

    def _make_blueprint(self):
        force_views = [
            rrb.TimeSeriesView(origin=f"Forces/{leg}", name=f"{leg} Leg Force (N)")
            for leg in ("FL", "FR", "RL", "RR")
        ]
        torque_views = [
            rrb.TimeSeriesView(origin=f"Torques/{leg}", name=f"{leg} Thigh Torque (Nm)")
            for leg in ("FL", "FR", "RL", "RR")
        ]
        control_views = [
            rrb.TimeSeriesView(origin="Control/base_x", name="Base X"),
            rrb.TimeSeriesView(origin="Control/base_z", name="Base Z"),
            rrb.TimeSeriesView(origin="Control/target_x_vel", name="Target X Velocity"),
            rrb.TimeSeriesView(origin="Control/target_z_pos", name="Target Body Height"),
            rrb.TimeSeriesView(origin="Control/swing_height", name="Swing Height"),
        ]
        step_views = [
            rrb.TimeSeriesView(origin="Step/distance", name="Step Distance"),
            rrb.TimeSeriesView(origin="Step/height", name="Step Height"),
            rrb.TimeSeriesView(origin="Step/confidence", name="Step Confidence"),
            rrb.TextDocumentView(origin="Step/state", name="Step State"),
        ]

        camera_views = [
            rrb.Spatial2DView(origin="Cameras/front_camera/rgb", name="Front RGB"),
            rrb.Spatial2DView(origin="Cameras/front_camera/depth", name="Front Depth"),
        ]
        if self.enable_depth_diagnostics:
            camera_views.append(rrb.Spatial2DView(origin="Perception/height_map_2d", name="Local Height Map"))
            camera_views.append(rrb.Spatial2DView(origin="Perception/world_height_memory_2d", name="World Height Memory"))

        tabs = [
            rrb.Vertical(*camera_views, name="Camera"),
            rrb.Spatial3DView(origin="World", name="Depth Perception 3D"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="Control/state", name="State"),
                rrb.Grid(*control_views),
                name="Control",
            ),
            rrb.Vertical(rrb.Grid(*step_views), name="Step"),
            rrb.Spatial3DView(origin="World/footholds", name="Footholds"),
        ]
        if self.enable_lidar:
            tabs.append(rrb.Spatial3DView(origin="World/radar", name="LiDAR Point Cloud"))
        tabs.extend(
            [
                rrb.Vertical(rrb.Grid(*force_views), name="Forces"),
                rrb.Vertical(rrb.Grid(*torque_views), name="Torques"),
                rrb.TimeSeriesView(origin="Body/Height", name="Body Height"),
            ]
        )

        return rrb.Blueprint(
            rrb.Tabs(
                *tabs,
                name="Go2",
            ),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=True),
        )

    def set_time(self, time_s: float) -> None:
        rr.set_time_seconds("sim_time", float(time_s))

    def log_depth_frame(self, frame) -> None:
        self.set_time(frame.time_s)
        camera_root = f"Cameras/{frame.camera_name}"

        if frame.rgb is not None:
            rr.log(f"{camera_root}/rgb", rr.Image(np.asarray(frame.rgb)))

        depth = np.asarray(frame.depth_m, dtype=np.float32)
        rr.log(
            f"{camera_root}/depth",
            rr.DepthImage(depth, meter=1.0, colormap="turbo", depth_range=[0.0, 5.0]),
        )

    def log_depth_diagnostics(self, cloud, height_map, obstacle, terrain_memory=None) -> None:
        self.set_time(height_map.time_s)
        points = np.asarray(cloud.points_world, dtype=np.float32)
        if points.shape[0] > 8000:
            step = max(1, points.shape[0] // 8000)
            points = points[::step]
        rr.log("World/depth_camera/points", rr.Points3D(points, radii=0.004))

        height_points = height_map.height_points_world()
        rr.log("World/depth_height_map/points", rr.Points3D(height_points, radii=0.018))
        rr.log("World/base", rr.Points3D([height_map.origin_world], radii=0.05))
        if terrain_memory is not None:
            memory_points = terrain_memory.height_points_world()
            rr.log("World/depth_height_memory/points", rr.Points3D(memory_points, radii=0.014))
            memory_img = terrain_memory.image_around(height_map.origin_world)
            rr.log(
                "Perception/world_height_memory_2d",
                rr.DepthImage(memory_img, meter=1.0, colormap="turbo", depth_range=[0.0, 0.5]),
            )

        height_img = height_map.image(min_height=0.0, max_height=0.5)
        rr.log(
            "Perception/height_map_2d",
            rr.DepthImage(height_img, meter=1.0, colormap="turbo", depth_range=[0.0, 0.5]),
        )

        distance = float(obstacle.distance_m) if np.isfinite(obstacle.distance_m) else 5.0
        rr.log("Control/obstacle_distance", rr.Scalars(distance))
        rr.log("Control/obstacle_height", rr.Scalars(float(obstacle.height_m)))
        rr.log("Control/terrain_delta", rr.Scalars(float(getattr(obstacle, "terrain_delta_m", obstacle.height_m))))
        rr.log("Control/terrain_confidence", rr.Scalars(float(obstacle.confidence)))

    def log_step_state(self, time_s, step_estimate, command=None) -> None:
        self.set_time(time_s)
        valid = bool(getattr(step_estimate, "valid", False))
        distance = float(getattr(step_estimate, "distance_m", 5.0))
        if not np.isfinite(distance):
            distance = 5.0
        height = float(getattr(step_estimate, "height_m", 0.0))
        confidence = float(getattr(step_estimate, "confidence", 0.0))
        rr.log("Step/distance", rr.Scalars(distance))
        rr.log("Step/height", rr.Scalars(height))
        rr.log("Step/confidence", rr.Scalars(confidence))
        state = getattr(command, "state", "UNKNOWN") if command is not None else "UNKNOWN"
        leg_group = getattr(command, "active_leg_group", "none") if command is not None else "none"
        forced_swing = getattr(command, "forced_swing_legs", ()) if command is not None else ()
        forced_stance = getattr(command, "forced_stance_legs", ()) if command is not None else ()
        text = (
            f"# {state}\n"
            f"- valid: {valid}\n"
            f"- distance: {distance:.3f} m\n"
            f"- height: {height:.3f} m\n"
            f"- confidence: {confidence:.2f}\n"
            f"- leg_group: {leg_group}\n"
            f"- forced_swing: {', '.join(forced_swing) if forced_swing else 'none'}\n"
            f"- forced_stance: {', '.join(forced_stance) if forced_stance else 'none'}\n"
        )
        rr.log("Step/state", rr.TextDocument(text, media_type="text/markdown"))
        if valid:
            edge_world_x = float(getattr(command, "step_edge_world_x", np.nan)) if command is not None else np.nan
            x = edge_world_x if np.isfinite(edge_world_x) else float(getattr(step_estimate, "edge_x_body", distance))
            rr.log(
                "World/step_edge",
                rr.LineStrips3D([np.asarray([[x, -0.35, height], [x, 0.35, height]], dtype=np.float32)], radii=0.01),
            )

    def log_lidar_frame(self, frame) -> None:
        if not self.enable_lidar:
            return
        self.set_time(frame.time_s)
        points = np.asarray(frame.points_world, dtype=np.float32)
        sensor_root = f"World/{frame.sensor_name}"

        rr.log(f"{sensor_root}/origin", rr.Points3D([frame.origin_world], radii=0.04))
        if points.size == 0:
            rr.log(f"{sensor_root}/points", rr.Points3D(np.empty((0, 3), dtype=np.float32)))
            return
        rr.log(f"{sensor_root}/points", rr.Points3D(points, radii=0.01))

    def log_sensor_forces(self, time_s, sensor_forces) -> None:
        self.set_time(time_s)
        for leg, value in sensor_forces.items():
            rr.log(f"Forces/{leg}", rr.Scalars(float(value)))

    def log_thigh_torques(self, time_s, torques) -> None:
        self.set_time(time_s)
        for leg, value in torques.items():
            rr.log(f"Torques/{leg}", rr.Scalars(float(value)))

    def log_body_height(self, time_s, height) -> None:
        self.set_time(time_s)
        rr.log("Body/Height", rr.Scalars(float(height)))

    def log_control_state(self, time_s, command, base_pos=None) -> None:
        self.set_time(time_s)
        if base_pos is not None:
            base = np.asarray(base_pos, dtype=float).reshape(3)
            rr.log("Control/base_x", rr.Scalars(float(base[0])))
            rr.log("Control/base_z", rr.Scalars(float(base[2])))
        rr.log("Control/target_x_vel", rr.Scalars(float(command.x_vel)))
        rr.log("Control/target_z_pos", rr.Scalars(float(command.z_pos)))
        rr.log("Control/swing_height", rr.Scalars(float(command.swing_height)))
        rr.log("Control/terrain_confidence", rr.Scalars(float(command.terrain_confidence)))
        if base_pos is not None:
            text = (
                f"# {command.state}\n"
                f"- base_x: {base[0]:.3f} m\n"
                f"- base_z: {base[2]:.3f} m\n"
            )
        else:
            text = f"# {command.state}\n"
        text += (
            f"- target_x_vel: {command.x_vel:.3f} m/s\n"
            f"- target_z_pos: {command.z_pos:.3f} m\n"
            f"- swing_height: {command.swing_height:.3f} m\n"
            f"- step_distance: {getattr(command, 'step_distance_m', 5.0):.3f} m\n"
            f"- step_height: {getattr(command, 'active_step_height', 0.0):.3f} m\n"
            f"- leg_group: {getattr(command, 'active_leg_group', 'none')}\n"
            f"- forced_swing: {', '.join(getattr(command, 'forced_swing_legs', ())) or 'none'}\n"
            f"- forced_stance: {', '.join(getattr(command, 'forced_stance_legs', ())) or 'none'}\n"
            f"- confidence: {command.terrain_confidence:.2f}\n"
        )
        rr.log("Control/state", rr.TextDocument(text, media_type="text/markdown"))

    def log_footholds(self, time_s, current_feet: dict, touchdown_feet: dict, foot_contacts: dict | None = None) -> None:
        self.set_time(time_s)
        legs = ("FL", "FR", "RL", "RR")
        current = np.asarray([current_feet[leg] for leg in legs], dtype=np.float32)
        planned = np.asarray([touchdown_feet[leg] for leg in legs], dtype=np.float32)
        if foot_contacts is None:
            current_colors = [0, 180, 255]
        else:
            current_colors = [
                [0, 220, 120] if foot_contacts.get(leg, False) else [0, 120, 255]
                for leg in legs
            ]
        rr.log(
            "World/footholds/current",
            rr.Points3D(current, radii=0.025, labels=list(legs), colors=current_colors),
        )
        rr.log(
            "World/footholds/planned_touchdown",
            rr.Points3D(planned, radii=0.035, labels=list(legs), colors=[255, 140, 0]),
        )
        strips = [np.vstack((current[i], planned[i])) for i in range(len(legs))]
        rr.log("World/footholds/foot_to_touchdown", rr.LineStrips3D(strips, radii=0.008, colors=[255, 255, 255]))
