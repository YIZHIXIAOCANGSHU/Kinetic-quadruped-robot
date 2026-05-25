from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObstacleAwareCommand:
    state: str
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float
    gait_duty: float
    gait_hz: float
    swing_height: float
    z_weight: float
    vz_weight: float
    obstacle_distance_m: float
    obstacle_height_m: float
    support_height_m: float
    memory_support_height_m: float
    terrain_delta_m: float
    terrain_confidence: float
    support_source: str


class ObstacleAwareCommandAdapter:
    def __init__(
        self,
        *,
        nominal_body_height_m: float = 0.27,
        approach_speed_mps: float = 0.15,
        climb_speed_mps: float = 0.12,
        descend_speed_mps: float = 0.10,
        min_obstacle_swing_m: float = 0.10,
        max_obstacle_swing_m: float = 0.22,
        body_height_rate_limit_mps: float = 0.15,
        confidence_threshold: float = 0.25,
        max_support_raise_m: float = 0.45,
        obstacle_memory_s: float = 1.5,
        obstacle_gait_hz: float = 2.0,
        obstacle_gait_duty: float = 0.65,
    ):
        self.nominal_body_height_m = float(nominal_body_height_m)
        self.approach_speed_mps = float(approach_speed_mps)
        self.climb_speed_mps = float(climb_speed_mps)
        self.descend_speed_mps = float(descend_speed_mps)
        self.min_obstacle_swing_m = float(min_obstacle_swing_m)
        self.max_obstacle_swing_m = float(max_obstacle_swing_m)
        self.body_height_rate_limit_mps = float(body_height_rate_limit_mps)
        self.confidence_threshold = float(confidence_threshold)
        self.max_support_raise_m = float(max_support_raise_m)
        self.obstacle_memory_s = float(obstacle_memory_s)
        self.obstacle_gait_hz = float(obstacle_gait_hz)
        self.obstacle_gait_duty = float(obstacle_gait_duty)
        self._z_pos = self.nominal_body_height_m
        self._state = "NORMAL"
        self._obstacle_memory_s = 0.0

    @property
    def state(self) -> str:
        return self._state

    def adapt(self, raw_cmd, ramped_x_vel: float, obstacle, dt: float) -> ObstacleAwareCommand:
        if float(raw_cmd.x_vel) <= 0.0 and abs(float(ramped_x_vel)) <= 1e-6:
            self._z_pos = float(raw_cmd.z_pos)
            self._state = "NORMAL"
            self._obstacle_memory_s = 0.0
            return ObstacleAwareCommand(
                state="NORMAL",
                x_vel=0.0,
                y_vel=float(raw_cmd.y_vel),
                z_pos=float(raw_cmd.z_pos),
                yaw_rate=float(raw_cmd.yaw_rate),
                gait_duty=float(raw_cmd.gait_duty),
                gait_hz=float(raw_cmd.gait_hz),
                swing_height=float(raw_cmd.swing_height),
                z_weight=float(raw_cmd.z_weight),
                vz_weight=float(raw_cmd.vz_weight),
                obstacle_distance_m=float("inf"),
                obstacle_height_m=0.0,
                support_height_m=0.0,
                memory_support_height_m=0.0,
                terrain_delta_m=0.0,
                terrain_confidence=0.0,
                support_source="idle",
            )

        confidence = float(getattr(obstacle, "confidence", 0.0)) if obstacle is not None else 0.0
        valid = bool(getattr(obstacle, "valid", False)) if obstacle is not None else False
        distance = float(getattr(obstacle, "distance_m", float("inf"))) if obstacle is not None else float("inf")
        height = float(getattr(obstacle, "height_m", 0.0)) if obstacle is not None else 0.0
        support_height = float(getattr(obstacle, "support_height_m", 0.0)) if obstacle is not None else 0.0
        memory_support_height = float(
            getattr(obstacle, "memory_support_height_m", support_height)
        ) if obstacle is not None else 0.0
        terrain_delta = float(getattr(obstacle, "terrain_delta_m", height)) if obstacle is not None else 0.0
        obstacle_delta = terrain_delta if abs(terrain_delta) > 1e-6 else height
        support_source = str(getattr(obstacle, "support_source", "none")) if obstacle is not None else "none"
        state_hint = str(getattr(obstacle, "state_hint", "")) if obstacle is not None else ""

        dt_s = max(float(dt), 0.0)
        reliable_obstacle = valid and confidence >= self.confidence_threshold and (
            (distance < 1.0 and abs(obstacle_delta) > 0.05) or support_height > 0.05
        )
        if reliable_obstacle:
            self._obstacle_memory_s = self.obstacle_memory_s
        else:
            self._obstacle_memory_s = max(0.0, self._obstacle_memory_s - dt_s)

        support_raise = min(max(support_height, 0.0), self.max_support_raise_m)
        support_z = self.nominal_body_height_m + support_raise
        gait_hz = float(raw_cmd.gait_hz)
        gait_duty = float(raw_cmd.gait_duty)

        if not valid or confidence < self.confidence_threshold:
            state = "NORMAL"
            x_vel = float(ramped_x_vel)
            swing_height = float(raw_cmd.swing_height)
            target_z = support_z if support_height > 0.03 else float(raw_cmd.z_pos)
            if self._obstacle_memory_s > 0.0:
                state = "APPROACH_UP"
                x_vel = min(x_vel, self.approach_speed_mps)
                swing_height = max(swing_height, self.min_obstacle_swing_m)
                gait_hz = self.obstacle_gait_hz
                gait_duty = self.obstacle_gait_duty
        elif obstacle_delta < -0.05 and distance < 0.70:
            state = "DESCENDING" if distance < 0.35 or state_hint == "DESCENDING" else "APPROACH_DOWN"
            x_vel = min(float(ramped_x_vel), self.descend_speed_mps)
            swing_height = min(
                0.18,
                max(float(raw_cmd.swing_height), self.min_obstacle_swing_m, abs(obstacle_delta) + 0.05),
            )
            target_z = support_z
            gait_hz = self.obstacle_gait_hz
            gait_duty = self.obstacle_gait_duty
        elif obstacle_delta > 0.05 and distance < 0.45:
            state = "CLIMBING"
            x_vel = min(float(ramped_x_vel), self.climb_speed_mps)
            swing_height = min(
                self.max_obstacle_swing_m,
                max(0.16, self.min_obstacle_swing_m, obstacle_delta + 0.08),
            )
            target_z = support_z
            gait_hz = self.obstacle_gait_hz
            gait_duty = self.obstacle_gait_duty
        elif obstacle_delta > 0.05 and distance < 1.0:
            state = "APPROACH_UP"
            x_vel = min(float(ramped_x_vel), self.approach_speed_mps)
            swing_height = min(
                self.max_obstacle_swing_m,
                max(0.16, self.min_obstacle_swing_m, obstacle_delta + 0.06),
            )
            target_z = support_z
            gait_hz = self.obstacle_gait_hz
            gait_duty = self.obstacle_gait_duty
        elif support_height > 0.05:
            state = "ON_PLATFORM"
            x_vel = min(float(ramped_x_vel), self.approach_speed_mps)
            swing_height = max(float(raw_cmd.swing_height), self.min_obstacle_swing_m)
            target_z = support_z
            gait_hz = self.obstacle_gait_hz
            gait_duty = self.obstacle_gait_duty
        else:
            state = "NORMAL"
            x_vel = float(ramped_x_vel)
            swing_height = float(raw_cmd.swing_height)
            target_z = float(raw_cmd.z_pos)

        max_delta = self.body_height_rate_limit_mps * max(float(dt), 0.0)
        if self._z_pos == self.nominal_body_height_m and abs(float(raw_cmd.z_pos) - self._z_pos) > 0.03:
            self._z_pos = float(raw_cmd.z_pos)
        if target_z > self._z_pos + max_delta:
            z_pos = self._z_pos + max_delta
        elif target_z < self._z_pos - max_delta:
            z_pos = self._z_pos - max_delta
        else:
            z_pos = target_z
        self._z_pos = float(z_pos)
        self._state = state

        return ObstacleAwareCommand(
            state=state,
            x_vel=x_vel,
            y_vel=float(raw_cmd.y_vel),
            z_pos=float(z_pos),
            yaw_rate=float(raw_cmd.yaw_rate),
            gait_duty=float(gait_duty),
            gait_hz=float(gait_hz),
            swing_height=float(swing_height),
            z_weight=float(raw_cmd.z_weight),
            vz_weight=float(raw_cmd.vz_weight),
            obstacle_distance_m=distance,
            obstacle_height_m=height,
            support_height_m=support_height,
            memory_support_height_m=memory_support_height,
            terrain_delta_m=obstacle_delta,
            terrain_confidence=confidence,
            support_source=support_source,
        )
