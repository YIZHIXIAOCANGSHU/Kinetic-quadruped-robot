from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepClimbCommand:
    state: str
    x_vel: float
    z_pos: float
    gait_hz: float
    gait_duty: float
    swing_height: float
    active_step_height: float
    active_leg_group: str
    step_distance_m: float
    step_confidence: float
    force_gait: bool
    forced_swing_legs: tuple[str, ...] = ()
    forced_stance_legs: tuple[str, ...] = ()
    step_edge_world_x: float = np.nan
    raw_step_distance_m: float = np.inf
    remembered_step_distance_m: float = np.inf
    transition_reason: str = ""


class StepClimbStateMachine:
    def __init__(
        self,
        *,
        nominal_body_height_m: float = 0.27,
        approach_speed_mps: float = 0.15,
        shift_speed_mps: float = 0.08,
        stop_distance_m: float = 0.35,
        min_state_duration_s: float = 0.35,
        front_lift_duration_s: float = 0.35,
        shift_duration_s: float = 0.55,
        rear_lift_duration_s: float = 0.35,
        recover_duration_s: float = 0.45,
        climb_gait_hz: float = 1.5,
        climb_gait_duty: float = 0.78,
    ):
        self.nominal_body_height_m = float(nominal_body_height_m)
        self.approach_speed_mps = float(approach_speed_mps)
        self.shift_speed_mps = float(shift_speed_mps)
        self.stop_distance_m = float(stop_distance_m)
        self.min_state_duration_s = float(min_state_duration_s)
        self.front_lift_duration_s = float(front_lift_duration_s)
        self.shift_duration_s = float(shift_duration_s)
        self.rear_lift_duration_s = float(rear_lift_duration_s)
        self.recover_duration_s = float(recover_duration_s)
        self.climb_gait_hz = float(climb_gait_hz)
        self.climb_gait_duty = float(climb_gait_duty)
        self.state = "WALK_FLAT"
        self._state_start_t = 0.0
        self._active_step_height = 0.0
        self._shift_start_x = 0.0
        self._active_edge_x_world = np.nan
        self._last_transition_reason = ""

    def update(
        self,
        time_s: float,
        step_estimate,
        foot_contacts: dict[str, bool],
        *,
        base_x: float,
        raw_x_vel: float,
        raw_z_pos: float,
        controller_ready: bool = True,
    ) -> StepClimbCommand:
        t = float(time_s)
        self._last_transition_reason = ""
        valid_step = bool(getattr(step_estimate, "valid", False))
        measured_distance = float(getattr(step_estimate, "distance_m", np.inf))
        height = float(getattr(step_estimate, "height_m", 0.0))
        confidence = float(getattr(step_estimate, "confidence", 0.0))
        all_feet_contact = all(bool(foot_contacts.get(leg, False)) for leg in ("FL", "FR", "RL", "RR"))

        if valid_step and np.isfinite(measured_distance):
            candidate_edge_x_world = float(base_x) + measured_distance
            if not np.isfinite(self._active_edge_x_world) or self.state == "WALK_FLAT":
                self._active_edge_x_world = candidate_edge_x_world
        remembered_distance = (
            float(self._active_edge_x_world - float(base_x))
            if np.isfinite(self._active_edge_x_world)
            else np.inf
        )
        raw_distance = measured_distance if valid_step and np.isfinite(measured_distance) else np.inf
        distance = remembered_distance if np.isfinite(remembered_distance) else raw_distance
        known_step = bool(np.isfinite(distance))
        climb_lift_states = ("LIFT_FL", "LIFT_FR", "LIFT_RL", "LIFT_RR")

        if not controller_ready and self.state in climb_lift_states + ("SHIFT_FORWARD", "RECOVER_WALK"):
            self._transition("STOP_BEFORE_STEP", t, "controller_not_ready")

        if self.state == "WALK_FLAT":
            if valid_step:
                self._active_step_height = height
                next_state = "STOP_BEFORE_STEP" if distance <= self.stop_distance_m else "APPROACH_STEP"
                self._transition(next_state, t, f"locked_distance={distance:.3f}m")
        elif self.state == "APPROACH_STEP":
            if valid_step:
                self._active_step_height = 0.8 * self._active_step_height + 0.2 * height
            if self._can_transition(t) and known_step and distance <= self.stop_distance_m:
                self._transition("STOP_BEFORE_STEP", t, f"locked_distance={distance:.3f}m")
        elif self.state == "STOP_BEFORE_STEP":
            if valid_step:
                self._active_step_height = 0.8 * self._active_step_height + 0.2 * height
            if self._can_transition(t) and all_feet_contact and controller_ready:
                self._transition("LIFT_FL", t, "all_feet_contact")
        elif self.state == "LIFT_FL":
            if t - self._state_start_t >= self.front_lift_duration_s:
                self._transition("LIFT_FR", t, f"FL_lift_elapsed={t - self._state_start_t:.3f}s")
        elif self.state == "LIFT_FR":
            if t - self._state_start_t >= self.front_lift_duration_s:
                self._shift_start_x = float(base_x)
                self._transition("SHIFT_FORWARD", t, f"FR_lift_elapsed={t - self._state_start_t:.3f}s")
        elif self.state == "SHIFT_FORWARD":
            moved = float(base_x) - self._shift_start_x
            target_move = max(0.12, min(0.22, self._active_step_height + 0.10))
            if t - self._state_start_t >= self.shift_duration_s or moved >= target_move:
                reason = f"moved={moved:.3f}m target={target_move:.3f}m"
                self._transition("LIFT_RL", t, reason)
        elif self.state == "LIFT_RL":
            if t - self._state_start_t >= self.rear_lift_duration_s:
                self._transition("LIFT_RR", t, f"RL_lift_elapsed={t - self._state_start_t:.3f}s")
        elif self.state == "LIFT_RR":
            if t - self._state_start_t >= self.rear_lift_duration_s:
                self._transition("RECOVER_WALK", t, f"RR_lift_elapsed={t - self._state_start_t:.3f}s")
        elif self.state == "RECOVER_WALK":
            if self._can_transition(t) and t - self._state_start_t >= self.recover_duration_s and all_feet_contact:
                self._transition("WALK_FLAT", t, "recover_complete")
                self._active_step_height = 0.0
                self._active_edge_x_world = np.nan

        return self._command(raw_x_vel, raw_z_pos, distance, confidence, raw_distance, remembered_distance)

    def _can_transition(self, time_s: float) -> bool:
        return float(time_s) - self._state_start_t >= self.min_state_duration_s

    def _transition(self, next_state: str, time_s: float, reason: str) -> None:
        if next_state != self.state:
            previous_state = self.state
            self.state = next_state
            self._state_start_t = float(time_s)
            self._last_transition_reason = f"{previous_state}->{next_state}: {reason}"

    def _command(
        self,
        raw_x_vel: float,
        raw_z_pos: float,
        distance: float,
        confidence: float,
        raw_distance: float,
        remembered_distance: float,
    ) -> StepClimbCommand:
        step_h = float(np.clip(self._active_step_height, 0.0, 0.35))
        state = self.state
        x_vel = float(raw_x_vel)
        z_pos = float(raw_z_pos)
        gait_hz = 3.0
        gait_duty = 0.60
        swing_height = 0.08
        leg_group = "none"
        forced_swing_legs: tuple[str, ...] = ()
        forced_stance_legs: tuple[str, ...] = ()

        if state == "APPROACH_STEP":
            x_vel = min(x_vel, self.approach_speed_mps)
            gait_hz = self.climb_gait_hz
            gait_duty = self.climb_gait_duty
            swing_height = max(0.12, step_h + 0.06)
        elif state == "STOP_BEFORE_STEP":
            x_vel = 0.0
            gait_hz = self.climb_gait_hz
            gait_duty = 1.0
            swing_height = max(0.12, step_h + 0.06)
            forced_stance_legs = ("FL", "FR", "RL", "RR")
        elif state in ("LIFT_FL", "LIFT_FR", "LIFT_RL", "LIFT_RR"):
            x_vel = 0.0
            gait_hz = self.climb_gait_hz
            gait_duty = self.climb_gait_duty
            swing_height = max(0.16, step_h + 0.08)
            leg = state.removeprefix("LIFT_")
            leg_group = leg
            forced_swing_legs = (leg,)
            forced_stance_legs = tuple(other for other in ("FL", "FR", "RL", "RR") if other != leg)
        elif state == "SHIFT_FORWARD":
            x_vel = self.shift_speed_mps
            z_pos = float(raw_z_pos) + min(step_h * 0.5, 0.08)
            gait_hz = self.climb_gait_hz
            gait_duty = 1.0
            swing_height = max(0.14, step_h + 0.06)
            leg_group = "none"
            forced_stance_legs = ("FL", "FR", "RL", "RR")
        elif state == "RECOVER_WALK":
            x_vel = min(float(raw_x_vel), self.approach_speed_mps)
            z_pos = float(raw_z_pos) + min(step_h, 0.18)
            gait_hz = self.climb_gait_hz
            gait_duty = 1.0
            swing_height = max(0.12, step_h + 0.04)
            forced_stance_legs = ("FL", "FR", "RL", "RR")

        return StepClimbCommand(
            state=state,
            x_vel=float(x_vel),
            z_pos=float(z_pos),
            gait_hz=float(gait_hz),
            gait_duty=float(gait_duty),
            swing_height=float(swing_height),
            active_step_height=step_h,
            active_leg_group=leg_group,
            step_distance_m=float(distance),
            step_confidence=float(confidence),
            force_gait=state in ("LIFT_FL", "LIFT_FR", "SHIFT_FORWARD", "LIFT_RL", "LIFT_RR"),
            forced_swing_legs=forced_swing_legs,
            forced_stance_legs=forced_stance_legs,
            step_edge_world_x=float(self._active_edge_x_world),
            raw_step_distance_m=float(raw_distance),
            remembered_step_distance_m=float(remembered_distance),
            transition_reason=self._last_transition_reason,
        )
