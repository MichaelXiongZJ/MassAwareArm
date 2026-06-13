"""LSPB joint-space trajectories for the tracking workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAYLOAD_COMP_RAMP_DOWN_TIME = 1.0


@dataclass(frozen=True)
class TrajectorySample:
    stage: str
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray
    gripper_closed: bool
    collect_mass_samples: bool
    compensate_payload: bool
    payload_comp_scale: float
    done: bool = False


@dataclass(frozen=True)
class TrajectorySegment:
    name: str
    start: np.ndarray
    goal: np.ndarray
    duration: float
    blend_time: float
    gripper_closed: bool
    collect_mass_samples: bool
    compensate_payload: bool
    payload_comp_ramp_down_time: float


class JointSegmentTrajectory:
    """LSPB joint-space trajectory made from named segments."""

    def __init__(
        self,
        segments: list[TrajectorySegment],
        final_q: np.ndarray,
        *,
        final_gripper_closed: bool,
        final_compensate_payload: bool = False,
    ) -> None:
        self.segments = segments
        self.final_q = final_q.copy()
        self.final_gripper_closed = final_gripper_closed
        self.final_compensate_payload = final_compensate_payload

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)

    def sample(self, elapsed: float) -> TrajectorySample:
        remaining = float(elapsed)
        for segment in self.segments:
            if remaining <= segment.duration:
                return sample_joint_segment(segment, remaining)
            remaining -= segment.duration
        return TrajectorySample(
            stage="done",
            q=self.final_q.copy(),
            q_dot=np.zeros_like(self.final_q),
            q_ddot=np.zeros_like(self.final_q),
            gripper_closed=self.final_gripper_closed,
            collect_mass_samples=False,
            compensate_payload=self.final_compensate_payload,
            payload_comp_scale=1.0 if self.final_compensate_payload else 0.0,
            done=True,
        )


class PickWeighTrajectory(JointSegmentTrajectory):
    """Tracking trajectory for pick and weighing only."""

    def __init__(
        self,
        q_initial: np.ndarray,
        q_grasp: np.ndarray,
        q_weigh: np.ndarray,
        move_to_grasp_time: float = 3.0,
        grasp_hold_time: float = 1.0,
        lift_to_weigh_time: float = 2.0,
        weigh_hold_time: float = 1.5,
        blend_time_fraction: float = 0.25,
        collect_lift_samples: bool = False,
    ) -> None:
        self.blend_time_fraction = float(np.clip(blend_time_fraction, 1e-6, 0.5))
        segments = [
            move_segment(
                "move_to_grasp",
                q_initial,
                q_grasp,
                move_to_grasp_time,
                self._blend_time(move_to_grasp_time),
                gripper_closed=False,
            ),
            hold_segment(
                "grasp",
                q_grasp,
                grasp_hold_time,
                self._blend_time(grasp_hold_time),
                gripper_closed=True,
            ),
            move_segment(
                "lift_to_weigh",
                q_grasp,
                q_weigh,
                lift_to_weigh_time,
                self._blend_time(lift_to_weigh_time),
                gripper_closed=True,
                collect_mass_samples=collect_lift_samples,
            ),
            hold_segment(
                "weigh_hold",
                q_weigh,
                weigh_hold_time,
                self._blend_time(weigh_hold_time),
                gripper_closed=True,
                collect_mass_samples=True,
            ),
        ]
        super().__init__(
            segments,
            q_weigh,
            final_gripper_closed=True,
        )

    def _blend_time(self, duration: float) -> float:
        return max(float(duration), 1e-9) * self.blend_time_fraction


class ReleaseTrajectory(JointSegmentTrajectory):
    """Tracking trajectory from weighing point to release."""

    def __init__(
        self,
        q_initial: np.ndarray,
        q_release_approach: np.ndarray,
        q_release: np.ndarray,
        move_to_release_time: float = 3.0,
        release_hold_time: float = 0.8,
        blend_time_fraction: float = 0.25,
    ) -> None:
        self.blend_time_fraction = float(np.clip(blend_time_fraction, 1e-6, 0.5))
        approach_time, adjust_time = split_duration_by_joint_distance(
            move_to_release_time,
            q_initial,
            q_release_approach,
            q_release,
        )
        segments = [
            move_segment(
                "move_to_release_approach",
                q_initial,
                q_release_approach,
                approach_time,
                self._blend_time(approach_time),
                gripper_closed=True,
                compensate_payload=True,
            ),
            move_segment(
                "move_to_release",
                q_release_approach,
                q_release,
                adjust_time,
                self._blend_time(adjust_time),
                gripper_closed=True,
                compensate_payload=True,
            ),
            hold_segment(
                "release",
                q_release,
                release_hold_time,
                self._blend_time(release_hold_time),
                gripper_closed=False,
                compensate_payload=True,
                payload_comp_ramp_down_time=min(
                    PAYLOAD_COMP_RAMP_DOWN_TIME,
                    release_hold_time,
                ),
            ),
        ]
        super().__init__(segments, q_release, final_gripper_closed=False)

    def _blend_time(self, duration: float) -> float:
        return max(float(duration), 1e-9) * self.blend_time_fraction


JointTrajectory = JointSegmentTrajectory


def make_pick_weigh_trajectory(
    *,
    q_initial: np.ndarray,
    q_grasp: np.ndarray,
    q_weigh: np.ndarray,
    move_to_grasp_time: float = 3.0,
    grasp_hold_time: float = 1.0,
    lift_to_weigh_time: float = 2.0,
    weigh_hold_time: float = 1.5,
    blend_time_fraction: float = 0.25,
    collect_lift_samples: bool = False,
) -> PickWeighTrajectory:
    return PickWeighTrajectory(
        q_initial,
        q_grasp,
        q_weigh,
        move_to_grasp_time=move_to_grasp_time,
        grasp_hold_time=grasp_hold_time,
        lift_to_weigh_time=lift_to_weigh_time,
        weigh_hold_time=weigh_hold_time,
        blend_time_fraction=blend_time_fraction,
        collect_lift_samples=collect_lift_samples,
    )


def make_release_trajectory(
    *,
    q_initial: np.ndarray,
    q_release_approach: np.ndarray,
    q_release: np.ndarray,
    move_to_release_time: float = 3.0,
    release_hold_time: float = 0.8,
    blend_time_fraction: float = 0.25,
) -> ReleaseTrajectory:
    return ReleaseTrajectory(
        q_initial,
        q_release_approach,
        q_release,
        move_to_release_time=move_to_release_time,
        release_hold_time=release_hold_time,
        blend_time_fraction=blend_time_fraction,
    )


def release_time(cli_value: float | None, profile) -> float:
    return profile.move_to_release_time if cli_value is None else float(cli_value)


def release_target(profile, cfg: dict, release_bin: str) -> np.ndarray:
    if release_bin == "heavy":
        target = profile.heavy_release_xyz
        fallback = cfg["poses"]["heavy_bin_drop"]
    else:
        target = profile.light_release_xyz
        fallback = cfg["poses"]["light_bin_drop"]
    return np.asarray(target if target is not None else fallback, dtype=float).copy()


def release_approach_target(profile, release_xyz: np.ndarray) -> np.ndarray:
    approach = np.asarray(release_xyz, dtype=float).copy()
    if profile.weigh_xyz is not None:
        approach[2] = float(profile.weigh_xyz[2])
    else:
        approach[2] = approach[2] + 0.18
    return approach


def move_segment(
    name: str,
    start: np.ndarray,
    goal: np.ndarray,
    duration: float,
    blend_time: float,
    *,
    gripper_closed: bool,
    collect_mass_samples: bool = False,
    compensate_payload: bool = False,
    payload_comp_ramp_down_time: float = 0.0,
) -> TrajectorySegment:
    return TrajectorySegment(
        name=name,
        start=start.copy(),
        goal=goal.copy(),
        duration=max(float(duration), 1e-9),
        blend_time=max(float(blend_time), 1e-9),
        gripper_closed=gripper_closed,
        collect_mass_samples=collect_mass_samples,
        compensate_payload=compensate_payload,
        payload_comp_ramp_down_time=max(float(payload_comp_ramp_down_time), 0.0),
    )


def hold_segment(
    name: str,
    q: np.ndarray,
    duration: float,
    blend_time: float,
    *,
    gripper_closed: bool,
    collect_mass_samples: bool = False,
    compensate_payload: bool = False,
    payload_comp_ramp_down_time: float = 0.0,
) -> TrajectorySegment:
    return move_segment(
        name,
        q,
        q,
        duration,
        blend_time,
        gripper_closed=gripper_closed,
        collect_mass_samples=collect_mass_samples,
        compensate_payload=compensate_payload,
        payload_comp_ramp_down_time=payload_comp_ramp_down_time,
    )


def sample_joint_segment(segment: TrajectorySegment, elapsed: float) -> TrajectorySample:
    pos_s, vel_s, acc_s = lspb(elapsed, segment.duration, segment.blend_time)
    delta = segment.goal - segment.start
    payload_comp_scale = payload_compensation_scale(segment, elapsed)
    return TrajectorySample(
        stage=segment.name,
        q=segment.start + pos_s * delta,
        q_dot=vel_s * delta,
        q_ddot=acc_s * delta,
        gripper_closed=segment.gripper_closed,
        collect_mass_samples=segment.collect_mass_samples,
        compensate_payload=segment.compensate_payload,
        payload_comp_scale=payload_comp_scale,
    )


def payload_compensation_scale(segment: TrajectorySegment, elapsed: float) -> float:
    if not segment.compensate_payload:
        return 0.0
    if segment.payload_comp_ramp_down_time <= 0.0:
        return 1.0
    return float(
        np.clip(
            1.0 - float(elapsed) / segment.payload_comp_ramp_down_time,
            0.0,
            1.0,
        )
    )


def split_duration_by_joint_distance(
    total_duration: float,
    start: np.ndarray,
    middle: np.ndarray,
    goal: np.ndarray,
) -> tuple[float, float]:
    total_duration = max(float(total_duration), 1e-9)
    first_distance = float(np.linalg.norm(middle - start))
    second_distance = float(np.linalg.norm(goal - middle))
    total_distance = first_distance + second_distance
    if total_distance <= 1e-9:
        return 0.5 * total_duration, 0.5 * total_duration
    first_duration = total_duration * first_distance / total_distance
    second_duration = total_duration - first_duration
    return max(first_duration, 1e-9), max(second_duration, 1e-9)


def lspb(elapsed: float, duration: float, blend_time: float) -> tuple[float, float, float]:
    duration = max(float(duration), 1e-9)
    time = float(np.clip(elapsed, 0.0, duration))
    blend = float(np.clip(blend_time, 1e-9, 0.5 * duration))

    if time <= 0.0:
        return 0.0, 0.0, 0.0
    if time >= duration:
        return 1.0, 0.0, 0.0

    cruise_velocity = 1.0 / (duration - blend)
    blend_acceleration = cruise_velocity / blend
    if time <= blend:
        return (
            0.5 * blend_acceleration * time**2,
            blend_acceleration * time,
            blend_acceleration,
        )
    if time <= duration - blend:
        return cruise_velocity * (time - 0.5 * blend), cruise_velocity, 0.0

    remaining = duration - time
    return (
        1.0 - 0.5 * blend_acceleration * remaining**2,
        blend_acceleration * remaining,
        -blend_acceleration,
    )
