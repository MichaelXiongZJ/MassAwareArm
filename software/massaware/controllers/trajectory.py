"""LSPB joint-space trajectories for the tracking workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    stage: str
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray
    gripper_closed: bool
    collect_mass_samples: bool
    compensate_payload: bool
    # Fraction (1->0) of the payload weight that is both physically applied and
    # compensated on this tick. It is 1.0 everywhere except the release ramp,
    # where the applied payload force and the feedforward are faded out together
    # so the commanded torque has no step when the gripper opens.
    payload_scale: float = 1.0
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
    payload_scale_start: float = 1.0
    payload_scale_end: float = 1.0


class JointSegmentTrajectory:
    """LSPB joint-space trajectory made from named segments."""

    def __init__(
        self,
        segments: list[TrajectorySegment],
        final_q: np.ndarray,
        *,
        final_gripper_closed: bool,
        final_compensate_payload: bool = False,
        final_payload_scale: float = 1.0,
    ) -> None:
        self.segments = segments
        self.final_q = final_q.copy()
        self.final_gripper_closed = final_gripper_closed
        self.final_compensate_payload = final_compensate_payload
        self.final_payload_scale = float(final_payload_scale)

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
            payload_scale=self.final_payload_scale,
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
        release_ramp_time: float = 0.5,
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
            # Hold at the release pose with the gripper still closed and fade the
            # payload weight and its feedforward to zero together. Because the
            # applied force and the compensation track each other throughout the
            # ramp, the arm stays on target and the commanded torque has no step
            # when the gripper finally opens in the next segment.
            hold_segment(
                "release_ramp",
                q_release,
                release_ramp_time,
                self._blend_time(release_ramp_time),
                gripper_closed=True,
                compensate_payload=True,
                payload_scale_start=1.0,
                payload_scale_end=0.0,
            ),
            hold_segment(
                "release",
                q_release,
                release_hold_time,
                self._blend_time(release_hold_time),
                gripper_closed=False,
                payload_scale_start=0.0,
                payload_scale_end=0.0,
            ),
        ]
        super().__init__(
            segments,
            q_release,
            final_gripper_closed=False,
            final_payload_scale=0.0,
        )

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
    release_ramp_time: float = 0.5,
    blend_time_fraction: float = 0.25,
) -> ReleaseTrajectory:
    return ReleaseTrajectory(
        q_initial,
        q_release_approach,
        q_release,
        move_to_release_time=move_to_release_time,
        release_hold_time=release_hold_time,
        release_ramp_time=release_ramp_time,
        blend_time_fraction=blend_time_fraction,
    )


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
    payload_scale_start: float = 1.0,
    payload_scale_end: float = 1.0,
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
        payload_scale_start=float(payload_scale_start),
        payload_scale_end=float(payload_scale_end),
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
    payload_scale_start: float = 1.0,
    payload_scale_end: float = 1.0,
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
        payload_scale_start=payload_scale_start,
        payload_scale_end=payload_scale_end,
    )


def sample_joint_segment(segment: TrajectorySegment, elapsed: float) -> TrajectorySample:
    pos_s, vel_s, acc_s = lspb(elapsed, segment.duration, segment.blend_time)
    delta = segment.goal - segment.start
    # Linear-in-time fade of the payload scale across the segment (independent of
    # the LSPB position blend), so the release ramp is a clean straight line.
    frac = float(np.clip(elapsed / max(segment.duration, 1e-9), 0.0, 1.0))
    payload_scale = (
        segment.payload_scale_start
        + (segment.payload_scale_end - segment.payload_scale_start) * frac
    )
    return TrajectorySample(
        stage=segment.name,
        q=segment.start + pos_s * delta,
        q_dot=vel_s * delta,
        q_ddot=acc_s * delta,
        gripper_closed=segment.gripper_closed,
        collect_mass_samples=segment.collect_mass_samples,
        compensate_payload=segment.compensate_payload,
        payload_scale=payload_scale,
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
