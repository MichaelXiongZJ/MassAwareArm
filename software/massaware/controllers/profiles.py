"""Tracking-controller profiles used by the new comparison runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingProfile:
    name: str
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    inverse_dynamics_kp: np.ndarray
    inverse_dynamics_kd: np.ndarray
    grasp_xyz: np.ndarray | None
    weigh_xyz: np.ndarray | None
    light_release_xyz: np.ndarray | None
    heavy_release_xyz: np.ndarray | None
    move_to_release_time: float
    grasp_hold_time: float
    weigh_hold_time_min: float
    release_hold_time: float
    blend_time_fraction: float
    collect_lift_samples: bool
    use_analytical_ik: bool


TRACKING_PROFILE = TrackingProfile(
    name="tracking",
    kp=np.array([1260.0, 1540.0, 1260.0, 504.0, 392.0, 280.0]),
    ki=np.zeros(6),
    kd=np.array([98.0, 112.0, 98.0, 49.0, 42.0, 35.0]),
    inverse_dynamics_kp=np.array([1800.0, 2200.0, 1800.0, 720.0, 560.0, 400.0]),
    inverse_dynamics_kd=np.array([70.0, 80.0, 70.0, 35.0, 30.0, 25.0]),
    grasp_xyz=np.array([0.55, 0.0, 0.45]),
    weigh_xyz=np.array([0.55, 0.0, 0.73]),
    light_release_xyz=np.array([0.0, 0.68, 0.50]),
    heavy_release_xyz=np.array([0.0, -0.68, 0.50]),
    move_to_release_time=6.0,
    grasp_hold_time=0.5,
    weigh_hold_time_min=2.0,
    release_hold_time=0.8,
    blend_time_fraction=0.25,
    collect_lift_samples=False,
    use_analytical_ik=True,
)


def tracking_profile(name: str, cfg: dict) -> TrackingProfile:
    if name in {"tracking", "external"}:
        return TRACKING_PROFILE
    if name != "main":
        raise ValueError(f"Unknown tracking profile '{name}'")
    return TrackingProfile(
        name="main",
        kp=np.asarray(cfg["controller"]["kp"], dtype=float),
        ki=np.asarray(cfg["controller"]["ki"], dtype=float),
        kd=np.asarray(cfg["controller"]["kd"], dtype=float),
        inverse_dynamics_kp=np.asarray(cfg["controller"]["kp"], dtype=float),
        inverse_dynamics_kd=np.asarray(cfg["controller"]["kd"], dtype=float),
        grasp_xyz=None,
        weigh_xyz=None,
        light_release_xyz=np.asarray(cfg["poses"]["light_bin_drop"], dtype=float),
        heavy_release_xyz=np.asarray(cfg["poses"]["heavy_bin_drop"], dtype=float),
        move_to_release_time=3.0,
        grasp_hold_time=1.0,
        weigh_hold_time_min=0.0,
        release_hold_time=0.8,
        blend_time_fraction=0.25,
        collect_lift_samples=False,
        use_analytical_ik=False,
    )


def controller_gains(
    profile: TrackingProfile,
    controller_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if controller_name == "pid_tracking":
        return profile.kp, profile.ki, profile.kd
    if controller_name == "inverse_dynamics":
        return (
            profile.inverse_dynamics_kp,
            np.zeros_like(profile.inverse_dynamics_kp),
            profile.inverse_dynamics_kd,
        )
    raise ValueError(f"Unknown controller '{controller_name}'")
