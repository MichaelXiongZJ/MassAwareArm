"""Shared pieces for joint-space tracking controllers."""

from __future__ import annotations

import numpy as np

from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot


class TrackingControllerBase:
    """Common gain, error, and payload-compensation behavior."""

    def __init__(
        self,
        env: MujocoEnv,
        robot: Robot,
        *,
        kp: np.ndarray | list[float],
        kd: np.ndarray | list[float],
        gravity: float = 9.81,
        ki: np.ndarray | list[float] | None = None,
    ) -> None:
        self.env = env
        self.robot = robot
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.gravity = float(gravity)
        self._base_gains = {
            "kp": self.kp.copy(),
            "kd": self.kd.copy(),
        }
        if ki is not None:
            self.ki = np.asarray(ki, dtype=float)
            self._base_gains["ki"] = self.ki.copy()

    def reset(self) -> None:
        pass

    def apply_overrides(self, overrides: dict) -> None:
        for key, value in overrides.items():
            if key not in self._base_gains:
                if key == "ki":
                    continue
                raise KeyError(f"Unknown controller gain override '{key}'")
            setattr(self, key, np.asarray(value, dtype=float))
        if overrides:
            self.reset()

    def clear_overrides(self) -> None:
        for key, value in self._base_gains.items():
            setattr(self, key, value.copy())
        self.reset()

    def tracking_errors(
        self,
        q_ref: np.ndarray,
        q_dot_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.env.get_arm_qpos()
        q_dot = self.env.get_arm_qvel()
        return q, q_dot, q_ref - q, q_dot_ref - q_dot

    def gravity_compensation_mask(
        self,
        gravity_mask: np.ndarray | None,
        q: np.ndarray,
    ) -> np.ndarray:
        return np.ones_like(q) if gravity_mask is None else gravity_mask

    def payload_compensation(self, payload_mass: float) -> np.ndarray:
        if payload_mass <= 0.0:
            return np.zeros_like(self.kp)
        upward_force = np.array([0.0, 0.0, payload_mass * self.gravity])
        q = self.env.get_arm_qpos()
        return self.robot.jacobian_ee(q).T @ upward_force
