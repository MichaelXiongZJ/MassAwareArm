"""Computed-torque trajectory-tracking controller."""

from __future__ import annotations

import numpy as np

from massaware.controllers.references import ControlOutput, JointReference
from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot


class InverseDynamicsTrackingController:
    """Joint-space computed-torque controller for the UR5e arm."""

    def __init__(
        self,
        env: MujocoEnv,
        robot: Robot,
        kp: np.ndarray | list[float],
        kd: np.ndarray | list[float],
        gravity: float = 9.81,
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

    def reset(self) -> None:
        pass

    def apply_overrides(self, overrides: dict) -> None:
        if "kp" in overrides:
            self.kp = np.asarray(overrides["kp"], dtype=float)
        if "kd" in overrides:
            self.kd = np.asarray(overrides["kd"], dtype=float)

    def clear_overrides(self) -> None:
        self.kp = self._base_gains["kp"].copy()
        self.kd = self._base_gains["kd"].copy()

    def command(
        self,
        reference: JointReference,
        *,
        gravity_mask: np.ndarray | None = None,
        payload_mass: float = 0.0,
    ) -> ControlOutput:
        q = self.env.get_arm_qpos()
        q_dot = self.env.get_arm_qvel()
        q_error = reference.q - q
        q_dot_error = reference.q_dot - q_dot

        q_ddot_cmd = reference.q_ddot + self.kd * q_dot_error + self.kp * q_error
        if gravity_mask is None:
            gravity_mask = np.ones_like(q)
        qfrc_bias = self.env.qfrc_bias
        tau_feedforward = self.env.mass_matrix() @ reference.q_ddot + qfrc_bias * gravity_mask
        tau_nominal = self.env.mass_matrix() @ q_ddot_cmd + qfrc_bias * gravity_mask
        tau_payload = self._payload_compensation(payload_mass)
        tau_cmd = tau_nominal + tau_payload
        self.env.set_arm_ctrl(tau_cmd)

        return ControlOutput(
            tau_cmd=tau_cmd.copy(),
            tau_feedforward=tau_feedforward.copy(),
            tau_nominal=tau_nominal.copy(),
            tau_payload=tau_payload.copy(),
            q_error=q_error,
            q_dot_error=q_dot_error,
        )

    def _payload_compensation(self, payload_mass: float) -> np.ndarray:
        if payload_mass <= 0.0:
            return np.zeros_like(self.kp)
        upward_force = np.array([0.0, 0.0, payload_mass * self.gravity])
        q = self.env.get_arm_qpos()
        return self.robot.jacobian_ee(q).T @ upward_force
