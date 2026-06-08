"""Joint-space PID trajectory-tracking controller."""

from __future__ import annotations

import numpy as np

from massaware.controllers.references import ControlOutput, JointReference
from massaware.mujoco_env import MujocoEnv


class TrackingPIDController:
    """PID controller that tracks joint position and velocity references."""

    def __init__(
        self,
        env: MujocoEnv,
        kp: np.ndarray | list[float],
        ki: np.ndarray | list[float],
        kd: np.ndarray | list[float],
    ) -> None:
        self.env = env
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self._base_gains = {
            "kp": self.kp.copy(),
            "ki": self.ki.copy(),
            "kd": self.kd.copy(),
        }
        self.integral_error = np.zeros_like(self.kp)

    def reset(self) -> None:
        self.integral_error.fill(0.0)

    def apply_overrides(self, overrides: dict) -> None:
        for key, value in overrides.items():
            if key not in self._base_gains:
                continue
            setattr(self, key, np.asarray(value, dtype=float))
        if overrides:
            self.reset()

    def clear_overrides(self) -> None:
        self.kp = self._base_gains["kp"].copy()
        self.ki = self._base_gains["ki"].copy()
        self.kd = self._base_gains["kd"].copy()
        self.reset()

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

        self.integral_error += q_error * self.env.dt
        tau_feedback = (
            self.kp * q_error
            + self.ki * self.integral_error
            + self.kd * q_dot_error
        )
        if gravity_mask is None:
            gravity_mask = np.ones_like(q)
        tau_feedforward = self.env.qfrc_bias * gravity_mask
        tau_payload = self._payload_compensation(payload_mass)
        tau_nominal = tau_feedback + tau_feedforward
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
        # Kept for interface symmetry. Payload compensation is intentionally
        # disabled in these experiments unless a runner supplies a mass.
        return np.zeros_like(self.kp) if payload_mass <= 0.0 else np.zeros_like(self.kp)
