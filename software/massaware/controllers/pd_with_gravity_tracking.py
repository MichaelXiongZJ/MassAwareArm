"""Joint-space PID trajectory-tracking controller."""

from __future__ import annotations

import numpy as np

from massaware.controllers.base import TrackingControllerBase
from massaware.controllers.references import ControlOutput, JointReference
from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot


class TrackingPIDController(TrackingControllerBase):
    """PID controller that tracks joint position and velocity references."""

    def __init__(
        self,
        env: MujocoEnv,
        robot: Robot,
        kp: np.ndarray | list[float],
        ki: np.ndarray | list[float],
        kd: np.ndarray | list[float],
        gravity: float = 9.81,
        allow_overrides: bool = True,
    ) -> None:
        super().__init__(
            env,
            robot,
            kp=kp,
            ki=ki,
            kd=kd,
            gravity=gravity,
            allow_overrides=allow_overrides,
        )
        self.integral_error = np.zeros_like(self.kp)

    def reset(self) -> None:
        self.integral_error.fill(0.0)

    def command(
        self,
        reference: JointReference,
        *,
        gravity_mask: np.ndarray | None = None,
        payload_mass: float = 0.0,
    ) -> ControlOutput:
        q, _, q_error, q_dot_error = self.tracking_errors(reference.q, reference.q_dot)

        self.integral_error += q_error * self.env.dt
        tau_feedback = (
            self.kp * q_error
            + self.ki * self.integral_error
            + self.kd * q_dot_error
        )
        gravity_mask = self.gravity_compensation_mask(gravity_mask, q)
        tau_gravity = self.env.gravity_torque(q) * gravity_mask
        tau_feedforward = tau_gravity
        tau_payload = self.payload_compensation(payload_mass)
        tau_nominal = tau_feedback + tau_feedforward
        tau_cmd_raw = tau_nominal + tau_payload
        tau_cmd_clipped = self.env.set_arm_ctrl(tau_cmd_raw)

        return ControlOutput(
            tau_cmd=tau_cmd_clipped.copy(),
            tau_cmd_raw=tau_cmd_raw.copy(),
            tau_cmd_clipped=tau_cmd_clipped.copy(),
            tau_feedback=tau_feedback.copy(),
            tau_feedforward=tau_feedforward.copy(),
            tau_nominal=tau_nominal.copy(),
            tau_payload=tau_payload.copy(),
            tau_gravity=tau_gravity.copy(),
            tau_bias=tau_gravity.copy(),
            q_error=q_error,
            q_dot_error=q_dot_error,
        )
