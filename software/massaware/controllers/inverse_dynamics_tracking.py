"""Computed-torque trajectory-tracking controller."""

from __future__ import annotations

import numpy as np

from massaware.controllers.base import TrackingControllerBase
from massaware.controllers.references import ControlOutput, JointReference
from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot


class InverseDynamicsTrackingController(TrackingControllerBase):
    """Joint-space computed-torque controller for the UR5e arm."""

    def __init__(
        self,
        env: MujocoEnv,
        robot: Robot,
        kp: np.ndarray | list[float],
        kd: np.ndarray | list[float],
        gravity: float = 9.81,
        allow_overrides: bool = True,
    ) -> None:
        super().__init__(
            env,
            robot,
            kp=kp,
            kd=kd,
            gravity=gravity,
            allow_overrides=allow_overrides,
        )

    def command(
        self,
        reference: JointReference,
        *,
        gravity_mask: np.ndarray | None = None,
        payload_mass: float = 0.0,
    ) -> ControlOutput:
        q, _, q_error, q_dot_error = self.tracking_errors(reference.q, reference.q_dot)

        q_ddot_cmd = reference.q_ddot + self.kd * q_dot_error + self.kp * q_error
        gravity_mask = self.gravity_compensation_mask(gravity_mask, q)
        qfrc_bias = self.env.qfrc_bias
        tau_gravity = self.env.gravity_torque(q)
        tau_coriolis = qfrc_bias - tau_gravity
        tau_bias = tau_coriolis + tau_gravity * gravity_mask
        mass_matrix = self.env.mass_matrix()
        tau_feedback_accel = self.kd * q_dot_error + self.kp * q_error
        tau_feedback = mass_matrix @ tau_feedback_accel
        tau_feedforward = mass_matrix @ reference.q_ddot + tau_bias
        tau_nominal = mass_matrix @ q_ddot_cmd + tau_bias
        tau_payload = self.payload_compensation(payload_mass)
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
            tau_gravity=(tau_gravity * gravity_mask).copy(),
            tau_bias=tau_bias.copy(),
            q_error=q_error,
            q_dot_error=q_dot_error,
        )
