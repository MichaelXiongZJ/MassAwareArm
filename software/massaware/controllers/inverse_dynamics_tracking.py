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
    ) -> None:
        super().__init__(env, robot, kp=kp, kd=kd, gravity=gravity)

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
        mass_matrix = self.env.mass_matrix()
        tau_feedforward = mass_matrix @ reference.q_ddot + qfrc_bias * gravity_mask
        tau_nominal = mass_matrix @ q_ddot_cmd + qfrc_bias * gravity_mask
        tau_payload = self.payload_compensation(payload_mass)
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
