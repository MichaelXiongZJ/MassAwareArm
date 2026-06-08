"""Inverse-dynamics residual mass estimator.

This estimator is adapted to the main single-cube mission pipeline. It uses
the per-tick EstimatorObs already assembled by TickLoop and does not require
an empty-gripper calibration pass.
"""

from __future__ import annotations

import numpy as np

from massaware.estimators.base import EstimateResult, Estimator, EstimatorObs
from massaware.estimators.registry import register


class InverseDynamicsResidualEstimator(Estimator):
    """Estimate payload mass from residual joint torque at the weighing pose."""

    name = "inverse_dynamics"
    requires_calibration = False

    def __init__(self, cfg: dict):
        self.min_samples = int(cfg.get("min_samples", 30))
        self.g = float(cfg.get("g", 9.81))
        self.max_q_dot_norm = float(cfg.get("max_q_dot_norm", 0.35))
        self.max_q_error_norm = float(cfg.get("max_q_error_norm", 0.25))
        self.min_leverage = float(cfg.get("min_leverage", 1e-6))
        self.min_mass = float(cfg.get("min_mass", 0.0))
        self.max_mass = float(cfg.get("max_mass", 5.0))
        self.torque_source = str(cfg.get("torque_source", "measured"))
        self.reset()

    def reset(self) -> None:
        self._gravity_torque_samples: list[np.ndarray] = []
        self._residual_samples: list[np.ndarray] = []
        self._rejected_samples = 0

    def update(self, obs: EstimatorObs) -> None:
        gravity_force_per_kg = np.array([0.0, 0.0, -self.g])
        gravity_torque_per_kg = obs.jacobian_ee.T @ gravity_force_per_kg

        q_dot_norm = float(np.linalg.norm(obs.q_dot))
        q_error_norm = float(np.linalg.norm(obs.q_ref - obs.q))
        leverage = float(gravity_torque_per_kg @ gravity_torque_per_kg)
        if (
            q_dot_norm > self.max_q_dot_norm
            or q_error_norm > self.max_q_error_norm
            or leverage < self.min_leverage
        ):
            self._rejected_samples += 1
            return

        tau_total = self._select_torque(obs)
        residual = tau_total - obs.qfrc_bias

        self._gravity_torque_samples.append(gravity_torque_per_kg)
        self._residual_samples.append(residual)

    def estimate(self) -> EstimateResult:
        sample_count = len(self._gravity_torque_samples)
        if sample_count < self.min_samples:
            raise RuntimeError(
                f"Need at least {self.min_samples} samples, got {sample_count}"
            )

        a_stack = np.stack(self._gravity_torque_samples, axis=0)
        residual_stack = np.stack(self._residual_samples, axis=0)

        numerator = float(np.sum(a_stack * residual_stack))
        denominator = float(np.sum(a_stack * a_stack))
        if denominator < self.min_leverage:
            raise RuntimeError("Not enough gravity leverage to estimate payload mass")

        raw_m_hat = -numerator / denominator
        m_hat = float(np.clip(raw_m_hat, self.min_mass, self.max_mass))

        residual_fit = residual_stack + m_hat * a_stack
        residual_rmse = float(np.sqrt(np.mean(residual_fit**2)))
        sigma = residual_rmse / max(np.sqrt(denominator / sample_count), 1e-9)

        return EstimateResult(
            m_hat=m_hat,
            sigma=sigma,
            diagnostics={
                "sample_count": sample_count,
                "rejected_samples": self._rejected_samples,
                "raw_m_hat": raw_m_hat,
                "residual_rmse": residual_rmse,
                "torque_source": self.torque_source,
            },
        )

    def _select_torque(self, obs: EstimatorObs) -> np.ndarray:
        if self.torque_source == "commanded":
            return obs.tau_cmd
        if self.torque_source == "measured":
            return obs.tau_meas
        raise ValueError(
            "inverse_dynamics.torque_source must be 'measured' or 'commanded'"
        )


register("inverse_dynamics", InverseDynamicsResidualEstimator)
