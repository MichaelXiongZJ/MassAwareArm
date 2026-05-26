"""PID-error mass estimator (single-joint, elbow).

Principle
---------
With the measurement joint's gravity compensation disabled and an integral
term active, the PID controller's steady-state `tau_cmd` at that joint must
balance the joint torque produced by gravity acting on everything distal to
it. The difference between loaded and unloaded steady-state torque is

    tau_cmd_loaded - tau_ss_empty = m * g * moment_arm

so the mass is

    m_hat = (mean(tau_cmd) - tau_ss_empty) / (g * moment_arm)

`tau_ss_empty` is captured at INIT via `calibrate()` (same pose, same mask,
empty gripper). `moment_arm` is hand-tuned in config and pinned for
later FK-derivation (see docs/PINNED_ISSUES.md P1).
"""

from __future__ import annotations

import numpy as np

from massaware.estimators.base import EstimateResult, Estimator, EstimatorObs
from massaware.estimators.registry import register

GRAVITY = 9.81  # m/s^2


class PIDErrorEstimator(Estimator):
    name = "pid_error"
    requires_calibration = True

    def __init__(self, cfg: dict):
        self.measurement_joint: int = int(cfg["measurement_joint"])
        self.moment_arm: float = float(cfg["moment_arm"])
        self.tau_ss_empty: float | None = None  # populated by load_calibration()
        self._samples: list[float] = []
        # Calibration accumulator (separate buffer from weigh samples).
        self._cal_samples: list[float] = []

        # Softened gains on the measurement joint avoid the high-stiffness
        # limit cycle the uncompensated joint produces with base kp/kd. The
        # mean tau_cmd is unchanged (still tracks gravity load exactly); the
        # softer loop just reduces the per-sample variance by ~10x.
        self._base_kp = np.asarray(cfg["controller_kp"], dtype=float)
        self._base_kd = np.asarray(cfg["controller_kd"], dtype=float)
        self._kp_soft_meas: float = float(cfg.get("kp_soft", 200.0))
        self._kd_soft_meas: float = float(cfg.get("kd_soft", 80.0))

    # ----- Controller / gravity hooks -----

    def gravity_comp_mask(self, n_joints: int) -> np.ndarray:
        """All joints compensated except the measurement joint."""
        mask = np.ones(n_joints)
        mask[self.measurement_joint] = 0.0
        return mask

    def controller_overrides(self) -> dict:
        """Soften the measurement joint only; other joints stay at base gains."""
        kp = self._base_kp.copy()
        kd = self._base_kd.copy()
        kp[self.measurement_joint] = self._kp_soft_meas
        kd[self.measurement_joint] = self._kd_soft_meas
        return {"kp": kp, "kd": kd}

    # ----- Weighing -----

    def reset(self) -> None:
        self._samples.clear()

    def update(self, obs: EstimatorObs) -> None:
        self._samples.append(float(obs.tau_cmd[self.measurement_joint]))

    def estimate(self) -> EstimateResult:
        if self.tau_ss_empty is None:
            raise RuntimeError("PIDErrorEstimator.estimate() called before calibration was loaded")
        if not self._samples:
            raise RuntimeError("PIDErrorEstimator.estimate() called with no samples")
        tau_arr = np.asarray(self._samples, dtype=float)
        tau_mean = float(tau_arr.mean())
        tau_std = float(tau_arr.std(ddof=1)) if tau_arr.size > 1 else 0.0
        m_hat = (tau_mean - self.tau_ss_empty) / (GRAVITY * self.moment_arm)
        sigma = tau_std / (GRAVITY * self.moment_arm) if tau_std > 0 else None
        return EstimateResult(
            m_hat=m_hat,
            sigma=sigma,
            diagnostics={
                "tau_mean": tau_mean,
                "tau_std": tau_std,
                "tau_ss_empty": self.tau_ss_empty,
                "moment_arm": self.moment_arm,
                "n_samples": int(tau_arr.size),
            },
        )

    # ----- Calibration (tick-driven by InitState) -----

    def start_calibration(self, ctx) -> None:  # noqa: ARG002
        self._cal_samples.clear()

    def update_calibration(self, obs: EstimatorObs) -> None:
        self._cal_samples.append(float(obs.tau_cmd[self.measurement_joint]))

    def finish_calibration(self) -> dict:
        if not self._cal_samples:
            raise RuntimeError("PIDErrorEstimator.finish_calibration() called with no samples")
        tau_ss = float(np.mean(self._cal_samples))
        # Persist a full 6-vector for human readability; only the meas joint is used.
        full = [0.0] * 6
        full[self.measurement_joint] = tau_ss
        return {"tau_ss_empty": full}

    def load_calibration(self, data: dict) -> None:
        if "tau_ss_empty" not in data:
            raise KeyError("calibration data missing 'tau_ss_empty' (required by pid_error)")
        arr = np.asarray(data["tau_ss_empty"], dtype=float)
        self.tau_ss_empty = float(arr[self.measurement_joint])


register("pid_error", PIDErrorEstimator)
