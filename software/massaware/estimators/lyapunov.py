"""Lyapunov (Spring-Sag) mass estimator.

Principle
---------
With ki=0 (so the proportional gains act as multi-dimensional springs), kd
softened (so the loop reaches a clean rest), and full gravity compensation
enabled, the arm sags from `q_empty` to `q_loaded` under the payload.

For the PD-controlled step response of a spring-damper system from one rest
state to another, work–energy bookkeeping gives, for any positive damping:

    W_grav = ΔE_spring + W_damper          (energy conservation)
    F = kp·disp at equilibrium             (force balance)
    W_grav = F · disp = kp · disp^2
    ΔE_spring = ½ kp · disp^2
    => W_damper = ½ kp · disp^2 = ΔE_spring  (exactly half dissipated)

Hence

    ΔE_spring = Σ_j ½ kp_j ( (q_loaded_j - q_ref_j)^2 - (q_empty_j - q_ref_j)^2 )
    Δh        = z_empty - z_loaded            (EE descends => Δh > 0)
    m_hat     = 2 · ΔE_spring / (g · Δh)

Calibration at INIT (same mask + overrides, empty gripper) captures
`q_empty` and `z_empty`. The same controller configuration is reused at WEIGH.
"""

from __future__ import annotations

import numpy as np

from massaware.estimators.base import EstimateResult, Estimator, EstimatorObs
from massaware.estimators.registry import register

GRAVITY = 9.81


class LyapunovEstimator(Estimator):
    name = "lyapunov"
    requires_calibration = True

    def __init__(self, cfg: dict):
        self._base_kp = np.asarray(cfg["controller_kp"], dtype=float)
        self._base_kd = np.asarray(cfg["controller_kd"], dtype=float)
        self._n = int(self._base_kp.shape[0])

        # Soften kp during weighing so sub-gram sag isn't dominated by sim
        # numerical noise. Defaults chosen so a 0.2 kg payload produces several
        # mm of EE-z sag, well above the floating-point / quasi-static floor.
        kp_soft_scale = float(cfg.get("kp_soft_scale", 0.1))
        kd_soft_scale = float(cfg.get("kd_soft_scale", 0.5))
        self._kp_soft = self._base_kp * kp_soft_scale
        self._kd_soft = self._base_kd * kd_soft_scale
        # The energy formula uses the *active* kp at the time of measurement.
        self._kp = self._kp_soft

        # Calibration baselines (filled by load_calibration / finish_calibration).
        self.q_empty: np.ndarray | None = None
        self.z_empty: float | None = None

        # Per-tick samples (weigh and calibration use separate buffers).
        self._q_samples: list[np.ndarray] = []
        self._z_samples: list[float] = []
        self._cal_q_samples: list[np.ndarray] = []
        self._cal_z_samples: list[float] = []

        # Remember the q_ref used for the most recent estimate, so the elastic
        # energy reference is the same one the controller is tracking. WeighState
        # always commands q_ref = ctx.weigh_qpos, but we read it from `obs.q_ref`
        # to avoid coupling to ctx.
        self._last_q_ref: np.ndarray | None = None

    # ----- Controller / gravity hooks -----

    def gravity_comp_mask(self, n_joints: int) -> np.ndarray:
        """Full gravity compensation: the arm should sag only from the payload."""
        return np.ones(n_joints)

    def controller_overrides(self) -> dict:
        """Softened P-only loop during weighing/calibration.

        - `kp` is softened so the arm visibly sags under the payload (otherwise
          numerical floor swamps the sub-mm signal at base kp ~ 2000).
        - `kd` is scaled down to keep the loop's damping ratio sensible.
        - `ki` is forced to zero so the proportional term acts as a pure spring,
          which is the physical model the energy formula assumes.
        """
        return {"kp": self._kp_soft, "kd": self._kd_soft, "ki": np.zeros(self._n)}

    # ----- Weighing (loaded hold) -----

    def reset(self) -> None:
        self._q_samples.clear()
        self._z_samples.clear()
        self._last_q_ref = None

    def update(self, obs: EstimatorObs) -> None:
        self._q_samples.append(obs.q.copy())
        self._last_q_ref = obs.q_ref.copy()
        self._z_samples.append(float(obs.ee_xyz[2]))

    def estimate(self) -> EstimateResult:
        if self.q_empty is None or self.z_empty is None:
            raise RuntimeError("LyapunovEstimator.estimate() called before calibration was loaded")
        if not self._q_samples or self._last_q_ref is None:
            raise RuntimeError("LyapunovEstimator.estimate() called with no samples")

        q_loaded = np.mean(np.stack(self._q_samples, axis=0), axis=0)
        z_loaded = float(np.mean(self._z_samples))
        return self._compute(q_loaded, z_loaded, self._last_q_ref)

    # ----- Calibration -----

    def start_calibration(self, ctx) -> None:  # noqa: ARG002
        self._cal_q_samples.clear()
        self._cal_z_samples.clear()

    def update_calibration(self, obs: EstimatorObs) -> None:
        self._cal_q_samples.append(obs.q.copy())
        self._cal_z_samples.append(float(obs.ee_xyz[2]))

    def finish_calibration(self) -> dict:
        if not self._cal_q_samples:
            raise RuntimeError("LyapunovEstimator.finish_calibration() called with no samples")
        q_empty = np.mean(np.stack(self._cal_q_samples, axis=0), axis=0)
        z_empty = float(np.mean(self._cal_z_samples))
        return {
            "q_empty": [float(x) for x in q_empty],
            "z_empty": z_empty,
        }

    def load_calibration(self, data: dict) -> None:
        if "q_empty" not in data or "z_empty" not in data:
            raise KeyError("calibration data missing 'q_empty' / 'z_empty' (required by lyapunov)")
        self.q_empty = np.asarray(data["q_empty"], dtype=float)
        self.z_empty = float(data["z_empty"])

    # ----- Internal math -----

    def _compute(self, q_loaded: np.ndarray, z_loaded: float, q_ref: np.ndarray) -> EstimateResult:
        e_loaded = 0.5 * self._kp * (q_loaded - q_ref) ** 2
        e_empty  = 0.5 * self._kp * (self.q_empty  - q_ref) ** 2
        delta_e = float(np.sum(e_loaded - e_empty))
        delta_h = float(self.z_empty - z_loaded)
        if abs(delta_h) < 1e-6:
            # Arm didn't sag — divide-by-near-zero would explode. Return 0 with
            # a high sigma so the classifier flags it as light if anything.
            return EstimateResult(
                m_hat=0.0, sigma=float("inf"),
                diagnostics={"delta_e": delta_e, "delta_h": delta_h, "warning": "no sag detected"},
            )
        # Factor of 2: under PD step response, the damper dissipates exactly
        # as much energy as the spring stores. See module docstring.
        m_hat = 2.0 * delta_e / (GRAVITY * delta_h)
        return EstimateResult(
            m_hat=m_hat,
            sigma=None,
            diagnostics={
                "delta_e": delta_e,
                "delta_h": delta_h,
                "q_loaded": [float(x) for x in q_loaded],
                "z_loaded": z_loaded,
                "n_samples": len(self._q_samples),
            },
        )


register("lyapunov", LyapunovEstimator)