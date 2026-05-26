"""Momentum-observer mass estimator (stateful, time-integrating).

Principle
---------
The equations of motion of the arm, with an unknown payload contributing a
joint-space disturbance `tau_ext`, are

    M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau + tau_ext.

Define the generalized momentum p = M(q) q_dot. Using the skew-symmetric
identity Mdot = C + C^T:

    p_dot = tau + tau_ext + C^T(q, q_dot) q_dot - g(q).

For the WEIGH hold the arm has settled, q_dot is small, and C q_dot is
negligible, so g(q) ≈ qfrc_bias (MuJoCo's gravity+Coriolis term).

The observer maintains a predicted momentum `pi` whose dynamics include the
feedback term r itself, so that the difference `p - pi` evolves as a first-
order low-pass on the external disturbance:

    d/dt(pi) = tau + r - qfrc_bias
    r        = K_O * (p - pi)

Taking the time derivative and substituting gives r_dot = K_O * (tau_ext - r)
in the continuous limit, a first-order filter with time constant 1/K_O.

The discrete-time form used in `update()` is forward Euler on `pi` followed
by the algebraic evaluation of `r`:

    pi_{k+1} = pi_k + dt * (tau_k + r_k - qfrc_bias_k)
    r_{k+1}  = K_O * (p_{k+1} - pi_{k+1})

with seeds pi_0 = p_0 and r_0 = 0.

Mass extraction
---------------
At rest the only external Cartesian force from a point-mass payload is
F_ext = -m g z_hat. The corresponding joint-torque vector is

    tau_ext = J_v^T F_ext = -m g * (row of J_v selecting EE z).

Calling that row J_z (a 6-vector), the converged residual obeys
r ≈ -m g J_z. The least-squares mass estimate, after subtracting the empty
arm baseline captured at calibration, is

    m_hat = - <r - r_empty, J_z> / (g * ||J_z||^2).
"""

from __future__ import annotations

import numpy as np

from massaware.estimators.base import EstimateResult, Estimator, EstimatorObs
from massaware.estimators.registry import register

GRAVITY = 9.81  # m / s^2


class MomentumObserver(Estimator):
    name = "momentum_observer"
    requires_calibration = True

    def __init__(self, cfg: dict):
        self._K_O: float = float(cfg.get("K_O", 50.0))
        self._burn_in_s: float = float(cfg.get("burn_in_s", 0.5))
        self._n: int = 6  # UR5e joints

        # Observer state. Reset for every WEIGH / calibration entry.
        self._r = np.zeros(self._n)        # residual estimate of tau_ext
        self._pi = np.zeros(self._n)       # integrated prediction of momentum
        self._initialized = False
        self._t_start: float | None = None
        self._t_prev: float | None = None

        # Sample buffers (populated after burn-in).
        self._r_samples: list[np.ndarray] = []
        self._jz_samples: list[np.ndarray] = []

        # Calibration outcome (loaded after INIT).
        self.r_empty: np.ndarray | None = None

    # ----- Controller / gravity hooks -----

    def gravity_comp_mask(self, n_joints: int) -> np.ndarray:
        """Full gravity compensation; the arm sags only from the payload."""
        return np.ones(n_joints)

    # controller_overrides(): inherits default {} — base PID is fine.

    # ----- Lifecycle -----

    def reset(self) -> None:
        self._r.fill(0.0)
        self._pi.fill(0.0)
        self._initialized = False
        self._t_start = None
        self._t_prev = None
        self._r_samples.clear()
        self._jz_samples.clear()

    def update(self, obs: EstimatorObs) -> None:
        """Advance the discrete observer recursion by one tick and, after the
        burn-in window, record (r, J_z) for the final mean."""
        p = obs.M @ obs.q_dot
        J_z = obs.jacobian_ee[2, :].copy()   # z-row of the 3x6 linear Jacobian

        if not self._initialized:
            self._pi = p.copy()
            self._r.fill(0.0)
            self._t_start = obs.t
            self._t_prev = obs.t
            self._initialized = True
            return

        dt = float(obs.t - self._t_prev) if self._t_prev is not None else 0.0
        self._t_prev = obs.t
        if dt <= 0.0:
            return  # duplicate or out-of-order sample; skip safely

        # De Luca form: advance pi by Euler with the previous-tick r feedback,
        # then compute r algebraically as K_O * (p - pi). This gives r the
        # first-order low-pass response r_dot = K_O*(tau_ext - r) in the
        # continuous limit, with time constant 1/K_O.
        self._pi = self._pi + dt * (obs.tau_cmd + self._r - obs.qfrc_bias)
        self._r = self._K_O * (p - self._pi)

        # Collect samples once the burn-in has elapsed.
        if (obs.t - self._t_start) >= self._burn_in_s:
            self._r_samples.append(self._r.copy())
            self._jz_samples.append(J_z)

    def estimate(self) -> EstimateResult:
        if self.r_empty is None:
            raise RuntimeError("MomentumObserver.estimate() called before calibration was loaded")
        if not self._r_samples:
            raise RuntimeError(
                "MomentumObserver.estimate() called with no post-burn-in samples; "
                "extend weigh_hold_s or shorten burn_in_s"
            )

        r_mean = np.mean(np.stack(self._r_samples, axis=0), axis=0)
        jz_mean = np.mean(np.stack(self._jz_samples, axis=0), axis=0)
        return self._project(r_mean, jz_mean)

    # ----- Calibration -----

    def start_calibration(self, ctx) -> None:  # noqa: ARG002
        self.reset()

    def update_calibration(self, obs: EstimatorObs) -> None:
        # Same recursion as the weigh hold. The post-burn-in residual stream
        # already lands in self._r_samples, so finish_calibration() reuses it
        # directly.
        self.update(obs)

    def finish_calibration(self) -> dict:
        if not self._r_samples:
            raise RuntimeError("MomentumObserver.finish_calibration() called with no samples")
        r_empty = np.mean(np.stack(self._r_samples, axis=0), axis=0)
        return {"r_empty": [float(x) for x in r_empty]}

    def load_calibration(self, data: dict) -> None:
        if "r_empty" not in data:
            raise KeyError("calibration data missing 'r_empty' (required by momentum_observer)")
        self.r_empty = np.asarray(data["r_empty"], dtype=float)

    # ----- Internal -----

    def _project(self, r_mean: np.ndarray, jz_mean: np.ndarray) -> EstimateResult:
        """Least-squares projection of (r - r_empty) onto -g * J_z."""
        delta_r = r_mean - self.r_empty
        denom = GRAVITY * float(np.dot(jz_mean, jz_mean))
        if denom < 1e-12:
            return EstimateResult(
                m_hat=0.0, sigma=float("inf"),
                diagnostics={"warning": "J_z is degenerate at weigh pose"},
            )
        m_hat = -float(np.dot(delta_r, jz_mean)) / denom

        # Sample stddev of m_hat by propagating per-tick residual scatter through
        # the projection. Cheap, useful for the diagnostics dump.
        r_arr = np.stack(self._r_samples, axis=0)
        per_sample = -((r_arr - self.r_empty) @ jz_mean) / denom
        sigma = float(per_sample.std(ddof=1)) if per_sample.size > 1 else None

        return EstimateResult(
            m_hat=m_hat,
            sigma=sigma,
            diagnostics={
                "K_O": self._K_O,
                "burn_in_s": self._burn_in_s,
                "n_samples": int(r_arr.shape[0]),
                "r_mean": [float(x) for x in r_mean],
                "r_empty": [float(x) for x in self.r_empty],
                "jz_mean": [float(x) for x in jz_mean],
            },
        )


register("momentum_observer", MomentumObserver)
