"""MomentumObserver math (no MuJoCo).

These tests drive the observer directly with synthetic EstimatorObs streams,
so they cover the discrete recursion, the burn-in gating, the calibration
contract, and the mass projection without any sim cost.
"""

import numpy as np
import pytest

from massaware.estimators.base import EstimatorObs
from massaware.estimators.momentum_observer import GRAVITY, MomentumObserver


def _make(K_O: float = 50.0, burn_in_s: float = 0.05) -> MomentumObserver:
    return MomentumObserver({
        "K_O": K_O,
        "burn_in_s": burn_in_s,
    })


def _obs(t: float, q_dot=None, tau_cmd=None, qfrc_bias=None, J_z=None, M=None) -> EstimatorObs:
    return EstimatorObs(
        t=t,
        q=np.zeros(6),
        q_dot=np.zeros(6) if q_dot is None else np.asarray(q_dot, dtype=float),
        tau_cmd=np.zeros(6) if tau_cmd is None else np.asarray(tau_cmd, dtype=float),
        tau_meas=np.zeros(6),
        qfrc_bias=np.zeros(6) if qfrc_bias is None else np.asarray(qfrc_bias, dtype=float),
        jacobian_ee=(
            np.zeros((3, 6)) if J_z is None
            else np.vstack([np.zeros((2, 6)), np.asarray(J_z, dtype=float)])
        ),
        q_ref=np.zeros(6),
        ee_xyz=np.zeros(3),
        M=np.eye(6) if M is None else np.asarray(M, dtype=float),
    )


# ---------- Convergence behaviour ----------

def test_residual_converges_to_constant_disturbance():
    """With q_dot = 0, p = 0 throughout. The observer sees tau - qfrc_bias = -D
    and must drive its internal r toward D (low-pass on the disturbance)."""
    obs = _make(K_O=50.0)
    D = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])  # disturbance on shoulder lift
    dt = 0.002
    # Run for ~6 time constants (6 / 50 = 0.12 s)
    n_steps = int(0.30 / dt)
    for k in range(n_steps):
        t = k * dt
        obs.update(_obs(t, tau_cmd=np.zeros(6), qfrc_bias=D))
    # r should have settled near D (well below 1% error).
    assert np.allclose(obs._r, D, atol=5e-2)


def test_residual_stays_near_zero_without_disturbance():
    obs = _make(K_O=50.0)
    dt = 0.002
    for k in range(500):  # 1 s
        obs.update(_obs(k * dt))
    assert np.max(np.abs(obs._r)) < 1e-6


def test_residual_tracks_disturbance_with_time_varying_mass_matrix():
    """Constant external disturbance while M(t) varies and q_dot != 0 — i.e.
    the arm is moving. The Mdot*q_dot correction must keep the residual locked
    on the disturbance instead of leaking the momentum change into it.

    Synthetic consistency: with q_dot constant, p = M(t) q_dot gives
    p_dot = Mdot q_dot exactly, so the exact dynamics
    p_dot = tau + tau_ext + Mdot q_dot - qfrc_bias require tau = -tau_ext
    (qfrc_bias = 0) for a disturbance tau_ext = D.
    """
    obs = _make(K_O=50.0)
    D = np.array([0.0, 2.0, 0.0, 1.0, 0.0, 0.0])
    q_dot = np.full(6, 0.3)
    dt = 0.002
    for k in range(int(0.50 / dt)):
        t = k * dt
        scale = 1.0 + 0.5 * np.sin(2.0 * np.pi * t)  # 1 Hz inertia swing
        obs.update(_obs(t, q_dot=q_dot, tau_cmd=-D, M=scale * np.eye(6)))
    # Without the correction the residual would carry the Mdot*q_dot term
    # (~0.9 N*m amplitude here); with it, r stays pinned to D.
    assert np.allclose(obs._r, D, atol=5e-2)


def test_burn_in_skips_initial_samples():
    obs = _make(K_O=50.0, burn_in_s=0.10)
    dt = 0.002
    for k in range(int(0.20 / dt)):
        obs.update(_obs(k * dt))
    # First 0.10 s rejected, ~0.10 s = 50 samples accepted (off-by-one tolerable).
    n = len(obs._r_samples)
    assert 45 <= n <= 55


# ---------- Calibration contract ----------

def test_load_calibration_rejects_missing_key():
    obs = _make()
    with pytest.raises(KeyError):
        obs.load_calibration({"not_r_empty": [0] * 6})


def test_load_calibration_stores_r_empty_as_array():
    obs = _make()
    obs.load_calibration({"r_empty": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    assert isinstance(obs.r_empty, np.ndarray)
    assert obs.r_empty.shape == (6,)


def test_estimate_without_calibration_raises():
    obs = _make()
    obs._r_samples.append(np.zeros(6))
    obs._jz_samples.append(np.array([0, 0.5, 0.4, 0, 0, 0]))
    with pytest.raises(RuntimeError):
        obs.estimate()


def test_estimate_without_samples_raises():
    obs = _make()
    obs.load_calibration({"r_empty": [0.0] * 6})
    with pytest.raises(RuntimeError):
        obs.estimate()


def test_calibration_roundtrip_produces_r_empty():
    obs = _make(K_O=50.0, burn_in_s=0.05)
    obs.start_calibration(ctx=None)
    dt = 0.002
    for k in range(int(0.30 / dt)):
        obs.update_calibration(_obs(k * dt))
    out = obs.finish_calibration()
    assert "r_empty" in out
    assert len(out["r_empty"]) == 6


# ---------- Mass projection ----------

def test_projection_recovers_known_mass():
    """Feed the observer a residual exactly equal to -m g J_z (post-calibration)
    and verify estimate() recovers m to numerical precision."""
    obs = _make()
    obs.load_calibration({"r_empty": [0.0] * 6})

    m_true = 0.4
    J_z = np.array([0.0, 0.49, 0.39, 0.0, 0.0, 0.0])  # plausible elbow-arm geometry
    r_target = -m_true * GRAVITY * J_z

    # Inject 100 identical samples; the projection averages and returns m_true.
    for _ in range(100):
        obs._r_samples.append(r_target.copy())
        obs._jz_samples.append(J_z.copy())

    result = obs.estimate()
    assert result.m_hat == pytest.approx(m_true, abs=1e-10)


def test_projection_handles_degenerate_jacobian():
    """If the EE z-Jacobian column is zero, the projection denominator is zero
    and the estimator must return 0 with sigma = inf rather than divide by zero."""
    obs = _make()
    obs.load_calibration({"r_empty": [0.0] * 6})
    for _ in range(10):
        obs._r_samples.append(np.ones(6))
        obs._jz_samples.append(np.zeros(6))
    result = obs.estimate()
    assert result.m_hat == 0.0
    assert result.sigma == float("inf")


def test_calibration_baseline_subtracts_cleanly():
    """If r_empty == r_loaded, the projection must return zero (no payload)."""
    obs = _make()
    bias = np.array([0.0, 0.7, 0.2, 0.0, 0.0, 0.0])
    obs.load_calibration({"r_empty": bias.tolist()})
    for _ in range(50):
        obs._r_samples.append(bias.copy())
        obs._jz_samples.append(np.array([0, 0.5, 0.4, 0, 0, 0]))
    result = obs.estimate()
    assert result.m_hat == pytest.approx(0.0, abs=1e-12)


def test_varying_jacobian_per_sample_fit_recovers_mass():
    """r_k = -m g J_z,k with J_z changing across samples (a moving pose):
    the sample-wise least squares must recover m exactly."""
    obs = _make()
    obs.load_calibration({"r_empty": [0.0] * 6})
    m_true = 0.4
    rng = np.random.default_rng(0)
    for _ in range(60):
        J = rng.uniform(-0.5, 0.5, size=6)
        obs._r_samples.append(-m_true * GRAVITY * J)
        obs._jz_samples.append(J)
    result = obs.estimate()
    assert result.m_hat == pytest.approx(m_true, abs=1e-10)


def test_per_sample_fit_gates_blind_poses():
    """Samples taken at poses with no vertical leverage carry no mass signal.
    The fit must skip them instead of letting their residual (e.g. unmodelled
    wrist bias) contaminate the estimate, which the old mean-then-project did."""
    obs = _make()
    obs.load_calibration({"r_empty": [0.0] * 6})
    m_true = 0.8
    J_good = np.array([0.0, 0.49, 0.39, 0.0, 0.0, 0.0])
    for _ in range(40):
        obs._r_samples.append(-m_true * GRAVITY * J_good)
        obs._jz_samples.append(J_good.copy())
    for _ in range(40):
        obs._r_samples.append(np.array([0.0, 3.0, -2.0, 1.0, 0.0, 0.0]))
        obs._jz_samples.append(np.zeros(6))
    result = obs.estimate()
    assert result.m_hat == pytest.approx(m_true, abs=1e-9)
    assert result.diagnostics["n_rejected"] == 40


# ---------- Hooks and metadata ----------

def test_gravity_comp_mask_is_all_ones():
    assert np.allclose(_make().gravity_comp_mask(6), 1.0)


def test_controller_overrides_is_empty():
    assert _make().controller_overrides() == {}


def test_reset_clears_state():
    obs = _make()
    dt = 0.002
    for k in range(int(0.30 / dt)):
        obs.update(_obs(k * dt, qfrc_bias=np.full(6, 1.0)))
    assert obs._r_samples  # something was collected
    assert np.any(obs._r)
    obs.reset()
    assert obs._r_samples == []
    assert np.all(obs._r == 0.0)
    assert obs._initialized is False
