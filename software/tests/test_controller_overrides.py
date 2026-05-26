"""PIDController.apply_overrides / clear_overrides semantics."""

import numpy as np
import pytest

from massaware.controller import PIDController


def _make() -> PIDController:
    kp = np.array([2000.0] * 6)
    kd = np.array([400.0] * 6)
    ki = np.zeros(6)
    return PIDController(kp=kp, ki=ki, kd=kd)


def test_apply_then_clear_restores_base_gains():
    c = _make()
    base_kp = c.kp.copy()
    base_kd = c.kd.copy()

    c.apply_overrides({"kp": np.full(6, 100.0), "kd": np.full(6, 30.0)})
    assert np.allclose(c.kp, 100.0)
    assert np.allclose(c.kd, 30.0)

    c.clear_overrides()
    assert np.allclose(c.kp, base_kp)
    assert np.allclose(c.kd, base_kd)


def test_apply_resets_integral_state():
    c = _make()
    # Pretend the integral wound up
    c._integral_err = np.array([1.0] * 6)

    c.apply_overrides({"kp": np.full(6, 100.0)})
    assert np.allclose(c._integral_err, 0.0)

    # clear_overrides also resets (so stale integral can't leak back into the
    # base controller after a measurement under softened gains).
    c._integral_err = np.array([2.0] * 6)
    c.clear_overrides()
    assert np.allclose(c._integral_err, 0.0)


def test_double_apply_without_clear_raises():
    c = _make()
    c.apply_overrides({"kp": np.full(6, 100.0)})
    with pytest.raises(RuntimeError):
        c.apply_overrides({"kp": np.full(6, 50.0)})


def test_clear_without_apply_is_noop():
    c = _make()
    base_kp = c.kp.copy()
    c.clear_overrides()  # must not raise
    assert np.allclose(c.kp, base_kp)


def test_empty_overrides_is_noop():
    c = _make()
    base_kp = c.kp.copy()
    c.apply_overrides({})
    assert np.allclose(c.kp, base_kp)
    c.clear_overrides()  # still safe


def test_unknown_key_raises():
    c = _make()
    with pytest.raises(KeyError):
        c.apply_overrides({"kx": np.zeros(6)})


def test_compute_uses_overridden_gains():
    c = _make()
    q = np.zeros(6)
    q_dot = np.zeros(6)
    q_ref = np.full(6, 0.01)  # 0.01 rad error
    qb = np.zeros(6)
    tau_base = c.compute(q, q_dot, q_ref, qb, dt=0.002, use_gravity_comp=False)
    expected_base = 2000.0 * 0.01
    assert np.allclose(tau_base, expected_base)

    c.apply_overrides({"kp": np.full(6, 200.0)})
    tau_soft = c.compute(q, q_dot, q_ref, qb, dt=0.002, use_gravity_comp=False)
    expected_soft = 200.0 * 0.01
    assert np.allclose(tau_soft, expected_soft)
