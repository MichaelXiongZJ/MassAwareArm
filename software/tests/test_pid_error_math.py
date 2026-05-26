"""PIDErrorEstimator math (no MuJoCo)."""

import numpy as np
import pytest

from massaware.estimators.base import EstimatorObs
from massaware.estimators.pid_error import GRAVITY, PIDErrorEstimator


def _make(measurement_joint=2, moment_arm=0.5):
    return PIDErrorEstimator({
        "measurement_joint": measurement_joint,
        "moment_arm": moment_arm,
        "controller_kp": np.array([2000.0] * 6),
        "controller_kd": np.array([400.0] * 6),
    })


def _obs(tau_at_meas: float, j: int = 2) -> EstimatorObs:
    tau = np.zeros(6)
    tau[j] = tau_at_meas
    return EstimatorObs(
        t=0.0,
        q=np.zeros(6),
        q_dot=np.zeros(6),
        tau_cmd=tau,
        tau_meas=np.zeros(6),
        qfrc_bias=np.zeros(6),
        jacobian_ee=np.zeros((3, 6)),
        q_ref=np.zeros(6),
        ee_xyz=np.zeros(3),
    )


def test_estimate_recovers_known_mass():
    e = _make()
    tau_empty = 20.0
    moment_arm = 0.5
    m_true = 0.4
    tau_loaded = tau_empty + m_true * GRAVITY * moment_arm

    e.load_calibration({"tau_ss_empty": [0, 0, tau_empty, 0, 0, 0]})
    for _ in range(500):
        e.update(_obs(tau_loaded))

    result = e.estimate()
    assert result.m_hat == pytest.approx(m_true, abs=1e-6)


def test_estimate_without_calibration_raises():
    e = _make()
    e.update(_obs(1.0))
    with pytest.raises(RuntimeError):
        e.estimate()


def test_estimate_without_samples_raises():
    e = _make()
    e.load_calibration({"tau_ss_empty": [0, 0, 20.0, 0, 0, 0]})
    with pytest.raises(RuntimeError):
        e.estimate()


def test_reset_clears_samples():
    e = _make()
    e.load_calibration({"tau_ss_empty": [0, 0, 20.0, 0, 0, 0]})
    for _ in range(10):
        e.update(_obs(22.0))
    e.reset()
    # Should require fresh samples after reset.
    with pytest.raises(RuntimeError):
        e.estimate()


def test_calibration_roundtrip():
    e = _make()
    e.start_calibration(ctx=None)
    for _ in range(100):
        e.update_calibration(_obs(20.0))
    out = e.finish_calibration()
    assert out["tau_ss_empty"][2] == pytest.approx(20.0)
    # All non-measurement entries are zero by design.
    for i, v in enumerate(out["tau_ss_empty"]):
        if i != 2:
            assert v == 0.0


def test_load_calibration_rejects_missing_key():
    e = _make()
    with pytest.raises(KeyError):
        e.load_calibration({"nope": [0] * 6})


def test_gravity_comp_mask_zeros_measurement_joint():
    e = _make(measurement_joint=2)
    mask = e.gravity_comp_mask(6)
    assert mask[2] == 0.0
    other = [v for i, v in enumerate(mask) if i != 2]
    assert all(v == 1.0 for v in other)


def test_controller_overrides_softens_only_measurement_joint():
    e = _make(measurement_joint=2)
    ov = e.controller_overrides()
    assert "kp" in ov and "kd" in ov
    # Measurement joint softened
    assert ov["kp"][2] == 200.0
    assert ov["kd"][2] == 80.0
    # Other joints unchanged from base
    for i in (0, 1, 3, 4, 5):
        assert ov["kp"][i] == 2000.0
        assert ov["kd"][i] == 400.0


def test_sigma_reported_for_noisy_samples():
    e = _make()
    e.load_calibration({"tau_ss_empty": [0, 0, 20.0, 0, 0, 0]})
    rng = np.random.default_rng(0)
    moment_arm = 0.5
    for _ in range(2000):
        e.update(_obs(20.0 + 0.5 * GRAVITY * moment_arm + 0.3 * rng.standard_normal()))
    r = e.estimate()
    assert r.sigma is not None and r.sigma > 0
    assert r.m_hat == pytest.approx(0.5, abs=0.05)
