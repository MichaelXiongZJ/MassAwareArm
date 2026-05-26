"""LyapunovEstimator math (no MuJoCo)."""

import numpy as np
import pytest

from massaware.estimators.base import EstimatorObs
from massaware.estimators.lyapunov import GRAVITY, LyapunovEstimator


def _make(kp=None, kd=None, kp_soft_scale=0.1, kd_soft_scale=0.5):
    if kp is None: kp = np.array([2000.0] * 6)
    if kd is None: kd = np.array([400.0] * 6)
    return LyapunovEstimator({
        "controller_kp": kp,
        "controller_kd": kd,
        "kp_soft_scale": kp_soft_scale,
        "kd_soft_scale": kd_soft_scale,
    })


def _obs(q: np.ndarray, ee_z: float, q_ref: np.ndarray) -> EstimatorObs:
    return EstimatorObs(
        t=0.0,
        q=q,
        q_dot=np.zeros(6),
        tau_cmd=np.zeros(6),
        tau_meas=np.zeros(6),
        qfrc_bias=np.zeros(6),
        jacobian_ee=np.zeros((3, 6)),
        q_ref=q_ref,
        ee_xyz=np.array([0.0, 0.0, ee_z]),
    )


def test_estimate_recovers_known_mass():
    """Single-joint synthetic case: shoulder_lift sags by `disp`, EE drops by Δh.

    Choose disp and Δh consistent with the PD-step factor-of-two rule:
        ΔE_spring = ½ kp_soft · disp²
        Δh        = ½ kp_soft · disp² / (m_true · g)
    """
    kp = np.array([2000.0] * 6)
    e = _make(kp=kp, kp_soft_scale=0.1)   # kp_soft = 200 per joint
    kp_soft = 200.0

    q_ref = np.zeros(6)
    q_empty = q_ref.copy()
    z_empty = 0.5

    m_true = 0.4
    j = 1  # shoulder_lift
    disp = 0.05  # 50 mrad sag at the measurement joint

    # PD step-response model: m·g·Δh = 2·ΔE_spring  =>  Δh = ΔE_spring/(m·g)·2 ... wait
    # Actually: m·g·Δh = 2·(½ kp_soft·disp²) = kp_soft·disp²
    delta_h = kp_soft * disp**2 / (m_true * GRAVITY)
    z_loaded = z_empty - delta_h
    q_loaded = q_ref.copy()
    q_loaded[j] = disp

    e.load_calibration({"q_empty": q_empty.tolist(), "z_empty": z_empty})

    for _ in range(500):
        e.update(_obs(q_loaded, z_loaded, q_ref))

    r = e.estimate()
    assert r.m_hat == pytest.approx(m_true, abs=1e-6)


def test_estimate_without_calibration_raises():
    e = _make()
    e.update(_obs(np.zeros(6), 0.5, np.zeros(6)))
    with pytest.raises(RuntimeError):
        e.estimate()


def test_estimate_no_sag_returns_zero():
    e = _make()
    q_ref = np.zeros(6)
    e.load_calibration({"q_empty": q_ref.tolist(), "z_empty": 0.5})
    # q_loaded == q_empty and z_loaded == z_empty -> no sag -> degenerate
    for _ in range(10):
        e.update(_obs(q_ref, 0.5, q_ref))
    r = e.estimate()
    assert r.m_hat == 0.0


def test_calibration_roundtrip():
    e = _make()
    q_ref = np.zeros(6)
    q_avg = np.full(6, 0.01)
    for _ in range(100):
        e.start_calibration(ctx=None) if False else None  # idempotent demo
    e.start_calibration(ctx=None)
    for _ in range(100):
        e.update_calibration(_obs(q_avg, 0.5, q_ref))
    data = e.finish_calibration()
    assert np.allclose(data["q_empty"], q_avg, atol=1e-12)
    assert data["z_empty"] == pytest.approx(0.5)


def test_load_calibration_rejects_missing_keys():
    e = _make()
    with pytest.raises(KeyError):
        e.load_calibration({"q_empty": [0.0] * 6})  # missing z_empty
    with pytest.raises(KeyError):
        e.load_calibration({"z_empty": 0.5})       # missing q_empty


def test_controller_overrides_zero_ki_soft_kp_kd():
    e = _make(kp_soft_scale=0.1, kd_soft_scale=0.5)
    ov = e.controller_overrides()
    assert np.allclose(ov["ki"], 0.0)
    assert np.allclose(ov["kp"], 200.0)
    assert np.allclose(ov["kd"], 200.0)


def test_full_gravity_mask():
    e = _make()
    assert np.allclose(e.gravity_comp_mask(6), 1.0)
