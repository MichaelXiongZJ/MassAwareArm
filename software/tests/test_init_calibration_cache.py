"""InitState calibration-cache load/save helpers (file-level, no MuJoCo)."""

import numpy as np
import pytest
import yaml

from massaware import planner


@pytest.fixture
def tmp_cal(tmp_path, monkeypatch):
    """Point CALIBRATION_PATH at a temp file for each test."""
    cal_path = tmp_path / "calibration.yaml"
    monkeypatch.setattr(planner, "CALIBRATION_PATH", cal_path)
    return cal_path


def test_missing_file_returns_none(tmp_cal):
    weigh_qpos = np.array([0, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    assert planner._load_cached_calibration(weigh_qpos) is None


def test_matching_pose_loads(tmp_cal):
    weigh_qpos = np.array([0, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    tmp_cal.write_text(yaml.safe_dump({
        "weigh_qpos_used": weigh_qpos.tolist(),
        "tau_ss_empty": [0, 0, 20.0, 0, 0, 0],
    }))
    cached = planner._load_cached_calibration(weigh_qpos)
    assert cached is not None
    assert cached["tau_ss_empty"][2] == 20.0


def test_mismatched_pose_rejected(tmp_cal):
    weigh_qpos = np.array([0, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    bad_pose = weigh_qpos.copy()
    bad_pose[0] = 0.5  # diff >> tol
    tmp_cal.write_text(yaml.safe_dump({
        "weigh_qpos_used": bad_pose.tolist(),
        "tau_ss_empty": [0] * 6,
    }))
    assert planner._load_cached_calibration(weigh_qpos) is None


def test_within_tolerance_accepted(tmp_cal):
    weigh_qpos = np.array([0, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    near = weigh_qpos.copy()
    near[3] += planner.CALIBRATION_QPOS_TOL / 2  # half the tol
    tmp_cal.write_text(yaml.safe_dump({
        "weigh_qpos_used": near.tolist(),
        "tau_ss_empty": [0, 0, 7.5, 0, 0, 0],
    }))
    assert planner._load_cached_calibration(weigh_qpos) is not None


def test_save_merges_with_existing_keys(tmp_cal):
    weigh_qpos = np.array([0, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    # First estimator writes tau_ss_empty.
    planner._save_calibration(weigh_qpos, {"tau_ss_empty": [0, 0, 22.0, 0, 0, 0]})
    # Second estimator writes q_empty/z_empty for the SAME pose -> keys merged.
    planner._save_calibration(weigh_qpos, {
        "q_empty": weigh_qpos.tolist(),
        "z_empty": 0.582,
    })
    raw = yaml.safe_load(tmp_cal.read_text())
    assert "tau_ss_empty" in raw
    assert "q_empty" in raw
    assert "z_empty" in raw
    assert raw["weigh_qpos_used"] == weigh_qpos.tolist()


def test_save_overwrites_when_pose_changes(tmp_cal):
    weigh_qpos_a = np.array([0, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    weigh_qpos_b = np.array([0.5, -1.5708, -1.5708, -1.5708, 1.5708, 0])
    planner._save_calibration(weigh_qpos_a, {"tau_ss_empty": [0, 0, 22.0, 0, 0, 0]})
    planner._save_calibration(weigh_qpos_b, {"q_empty": weigh_qpos_b.tolist(), "z_empty": 0.6})
    raw = yaml.safe_load(tmp_cal.read_text())
    # Stale key from pose A must be evicted because pose changed.
    assert "tau_ss_empty" not in raw
    assert "q_empty" in raw
    assert raw["weigh_qpos_used"] == weigh_qpos_b.tolist()
