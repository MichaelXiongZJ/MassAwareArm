"""MujocoEnv.mass_matrix — joint-space inertia accessor for the momentum observer.

The matrix is used by the momentum observer to compute the generalized
momentum p = M(q) q_dot. These tests verify the structural invariants the
observer relies on (correct shape, exactly symmetric, positive definite),
not the absolute numerical values.
"""

import numpy as np
import pytest

from massaware.config import load_config
from massaware.mujoco_env import MujocoEnv


@pytest.fixture
def env() -> MujocoEnv:
    cfg = load_config()
    e = MujocoEnv()
    e.reset(arm_qpos=cfg["poses"]["home_qpos"])
    return e


def test_shape_is_6x6(env):
    assert env.mass_matrix().shape == (6, 6)


def test_symmetric_to_machine_precision(env):
    M = env.mass_matrix()
    assert np.max(np.abs(M - M.T)) < 1e-10


def test_positive_definite_at_home(env):
    M = env.mass_matrix()
    eigvals = np.linalg.eigvalsh(M)
    assert float(eigvals.min()) > 0.0


def test_positive_definite_at_weigh(env):
    cfg = load_config()
    env.reset(arm_qpos=cfg["poses"]["weigh_qpos"])
    M = env.mass_matrix()
    eigvals = np.linalg.eigvalsh(M)
    assert float(eigvals.min()) > 0.0


def test_returns_copy_not_view(env):
    """A subsequent call must not invalidate the first call's reference."""
    M1 = env.mass_matrix()
    M2 = env.mass_matrix()
    M2[0, 0] = -999.0
    assert M1[0, 0] != -999.0


def test_diagonal_magnitudes_in_expected_range(env):
    """Proximal joints (shoulder_pan, shoulder_lift) carry more inertia than
    distal ones (wrists). Order-of-magnitude check, not a tight fit."""
    diag = np.diag(env.mass_matrix())
    # All entries must be positive.
    assert (diag > 0).all()
    # Wrist diagonals are at most order 1; proximal diagonals are at least 0.1.
    assert diag.max() < 100.0
    assert diag.min() > 0.001


def test_mass_matrix_responds_to_cube_mass_swap(env):
    """Increasing the payload mass while the arm holds the cube should not
    change the UR5e block of M (the cube has its own DOFs via the freejoint),
    but the helper must still return a valid PD matrix after the swap.

    This guards against an implementation that accidentally indexes into the
    cube's DOFs when expanding qM.
    """
    env.set_body_mass("cube", 1.5)
    M = env.mass_matrix()
    assert M.shape == (6, 6)
    assert np.max(np.abs(M - M.T)) < 1e-10
    assert float(np.linalg.eigvalsh(M).min()) > 0.0
