"""MujocoEnv.set_body_mass — runtime mass mutation for the payload cube.

These tests require MuJoCo (model load + step), so they are slower than the
pure-math tests but still finish in well under a second per case.
"""

import mujoco
import numpy as np
import pytest

from massaware.mujoco_env import MujocoEnv


CUBE_BODY = "cube"


@pytest.fixture
def env() -> MujocoEnv:
    return MujocoEnv()


def test_set_body_mass_updates_model_array(env):
    bid = env.model.body(CUBE_BODY).id
    old = env.set_body_mass(CUBE_BODY, 0.7)
    assert old == pytest.approx(0.5)                     # default in scene.xml
    assert float(env.model.body_mass[bid]) == pytest.approx(0.7)


def test_set_body_mass_scales_inertia_linearly(env):
    bid = env.model.body(CUBE_BODY).id
    inertia_before = env.model.body_inertia[bid].copy()
    env.set_body_mass(CUBE_BODY, 1.5)                    # 3x the default
    inertia_after = env.model.body_inertia[bid].copy()
    assert np.allclose(inertia_after, inertia_before * 3.0)


def test_set_body_mass_changes_gravity_bias(env):
    """After a mass swap, the cube's contribution to qfrc_bias must update.

    With the cube free-floating in the air, MuJoCo's qfrc_bias at the cube's
    free-joint DOFs should equal m*g along the z axis.
    """
    bid = env.model.body(CUBE_BODY).id
    # Move the cube above the table so gravity is the only force.
    env.reset()
    qpos_adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
    env.data.qpos[qpos_adr + 2] = 1.5                    # z high enough to be airborne
    mujoco.mj_forward(env.model, env.data)

    dof_adr = env.model.jnt_dofadr[env.model.body_jntadr[bid]]
    # qfrc_bias on the free joint's translational z DOF == m*g (sign convention)
    bias_before = float(env.data.qfrc_bias[dof_adr + 2])
    m_before = float(env.model.body_mass[bid])
    assert bias_before == pytest.approx(m_before * 9.81, rel=1e-3)

    env.set_body_mass(CUBE_BODY, 1.0)
    mujoco.mj_forward(env.model, env.data)
    bias_after = float(env.data.qfrc_bias[dof_adr + 2])
    assert bias_after == pytest.approx(1.0 * 9.81, rel=1e-3)


def test_multiple_swaps_are_idempotent_to_final_value(env):
    bid = env.model.body(CUBE_BODY).id
    inertia_orig = env.model.body_inertia[bid].copy()

    # Round-trip: 0.5 -> 0.2 -> 0.5
    env.set_body_mass(CUBE_BODY, 0.2)
    env.set_body_mass(CUBE_BODY, 0.5)

    assert float(env.model.body_mass[bid]) == pytest.approx(0.5)
    # And inertia should be back to the original (within FP noise).
    assert np.allclose(env.model.body_inertia[bid], inertia_orig, rtol=1e-12)


def test_swap_then_reset_preserves_swap(env):
    """The mass change must survive `env.reset()` — it lives on the model,
    not on the data."""
    env.set_body_mass(CUBE_BODY, 0.3)
    env.reset()
    bid = env.model.body(CUBE_BODY).id
    assert float(env.model.body_mass[bid]) == pytest.approx(0.3)


def test_swap_rejects_zero_or_negative(env):
    with pytest.raises(ValueError):
        env.set_body_mass(CUBE_BODY, 0.0)


def test_swap_on_unknown_body_raises(env):
    with pytest.raises(KeyError):
        env.set_body_mass("nonexistent_body", 0.5)
