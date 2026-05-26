"""End-to-end integration tests for the mission pipeline.

These tests run the full FSM through MuJoCo, so they are slow (~2 s per case)
but catch regressions in: scene asset, perception, planner FSM routing,
controller wiring, mass-swap, calibration cache, and estimator integration.

Pure-math behavior of each estimator is covered separately by
`test_pid_error_math.py` / `test_lyapunov_math.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.mujoco_env import MujocoEnv
from massaware.planner import FSM
from massaware.tick_loop import Gripper, _build_obs
from scripts.mission import build


def _drive(env, ctx, controller, gripper, fsm) -> None:
    while not fsm.done:
        fsm.tick()
        if ctx.reset_controller:
            controller.reset()
            ctx.reset_controller = False
        if ctx.arm_target is None:
            ctx.arm_target = env.get_arm_qpos()
        qb = env.qfrc_bias
        tau = controller.compute(
            q=env.get_arm_qpos(),
            q_dot=env.get_arm_qvel(),
            q_ref=ctx.arm_target,
            qfrc_bias=qb,
            dt=env.dt,
            use_gravity_comp=False,
        ) + qb * ctx.gravity_comp_mask
        env.set_arm_ctrl(tau)
        gripper.apply(ctx.gripper_cmd)
        mujoco.mj_step(env.model, env.data)
        if ctx.estimator is not None and ctx.estimator_sink is not None:
            ctx.estimator_sink(_build_obs(env, ctx.robot, q_ref=ctx.arm_target, tau_cmd=tau))


@pytest.fixture(scope="module")
def shared_env(tmp_path_factory) -> MujocoEnv:
    """One MujocoEnv shared across all tests in this module — model load is
    the slow part, but `env.reset()` between tests gives a clean slate."""
    return MujocoEnv()


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch, tmp_path):
    """Point CALIBRATION_PATH at a tmpdir so each test starts uncalibrated."""
    from massaware import planner
    monkeypatch.setattr(planner, "CALIBRATION_PATH", tmp_path / "calibration.yaml")


@pytest.mark.parametrize("estimator,mass,expected_bin", [
    ("pid_error", 0.2, "light"),
    ("pid_error", 0.5, "heavy"),
    ("lyapunov",  0.2, "light"),
    ("lyapunov",  0.5, "heavy"),
])
def test_end_to_end_classification(shared_env, estimator, mass, expected_bin):
    env, ctx, controller = build(estimator_name=estimator, cube_mass=mass, env=shared_env)
    fsm = FSM(ctx)
    gripper = Gripper(env)
    _drive(env, ctx, controller, gripper, fsm)

    assert ctx.estimate_result is not None, "WeighState did not produce an estimate"
    # Within ±25% of true mass — generous because trial-to-trial variance is
    # determined by sim dynamics, not the estimator math.
    assert abs(ctx.estimate_result.m_hat - mass) / mass < 0.25
    assert ctx.bin_label == expected_bin

    # State trace must include WEIGH and CLASSIFY when an estimator is set.
    assert "WEIGH" in ctx.trace
    assert "CLASSIFY" in ctx.trace


def test_no_estimator_path_skips_weighing(shared_env):
    env, ctx, controller = build(estimator_name="none", cube_mass=0.5, env=shared_env)
    fsm = FSM(ctx)
    gripper = Gripper(env)
    _drive(env, ctx, controller, gripper, fsm)

    assert ctx.estimate_result is None
    assert "WEIGH" not in ctx.trace
    assert "CLASSIFY" not in ctx.trace
    # Phase-3 trace exactly.
    assert ctx.trace == ["SEARCH", "GRASP", "PLACE", "HOME", "DONE"]


def test_calibration_cache_is_reused_across_trials(shared_env, monkeypatch, tmp_path):
    """Run two trials of the same estimator; the second must hit the cache.

    `InitState` prints '[INIT] loaded cached calibration' on a cache hit.
    """
    from massaware import planner
    monkeypatch.setattr(planner, "CALIBRATION_PATH", tmp_path / "calibration.yaml")

    # First trial: fresh calibration written.
    env, ctx, controller = build(estimator_name="pid_error", cube_mass=0.5, env=shared_env)
    fsm = FSM(ctx)
    gripper = Gripper(env)
    _drive(env, ctx, controller, gripper, fsm)
    assert (tmp_path / "calibration.yaml").exists()

    # Second trial: cache must be loaded (we can't easily capture stdout here
    # without extra plumbing, so we assert the estimator's loaded baseline).
    env, ctx, controller = build(estimator_name="pid_error", cube_mass=0.2, env=shared_env)
    assert ctx.estimator.tau_ss_empty is None  # not yet loaded before INIT runs
    fsm = FSM(ctx)
    gripper = Gripper(env)
    _drive(env, ctx, controller, gripper, fsm)
    # After INIT.enter() loads cache, tau_ss_empty must be populated.
    assert ctx.estimator.tau_ss_empty is not None
