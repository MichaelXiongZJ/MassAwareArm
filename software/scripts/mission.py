"""Single-mission runner.

Runs one full pipeline pass against the current scene:

    INIT -> SEARCH -> GRASP -> WEIGH -> CLASSIFY -> PLACE -> HOME -> DONE

When the active estimator is `null` (CLI: `--estimator none`), the pipeline
reverts to the Phase 3 SEARCH -> GRASP -> PLACE -> HOME flow and always drops
the cube in the light bin.

For batch sweeps over (estimator, mass) pairs, use `verify_estimators.py`
instead — that script reuses the same `build()` helper but drives many runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.config import load_config
from massaware.controller import PIDController
from massaware.estimators import Estimator, build as build_estimator
# Importing the concrete estimator modules triggers their self-registration.
from massaware.estimators import pid_error as _pid_error  # noqa: F401
from massaware.estimators import lyapunov as _lyapunov    # noqa: F401
from massaware.mujoco_env import MujocoEnv
from massaware.perception import GroundTruthPerception
from massaware.planner import FSM, PlannerContext
from massaware.robot import Robot
from massaware.tick_loop import Gripper, TickLoop

# Default cube body and mass when no override is given on the CLI.
DEFAULT_CUBE_BODY = "cube"


def build_controller(cfg: dict) -> PIDController:
    """Initialize the PID controller from `cfg['controller']`."""
    c = cfg["controller"]
    return PIDController(kp=c["kp"], ki=c["ki"], kd=c["kd"])


def build_estimator_from_cfg(cfg: dict, name_override: str | None = None) -> Estimator | None:
    """Build the active estimator.

    `name_override` (when not None) takes precedence over `estimator.name` in
    the YAML. Pass the literal string ``"none"`` (case-insensitive) or an
    empty string to force the no-estimator path even when YAML names one.
    """
    est_cfg = cfg.get("estimator", {}) or {}
    name = name_override if name_override is not None else est_cfg.get("name")
    if not name or str(name).lower() == "none":
        return None
    runtime_cfg = dict(est_cfg.get(name, {}) or {})
    runtime_cfg["controller_kp"] = np.asarray(cfg["controller"]["kp"], dtype=float)
    runtime_cfg["controller_kd"] = np.asarray(cfg["controller"]["kd"], dtype=float)
    return build_estimator(name, runtime_cfg)


def build(
    estimator_name: str | None = None,
    cube_mass: float | None = None,
    env: MujocoEnv | None = None,
) -> tuple[MujocoEnv, PlannerContext, PIDController]:
    """Initialize environment, controller, estimator, and planner context.

    Args:
        estimator_name: override `estimator.name` from YAML for this build.
        cube_mass: if not None, set the cube's body mass before resetting.
        env: reuse an existing `MujocoEnv` (e.g. across trials) instead of
            instantiating a fresh one. The env is reset to home and the cube
            is re-massed if `cube_mass` is supplied.

    Returns: `(env, ctx, controller)`.
    """
    cfg = load_config()

    if env is None:
        env = MujocoEnv()
    if cube_mass is not None:
        env.set_body_mass(DEFAULT_CUBE_BODY, float(cube_mass))
    env.reset(arm_qpos=cfg["poses"]["home_qpos"])

    robot = Robot(env)
    controller = build_controller(cfg)
    estimator = build_estimator_from_cfg(cfg, name_override=estimator_name)

    weigh_cfg = cfg.get("weigh", {}) or {}
    ctx = PlannerContext(
        env=env,
        robot=robot,
        perception=GroundTruthPerception(env),
        controller=controller,
        target_color="grey",
        home_qpos=cfg["poses"]["home_qpos"],
        weigh_qpos=cfg["poses"]["weigh_qpos"],
        light_bin_drop=cfg["poses"]["light_bin_drop"],
        heavy_bin_drop=cfg["poses"]["heavy_bin_drop"],
        estimator=estimator,
        mass_threshold=float(cfg["classifier"]["mass_threshold"]),
        weigh_hold_s=float(weigh_cfg.get("hold_seconds", 1.0)),
    )
    return env, ctx, controller


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--viewer", action="store_true",
                    help="open MuJoCo passive viewer for this run")
    ap.add_argument("--mass", type=float, default=None,
                    help="override cube mass (kg) for this run; "
                         "if omitted, the scene XML default is used")
    ap.add_argument("--estimator", default=None,
                    help="override estimator.name from YAML for this run; "
                         "use 'none' to force the no-estimator pipeline")
    args = ap.parse_args()

    env, ctx, controller = build(estimator_name=args.estimator, cube_mass=args.mass)
    cube_bid = env.model.body(DEFAULT_CUBE_BODY).id
    initial_pos = env.data.xpos[cube_bid].copy()
    cube_mass = float(env.model.body_mass[cube_bid])

    gripper = Gripper(env)
    fsm = FSM(ctx)

    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat = np.array([0.25, 0.0, 0.4])
            viewer.cam.distance = 2
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -20
            loop = TickLoop(env, fsm, gripper, controller, ctx.robot, viewer=viewer)
            loop.run()
            _print_summary(env, cube_bid, initial_pos, cube_mass, ctx)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    else:
        loop = TickLoop(env, fsm, gripper, controller, ctx.robot)
        loop.run()
        _print_summary(env, cube_bid, initial_pos, cube_mass, ctx)
    return 0


def _print_summary(
    env: MujocoEnv,
    cube_bid: int,
    initial_pos: np.ndarray,
    cube_mass: float,
    ctx: PlannerContext,
) -> None:
    final_pos = env.data.xpos[cube_bid].copy()
    print("\n=== Mission summary ===")
    print(f"cube mass       : {cube_mass:.3f} kg")
    print(f"initial position: {np.round(initial_pos, 3)}")
    print(f"final position  : {np.round(final_pos, 3)}")
    print(f"state trace     : {' -> '.join(ctx.trace)}")
    if ctx.estimate_result is not None:
        r = ctx.estimate_result
        sigma_s = f"{r.sigma:.4f}" if r.sigma is not None else "n/a"
        err_pct = 100.0 * (r.m_hat - cube_mass) / cube_mass if cube_mass else float("nan")
        print(f"estimator       : {ctx.estimator.name}")
        print(f"m_hat           : {r.m_hat:.4f} kg  (sigma={sigma_s}, err={err_pct:+.1f}%)")
        print(f"classified bin  : {ctx.bin_label}")


if __name__ == "__main__":
    raise SystemExit(main())
