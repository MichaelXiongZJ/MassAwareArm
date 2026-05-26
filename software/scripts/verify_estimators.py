"""Sub-phase E verification driver.

Sweeps every (estimator, mass) pair through the full FSM and prints a table
comparing `m_hat` to ground truth. Asserts each trial classifies the cube
into the correct bin. Exits non-zero on any failure.

The scene contains a single grey cube; its mass is mutated between trials
via `MujocoEnv.set_body_mass()` so every trial sees identical kinematics —
the only thing that changes is the payload mass.

Usage:
    python software/scripts/verify_estimators.py
    python software/scripts/verify_estimators.py --viewer       # one window, all runs
    python software/scripts/verify_estimators.py --clear-cache  # force fresh calibration
    python software/scripts/verify_estimators.py --estimator lyapunov --masses 0.1,0.2,0.3,0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.planner import FSM
from massaware.tick_loop import Gripper, _build_obs
from scripts.mission import build

# (true mass kg, expected bin given default threshold 0.35)
DEFAULT_MASSES: list[tuple[float, str]] = [
    (0.2, "light"),
    (0.5, "heavy"),
]
DEFAULT_ESTIMATORS: list[str] = ["pid_error", "lyapunov"]


def _drive(env, ctx, controller, gripper, fsm, viewer=None) -> None:
    """Tick the FSM + controller until done. Mirrors TickLoop's body."""
    while not fsm.done:
        if viewer is not None and not viewer.is_running():
            break
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
        t0 = time.perf_counter() if viewer is not None else 0.0
        mujoco.mj_step(env.model, env.data)
        if ctx.estimator is not None and ctx.estimator_sink is not None:
            ctx.estimator_sink(_build_obs(env, ctx.robot, q_ref=ctx.arm_target, tau_cmd=tau))
        if viewer is not None:
            viewer.sync()
            remaining = env.dt - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)


def _run_trial(env, estimator_name: str, mass: float, viewer=None) -> dict:
    """Run one trial: reset env, mass-swap cube, build estimator + FSM, drive."""
    env, ctx, controller = build(
        estimator_name=estimator_name,
        cube_mass=mass,
        env=env,
    )
    fsm = FSM(ctx)
    gripper = Gripper(env)

    _drive(env, ctx, controller, gripper, fsm, viewer=viewer)

    return {
        "estimator": estimator_name,
        "mass": mass,
        "m_hat": float(ctx.estimate_result.m_hat) if ctx.estimate_result else float("nan"),
        "bin_label": ctx.bin_label,
    }


def _parse_masses(spec: str) -> list[tuple[float, str]]:
    """Parse `--masses 0.1,0.2,0.5` into [(0.1, ?), ...] (expected bin from default threshold)."""
    from massaware.config import load_config
    threshold = float(load_config()["classifier"]["mass_threshold"])
    rows: list[tuple[float, str]] = []
    for tok in spec.split(","):
        m = float(tok.strip())
        rows.append((m, "heavy" if m >= threshold else "light"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimator", default=None,
                    help="run a single estimator instead of all known")
    ap.add_argument("--masses", default=None,
                    help="comma-separated list of cube masses (kg); "
                         "default is 0.2,0.5")
    ap.add_argument("--clear-cache", action="store_true",
                    help="delete configs/calibration.yaml before starting")
    ap.add_argument("--viewer", action="store_true",
                    help="open one MuJoCo viewer for the whole sweep")
    args = ap.parse_args()

    cal_path = Path(__file__).resolve().parents[1] / "configs" / "calibration.yaml"
    if args.clear_cache and cal_path.exists():
        cal_path.unlink()

    estimators = [args.estimator] if args.estimator else DEFAULT_ESTIMATORS
    masses = _parse_masses(args.masses) if args.masses else DEFAULT_MASSES

    # Build a single MujocoEnv up-front; reuse it across every trial so the
    # viewer (if any) sees one continuous session and we don't reload the
    # model XML on each iteration.
    from massaware.mujoco_env import MujocoEnv
    env = MujocoEnv()

    rows: list[dict] = []
    failures: list[str] = []
    t_start = time.time()

    def _sweep(viewer=None):
        for est_name in estimators:
            for mass, expected_bin in masses:
                print(f"\n>>> {est_name} / mass={mass} kg (expected={expected_bin})")
                r = _run_trial(env, est_name, mass, viewer=viewer)
                r["expected_bin"] = expected_bin
                r["err_pct"] = 100.0 * (r["m_hat"] - mass) / mass if mass else float("nan")
                rows.append(r)
                if r["bin_label"] != expected_bin:
                    failures.append(
                        f"{est_name}/mass={mass}: classified {r['bin_label']!r}, expected {expected_bin!r}"
                    )

    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat = [0.25, 0.0, 0.4]
            v.cam.distance = 2
            v.cam.azimuth = 180
            v.cam.elevation = -20
            _sweep(viewer=v)
            print("\n(sweep complete; close the viewer window to exit)")
            while v.is_running():
                v.sync()
                time.sleep(0.02)
    else:
        _sweep()

    elapsed = time.time() - t_start

    print("\n" + "=" * 72)
    print(f"{'estimator':<12} {'mass':>6} {'m_hat':>8} {'err':>7} {'bin':>6} {'expected':>8} {'OK':>5}")
    print("-" * 72)
    for r in rows:
        ok = "PASS" if r["bin_label"] == r["expected_bin"] else "FAIL"
        print(f"{r['estimator']:<12} {r['mass']:>6.2f} {r['m_hat']:>8.4f} "
              f"{r['err_pct']:>+6.1f}% {r['bin_label']:>6} {r['expected_bin']:>8} {ok:>5}")
    print("=" * 72)
    print(f"sweep took {elapsed:.1f} s")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll trials classified correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
