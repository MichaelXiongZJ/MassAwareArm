"""Estimator weight-estimation sweep.

Runs every (estimator, mass) pair through the full FSM and prints a table
comparing `m_hat` to ground truth. The focus is the quality of the mass
estimate across a wide range of payloads. Bin classification is reported
for reference but does not gate success.

The scene contains a single grey cube; its mass is mutated between trials
via `MujocoEnv.set_body_mass()` so every trial sees identical kinematics
and the only thing that changes is the payload mass.

Usage:
    python software/scripts/verify_estimators.py
    python software/scripts/verify_estimators.py --viewer
    python software/scripts/verify_estimators.py --clear-cache
    python software/scripts/verify_estimators.py --estimator lyapunov \
        --masses 0.05,0.1,0.2,0.35,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.planner import FSM
from massaware.tick_loop import Gripper, _build_obs
from scripts.mission import build

# Default mass sweep: geometric spacing from 10 g to 2.5 kg, 25 points.
# Multiplicative steps give automatic density at the light end (where the
# sag-based method's noise floor lives) without losing coverage of the heavy
# edge. The ratio between successive trials is about 1.26x.
M_MIN, M_MAX, N_MASSES = 0.01, 3, 20
DEFAULT_MASSES: list[float] = [
    round(float(m), 4) for m in np.geomspace(M_MIN, M_MAX, num=N_MASSES) # log
    # round(float(m), 4) for m in np.linspace(M_MIN, M_MAX, num=N_MASSES) # linear
]
DEFAULT_ESTIMATORS: list[str] = ["pid_error", "lyapunov", "momentum_observer"]

# Loose tolerance on |err%|. Trials outside this are reported but the script
# does not fail the run on them; the focus is on the per-trial numbers, not
# on a pass/fail gate.
ERROR_PCT_WARN = 25.0


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

    r = ctx.estimate_result
    return {
        "estimator": estimator_name,
        "mass": mass,
        "m_hat": float(r.m_hat) if r is not None else float("nan"),
        "sigma": float(r.sigma) if (r is not None and r.sigma is not None) else float("nan"),
        "bin_label": ctx.bin_label,
    }


def _parse_masses(spec: str) -> list[float]:
    """Parse `--masses 0.1,0.2,0.5` into [0.1, 0.2, 0.5]."""
    return [float(tok.strip()) for tok in spec.split(",") if tok.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimator", default=None,
                    help="run a single estimator instead of all known")
    ap.add_argument("--masses", default=None,
                    help="comma-separated list of cube masses (kg); "
                         "default is 25 geometrically-spaced masses from 10g to 2.5kg")
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
    t_start = time.time()

    def _sweep(viewer=None):
        for est_name in estimators:
            for mass in masses:
                print(f"\n>>> {est_name} / mass={mass} kg")
                r = _run_trial(env, est_name, mass, viewer=viewer)
                r["err_pct"] = 100.0 * (r["m_hat"] - mass) / mass if mass else float("nan")
                rows.append(r)

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

    # Per-trial table.
    print("\n" + "=" * 74)
    print(f"{'estimator':<18} {'true (kg)':>10} {'m_hat (kg)':>11} "
          f"{'err':>7} {'sigma':>8} {'note':>10}")
    print("-" * 74)
    for r in rows:
        note = "" if abs(r["err_pct"]) <= ERROR_PCT_WARN else "high err"
        sigma_s = f"{r['sigma']:8.4f}" if r["sigma"] == r["sigma"] else "     n/a"
        print(f"{r['estimator']:<18} {r['mass']:>10.3f} {r['m_hat']:>11.4f} "
              f"{r['err_pct']:>+6.1f}% {sigma_s} {note:>10}")
    print("=" * 74)

    # Per-estimator summary: mean signed err and RMSE across the swept masses.
    print(f"\n{'estimator':<18} {'mean err':>10} {'mean |err|':>12} {'RMSE %':>10}")
    print("-" * 52)
    for est_name in estimators:
        errs = [r["err_pct"] for r in rows if r["estimator"] == est_name]
        if not errs:
            continue
        mean_err = sum(errs) / len(errs)
        mean_abs = sum(abs(e) for e in errs) / len(errs)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
        print(f"{est_name:<18} {mean_err:>+9.1f}% {mean_abs:>11.1f}% {rmse:>9.1f}%")

    print(f"\nsweep took {elapsed:.1f} s over {len(rows)} trials")

    # ── CSV output ────────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    csv_path = results_dir / f"sweep_{stamp}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["estimator", "true_mass", "m_hat", "sigma", "err_pct"])
        for r in rows:
            sigma = r["sigma"] if r["sigma"] == r["sigma"] else ""
            writer.writerow([r["estimator"], r["mass"], r["m_hat"], sigma, f"{r['err_pct']:.4f}"])
    print(f"wrote {csv_path.relative_to(results_dir.parent)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
