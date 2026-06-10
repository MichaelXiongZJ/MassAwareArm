"""Compare tracking controllers across mass estimators."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

from massaware.config import load_config
from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot
from massaware.tick_loop import Gripper

from mission_tracking import (
    CUBE_BODY,
    make_controller,
    make_estimator,
    make_pick_weigh_plan,
    release_time,
    run_calibration,
    run_tracking_mission,
)
from massaware.controllers.profiles import tracking_profile

CONTROLLERS = ["pid_tracking", "inverse_dynamics"]
ESTIMATORS = ["pid_error", "lyapunov", "momentum_observer", "inverse_dynamics"]
MASSES = [0.1, 0.5, 1.0, 2.0, 3.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--estimators", default=",".join(ESTIMATORS))
    parser.add_argument("--masses", default=",".join(str(m) for m in MASSES))
    parser.add_argument("--profile", choices=["tracking", "main"], default="tracking")
    parser.add_argument("--move-to-grasp-time", type=float, default=3.0)
    parser.add_argument("--lift-to-weigh-time", type=float, default=2.0)
    parser.add_argument("--move-to-release-time", type=float, default=None)
    parser.add_argument("--weigh-time", type=float, default=None)
    parser.add_argument("--calibration-time", type=float, default=3.0)
    parser.add_argument(
        "--disable-controller-overrides",
        action="store_true",
        help="ignore estimator-requested gain overrides during weighing/calibration",
    )
    parser.add_argument(
        "--collect-lift-samples",
        action="store_true",
        help="feed estimators during the lift-to-weigh motion (pause-free weighing)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    profile = tracking_profile(args.profile, cfg)
    rows: list[dict] = []
    t0 = time.time()

    for controller_name in parse_csv(args.controllers):
        for estimator_name in parse_csv(args.estimators):
            for mass in parse_float_csv(args.masses):
                print(
                    f"\n>>> controller={controller_name} estimator={estimator_name} "
                    f"mass={mass:.3f}kg"
                )
                rows.append(
                    run_trial(
                        cfg,
                        controller_name,
                        estimator_name,
                        mass,
                        profile_name=args.profile,
                        move_to_grasp_time=args.move_to_grasp_time,
                        lift_to_weigh_time=args.lift_to_weigh_time,
                        move_to_release_time=args.move_to_release_time,
                        weigh_time=args.weigh_time,
                        calibration_time=args.calibration_time,
                        allow_controller_overrides=not args.disable_controller_overrides,
                        collect_lift_samples=True if args.collect_lift_samples else None,
                    )
                )

    print_results(rows)
    print(
        f"\ncomparison took {time.time() - t0:.1f}s over {len(rows)} trials "
        f"(profile={profile.name})"
    )
    return 0


def run_trial(
    cfg: dict,
    controller_name: str,
    estimator_name: str,
    mass: float,
    *,
    profile_name: str,
    move_to_grasp_time: float,
    lift_to_weigh_time: float,
    move_to_release_time: float | None,
    weigh_time: float | None,
    calibration_time: float,
    allow_controller_overrides: bool,
    collect_lift_samples: bool | None = None,
) -> dict:
    try:
        env = MujocoEnv()
        env.set_body_mass(CUBE_BODY, mass)
        env.reset(arm_qpos=cfg["poses"]["home_qpos"])
        robot = Robot(env)
        gripper = Gripper(env)
        profile = tracking_profile(profile_name, cfg)
        controller = make_controller(
            controller_name,
            env,
            robot,
            cfg,
            profile,
            allow_overrides=allow_controller_overrides,
        )
        estimator = make_estimator(estimator_name, cfg, profile, controller_name)

        if estimator.requires_calibration:
            run_calibration(
                env,
                robot,
                controller,
                estimator,
                cfg["poses"]["weigh_qpos"],
                hold_time=calibration_time,
            )
            controller.clear_overrides()
            env.reset(arm_qpos=cfg["poses"]["home_qpos"])

        cube_xyz = env.data.xpos[env.model.body(CUBE_BODY).id].copy()
        trajectory = make_pick_weigh_plan(
            env,
            robot,
            cfg,
            profile,
            cube_xyz,
            move_to_grasp_time=move_to_grasp_time,
            lift_to_weigh_time=lift_to_weigh_time,
            weigh_time=weigh_time,
            collect_lift_samples=collect_lift_samples,
        )
        result = run_tracking_mission(
            env,
            robot,
            gripper,
            controller,
            estimator,
            cfg,
            trajectory,
            payload_mass=mass,
            move_to_release_time=release_time(move_to_release_time, profile),
            profile=profile,
        )
        m_hat = float(result["m_hat"])
        err_pct = 100.0 * (m_hat - mass) / mass if mass > 0 else math.nan
        return {
            "controller": controller_name,
            "estimator": estimator_name,
            "mass": mass,
            "m_hat": m_hat,
            "err_pct": err_pct,
            "sigma": float(result["sigma"]),
            "bin_label": result["bin_label"],
            "ok": bool(result["completed"]),
            "error": "",
        }
    except Exception as exc:
        return {
            "controller": controller_name,
            "estimator": estimator_name,
            "mass": mass,
            "m_hat": math.nan,
            "err_pct": math.nan,
            "sigma": math.nan,
            "bin_label": None,
            "ok": False,
            "error": repr(exc),
        }


def print_results(rows: list[dict]) -> None:
    print("\n" + "=" * 104)
    print(
        f"{'controller':<18} {'estimator':<18} {'true':>7} {'m_hat':>8} "
        f"{'err%':>9} {'sigma':>9} {'bin':>7} {'status':>8}"
    )
    print("-" * 104)
    for row in rows:
        sigma = "n/a" if math.isnan(row["sigma"]) else f"{row['sigma']:.3f}"
        err = "n/a" if math.isnan(row["err_pct"]) else f"{row['err_pct']:+.1f}%"
        status = "ok" if row["ok"] else "FAIL"
        print(
            f"{row['controller']:<18} {row['estimator']:<18} "
            f"{row['mass']:>7.3f} {row['m_hat']:>8.3f} {err:>9} "
            f"{sigma:>9} {str(row['bin_label'] or ''):>7} {status:>8}"
            f" {row['error'] if not row['ok'] else ''}"
        )

    print("\nSummary, excluding failed rows")
    print(f"{'controller':<18} {'estimator':<18} {'mean |err|':>12} {'RMSE':>10}")
    print("-" * 62)
    keys = sorted({(row["controller"], row["estimator"]) for row in rows})
    for controller, estimator in keys:
        errs = [
            row["err_pct"]
            for row in rows
            if row["controller"] == controller
            and row["estimator"] == estimator
            and row["ok"]
            and not math.isnan(row["err_pct"])
        ]
        if not errs:
            print(f"{controller:<18} {estimator:<18} {'n/a':>12} {'n/a':>10}")
            continue
        mean_abs = sum(abs(err) for err in errs) / len(errs)
        rmse = math.sqrt(sum(err * err for err in errs) / len(errs))
        print(f"{controller:<18} {estimator:<18} {mean_abs:>11.1f}% {rmse:>9.1f}%")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


if __name__ == "__main__":
    raise SystemExit(main())
