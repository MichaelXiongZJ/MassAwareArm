"""Sweep payload masses to estimate tracking-controller payload limits."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))
if str(SOFTWARE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT / "scripts"))

from massaware.config import load_config
from massaware.controllers.profiles import controller_gains, tracking_profile
from mission_tracking import release_time
from plot_tracking_controller_errors import run_controller_trace


CONTROLLERS = ["pid_tracking", "inverse_dynamics"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--profile", choices=["tracking", "main"], default="tracking")
    parser.add_argument("--masses", default="0.1,0.5,1,2,3,5,7.5,10,12.5,15,20")
    parser.add_argument("--payload-comp-mode", choices=["oracle", "estimated", "none"], default="oracle")
    parser.add_argument("--move-to-grasp-time", type=float, default=3.0)
    parser.add_argument("--lift-to-weigh-time", type=float, default=2.0)
    parser.add_argument("--move-to-release-time", type=float, default=None)
    parser.add_argument("--weigh-time", type=float, default=None)
    parser.add_argument("--joint-peak-deg", type=float, default=2.0)
    parser.add_argument("--torque-ratio", type=float, default=0.95)
    parser.add_argument(
        "--optimize-gains",
        action="store_true",
        help="search gain scales and report the largest payload that can meet tracking thresholds",
    )
    parser.add_argument("--kp-scales", default="0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0")
    parser.add_argument("--kd-scales", default="0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument(
        "--ignore-torque-limit",
        action="store_true",
        help="when optimizing gains, pass/fail uses tracking thresholds only",
    )
    parser.add_argument(
        "--disable-controller-overrides",
        action="store_true",
        help="ignore estimator-requested gain overrides during weighing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    profile = tracking_profile(args.profile, cfg)
    controllers = parse_csv(args.controllers)
    masses = parse_float_csv(args.masses)
    if args.optimize_gains:
        optimize_gains(cfg, profile, controllers, masses, args)
        return 0
    rows = []

    for controller in controllers:
        for mass in masses:
            rows.append(run_case(cfg, profile, controller, mass, args))

    print_results(rows, args)
    return 0


def optimize_gains(
    cfg: dict,
    profile,
    controllers: list[str],
    masses: list[float],
    args: argparse.Namespace,
) -> None:
    kp_scales = parse_float_csv(args.kp_scales)
    kd_scales = parse_float_csv(args.kd_scales)
    print(
        "gain optimization pass criteria: "
        f"joint_peak <= {args.joint_peak_deg:.3g} deg"
        + (
            ", torque ignored"
            if args.ignore_torque_limit
            else f", max |tau|/limit <= {args.torque_ratio:.3g}"
        )
    )
    print(f"masses={masses}")
    print(f"kp_scales={kp_scales}")
    print(f"kd_scales={kd_scales}")

    for controller in controllers:
        best: dict | None = None
        rows = []
        for mass in masses:
            best_for_mass = None
            for kp_scale in kp_scales:
                for kd_scale in kd_scales:
                    row = run_gain_case(
                        cfg,
                        profile,
                        controller,
                        mass,
                        kp_scale,
                        kd_scale,
                        args,
                    )
                    rows.append(row)
                    if not row["passed"]:
                        continue
                    if best_for_mass is None or row["score"] < best_for_mass["score"]:
                        best_for_mass = row
            if best_for_mass is not None:
                best = best_for_mass

        print(f"\n{controller}")
        print(
            f"{'mass':>7} {'kp_s':>6} {'kd_s':>6} {'joint_peak':>11} "
            f"{'ee_peak':>10} {'max_tau':>10} {'tau/lim':>9} {'score':>9}"
        )
        print("-" * 82)
        passing = [row for row in rows if row["passed"]]
        for row in passing:
            print(
                f"{row['mass']:>7.3f} {row['kp_scale']:>6.3f} {row['kd_scale']:>6.3f} "
                f"{fmt(row['joint_peak']):>11} {fmt(row['ee_peak']):>10} "
                f"{fmt(row['max_tau']):>10} {fmt(row['torque_ratio']):>9} "
                f"{fmt(row['score']):>9}"
            )
        if best is None:
            print("  no gain combination passed")
        else:
            print(
                "  best sampled payload: "
                f"{best['mass']:.3f} kg with kp_scale={best['kp_scale']:.3f}, "
                f"kd_scale={best['kd_scale']:.3f}, "
                f"max_tau={best['max_tau']:.2f} N-m, "
                f"tau/lim={best['torque_ratio']:.3f}"
            )


def run_gain_case(
    cfg: dict,
    profile,
    controller: str,
    mass: float,
    kp_scale: float,
    kd_scale: float,
    args: argparse.Namespace,
) -> dict:
    from dataclasses import replace

    kp, _ki, kd = controller_gains(profile, controller)
    if controller == "pid_tracking":
        active_profile = replace(
            profile,
            kp=kp * kp_scale,
            kd=kd * kd_scale,
        )
    else:
        active_profile = replace(
            profile,
            inverse_dynamics_kp=kp * kp_scale,
            inverse_dynamics_kd=kd * kd_scale,
        )
    row = run_case(cfg, active_profile, controller, mass, args)
    torque_ok = args.ignore_torque_limit or row["torque_ratio"] <= args.torque_ratio
    tracking_ok = row["joint_peak"] <= args.joint_peak_deg
    row["passed"] = tracking_ok and torque_ok and not row["error"]
    row["kp_scale"] = kp_scale
    row["kd_scale"] = kd_scale
    row["max_tau"] = row.pop("max_tau", np.nan)
    row["score"] = (
        row["joint_peak"]
        + 0.05 * row["ee_peak"]
        + 0.1 * row["torque_ratio"]
    )
    return row


def run_case(cfg: dict, profile, controller: str, mass: float, args: argparse.Namespace) -> dict:
    try:
        trace = run_controller_trace(
            cfg,
            profile,
            controller,
            mass=mass,
            payload_comp_mode=args.payload_comp_mode,
            move_to_grasp_time=args.move_to_grasp_time,
            lift_to_weigh_time=args.lift_to_weigh_time,
            move_to_release_time=release_time(args.move_to_release_time, profile),
            weigh_time=args.weigh_time,
            allow_controller_overrides=not args.disable_controller_overrides,
        )
        joint_err_deg = np.rad2deg(trace.q_ref - trace.q)
        ee_err_mm = (trace.desired_position - trace.actual_position) * 1000.0
        torque_limit = np.maximum(np.abs(trace.torque_min), np.abs(trace.torque_max))
        max_tau = float(np.max(np.abs(trace.tau)))
        torque_ratio = np.max(np.abs(trace.tau) / torque_limit)
        joint_peak = float(np.max(np.abs(joint_err_deg)))
        joint_rms = float(np.sqrt(np.mean(joint_err_deg**2)))
        ee_peak = float(np.max(np.linalg.norm(ee_err_mm, axis=1)))
        ee_rms = float(np.sqrt(np.mean(ee_err_mm**2)))
        passed = (
            joint_peak <= args.joint_peak_deg
            and torque_ratio <= args.torque_ratio
        )
        return {
            "controller": controller,
            "mass": mass,
            "joint_rms": joint_rms,
            "joint_peak": joint_peak,
            "ee_rms": ee_rms,
            "ee_peak": ee_peak,
            "max_tau": max_tau,
            "torque_ratio": float(torque_ratio),
            "passed": passed,
            "error": "",
        }
    except Exception as exc:
        return {
            "controller": controller,
            "mass": mass,
            "joint_rms": math.nan,
            "joint_peak": math.nan,
            "ee_rms": math.nan,
            "ee_peak": math.nan,
            "max_tau": math.nan,
            "torque_ratio": math.nan,
            "passed": False,
            "error": repr(exc),
        }


def print_results(rows: list[dict], args: argparse.Namespace) -> None:
    print(
        "pass criteria: "
        f"joint_peak <= {args.joint_peak_deg:.3g} deg, "
        f"max |tau|/limit <= {args.torque_ratio:.3g}, "
        "EE error reported only, "
        f"payload={args.payload_comp_mode}"
    )
    print(
        f"{'controller':<18} {'mass':>7} {'joint_rms':>10} {'joint_peak':>11} "
        f"{'ee_rms':>10} {'ee_peak':>10} {'tau/lim':>9} {'status':>8}"
    )
    print("-" * 95)
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        if row["error"]:
            status = "ERROR"
        print(
            f"{row['controller']:<18} {row['mass']:>7.3f} "
            f"{fmt(row['joint_rms']):>10} {fmt(row['joint_peak']):>11} "
            f"{fmt(row['ee_rms']):>10} {fmt(row['ee_peak']):>10} "
            f"{fmt(row['torque_ratio']):>9} {status:>8}"
            f" {row['error']}"
        )

    print("\nEstimated payload limit from sampled masses:")
    for controller in sorted({row["controller"] for row in rows}):
        passed = [
            row["mass"]
            for row in rows
            if row["controller"] == controller and row["passed"]
        ]
        limit = max(passed) if passed else None
        print(f"  {controller:<18} {limit if limit is not None else 'none'} kg")


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


if __name__ == "__main__":
    raise SystemExit(main())
