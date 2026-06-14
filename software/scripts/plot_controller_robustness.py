"""Plot payload and speed robustness for tracking controllers."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))
if str(SOFTWARE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT / "scripts"))

from massaware.config import load_config
from massaware.controllers.profiles import tracking_profile
from massaware.controllers.trajectory import release_time

from plot_tracking_controller_errors import (
    controller_label,
    run_controller_trace,
    select_time_window,
    tracking_metrics,
)


CONTROLLERS = ["pid_tracking", "inverse_dynamics"]
COLORS = {
    "pid_tracking": "tab:blue",
    "inverse_dynamics": "tab:orange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--profile", choices=["tracking", "main"], default="tracking")
    parser.add_argument(
        "--sweeps",
        choices=["payload", "speed", "all"],
        default="all",
        help="which robustness plots to generate",
    )
    parser.add_argument("--masses", default="0.1,0.5,1,2,3,5,7.5,10,12.5,15,20")
    parser.add_argument("--speed-scales", default="0.5,0.75,1,1.25,1.5,2,2.5,3")
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--payload-comp-mode", choices=["oracle", "estimated", "none"], default="oracle")
    parser.add_argument("--time-window", choices=["full", "release"], default="release")
    parser.add_argument("--move-to-grasp-time", type=float, default=3.0)
    parser.add_argument("--lift-to-weigh-time", type=float, default=2.0)
    parser.add_argument("--move-to-release-time", type=float, default=None)
    parser.add_argument("--weigh-time", type=float, default=None)
    parser.add_argument("--joint-peak-deg", type=float, default=2.0)
    parser.add_argument("--torque-ratio", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default=str(SOFTWARE_ROOT / "output" / "controller_robustness"),
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweeps in {"payload", "all"}:
        rows = run_payload_sweep(cfg, profile, controllers, args)
        print_results("Payload robustness", rows, "mass", "kg", args)
        plot_metric_sweep(
            rows,
            x_key="mass",
            x_label="Payload mass [kg]",
            output_path=output_dir / "payload_robustness_joint_error.png",
            metric_key="joint_peak",
            metric_label="Peak joint error [deg]",
            threshold=args.joint_peak_deg,
        )
        plot_metric_sweep(
            rows,
            x_key="mass",
            x_label="Payload mass [kg]",
            output_path=output_dir / "payload_robustness_torque_ratio.png",
            metric_key="torque_ratio",
            metric_label="Max |tau| / actuator limit",
            threshold=args.torque_ratio,
        )

    if args.sweeps in {"speed", "all"}:
        rows = run_speed_sweep(cfg, profile, controllers, args)
        print_results("Speed robustness", rows, "speed_scale", "x", args)
        plot_metric_sweep(
            rows,
            x_key="speed_scale",
            x_label="Release motion speed scale",
            output_path=output_dir / "speed_robustness_joint_error.png",
            metric_key="joint_peak",
            metric_label="Peak joint error [deg]",
            threshold=args.joint_peak_deg,
        )
        plot_metric_sweep(
            rows,
            x_key="speed_scale",
            x_label="Release motion speed scale",
            output_path=output_dir / "speed_robustness_torque_ratio.png",
            metric_key="torque_ratio",
            metric_label="Max |tau| / actuator limit",
            threshold=args.torque_ratio,
        )

    print(f"wrote robustness plots to {output_dir}")
    return 0


def run_payload_sweep(
    cfg: dict,
    profile,
    controllers: list[str],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for controller in controllers:
        for mass in parse_float_csv(args.masses):
            rows.append(
                run_case(
                    cfg,
                    profile,
                    controller,
                    mass=mass,
                    move_to_release_time=release_time(args.move_to_release_time, profile),
                    sweep_name="payload",
                    sweep_value=mass,
                    args=args,
                )
            )
    return rows


def run_speed_sweep(
    cfg: dict,
    profile,
    controllers: list[str],
    args: argparse.Namespace,
) -> list[dict]:
    base_release_time = release_time(args.move_to_release_time, profile)
    rows = []
    for controller in controllers:
        for speed_scale in parse_float_csv(args.speed_scales):
            if speed_scale <= 0.0:
                raise ValueError("speed scales must be positive")
            rows.append(
                run_case(
                    cfg,
                    profile,
                    controller,
                    mass=args.mass,
                    move_to_release_time=base_release_time / speed_scale,
                    sweep_name="speed_scale",
                    sweep_value=speed_scale,
                    args=args,
                )
            )
    return rows


def run_case(
    cfg: dict,
    profile,
    controller: str,
    *,
    mass: float,
    move_to_release_time: float,
    sweep_name: str,
    sweep_value: float,
    args: argparse.Namespace,
) -> dict:
    row = {
        "controller": controller,
        "mass": mass,
        "speed_scale": math.nan,
        "error": "",
    }
    row[sweep_name] = sweep_value
    try:
        full_trace = run_controller_trace(
            cfg,
            profile,
            controller,
            mass=mass,
            payload_comp_mode=args.payload_comp_mode,
            move_to_grasp_time=args.move_to_grasp_time,
            lift_to_weigh_time=args.lift_to_weigh_time,
            move_to_release_time=move_to_release_time,
            weigh_time=args.weigh_time,
            allow_controller_overrides=not args.disable_controller_overrides,
        )
        trace = select_time_window(full_trace, args.time_window)
        row.update(tracking_metrics(trace))
        row["passed"] = (
            row["joint_peak"] <= args.joint_peak_deg
            and row["torque_ratio"] <= args.torque_ratio
        )
    except Exception as exc:
        row.update(
            {
                "joint_rms": math.nan,
                "joint_peak": math.nan,
                "ee_rms": math.nan,
                "ee_peak": math.nan,
                "max_tau": math.nan,
                "torque_ratio": math.nan,
                "passed": False,
                "error": repr(exc),
            }
        )
    return row


def plot_metric_sweep(
    rows: list[dict],
    *,
    x_key: str,
    x_label: str,
    output_path: Path,
    metric_key: str,
    metric_label: str,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.9, 4.95))
    controllers = sorted({row["controller"] for row in rows})
    for controller in controllers:
        controller_rows = sorted(
            (row for row in rows if row["controller"] == controller),
            key=lambda row: row[x_key],
        )
        x = np.array([row[x_key] for row in controller_rows], dtype=float)
        y = np.array([row[metric_key] for row in controller_rows], dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=COLORS.get(controller),
            label=controller_label(controller),
        )
    ax.axhline(threshold, linestyle="--", color="black", linewidth=1.0, label="limit")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label.replace("[", " ["))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_results(
    title: str,
    rows: list[dict],
    x_key: str,
    x_unit: str,
    args: argparse.Namespace,
) -> None:
    print(f"\n{title}")
    print(
        "pass criteria: "
        f"joint_peak <= {args.joint_peak_deg:.3g} deg, "
        f"max |tau|/limit <= {args.torque_ratio:.3g}, "
        f"window={args.time_window}, payload={args.payload_comp_mode}"
    )
    print(
        f"{'controller':<26} {x_key:>12} {'joint_peak':>11} "
        f"{'ee_peak':>10} {'tau/lim':>9} {'status':>8}"
    )
    print("-" * 84)
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        if row["error"]:
            status = "ERROR"
        print(
            f"{controller_label(row['controller']):<26} "
            f"{fmt(row[x_key]) + ' ' + x_unit:>12} "
            f"{fmt(row['joint_peak']):>11} {fmt(row['ee_peak']):>10} "
            f"{fmt(row['torque_ratio']):>9} {status:>8} {row['error']}"
        )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
