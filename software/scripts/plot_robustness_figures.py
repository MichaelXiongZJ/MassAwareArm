"""Regenerate the four controller robustness figures for the report.

Sweeps payload mass (at fixed motion speed) and motion speed (at fixed payload),
runs both tracking controllers with oracle payload compensation, and plots the
carry-phase performance against the controller design requirements:

  - torque feasibility C2: peak commanded |tau| / actuator limit, with 0.95 line
  - tracking accuracy  C1: peak |joint error| [deg], with +/-2 deg line

Four figures, two controllers per figure, markers on every data point:
  robustness_torque_vs_payload.png   robustness_error_vs_payload.png
  robustness_torque_vs_speed.png     robustness_error_vs_speed.png

Usage:
    python software/scripts/plot_robustness_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from massaware.config import load_config  # noqa: E402
from massaware.controllers.profiles import tracking_profile  # noqa: E402
from mission_tracking import release_time  # noqa: E402
from paper_style import CONTROLLER_STYLE, use_paper_style  # noqa: E402
import plot_tracking_controller_errors as pe  # noqa: E402

CONTROLLERS = ["pid_tracking", "inverse_dynamics"]
OUT_DIR = SOFTWARE_ROOT.parent.parent / "Paper"


def carry_mask(trace) -> np.ndarray:
    """Boolean mask over the carrying phase (up to the gripper-open release)."""
    t = trace.time
    release_start = next(
        (tt for tt, stage in trace.stage_transitions if stage == "release"), None
    )
    return t < release_start if release_start is not None else np.ones_like(t, bool)


def trial_metrics(trace) -> dict:
    mask = carry_mask(trace)
    err_deg = np.rad2deg(np.abs(trace.q_ref - trace.q))[mask]
    ratio = (np.abs(trace.tau) / trace.torque_max[None, :])[mask]
    # Mean end-effector speed over the carry phase.
    pos = trace.actual_position[mask]
    t = trace.time[mask]
    if pos.shape[0] > 2:
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        ee_speed = float(step.sum() / max(t[-1] - t[0], 1e-9))
    else:
        ee_speed = float("nan")
    return {
        "peak_err_deg": float(err_deg.max()),
        "peak_ratio": float(ratio.max()),
        "ee_speed": ee_speed,
    }


def run_point(cfg, profile, controller, *, mass, move_to_release_time) -> dict:
    trace = pe.run_controller_trace(
        cfg, profile, controller,
        mass=mass, payload_comp_mode="oracle",
        move_to_grasp_time=3.0, lift_to_weigh_time=2.0,
        move_to_release_time=move_to_release_time, weigh_time=None,
        allow_controller_overrides=True,
    )
    return trial_metrics(trace)


def sweep(cfg, profile, *, masses, times) -> dict:
    data = {c: {"mass": [], "time": [], "speed": [],
                "err": [], "ratio": []} for c in CONTROLLERS}
    nominal_time = release_time(4.0, profile)
    nominal_mass = 1.5
    for c in CONTROLLERS:
        for m in masses:
            r = run_point(cfg, profile, c, mass=m, move_to_release_time=nominal_time)
            data[c]["mass"].append(m)
            data[c]["err"].append(r["peak_err_deg"])
            data[c]["ratio"].append(r["peak_ratio"])
            print(f"payload {c:18s} m={m:5.2f}kg  err={r['peak_err_deg']:.2f}deg  ratio={r['peak_ratio']:.2f}")
    for c in CONTROLLERS:
        for tt in times:
            r = run_point(cfg, profile, c, mass=nominal_mass, move_to_release_time=tt)
            data[c]["time"].append(tt)
            data[c]["speed"].append(r["ee_speed"])
            data[c]["serr"] = data[c].get("serr", []) + [r["peak_err_deg"]]
            data[c]["sratio"] = data[c].get("sratio", []) + [r["peak_ratio"]]
            print(f"speed   {c:18s} t={tt:4.2f}s  v={r['ee_speed']:.2f}m/s  err={r['peak_err_deg']:.2f}deg  ratio={r['peak_ratio']:.2f}")
    return data


def fig_vs_payload(data, key, ylabel, hline, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for c in CONTROLLERS:
        st = CONTROLLER_STYLE[c]
        ax.plot(data[c]["mass"], data[c][key], color=st["color"], marker=st["marker"],
                ls=st["ls"], label=st["label"])
    if hline is not None:
        ax.axhline(hline[0], color="black", ls="--", lw=1.2, label=hline[1])
    ax.set_xlabel("payload mass [kg]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_vs_speed(data, key, ylabel, hline, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for c in CONTROLLERS:
        st = CONTROLLER_STYLE[c]
        ax.plot(data[c]["time"], data[c][key], color=st["color"], marker=st["marker"],
                ls=st["ls"], label=st["label"])
    if hline is not None:
        ax.axhline(hline[0], color="black", ls="--", lw=1.2, label=hline[1])
    ax.set_xlabel("move-to-release time [s]  (shorter = faster motion)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    use_paper_style()
    cfg = load_config()
    profile = tracking_profile("tracking", cfg)

    masses = [0.001, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    times = [0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    data = sweep(cfg, profile, masses=masses, times=times)

    fig_vs_payload(data, "ratio", "peak commanded torque / limit",
                   (0.95, "0.95 limit (C2)"),
                   args.out_dir / "robustness_torque_vs_payload.png")
    fig_vs_payload(data, "err", "peak joint position error [deg]",
                   (2.0, "2 deg requirement (C1)"),
                   args.out_dir / "robustness_error_vs_payload.png")
    # Re-key the speed series for the plotting helpers.
    for c in CONTROLLERS:
        data[c]["ratio_speed"] = data[c]["sratio"]
        data[c]["err_speed"] = data[c]["serr"]
    fig_vs_speed(data, "ratio_speed", "peak commanded torque / limit",
                 (0.95, "0.95 limit (C2)"),
                 args.out_dir / "robustness_torque_vs_speed.png")
    fig_vs_speed(data, "err_speed", "peak joint position error [deg]",
                 (2.0, "2 deg requirement (C1)"),
                 args.out_dir / "robustness_error_vs_speed.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
