"""Grasp-pipeline integration figure: |rel error| vs mass, per controller.

Companion to the force-injection integration figure. Reads the CSV written by
`integration_grasp_sweep.py` (physically-grasped FSM, two weigh-hold control
laws x three estimators) and plots absolute relative estimation error against
true mass on log-log axes, one panel per controller, with the 5 % requirement
(E1) line. Unlike the force-injection figure, the residual estimators here sit
at their true accuracy rather than the floating-point floor.

Usage:
    python software/scripts/plot_integration_grasp_figure.py [CSV] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import ESTIMATOR_STYLE, use_paper_style  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path("/Users/michael/GitHub/6a2888bb674827a76320532a")

# Panels left-to-right; titles match the paper's controller naming.
CONTROLLER_ORDER = [
    ("pd_gravity", "(a) PD+G weigh-hold"),
    ("inverse_dynamics", "(b) Computed-torque weigh-hold"),
]
ESTIMATOR_ORDER = ["pid_error", "momentum_observer", "inverse_dynamics"]


def _latest_csv() -> Path:
    candidates = sorted((REPO_ROOT / "results").glob("integration_grasp_*.csv"))
    if not candidates:
        raise SystemExit("no integration_grasp_*.csv found in results/")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=PAPER_DIR / "integration_grasp_relative_error.png"
    )
    args = parser.parse_args()
    csv_path = args.csv or _latest_csv()

    # data[(controller, estimator)] = list of (true_mass, |err_pct|)
    data: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            key = (row["controller"], row["estimator"])
            data[key].append((float(row["true_mass"]), abs(float(row["err_pct"]))))

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    for ax, (controller, title) in zip(axes, CONTROLLER_ORDER):
        for est in ESTIMATOR_ORDER:
            pts = sorted(data.get((controller, est), []))
            if not pts:
                continue
            m = np.array([p[0] for p in pts])
            err = np.array([max(p[1], 1e-3) for p in pts])  # floor for log axis
            st = ESTIMATOR_STYLE[est]
            ax.loglog(m, err, marker=st["marker"], ls=st["ls"], color=st["color"],
                      label=st["label"])
        ax.axhline(5.0, color="black", ls="--", lw=1.2)
        ax.set_xlabel("true payload mass [kg]")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("absolute relative error [%]")
    # 5 % requirement label once, plus the estimator legend.
    axes[0].plot([], [], color="black", ls="--", lw=1.2, label="5% requirement (E1)")
    axes[0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out} from {csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
