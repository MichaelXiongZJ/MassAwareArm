"""Produce the report's standalone estimator-accuracy figure from a sweep CSV.

Two stacked panels sized for a single IEEE column:
  (a) response curve (estimated vs true mass) with the ideal 1:1 line,
  (b) absolute relative error on log-log axes with the 5 % requirement line.

Also prints the per-estimator summary (mean error, mean |error|, RMSE, all in
percent of true mass) used for the report's summary table.

Usage:
    python software/scripts/plot_paper_estimator_figure.py [CSV] [--out PATH]
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
DEFAULT_CSV = REPO_ROOT / "results" / "sweep_20260611_044619.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT.parent / "Paper" / "estimator_accuracy_standalone.png",
    )
    args = parser.parse_args()

    rows = defaultdict(list)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["estimator"]].append(
                (float(row["true_mass"]), float(row["m_hat"]), float(row["err_pct"]))
            )

    print(f"{'estimator':<20} {'mean err %':>10} {'mean |err| %':>12} {'RMSE %':>8}")
    for name, data in rows.items():
        err = np.array([e for _, _, e in data])
        print(
            f"{name:<20} {err.mean():>+10.1f} {np.abs(err).mean():>12.1f} "
            f"{np.sqrt((err**2).mean()):>8.1f}"
        )

    use_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 7.4))

    for name, st in ESTIMATOR_STYLE.items():
        if name not in rows:
            continue
        data = sorted(rows[name])
        m_true = np.array([t for t, _, _ in data])
        m_hat = np.array([h for _, h, _ in data])
        err = np.array([e for _, _, e in data])
        ax1.plot(m_true, m_hat, marker=st["marker"], ls=st["ls"], color=st["color"],
                 label=st["label"])
        ax2.loglog(m_true, np.abs(err), marker=st["marker"], ls=st["ls"],
                   color=st["color"], label=st["label"])

    lims = [0, 3.1]
    ax1.plot(lims, lims, ":", color="gray", lw=1.2, label="ideal ($\\hat{m}=m$)")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("true payload mass [kg]")
    ax1.set_ylabel("estimated mass $\\hat{m}$ [kg]")
    ax1.set_title("(a) Response curve")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.axhline(5.0, color="black", ls="--", lw=1.2, label="5 % requirement")
    ax2.set_xlabel("true payload mass [kg]")
    ax2.set_ylabel("absolute relative error [%]")
    ax2.set_title("(b) Relative estimation error")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
