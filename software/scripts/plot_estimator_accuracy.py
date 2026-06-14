"""Paper-ready estimator-accuracy figures from estimator_accuracy_*.csv.

Reads a CSV written by compare_tracking_controllers.py and produces three
figures, each with one subplot per controller:

  1. <stem>_estimated_vs_true.png   m_hat vs true mass, with the ideal line
  2. <stem>_signed_error.png        signed error (m_hat - m) in grams
  3. <stem>_relative_error.png      |error| in percent, log scale

Usage:
    python software/scripts/plot_estimator_accuracy.py                 # latest CSV
    python software/scripts/plot_estimator_accuracy.py results/estimator_accuracy_X.csv
    python software/scripts/plot_estimator_accuracy.py --estimators pid_error,momentum_observer,inverse_dynamics
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

CONTROLLER_TITLE = {
    "pid_tracking": "PD+G tracking controller",
    "inverse_dynamics": "Inverse dynamics (computed-torque) controller",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", default=None,
                        help="estimator_accuracy CSV (default: newest in results/)")
    parser.add_argument("--estimators", default=None,
                        help="comma-separated subset to plot (default: all in the CSV)")
    parser.add_argument("--controllers", default=None,
                        help="comma-separated subset to plot (default: all in the CSV)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--rename", action="append", default=[],
                        metavar="ESTIMATOR=LABEL",
                        help="override a legend label, e.g. --rename pid_error=sPID "
                        "(repeatable)")
    return parser.parse_args()


def find_latest_csv() -> Path:
    candidates = sorted(RESULTS_DIR.glob("estimator_accuracy_*.csv"))
    if not candidates:
        raise FileNotFoundError("no results/estimator_accuracy_*.csv found")
    return candidates[-1]


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("completed") == "1"]
    if not rows:
        raise ValueError(f"{path} contains no completed trials")
    return rows


def series(rows: list[dict], controller: str, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted(
        (float(r["true_mass_kg"]), float(r["m_hat_kg"]))
        for r in rows
        if r["controller"] == controller and r["estimator"] == estimator
    )
    m = np.array([p[0] for p in pairs])
    m_hat = np.array([p[1] for p in pairs])
    return m, m_hat


def annotate_mode(fig: plt.Figure, rows: list[dict]) -> None:
    mode = rows[0].get("weigh_mode", "hold")
    hold = rows[0].get("weigh_time_s", "")
    profile = rows[0].get("profile", "")
    if not profile or not hold:
        return  # older CSVs lack these columns; omit rather than print "?"
    mode_text = (
        f"weighing during lift motion (hold {hold} s)"
        if mode == "lift"
        else f"stationary weigh hold ({hold} s)"
    )
    fig.text(0.99, 0.005, f"{profile} profile, {mode_text}",
             ha="right", va="bottom", fontsize=9, color="0.45")


def make_figure(controllers: list[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(
        len(controllers), 1, figsize=(6.4, 3.4 * len(controllers)),
        sharex=True, constrained_layout=True,
    )
    axes = [axes] if len(controllers) == 1 else list(axes)
    return fig, axes


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_estimated_vs_true(rows, controllers, estimators, out: Path, dpi: int) -> None:
    fig, axes = make_figure(controllers)
    for ax, controller in zip(axes, controllers):
        lo, hi = np.inf, -np.inf
        for estimator in estimators:
            m, m_hat = series(rows, controller, estimator)
            if m.size == 0:
                continue
            st = ESTIMATOR_STYLE[estimator]
            ax.plot(m, m_hat, color=st["color"], marker=st["marker"], ls=st["ls"],
                    label=st["label"])
            lo, hi = min(lo, m.min()), max(hi, m.max())
        ax.plot([lo, hi], [lo, hi], color="black", ls=":", lw=1.0,
                alpha=0.6, label=r"ideal ($\hat{m} = m$)", zorder=0)
        style_axis(ax, CONTROLLER_TITLE.get(controller, controller), "estimated mass [kg]")
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("true payload mass [kg]")
    annotate_mode(fig, rows)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def plot_signed_error(rows, controllers, estimators, out: Path, dpi: int) -> None:
    fig, axes = make_figure(controllers)
    for ax, controller in zip(axes, controllers):
        for estimator in estimators:
            m, m_hat = series(rows, controller, estimator)
            if m.size == 0:
                continue
            st = ESTIMATOR_STYLE[estimator]
            ax.plot(m, (m_hat - m) * 1000.0, color=st["color"], marker=st["marker"],
                    ls=st["ls"], label=st["label"])
        ax.axhline(0.0, color="black", ls=":", lw=1.0, alpha=0.6, zorder=0)
        style_axis(ax, CONTROLLER_TITLE.get(controller, controller),
                   r"error  $\hat{m} - m$  [g]")
    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("true payload mass [kg]")
    annotate_mode(fig, rows)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def plot_relative_error(rows, controllers, estimators, out: Path, dpi: int) -> None:
    fig, axes = make_figure(controllers)
    for ax, controller in zip(axes, controllers):
        for estimator in estimators:
            m, m_hat = series(rows, controller, estimator)
            if m.size == 0:
                continue
            st = ESTIMATOR_STYLE[estimator]
            rel = np.abs(m_hat - m) / m * 100.0
            # Floor at the float-precision level so log scale stays finite.
            ax.semilogy(m, np.maximum(rel, 1e-7), color=st["color"], marker=st["marker"],
                        ls=st["ls"], label=st["label"])
        ax.axhline(5.0, color="black", ls="--", lw=1.0, alpha=0.6, zorder=0,
                   label="5 % requirement")
        style_axis(ax, CONTROLLER_TITLE.get(controller, controller), "|error| [%]")
    axes[0].legend(frameon=False, loc="best", ncol=2)
    axes[-1].set_xlabel("true payload mass [kg]")
    annotate_mode(fig, rows)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    args = parse_args()
    use_paper_style()
    csv_path = Path(args.csv) if args.csv else find_latest_csv()
    rows = load_rows(csv_path)

    for spec in args.rename:
        estimator, _, label = spec.partition("=")
        if estimator not in ESTIMATOR_STYLE or not label:
            raise SystemExit(f"--rename expects ESTIMATOR=LABEL with a known estimator, got '{spec}'")
        ESTIMATOR_STYLE[estimator]["label"] = label

    present = defaultdict(set)
    for row in rows:
        present[row["controller"]].add(row["estimator"])
    controllers = (
        [c.strip() for c in args.controllers.split(",")]
        if args.controllers else sorted(present, reverse=True)  # pid_tracking first
    )
    estimators = (
        [e.strip() for e in args.estimators.split(",")]
        if args.estimators
        else [e for e in ESTIMATOR_STYLE if any(e in present[c] for c in controllers)]
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    print(f"plotting {csv_path.name}: controllers={controllers}, estimators={estimators}")
    plot_estimated_vs_true(rows, controllers, estimators,
                           out_dir / f"{stem}_estimated_vs_true.png", args.dpi)
    plot_signed_error(rows, controllers, estimators,
                      out_dir / f"{stem}_signed_error.png", args.dpi)
    plot_relative_error(rows, controllers, estimators,
                        out_dir / f"{stem}_relative_error.png", args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
