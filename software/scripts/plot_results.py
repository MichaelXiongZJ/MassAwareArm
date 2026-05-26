#!/usr/bin/env python3
"""Sweep result plotting utility.

Parses a verification sweep CSV file and generates a professional 2x2 panel 
dashboard showcasing both linear and logarithmic scales for linearity and 
percentage error. This fully demonstrates the operational regimes, noise floors,
and high-load convergence of the PID-Error, Lyapunov, and Momentum Observer estimators.

Usage:
    python software/scripts/plot_results.py results/sweep_YYYYMMDD_HHMMSS.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def generate_dashboard(csv_path: Path):
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1

    # Load data using standard library csv module to avoid extra dependencies (like pandas)
    data = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            data.append({
                "estimator": row["estimator"],
                "true_mass": float(row["true_mass"]),
                "m_hat": float(row["m_hat"]),
                "sigma": float(row["sigma"]) if row["sigma"] else np.nan,
                "err_pct": float(row["err_pct"])
            })

    # Modern, elegant plotting aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Create the 2x2 panel dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=200)
    
    # Color palette
    colors = {
        "pid_error": "#1f77b4",       # Sleek Steel Blue
        "lyapunov": "#2ca02c",        # Organic Green
        "momentum_observer": "#9467bd" # Vibrant Purple
    }
    
    # Estimator labels for legend
    labels = {
        "pid_error": "PID-Error",
        "lyapunov": "Lyapunov",
        "momentum_observer": "Momentum"
    }

    # Group data by estimator
    grouped = {}
    for row in data:
        grouped.setdefault(row["estimator"], []).append(row)

    # Sort each group by true_mass for continuous lines
    for est in grouped:
        grouped[est].sort(key=lambda x: x["true_mass"])

    # Extract unique sorted true masses for perfect fit line
    all_masses = np.sort(list(set(row["true_mass"] for row in data)))
    
    # ────────────────────────────────────────────────────────────────────────
    # PANEL A: RESPONSE CURVE (LINEAR SCALE)
    # ────────────────────────────────────────────────────────────────────────
    ax_a = axes[0, 0]
    ax_a.set_xlim(left=0, right = 3.1)
    ax_a.plot(all_masses, all_masses, 'k--', alpha=0.5, label="Perfect Fit (1:1)")
    for est, rows in grouped.items():
        tm = [r["true_mass"] for r in rows]
        mh = [r["m_hat"] for r in rows]
        ax_a.plot(tm, mh, 'o-', color=colors.get(est, "#888"),
                  label=labels.get(est, est), markersize=5, linewidth=1.5)
    ax_a.set_xlabel("True Mass (kg)", fontsize=11, fontweight='semibold')
    ax_a.set_ylabel("Estimated Mass (kg)", fontsize=11, fontweight='semibold')
    ax_a.set_title("Response Curve (Linear Scale)", fontsize=12, fontweight='bold', pad=10)
    ax_a.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # ────────────────────────────────────────────────────────────────────────
    # PANEL B: RESPONSE CURVE (LOG SCALE)
    # ────────────────────────────────────────────────────────────────────────
    ax_b = axes[0, 1]
    ax_b.loglog(all_masses, all_masses, 'k--', alpha=0.5, label="Perfect Fit (1:1)")
    for est, rows in grouped.items():
        tm = [r["true_mass"] for r in rows]
        mh = [r["m_hat"] for r in rows]
        ax_b.loglog(tm, mh, 'o-', color=colors.get(est, "#888"),
                    label=labels.get(est, est), markersize=5, linewidth=1.5)
    ax_b.set_xlabel("True Mass (kg)", fontsize=11, fontweight='semibold')
    ax_b.set_ylabel("Estimated Mass (kg)", fontsize=11, fontweight='semibold')
    ax_b.set_title("Response Curve (log-log)", fontsize=12, fontweight='bold', pad=10)
    ax_b.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # ────────────────────────────────────────────────────────────────────────
    # PANEL C: PERCENT ERROR (LINEAR SCALE)
    # ────────────────────────────────────────────────────────────────────────
    ax_c = axes[1, 0]
    ax_c.set_xlim(left=0, right = 3.1)
    ax_c.axhline(0, color='black', linestyle='-', alpha=0.6)
    for est, rows in grouped.items():
        tm = [r["true_mass"] for r in rows]
        # Clip relative error for Lyapunov at the low end to keep the plot readable
        y_val = [np.clip(r["err_pct"], -50, 150) if est == "lyapunov" else r["err_pct"] for r in rows]
        ax_c.plot(tm, y_val, 's-', color=colors.get(est, "#888"),
                  label=labels.get(est, est), markersize=5, linewidth=1.5)
    ax_c.set_xlabel("True Mass (kg)", fontsize=11, fontweight='semibold')
    ax_c.set_ylabel("Relative Error (%)", fontsize=11, fontweight='semibold')
    ax_c.set_title("Relative Error (Linear X-Axis)", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_ylim(-15, 15)
    ax_c.legend(frameon=True, facecolor='white', edgecolor='none')

    # ────────────────────────────────────────────────────────────────────────
    # PANEL D: PERCENT ERROR (LOG SCALE)
    # ────────────────────────────────────────────────────────────────────────
    ax_d = axes[1, 1]
    ax_d.axhline(0, color='black', linestyle='-', alpha=0.6)
    for est, rows in grouped.items():
        tm = [r["true_mass"] for r in rows]
        y_val = [np.clip(r["err_pct"], -50, 150) if est == "lyapunov" else r["err_pct"] for r in rows]
        ax_d.semilogx(tm, y_val, 's-', color=colors.get(est, "#888"),
                      label=labels.get(est, est), markersize=5, linewidth=1.5)
    ax_d.set_xlabel("True Mass (kg)", fontsize=11, fontweight='semibold')
    ax_d.set_ylabel("Relative Error (%)", fontsize=11, fontweight='semibold')
    ax_d.set_title("Relative Error (log X-axis)", fontsize=12, fontweight='bold', pad=10)
    ax_d.set_ylim(-15, 15)
    ax_d.legend(frameon=True, facecolor='white', edgecolor='none')

    # Final Layout & Titles
    plt.suptitle(f"Estimator Performance Case Study Dashboard\nSource: {csv_path.name}", 
                 fontsize=15, fontweight='bold', y=0.97)
    plt.tight_layout()
    
    out_dir = csv_path.parent
    out_path = out_dir / f"{csv_path.stem}_plots.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Successfully generated case study dashboard at:\n  {out_path}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate linear/log plots for estimator sweeps.")
    parser.add_argument("csv_path", type=str, help="Path to the sweep CSV file")
    args = parser.parse_args()
    return generate_dashboard(Path(args.csv_path))

if __name__ == "__main__":
    sys.exit(main())
