"""Shared matplotlib styling for the report figures.

Every paper figure imports from here so the estimator/controller colors,
markers, fonts, and unit conventions are identical across the document.
Captions carry the figure title in IEEE style, so plots set axis labels but
never a figure-level title.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# One visual identity per estimator, used by every estimator figure.
ESTIMATOR_STYLE = {
    "pid_error":         {"label": "sPID",               "color": "#1f77b4", "marker": "o", "ls": "-"},
    "lyapunov":          {"label": "Energy-balance",      "color": "#d62728", "marker": "s", "ls": "-"},
    "momentum_observer": {"label": "Momentum observer",   "color": "#9467bd", "marker": "^", "ls": "-"},
    "inverse_dynamics":  {"label": "Inverse dynamics",    "color": "#2ca02c", "marker": "D", "ls": "--"},
}

# One visual identity per tracking controller.
CONTROLLER_STYLE = {
    "pid_tracking":     {"label": "PD+G controller",            "color": "#1f77b4", "marker": "o", "ls": "-"},
    "inverse_dynamics": {"label": "Inverse dynamics controller", "color": "#d62728", "marker": "s", "ls": "--"},
}

# Per-joint colors (q1..q6) for time-history figures.
JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
JOINT_LABELS = [
    r"$q_1$ shoulder pan",
    r"$q_2$ shoulder lift",
    r"$q_3$ elbow",
    r"$q_4$ wrist 1",
    r"$q_5$ wrist 2",
    r"$q_6$ wrist 3",
]


def use_paper_style() -> None:
    """Apply the shared rcParams. Call once at the top of each plot script."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
