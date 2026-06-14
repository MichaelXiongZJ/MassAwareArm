"""Log and plot the momentum-observer residual r(t) during the weigh hold.

Runs one FSM mission with the momentum-observer estimator and a known cube
mass, records the residual vector and filtered regressor at every tick of the
weigh hold, and renders the report figure:

  (a) per-joint residual components r_i(t),
  (b) the instantaneous mass solution m(t) against the true mass.

Usage:
    python software/scripts/plot_observer_convergence.py [--mass 1.0] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from massaware.tick_loop import Gripper, TickLoop  # noqa: E402
from massaware.planner import FSM  # noqa: E402
from paper_style import JOINT_LABELS, use_paper_style  # noqa: E402

import mission  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAVITY = 9.81


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT.parent / "Paper" / "momentum_residual_convergence.png",
    )
    args = parser.parse_args()

    env, ctx, controller = mission.build(
        estimator_name="momentum_observer", cube_mass=args.mass
    )
    est = ctx.estimator
    burn_in_s = est._burn_in_s

    log_t: list[float] = []
    log_r: list[np.ndarray] = []
    log_jz: list[np.ndarray] = []

    orig_update = est.update

    def hooked_update(obs):
        orig_update(obs)
        # Only record the loaded weigh hold (after calibration is in place).
        if est.r_empty is not None and est._initialized:
            log_t.append(obs.t)
            log_r.append(est._r.copy())
            log_jz.append(est._jz_filt.copy())

    est.update = hooked_update

    loop = TickLoop(env, FSM(ctx), Gripper(env), controller, ctx.robot)
    loop.run()

    if ctx.estimate_result is not None:
        print(f"m_hat = {ctx.estimate_result.m_hat:.4f} kg (true {args.mass} kg)")
    if not log_t:
        raise RuntimeError("no weigh-hold samples were recorded")

    t = np.asarray(log_t) - log_t[0]
    r = np.stack(log_r, axis=0)
    jz = np.stack(log_jz, axis=0)

    delta = r - est.r_empty
    leverage = np.einsum("ij,ij->i", jz, jz)
    m_inst = -np.einsum("ij,ij->i", delta, jz) / (GRAVITY * leverage)

    use_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 6.8), sharex=True)

    for j in range(6):
        ax1.plot(t, r[:, j], lw=1.5, label=JOINT_LABELS[j])
    ax1.axvspan(0, burn_in_s, color="gray", alpha=0.15)
    ax1.text(0.24, 0.07, "burn-in (discarded)", ha="center", fontsize=10,
             transform=ax1.transAxes)
    ax1.set_ylabel("residual $r_i$ [N$\\cdot$m]")
    ax1.set_title("(a) Observer residual convergence")
    ax1.legend(ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(t, m_inst, lw=1.6, color="#9467bd",
             label="instantaneous $\\hat{m}(t)$")
    ax2.axhline(args.mass, color="black", ls="--", lw=1.2,
                label=f"true mass {args.mass:g} kg")
    ax2.axvspan(0, burn_in_s, color="gray", alpha=0.15)
    ax2.set_xlabel("time since start of weigh hold [s]")
    ax2.set_ylabel("mass estimate [kg]")
    ax2.set_title("(b) Per-tick mass solution")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}  ({len(t)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
