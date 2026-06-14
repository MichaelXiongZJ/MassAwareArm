"""Integration sweep in the PHYSICALLY-GRASPED (FSM) pipeline.

The paper's integration study (Fig. 8) runs each (controller, estimator) pair
in the *tracking* pipeline, where the payload is injected as a known force at
the end-effector. There the residual-based estimators recover that same force
through the same Jacobian, so they sit at the floating-point noise floor and
the figure cannot rank them or expose any controller dependence.

This script re-runs the same controller x estimator sweep but in the FSM
pipeline, where the cube is *physically grasped* by the Robotiq 2F-85 (real
contact, gripper-linkage spring torques). That is the realistic setting in
which the estimators show their true accuracy. Two controllers drive the same
grasp-weigh-place mission:

  * pd_gravity        - legacy PD + gravity compensation (default.yaml gains)
  * inverse_dynamics  - computed-torque setpoint law: the feedback acceleration
                        Kp e - Kd qdot is shaped by the joint-space inertia
                        M(q) before being commanded, with the bias added by the
                        drive loop. At the stationary weigh hold this is the
                        same computed-torque law used in the paper's tracking
                        controller, reduced to qdot_d = qddot_d = 0.

Both controllers honor the estimator's apply_overrides()/clear_overrides() and
per-joint gravity mask exactly as the FSM expects, so the estimators run
unmodified.

Usage:
    python software/scripts/integration_grasp_sweep.py
    python software/scripts/integration_grasp_sweep.py --masses 0.1,0.5,1,2,3 \
        --estimators pid_error,momentum_observer,inverse_dynamics
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.estimators import inverse_dynamics as _inverse_dynamics  # noqa: F401  (registration)
from massaware.planner import FSM
from massaware.tick_loop import Gripper, _build_obs
from scripts.mission import build

# ---------------------------------------------------------------------------
# Sweep grid (parallels the paper's integration study: 3 estimators, 2
# controllers, geometric mass sweep 10 g - 3 kg).
# ---------------------------------------------------------------------------
CONTROLLERS = ["pd_gravity", "inverse_dynamics"]
ESTIMATORS = ["pid_error", "momentum_observer", "inverse_dynamics"]
M_MIN, M_MAX, N_MASSES = 0.01, 3.0, 16
DEFAULT_MASSES = [round(float(m), 4) for m in np.geomspace(M_MIN, M_MAX, num=N_MASSES)]

# Computed-torque acceleration gains (omega_n^2, 1/s^2) used by the paper's
# inverse-dynamics controller; see eqs. (16)-(17). At the weigh hold these act
# through M(q) to set the closed-loop error dynamics e'' + Kd e' + Kp e = 0.
ID_KP = np.array([1800.0, 2200.0, 1800.0, 720.0, 560.0, 400.0])
ID_KD = np.array([28.0, 32.0, 28.0, 14.0, 12.0, 10.0])


class InverseDynamicsSetpointController:
    """Computed-torque setpoint controller with the PIDController interface.

    Drop-in for the FSM drive loop: compute() returns the inertia-shaped
    feedback torque M(q) (Kp e + Ki int(e) - Kd qdot); the drive loop adds the
    bias qfrc_bias * gravity_mask separately, exactly as for the PD controller.
    """

    def __init__(self, env, robot, kp, ki, kd):
        self.env = env
        self.robot = robot
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self._integral_err = np.zeros(6)
        self._base_gains: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self._integral_err.fill(0.0)

    def apply_overrides(self, overrides: dict) -> None:
        if self._base_gains is not None:
            raise RuntimeError("overrides already active; clear first")
        if not overrides:
            return
        self._base_gains = {"kp": self.kp, "ki": self.ki, "kd": self.kd}
        for key, value in overrides.items():
            if key not in ("kp", "ki", "kd"):
                raise KeyError(f"unknown gain '{key}'")
            setattr(self, key, np.asarray(value, dtype=float))
        self.reset()

    def clear_overrides(self) -> None:
        if self._base_gains is None:
            return
        self.kp = self._base_gains["kp"]
        self.ki = self._base_gains["ki"]
        self.kd = self._base_gains["kd"]
        self._base_gains = None
        self.reset()

    def compute(self, q, q_dot, q_ref, qfrc_bias, dt, *, use_gravity_comp=False):
        err = q_ref - q
        self._integral_err += err * dt
        accel_cmd = (self.kp * err) + (self.ki * self._integral_err) - (self.kd * q_dot)
        tau = self.env.mass_matrix() @ accel_cmd
        if use_gravity_comp:
            tau = tau + qfrc_bias
        return tau


def _drive(env, ctx, motion_controller, hold_controller, gripper, fsm) -> None:
    """Tick the FSM until done, mirroring verify_estimators._drive.

    Grasping and all gross motion use `motion_controller` (the reliable PD law).
    Whenever the FSM has an estimator sink active -- i.e. during the empty-arm
    calibration hold and the weigh hold, the only intervals where the estimate
    is formed -- the command is produced by `hold_controller` instead, when one
    is supplied. The cube is therefore grasped identically in every trial and
    the ONLY variable is the control law holding the weigh pose while the
    estimator samples. This isolates the controller's effect on the estimate,
    which is exactly the integration question.
    """
    overrides_target = ctx.controller  # FSM applies overrides to this object
    while not fsm.done:
        fsm.tick()
        sampling = ctx.estimator is not None and ctx.estimator_sink is not None
        controller = hold_controller if (hold_controller is not None and sampling) else motion_controller
        if ctx.reset_controller:
            motion_controller.reset()
            if hold_controller is not None:
                hold_controller.reset()
            ctx.reset_controller = False
        if ctx.arm_target is None:
            ctx.arm_target = env.get_arm_qpos()
        qb = env.qfrc_bias
        tau = controller.compute(
            q=env.get_arm_qpos(),
            q_dot=env.get_arm_qvel(),
            q_ref=ctx.arm_target,
            qfrc_bias=qb,
            dt=env.dt,
            use_gravity_comp=False,
        ) + qb * ctx.gravity_comp_mask
        tau = env.set_arm_ctrl(tau)
        gripper.apply(ctx.gripper_cmd)
        mujoco.mj_step(env.model, env.data)
        if sampling:
            ctx.estimator_sink(_build_obs(env, ctx.robot, q_ref=ctx.arm_target, tau_cmd=tau))


def run_trial(env, controller_name: str, estimator_name: str, mass: float) -> dict:
    env, ctx, default_controller = build(
        estimator_name=estimator_name, cube_mass=mass, env=env
    )
    if controller_name == "pd_gravity":
        hold_controller = None  # PD holds and moves
    elif controller_name == "inverse_dynamics":
        hold_controller = InverseDynamicsSetpointController(
            env, ctx.robot, ID_KP, np.zeros(6), ID_KD
        )
    else:
        raise ValueError(f"unknown controller '{controller_name}'")
    # The FSM applies estimator overrides/masks through ctx.controller; point it
    # at whichever controller is actually active during the weigh hold.
    ctx.controller = hold_controller if hold_controller is not None else default_controller
    fsm = FSM(ctx)
    gripper = Gripper(env)
    try:
        _drive(env, ctx, default_controller, hold_controller, gripper, fsm)
        r = ctx.estimate_result
        m_hat = float(r.m_hat) if r is not None else float("nan")
        sigma = float(r.sigma) if (r is not None and r.sigma is not None) else float("nan")
        err_pct = 100.0 * (m_hat - mass) / mass if mass else float("nan")
        return {
            "controller": controller_name, "estimator": estimator_name,
            "mass": mass, "m_hat": m_hat, "sigma": sigma,
            "err_pct": err_pct, "bin_label": ctx.bin_label, "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "controller": controller_name, "estimator": estimator_name,
            "mass": mass, "m_hat": float("nan"), "sigma": float("nan"),
            "err_pct": float("nan"), "bin_label": None, "error": repr(exc),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--controllers", default=",".join(CONTROLLERS))
    ap.add_argument("--estimators", default=",".join(ESTIMATORS))
    ap.add_argument("--masses", default=None)
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    cal_path = Path(__file__).resolve().parents[1] / "configs" / "calibration.yaml"
    if args.clear_cache and cal_path.exists():
        cal_path.unlink()

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    estimators = [e.strip() for e in args.estimators.split(",") if e.strip()]
    masses = (
        [float(m) for m in args.masses.split(",")] if args.masses else DEFAULT_MASSES
    )

    from massaware.mujoco_env import MujocoEnv
    env = MujocoEnv()

    rows: list[dict] = []
    t0 = time.time()
    for controller_name in controllers:
        for estimator_name in estimators:
            for mass in masses:
                print(f"\n>>> controller={controller_name} estimator={estimator_name} mass={mass}kg")
                rows.append(run_trial(env, controller_name, estimator_name, mass))

    # Per-trial table.
    print("\n" + "=" * 96)
    print(f"{'controller':<18} {'estimator':<18} {'true':>8} {'m_hat':>9} {'err':>8} {'bin':>7} {'status':>8}")
    print("-" * 96)
    for r in rows:
        err = "n/a" if math.isnan(r["err_pct"]) else f"{r['err_pct']:+.1f}%"
        status = "ok" if not r["error"] else "FAIL"
        print(f"{r['controller']:<18} {r['estimator']:<18} {r['mass']:>8.3f} "
              f"{r['m_hat']:>9.4f} {err:>8} {str(r['bin_label'] or ''):>7} {status:>8} "
              f"{r['error']}")

    # Per-(controller, estimator) summary: mean |err| and RMSE over the sweep.
    print(f"\n{'controller':<18} {'estimator':<18} {'mean err':>10} {'mean |err|':>12} {'RMSE':>9}")
    print("-" * 70)
    for controller_name in controllers:
        for estimator_name in estimators:
            errs = [r["err_pct"] for r in rows
                    if r["controller"] == controller_name
                    and r["estimator"] == estimator_name
                    and not math.isnan(r["err_pct"])]
            if not errs:
                print(f"{controller_name:<18} {estimator_name:<18} {'n/a':>10} {'n/a':>12} {'n/a':>9}")
                continue
            mean_err = sum(errs) / len(errs)
            mean_abs = sum(abs(e) for e in errs) / len(errs)
            rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
            print(f"{controller_name:<18} {estimator_name:<18} {mean_err:>+9.1f}% "
                  f"{mean_abs:>11.1f}% {rmse:>8.1f}%")

    elapsed = time.time() - t0
    print(f"\nsweep took {elapsed:.1f}s over {len(rows)} trials")

    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    csv_path = results_dir / f"integration_grasp_{stamp}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["controller", "estimator", "true_mass", "m_hat", "sigma", "err_pct", "bin_label", "error"])
        for r in rows:
            sigma = r["sigma"] if r["sigma"] == r["sigma"] else ""
            writer.writerow([r["controller"], r["estimator"], r["mass"], r["m_hat"],
                             sigma, f"{r['err_pct']:.4f}", r["bin_label"] or "", r["error"]])
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
