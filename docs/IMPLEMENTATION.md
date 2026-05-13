# MassAwareArm — Implementation Roadmap

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Build order is **bottom-up to first end-to-end demo, then sideways to add estimators**. Each phase ends with a concrete "done check" you can run before moving on.

> **Guiding principle:** get one cube sorted end-to-end as fast as possible with the simplest estimator, *then* swap in the others. Avoid building modules you can't yet test.

---

## Phase 0 — Setup (½ day)

| Step | Done check |
|---|---|
| Create a Python 3.11 venv named `.env`; install `requirements.txt` | `python -c "import mujoco; print(mujoco.__version__)"` |
| UR5e + umi_gripper assets are vendored under `software/assets/{ur5e,gripper}/` | `python -m mujoco.viewer --mjcf=software/assets/scene.xml` opens |
| Create empty package skeleton: `massaware/__init__.py`, `configs/`, `scripts/` | `python -c "import massaware"` |

---

## Phase 1 — Scene & Sim Wrapper (1–2 days)

**Goal:** load a complete scene from Python, step it, read sensors.

1. Extend `assets/scene.xml`:
   - Add the gripper as a child of the UR5e wrist body.
   - Confirm `<actuator>` entries cover all 6 UR5e joints (the vendor `ur5e.xml` already provides these).
   - Add `<sensor>` entries: `jointpos`, `jointvel`, `actuatorfrc` per joint.
   - Add an overhead `<camera name="overhead">` (you won't use it yet, just reserve it).
2. Implement `massaware/mujoco_env.py`:
   - `class MujocoEnv` wrapping `mj_model`, `mj_data`.
   - Methods: `reset()`, `step()`, `get_sensor(name)`, `get_qpos()`, `get_qvel()`.
3. Implement `massaware/robot.py`:
   - `fk(q)`, `jacobian_ee(q)` — wrappers around `mj_kinematics` / `mj_jacSite`. Operate on the full 6-DOF joint vector.
   - `ik(target_xyz, q_seed)` — numerical IK (damped least-squares). With 6 DOF and a 3-D position target the IK is underdetermined; seed from `HOME_QPOS` (or current pose) to bias toward consistent gripper orientation. Clip per-iteration step and wrap the final result to the seed's 2π branch to avoid joint-wrap-around solutions.

**Done check:** `scripts/run.py` (stub) loads scene, steps 1000 ticks, prints final EE position; gravity makes the arm sag if actuators are disabled.

---

## Phase 2 — Controller (1 day)

**Goal:** the arm holds a commanded pose.

1. Implement `massaware/controller.py`:
   - `PIDController(kp, ki, kd, use_gravity_comp=False)`.
   - `compute(q, q_dot, q_ref) → tau_cmd`. When `use_gravity_comp=True`, add `qfrc_bias` from MuJoCo.
   - Internal integral state, with `reset()`.
2. Wire it into `mujoco_env.step()` at 500 Hz inner loop (N controller ticks per env step).
3. Hand-tune gains for all 6 joints empirically.

**Done check:** with empty gripper, PID holds an arbitrary pose; `‖q_ref - q‖ < 1e-3 rad` after 0.5 s settle. Disable gravity comp during this test.

---

## Phase 3 — Primitives & Bare FSM (2–3 days)

**Goal:** pick-and-place a cube end-to-end with **no weighing yet**.

1. Implement `massaware/perception/base.py` + `groundtruth.py`:
   - `CubeDetection(name, color, xyz, quat)`.
   - `GroundTruthPerception.detect()` reads cube poses straight from MuJoCo.
2. Implement `massaware/planner.py` motion primitives:
   - `move_to_joint(q_target, timeout)`
   - `move_to_cartesian(xyz, timeout)` → IK → `move_to_joint`
   - `grasp()`, `release()` — drive the gripper actuator to closed/open.
   - `settle(tol, timeout)` — wait until `‖q_dot‖ < tol`.
3. Implement the FSM in the same file:
   - States: `SEARCH`, `GRASP`, `PLACE`, `HOME` (skip WEIGH/CLASSIFY for now).
   - Hard-code `PLACE` to always go to the light bin.
4. Write `scripts/run.py` to instantiate everything and tick the FSM.

**Done check:** run it, watch one cube travel `home → cube → light_bin → home`. Print the state trace.

---

## Phase 4 — First Estimator: Inverse-Dynamics Residual (2 days)

**Why this one first:** controller-agnostic, no calibration step, uses `qfrc_bias` and `mj_jac` directly. Fastest path to a working "sort by weight" demo.

1. Implement `massaware/estimators/base.py`:
   - `Estimator` ABC, `EstimatorObs`, `EstimateResult` exactly as in ARCHITECTURE.md §3.
2. Implement `massaware/estimators/inverse_dynamics.py`:
   - In `update(obs)`, accumulate samples of `tau_resid = tau_meas - qfrc_bias` and `a = J_vᵀ · [0, 0, -g]`.
   - In `estimate()`, return `m_hat = (mean(a)·mean(tau_resid)) / (mean(a)·mean(a))`.
3. Implement `massaware/estimators/registry.py` (a name → class dict + `build(name, cfg)`).
4. Implement `massaware/classify.py`: `threshold_classifier(m_hat, thresh) → "light" | "heavy"`.
5. Add `WEIGH` and `CLASSIFY` states to `planner.py`. Pick `WEIGH_POSE` (arm ~horizontal, maximizes shoulder moment arm).
6. Pick `WEIGHT_THRESHOLD` between cube masses (e.g. 0.35 kg given 0.2 / 0.5 / 1.0 kg).

**Done check:** all three cubes get sorted to the correct bin in 3-of-3 runs. Print `(true_mass, m_hat, bin)` for each.

---

## Phase 5 — Calibration + PID-Error Estimator (1–2 days)

**Goal:** add the second baseline estimator and prove the swap.

1. Write `scripts/calibrate.py`:
   - Move empty arm to `WEIGH_POSE`, hold with pure PID (no gravity comp), record `tau_ss_empty` over N samples.
   - Save to `configs/calibration.yaml`.
2. Add `INIT` state to the FSM that loads (or runs) calibration before `SEARCH`.
3. Implement `massaware/estimators/pid_error.py`:
   - `update(obs)`: collect `tau_cmd` at shoulder (or weighted sum across load-bearing joints).
   - `estimate()`: `m_hat = (mean(tau_cmd) - tau_ss_empty) / (g · moment_arm_at_WEIGH_POSE)`.
4. Switch estimator via `configs/default.yaml` only — *zero* other code changes.

**Done check:** with `estimator: pid_error` in config, run the full pipeline; sorting still works. Compare `m_hat` from both estimators on the same cube — both should be in the right ballpark (`±20%` for PID-error, `±5%` for inverse-dynamics).

---

## Phase 6 — Momentum Observer (1–2 days)

**Goal:** validate that the `Estimator` interface handles a *stateful* time-integrating estimator without contortion.

1. Implement `massaware/estimators/momentum_observer.py`:
   - Maintain integral state `r(t)` updated each `update(obs)`:
     `r̂_dot = K_O · (p - ∫(tau_meas - qfrc_bias + r̂) dt - r̂)`
   - In `estimate()`, project converged residual onto `Jᵀg` to get `m_hat`.
2. Run during `WEIGH` (and optionally during `LIFT` for a mid-trajectory estimate).

**Done check:** with `estimator: momentum_observer`, sorting still works. Plot `r̂(t)` during a hold — it should converge within ~0.5 s.

---

## Phase 7 — Case Study & Logging (1–2 days)

**Goal:** produce the comparison plot from slide 8.

1. Implement `massaware/log.py`:
   - Append-only CSV per run: timestamp, state, `q`, `q_dot`, `tau_cmd`, `tau_meas`, `m_hat`, `sigma`, true mass.
2. Write `scripts/compare_estimators.py`:
   - Sweep: each estimator × each cube × N trials (e.g. 10).
   - For each trial, reset, run pipeline, log `(true_mass, m_hat)`.
   - Plot estimated-vs-true with error bars per estimator (matplotlib).
3. Save plot to `docs/figures/estimator_comparison.png`.

**Done check:** the produced plot looks like slide 8 (multiple methods, all hugging the `y = x` line, error bars visible).

---

## Phase 8 — Stretch Goals (time permitting)

- **Camera-CV perception:** implement `perception/camera_cv.py` (HSV thresholding + blob centroids on the overhead camera frame). Swap via config; verify pipeline still passes.
- **Re-weigh on uncertainty:** add `RE_WEIGH` state triggered when `sigma > threshold`.
- **Multi-class classifier:** more than two bins, with thresholds learned from a small calibration sweep.
- **Adaptive gripping force:** scale gripper close torque by `m_hat`.
- **Animation pass:** Yonghao's trajectory recording + replay.

---

## Suggested Build Order at a Glance

```
Phase 0  Setup
   │
Phase 1  Scene + Sim Wrapper + Robot Model
   │
Phase 2  PID Controller
   │
Phase 3  Primitives + Bare FSM (pick-and-place, no weighing)
   │
Phase 4  Inverse-Dynamics Estimator  ←── first working sort demo
   │
Phase 5  Calibration + PID-Error Estimator
   │
Phase 6  Momentum Observer
   │
Phase 7  Logging + Case-Study Plot   ←── deliverable
   │
Phase 8  Stretch goals
```

Estimated total: **2–3 weeks** of focused work for phases 0–7.

---

## Daily Rhythm Recommendations

- **Always run `scripts/run.py` end-to-end before committing.** If the pipeline breaks at any phase, fix it before adding more.
- **Hard-code first, configure later.** Move values into `configs/default.yaml` only after the hard-coded version works.
- **Log everything from Phase 4 onward.** The case-study plot only needs the data you've already been logging — don't bolt logging on at the end.
- **One estimator file per PR.** Reviewers can confirm the interface holds.
