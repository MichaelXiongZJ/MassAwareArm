# MassAwareArm

**Autonomous Mass-Aware Sorting System** — MAE C163C/C263C Final Project, Team 8 (Armstrong), Spring 2026.

A simulated UR5e with a Robotiq 2F-85 gripper picks a cube off a conveyor in MuJoCo, infers the cube's mass **without any force/torque sensor or external scale**, classifies it light/heavy against a threshold, and drops it in the matching bin. The point of the project is not the pick-and-place: it is the side-by-side comparison of mass estimators and tracking controllers that all share one interface and are judged against common, quantified requirements.

The full write-up — per-method derivations, system design, and figures — is in [docs/references/8_Final Report.pdf](docs/references/8_Final%20Report.pdf) (with the proposal and final presentation alongside it in [docs/references/](docs/references/)).

## What's compared

**Four mass estimators**, each reading a different signal so they fail in different ways:

| Estimator (`name`) | Reads | Calibration | One-line physics |
|---|---|---|---|
| `pid_error` (sPID) | elbow steady torque | yes | Disable gravity comp at one joint; its steady torque minus the empty baseline equals `m·g·d` |
| `lyapunov` (energy-balance / spring-sag) | joint positions, EE height | yes | Soften the gains; the arm sags; an energy balance gives `m = 2ΔE_spring/(g·Δh)` |
| `momentum_observer` | momentum residual | yes | Generalized-momentum disturbance observer; residual → external torque, projected onto the EE Jacobian. **Weighs during motion** |
| `inverse_dynamics` | per-tick residual torque | **no** | Regress `τ − bias` against the per-kg gravity regressor; gates out dynamic samples, so it weighs during gentle motion with no baseline |

**Two trajectory-tracking controllers** for the pick-and-place motion, with the estimated mass fed forward as payload compensation: `pid_tracking` (PD + gravity comp) and `inverse_dynamics` (computed-torque).

**Headline results** (10 g–3 kg sweep): the **momentum observer is the most accurate estimator (0.9 % RMSE)** and the only one meeting the 5 % requirement over the full range; the **inverse-dynamics controller** holds the tracking and torque limits over a wider range of motion speeds than the PD+gravity baseline. The recommended architecture is **inverse-dynamics control with momentum-observer estimation**.

Design requirements the system is graded against: **C1** joint error ≤ ±2°, **C2** commanded torque ≤ 0.95× actuator limit, **E1** estimation error < 5 %, **E2** payload range 10 g–3 kg.

## Two pipelines

The same estimators run behind a single per-tick observation contract in either of two mission pipelines:

| | Weighing pipeline (legacy FSM) | Tracking pipeline |
|---|---|---|
| Entry | `mission.py`, `verify_estimators.py` | `mission_tracking.py`, `compare_tracking_controllers.py` |
| Motion | discrete FSM steps, setpoint jumps | smooth LSPB joint trajectories with `q, q̇, q̈` references |
| Weighing | dedicated WEIGH state, stationary hold | flagged trajectory segment — can sample **during the lift** |
| Payload | cube physically grasped by the 2F-85 (contact) | cube pinned to the EE, weight applied as a pure force |

## Setup

Python 3.11 is recommended.

```bash
py -3.11 -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# macOS / Linux:          source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Smoke-check that MuJoCo is wired up — the viewer should show the UR5e on a pedestal, a conveyor with a grey cube, and two coloured drop bins:

```bash
python -m mujoco.viewer --mjcf=software/assets/scene.xml
```

## Running

**Weighing pipeline.** `verify_estimators.py` runs every (estimator × cube mass) pair through the full FSM and prints a per-trial table plus a per-estimator summary of mean error, mean absolute error, and RMSE. The default grid is 4 estimators × 20 geometrically-spaced masses (10 g–3 kg). It reuses one MuJoCo environment across trials and writes a timestamped CSV to the gitignored `results/`.

```bash
python software/scripts/verify_estimators.py                    # full sweep
python software/scripts/verify_estimators.py --clear-cache      # recompute empty-arm baselines
python software/scripts/verify_estimators.py --estimator momentum_observer --masses 0.1,0.3,0.5
python software/scripts/plot_results.py results/sweep_*.csv     # 2x2 dashboard
```

A single mission (one pick, weigh, classify, drop) — the cube body is re-massed at startup:

```bash
python software/scripts/mission.py --estimator pid_error --mass 0.35 --viewer
python software/scripts/mission.py --estimator none      # skip weighing; always drops in the light bin
```

**Tracking pipeline.** Smooth LSPB trajectories with a choice of controller and estimator (`--profile tracking` uses the report's hand-tuned gains):

```bash
python software/scripts/mission_tracking.py --controller inverse_dynamics \
    --estimator momentum_observer --mass 0.5 --profile tracking --viewer
python software/scripts/compare_tracking_controllers.py --profile tracking --masses 0.5
python software/scripts/sweep_tracking_payload_limits.py    # pass/fail vs C1 (≤2°) and C2 (≤0.95)
```

`integration_grasp_sweep.py` re-runs the controller × estimator integration study in the *physically grasped* FSM pipeline, where contact forces (not a clean injected force) expose each estimator's true accuracy. The `plot_*` scripts regenerate the report figures from sweep CSVs.

## Tests

```bash
python -m pytest software/tests -q
```

Around 70 tests covering the estimator maths (no MuJoCo), the calibration cache, controller overrides, the registry, the classifier, and end-to-end FSM pipeline runs through MuJoCo — roughly six seconds total.

## Adding an estimator or controller

The estimator interface is in [software/massaware/estimators/base.py](software/massaware/estimators/base.py). A subclass declares whether it needs calibration, optionally overrides its per-joint gravity-comp mask and the controller gains it wants while measuring, and implements `reset`/`update`/`estimate` (plus the calibration trio when needed). Estimators self-register at import, so a new one costs one file in `estimators/`, a `register("<name>", <Class>)` call, an optional YAML block in `configs/default.yaml`, and an import in the entry script. A new tracking controller subclasses the base in [software/massaware/controllers/](software/massaware/controllers/) and adds a branch in `mission_tracking.make_controller` plus gains in `profiles.py`. Nothing else in the planner or scene changes.

## Project layout

```
docs/                          report PDFs (references/) and an example sweep (examples/)
results/                       gitignored sweep CSVs
software/
├── assets/                    MuJoCo scene + vendored UR5e and Robotiq 2F-85 (new_gripper/2f85.xml)
├── massaware/
│   ├── mujoco_env.py          sim wrapper: state, qfrc_bias, mass matrix, runtime mass-swap
│   ├── robot.py               FK / IK / EE Jacobian
│   ├── controller.py          legacy setpoint PID (FSM pipeline)
│   ├── planner.py             FSM (INIT, SEARCH, GRASP, WEIGH, CLASSIFY, PLACE, HOME, ERROR)
│   ├── tick_loop.py           single mj_step owner for the FSM pipeline; builds EstimatorObs
│   ├── classify.py            threshold classifier
│   ├── config.py              YAML loader (pose degrees → radians)
│   ├── perception/            ground-truth backend now, CV backend later
│   ├── estimators/            base + registry + pid_error, lyapunov, momentum_observer, inverse_dynamics
│   └── controllers/           tracking pipeline: LSPB trajectories, analytical IK, payload model, 2 controllers
├── configs/                   default.yaml + autogenerated calibration.yaml
├── scripts/                   mission*, verify_estimators, compare/sweep, plotting
└── tests/                     pytest suite
requirements.txt
```
