# MassAwareArm

MAE C263C Project, Team 8 Armstrong. A simulated UR5e in MuJoCo picks a cube off a conveyor, infers the cube's mass without a force sensor or any external scale, and drops it in the correct bin. The point of the project is not the pick-and-place: it is the comparison between several mass estimators that can be slotted into the same arm and judged side by side.

Three estimators are implemented. The PID-error method reads the controller's steady-state torque command at a single joint while gravity compensation is selectively disabled, and converts that torque to a payload mass through a known geometric moment arm. The Lyapunov method leaves gravity compensation on, softens the controller into a multi-dimensional spring, lets the arm sag a few millimetres under the payload, and recovers the mass from the energy balance between spring storage and gravitational potential. The momentum observer integrates the joint-space dynamics through a generalized-momentum residual whose steady-state value is the external joint torque, and recovers the mass by projecting that residual onto the end-effector Jacobian. The three methods read different sensor channels (joint torque, joint position, and time-integrated momentum), so they fail in different ways. That is the property the case study is designed to expose.

If you want the underlying maths and physics for each method, see:

- [docs/PID_ERROR.md](docs/PID_ERROR.md) for the PID-error estimator
- [docs/LYAPUNOV.md](docs/LYAPUNOV.md) for the Lyapunov (spring-sag) estimator
- [docs/MOMENTUM_OBSERVER.md](docs/MOMENTUM_OBSERVER.md) for the momentum-observer estimator
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layered system design

## Setup

Python 3.11 is recommended.

Create and activate a virtual environment, then install:

```bash
py -3.11 -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# macOS / Linux:          source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

A quick smoke check that MuJoCo is wired up:

```bash
python -m mujoco.viewer --mjcf=software/assets/scene.xml
```

The viewer should open with the UR5e on a pedestal, a conveyor in front of it, a single grey cube on the conveyor, and two coloured drop zones for light and heavy bins.

## Running the estimator verification sweep

`scripts/verify_estimators.py` runs every combination of estimator and cube mass through the full FSM, prints a per-trial table of true mass against estimated mass, and finishes with a per-estimator summary of mean error, mean absolute error, and RMSE. The script reuses a single MuJoCo environment across trials so the whole sweep runs in roughly twenty-five seconds for the default 75-trial grid (three estimators times 25 geometrically-spaced masses from 10 g to 2.5 kg).

```bash
# Default sweep: all three estimators, 25 masses, 10 g to 2.5 kg
python software/scripts/verify_estimators.py

# Start with a clean calibration cache (recomputes baselines in-sim):
python software/scripts/verify_estimators.py --clear-cache

# Custom mass list, restricted to one estimator:
python software/scripts/verify_estimators.py --estimator lyapunov --masses 0.1,0.2,0.3,0.5,0.8

# Watch the whole sweep in one continuous viewer window:
python software/scripts/verify_estimators.py --viewer
```

Trials with absolute error above a 25% threshold are tagged with a `high err` note in the table; the script still exits zero (the focus is on the response curve, not a pass/fail gate). Change `M_MIN`, `M_MAX`, or `N_MASSES` at the top of the script to widen, narrow, or densify the sweep.

## Running a specific mission

A single mission is one pickup, one weigh, one classification, one drop. The script is `software/scripts/mission.py`.

```bash
# Default: uses estimator and cube mass from configs/default.yaml
python software/scripts/mission.py

# Specify the cube mass for this run (the body is re-massed at startup):
python software/scripts/mission.py --mass 0.35

# Override the estimator for this run only, ignoring YAML:
python software/scripts/mission.py --estimator pid_error
python software/scripts/mission.py --estimator lyapunov

# Watch it run in MuJoCo's passive viewer:
python software/scripts/mission.py --viewer --mass 0.5

# Skip the estimator entirely. Falls back to the Phase 3 pipeline,
# which just picks up the cube and always drops it in the light bin:
python software/scripts/mission.py --estimator none
```

The summary printed at the end gives the true mass, the estimated mass, the per-trial error in percent, and the bin label that the classifier picked.

## Running the test suite

Unit tests sit in `software/tests/` and cover the estimator maths (no MuJoCo), the calibration cache, the controller's override hooks, the registry, the classifier, and a small set of end-to-end pipeline tests that exercise the full FSM through MuJoCo.

```bash
python -m pytest software/tests -q
```

The whole suite (around 75 tests) finishes in roughly six seconds.

## How to swap in a new estimator

The estimator interface lives in `software/massaware/estimators/base.py`. The contract is small: a subclass declares whether it needs calibration, optionally overrides the per-joint gravity-compensation mask and the controller gains it wants while it is measuring, and implements four methods (`reset`, `update`, `estimate`, plus the calibration trio when needed). Concrete estimators self-register at import time, so the pipeline picks them up automatically once their module is imported in `mission.py`.

A minimal new estimator therefore costs one new file in `software/massaware/estimators/`, one `register("<name>", <Class>)` call at module bottom, one config block in `configs/default.yaml` if the estimator has tunables, and one import line in `mission.py`. Nothing in the planner, controller, or scene changes.

## Project layout

```
docs/                          design notes and per-estimator write-ups
software/
├── assets/                    MuJoCo scene plus the vendored UR5e and 2f85 gripper
├── massaware/
│   ├── mujoco_env.py          thin sim wrapper, runtime mass-swap helper
│   ├── robot.py               FK / IK / Jacobian
│   ├── controller.py          joint-space PID with per-estimator gain overrides
│   ├── planner.py             FSM (INIT, SEARCH, GRASP, WEIGH, CLASSIFY, PLACE, HOME, ERROR)
│   ├── tick_loop.py           the single owner of mj_step plus the estimator dispatch
│   ├── classify.py            threshold classifier
│   ├── perception/            ground-truth backend now, CV backend later
│   └── estimators/            base interface, registry, pid_error.py, lyapunov.py, momentum_observer.py
├── configs/                   default.yaml plus the autogenerated calibration.yaml
├── scripts/                   mission.py, verify_estimators.py
└── tests/                     pytest suite
requirements.txt
```