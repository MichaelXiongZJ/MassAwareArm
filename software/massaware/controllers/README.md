# Tracking Controller Additions

This folder contains the added trajectory-tracking controller workflow. It is
kept separate from the original `massaware/controller.py`, planner, config, and
mission files so it can be copied into another main branch with minimal changes.

## Added Files

Controller package:

```text
software/massaware/controllers/
  __init__.py
  attachment.py
  ik.py
  inverse_dynamics_tracking.py
  pid_tracking.py
  profiles.py
  references.py
  trajectory.py
```

Estimator addition:

```text
software/massaware/estimators/inverse_dynamics.py
```

Scripts:

```text
software/scripts/mission_inverse_dynamics_estimator.py
software/scripts/mission_tracking.py
software/scripts/compare_tracking_controllers.py
software/scripts/plot_tracking_controller_errors.py
```

## What Each Script Does

`mission_inverse_dynamics_estimator.py`

Runs the original main mission pipeline while registering and defaulting to the
`inverse_dynamics` estimator. This is for testing the ID estimator in the
original main control flow.

`mission_tracking.py`

Runs the new trajectory-tracking workflow. It supports:

```text
controllers: pid_tracking, inverse_dynamics
estimators:  pid_error, lyapunov, momentum_observer, inverse_dynamics
profiles:    tracking, main
```

`compare_tracking_controllers.py`

Runs batch comparisons over controller, estimator, and mass combinations.

`plot_tracking_controller_errors.py`

Plots controller tracking performance: joint angles, joint errors, torques,
end-effector position, end-effector error, and orientation error.

## Profiles

`--profile tracking`

Uses the external tracking-controller style setup:

```text
kp/kd: external tracking gains
IK: analytical IK
task waypoints: fixed grasp/weigh/release points
sampling: lift_to_weigh + weigh_hold
release: payload compensation enabled after mass estimate
```

`--profile main`

Uses main's default gains, main bin positions, position-only IK, and weigh-hold
sampling.

## Merge Into Another Main Branch

From the target main branch, copy these files:

```text
software/massaware/controllers/
software/massaware/estimators/inverse_dynamics.py
software/scripts/mission_inverse_dynamics_estimator.py
software/scripts/mission_tracking.py
software/scripts/compare_tracking_controllers.py
software/scripts/plot_tracking_controller_errors.py
```

Do not overwrite original main files unless needed. These additions are designed
to work by importing/registering the new estimator inside the new scripts.

## Test Commands

Run from the project root:

```bash
python -m py_compile software/massaware/controllers/*.py software/massaware/estimators/inverse_dynamics.py software/scripts/mission_inverse_dynamics_estimator.py software/scripts/mission_tracking.py software/scripts/compare_tracking_controllers.py software/scripts/plot_tracking_controller_errors.py
```

Single tracking smoke test:

```bash
python software/scripts/mission_tracking.py --controller inverse_dynamics --estimator inverse_dynamics --mass 0.5 --profile tracking
```

Small controller-estimator matrix:

```bash
python software/scripts/compare_tracking_controllers.py --profile tracking --masses 0.5
```

Mass sweep:

```bash
python software/scripts/compare_tracking_controllers.py --profile tracking --masses 0.1,0.5,1.0,2.0,5.0,10.0
```

Controller error plots:

```bash
python software/scripts/plot_tracking_controller_errors.py --mass 0.5 --profile tracking
```

Viewer:

```bash
python software/scripts/mission_tracking.py --viewer --controller inverse_dynamics --estimator inverse_dynamics --mass 0.5 --profile tracking
```

Original main flow with inverse-dynamics estimator:

```bash
python software/scripts/mission_inverse_dynamics_estimator.py --mass 0.5
```

If the environment only has `python3`, replace `python` with `python3`.
