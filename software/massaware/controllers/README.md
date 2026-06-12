# Tracking Controllers

This directory contains the joint-space tracking controller code used by
`software/scripts/mission_tracking.py` and the controller comparison/plotting
scripts. It is separate from the older FSM pipeline in `massaware/controller.py`
and `software/scripts/mission.py`.

## Files

```text
base.py                       shared gain overrides, tracking error, payload comp
pd_with_gravity_tracking.py   PD tracking controller with gravity comp
inverse_dynamics_tracking.py  computed-torque tracking controller
references.py                 controller input/output dataclasses
trajectory.py                 pick-weigh-release trajectories and release helpers
ik.py                         UR5e analytical IK plus fallback IK wrapper
attachment.py                 temporary payload attachment/force model
profiles.py                   controller gains and tracking experiment profiles
```

## Runtime Flow

`mission_tracking.py` builds a controller, estimator, pick/weigh trajectory, and
release trajectory:

```text
pick/weigh trajectory -> estimate payload mass -> choose bin -> release trajectory
```

Payload compensation is disabled during pick/weigh. During release, the
controller uses the estimated payload mass, or the true mass when a script is
run with oracle payload compensation.

## Main Scripts

Run one tracking mission with viewer:

```bash
.env/bin/python software/scripts/mission_tracking.py \
  --viewer \
  --controller inverse_dynamics \
  --estimator inverse_dynamics \
  --mass 0.5
```

Plot controller tracking errors. Use `--time-window full` for the whole mission
or `--time-window release` for only the payload-carrying release motion:

```bash
.env/bin/python software/scripts/plot_tracking_controller_errors.py \
  --time-window release \
  --payload-comp-mode oracle \
  --disable-controller-overrides
```

Plot payload and speed robustness:

```bash
.env/bin/python software/scripts/plot_controller_robustness.py \
  --sweeps all \
  --payload-comp-mode oracle \
  --disable-controller-overrides
```

## Notes

- `pid_tracking` uses feedback plus gravity and payload compensation.
- `inverse_dynamics` uses computed torque with mass matrix, bias terms, and
  payload compensation.
- Default pass/fail checks use peak joint error <= 2 deg and
  max `|tau_cmd_raw| / actuator_limit` <= 0.95.
