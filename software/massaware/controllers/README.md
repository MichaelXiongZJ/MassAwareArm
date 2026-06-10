# Tracking Controllers

This package contains the joint-space tracking controllers and helper code used
by the controller comparison scripts. It is separate from the original
`massaware/controller.py` pipeline.

## Files

```text
software/massaware/controllers/
  base.py                         shared gains, errors, overrides, payload comp.
  pid_tracking.py                 PD/PID + gravity + payload compensation
  inverse_dynamics_tracking.py    computed-torque tracking controller
  references.py                   reference/output dataclasses
  trajectory.py                   pick-weigh-release joint trajectories
  ik.py                           UR5e IK helper
  attachment.py                   payload attachment force model
  profiles.py                     tracking profiles and controller gains
```

## Controllers

PID tracking:

```text
u = Kp(qd - q) + Ki∫(qd - q)dt + Kd(qdot_d - qdot)
  + tau_g(q) + tau_p(q, m_hat)
```

Inverse dynamics tracking:

```text
qddot_cmd = qddot_d + Kd(qdot_d - qdot) + Kp(qd - q)
u = M(q)qddot_cmd + n(q, qdot) + tau_p(q, m_hat)
```

Payload compensation is gravity-only:

```text
tau_p(q, m_hat) = J(q)^T [0, 0, m_hat g]^T
```

Controller outputs distinguish raw torque demand from actuator-limited command:

```text
tau_cmd_raw      controller torque demand
tau_cmd_clipped  command after actuator ctrlrange clipping
tau_cmd          alias for tau_cmd_clipped
```

## Payload Timing

Payload compensation is disabled before and during weighing. After weighing,
release motion uses the estimated mass, or the true mass in oracle experiments.

```text
pick / lift / weigh_hold: no payload compensation
release motion: payload compensation enabled
release after opening: no payload compensation
```

Estimator-requested gain overrides can be disabled with:

```bash
--disable-controller-overrides
```

## Main Commands

Controller tracking plots, using oracle payload mass and fixed controller gains:

```bash
.env/bin/python software/scripts/plot_tracking_controller_errors.py \
  --payload-comp-mode oracle \
  --disable-controller-overrides
```

Include end-effector plots only when needed:

```bash
.env/bin/python software/scripts/plot_tracking_controller_errors.py \
  --include-ee-plots
```

Payload sweep with joint tracking and torque-limit pass/fail:

```bash
.env/bin/python software/scripts/sweep_tracking_payload_limits.py \
  --payload-comp-mode oracle \
  --disable-controller-overrides
```

Full controller/estimator matrix:

```bash
.env/bin/python software/scripts/compare_tracking_controllers.py \
  --profile tracking \
  --masses 0.5
```

Viewer:

```bash
.env/bin/python software/scripts/mission_tracking.py \
  --viewer \
  --controller inverse_dynamics \
  --estimator inverse_dynamics \
  --mass 0.5 \
  --profile tracking
```

## Metrics

For joint-space controller comparisons, pass/fail uses:

```text
joint peak error <= 2 deg
max |tau_cmd_raw| / actuator_limit <= 0.95
```

Metrics are scored over the carry phases only (everything before the
`release` stage). Detaching the payload re-enables cube collision while the
cube still overlaps the gripper pads, and the resulting one-tick contact
impulse would otherwise dominate the peak metrics.

End-effector error is reported as a diagnostic, not used as the default
pass/fail criterion.
