# PID-Error Mass Estimator

This note explains why the PID-error method works, how it is set up on a 6-DOF arm, and where its sensitivities sit.

## 1. The underlying physics

Pick one joint and call it the measurement joint. With a payload attached at the end-effector, the static joint torque required to hold that joint at a fixed reference angle is the sum of two terms: the gravity torque from the arm's own links distal to the joint, and the gravity torque from the payload. The arm-gravity term is a function of the arm's pose, which is held fixed at the weigh configuration, so it is a constant. The payload term is

$$
\tau_{\text{payload}}(q) \;=\; m \, g \, d(q),
$$

where $m$ is the unknown payload mass, $g$ is gravitational acceleration, and $d(q)$ is the horizontal moment arm from the joint's rotation axis to the payload's centre of mass at the weigh pose. For a revolute joint whose axis is the unit vector $\hat{a}$, with payload position $r_{\text{COM}}$ measured from the joint origin, the moment arm comes from projecting the gravity wrench through the joint's twist:

$$
d(q) \;=\; \bigl(r_{\text{COM}}(q) \times \hat{g}\bigr) \cdot \hat{a},
\qquad \hat{g} = (0, 0, -1).
$$

For the UR5e in this project, with the elbow chosen as the measurement joint and the weigh pose set to the home configuration, $d \approx 0.49$ m by direct evaluation of forward kinematics.

The estimator extracts $m$ by comparing the steady-state commanded torque at the measurement joint, $\tau_{\text{cmd}}$, against a baseline captured with the same pose and the same controller configuration but with an empty gripper. Let $\tau_{\text{ss,empty}}$ denote that baseline. Then

$$
\boxed{\; \hat{m} \;=\; \dfrac{\overline{\tau_{\text{cmd}}}\;-\;\tau_{\text{ss,empty}}}{g \, d}. \;}
$$

The overbar is a time average over a brief hold at the weigh pose. The denominator is purely geometric, so the entire calibration burden sits in the scalar $\tau_{\text{ss,empty}}$, which absorbs unmodelled friction, controller offsets, and any residual error in MuJoCo's `qfrc_bias` at the measurement joint.

## 2. Why a single uncompensated joint

In normal operation the controller adds the full gravity-bias vector $q_{\text{frc,bias}}(q)$ to its PD output, so each joint is automatically held against the arm's self-weight by the model rather than by the PID error. If gravity compensation is left on at every joint, the steady-state error at every joint is essentially zero, and there is no torque signal carrying the payload weight: the model just absorbs it.

The PID-error method disables the gravity-compensation feedforward on exactly one joint. The mask is a 0 on that joint and 1 everywhere else. With this setup, the other five joints still hold position essentially perfectly, but the measurement joint must develop a non-zero steady-state error so that its proportional term generates the torque needed to counteract gravity. That steady-state torque is what the estimator reads.

Choosing one joint, rather than all six, also localises the signal. With a single uncompensated joint the integral of arm dynamics across the kinematic chain does not need to be inverted at runtime: the moment arm $d(q)$ is a single scalar, computable once from FK at the fixed weigh pose. The elbow is the practical choice over the shoulder lift because it sits closer to the payload, which makes the signal-to-noise ratio less sensitive to any residual error in the model of the more proximal links.

## 3. Why the gains are softened during the weigh

There is one practical obstacle that the textbook formulation ignores. Running the UR5e's tuned position-control gains ($k_p \approx 2000$, $k_d \approx 400$) with a single uncompensated joint produces a hard limit cycle. The mean joint torque is correct, and indeed matches `qfrc_bias` to four significant figures, but the per-sample standard deviation reaches several tens of newton-metres, peak-to-peak well over one hundred. For a 0.2 kg cube, whose true torque contribution is just under one newton-metre, this noise floor swallows the signal.

The estimator therefore declares a `controller_overrides()` payload that softens the proportional and derivative gains on the measurement joint while leaving the other five joints stiff. The default in code is $k_p = 200$ and $k_d = 80$ at the elbow. With these gains the joint reaches a clean steady state, the standard deviation drops by roughly an order of magnitude, and the integral of the variance over a 1 s hold gives a standard error of the mean well below 0.2 N·m. That is comfortably below the signal even at the lightest payload tested.

Softening the loop changes the static stiffness but not the static equilibrium torque. At rest the proportional torque must still equal the disturbance torque, which is what the estimator reads. The softening therefore moves the joint farther from its reference (a few degrees instead of a few milliradians), without biasing $\hat{m}$.

## 4. Calibration

Calibration is the procedure that fixes the scalar $\tau_{\text{ss,empty}}$. At init, the planner moves the arm to the weigh pose with an empty gripper, applies the same gravity mask and the same softened gains that will be used during weighing, waits for the joint to settle, and averages $\tau_{\text{cmd}}$ at the measurement joint over a three-second hold. The result is persisted to `configs/calibration.yaml` and is keyed by the weigh pose, so any change to the pose invalidates the cache automatically.

The calibration step is not merely a numerical convenience. It also absorbs every static error the math above pretends does not exist: small inaccuracies in `qfrc_bias` arising from MuJoCo's inertial parameters, motor friction in the actuator model, and any constant offsets introduced by the integration step or the controller's discretisation. As long as the empty and loaded measurements share the same pose, the same mask, and the same gains, those constant errors cancel exactly, leaving only $m \, g \, d$.

## 5. Sensitivities and caveats

Three sensitivities matter in practice. First, the moment arm $d$ enters the denominator linearly, so a 10% error in $d$ becomes a 10% error in $\hat{m}$. The current value is hand-tuned in `configs/default.yaml` to match the forward-kinematics result at the weigh pose; see PINNED_ISSUES.md (P1) for the plan to derive it automatically.

Second, the estimator assumes that the arm reaches a static equilibrium during the hold. The non-fatal `SettleStep` accommodates the residual limit cycle that remains under softened gains, and the averaging window is long enough that the residual oscillation drops out of the mean. If the hold were too short relative to the closed-loop time constant, the mean would still carry transient bias.

Third, the method is brittle to changes in `qfrc_bias` at the non-measurement joints. The mask leaves those joints fully compensated, which is the right choice as long as MuJoCo's gravity model is accurate. If the arm geometry is modified without recomputing inertial parameters, the measurement joint inherits any residual gravity disturbance from the rest of the chain.

The method is, by construction, a quasi-static one. Anything that requires a dynamic signature, such as inertia, is invisible to it. That is the dividing line between this estimator and the momentum-observer method planned for the next phase.
