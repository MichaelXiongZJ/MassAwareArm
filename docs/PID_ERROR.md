# PID-Error Mass Estimator

This note explains the PID-error method as implemented in `software/massaware/estimators/pid_error.py`. The math is the main subject. The implementation gets one short section near the end. There is also an intuition section up top for readers with no robotics background.

## 1. Intuition

Hold a hand out in front of you, palm up and arm straight. If you relax your shoulder, gravity wins and your arm falls. To keep the arm horizontal, you have to clench the muscles in your shoulder just hard enough that they push back against gravity with exactly the right force. Now place an apple on your palm. To keep the arm level, you have to clench harder. If you knew exactly how hard you were pushing before and after the apple was placed, you could work out the apple's weight just from the difference in muscle effort.

The PID-error estimator does this with the robot arm. Normally the controller has a precomputed "gravity feedforward" that exactly cancels the weight of every link in the arm, so the controller does not have to do any real work to hold the arm up. For this estimator, the feedforward is turned off at exactly one joint, the elbow. The elbow now has to push for real to hold the arm against gravity. The harder the elbow pushes, the heavier the load below it. By comparing the elbow's effort with a cube in the gripper against its effort with an empty gripper, the mass of the cube is recovered.

It is essentially using the robot's controller as a digital scale. The read-out is the torque command at one joint, instead of a reading on a dial.

## 2. The physics

Pick one joint and call it the measurement joint. With the arm held at a fixed pose, the static torque at this joint must balance all the gravity loads acting below it. There are two such loads. First, the weight of the arm links distal to the joint. Second, the weight of the payload in the gripper. Since we have committed to weighing at a single fixed pose, the arm contribution is a constant. The payload contribution is

$$\tau_{\text{payload}} \;=\; m \, g \, d(q),$$

where $m$ is the unknown payload mass, $g = 9.81$ m/s$^2$ is the gravitational acceleration, and $d(q)$ is the horizontal distance from the joint's rotation axis to the payload's centre of mass. This horizontal distance is the moment arm.

There is a clean way to compute the moment arm from forward kinematics without ever leaving the model. For a revolute joint whose axis of rotation is the unit vector $\hat{a}$, with the payload sitting at a position $r(q)$ measured from the joint origin, the torque produced by gravity about the joint axis is

$$\tau \;=\; m \, (r \times \hat{g}) \cdot \hat{a}, \qquad \hat{g} = (0, 0, -1).$$

Stripping out the unknown $m$ gives the moment arm:

$$d(q) \;=\; (r(q) \times \hat{g}) \cdot \hat{a}.$$

For the UR5e at its weigh pose (the home configuration, with shoulder lift and elbow both at right angles), and with the elbow chosen as the measurement joint, this evaluation gives $d \approx 0.49$ m.

The estimator extracts $m$ by comparing the steady-state commanded torque at the measurement joint between two cases. With an empty gripper, the joint sits at a baseline torque $\tau_{\text{ss,empty}}$, which absorbs the arm's own weight contribution at the measurement joint plus whatever static modelling errors the simulator carries. With a loaded gripper, the joint commands an averaged torque $\overline{\tau_{\text{cmd}}}$. The difference is exactly the contribution from the payload:

$$\overline{\tau_{\text{cmd}}} \;-\; \tau_{\text{ss,empty}} \;=\; m \, g \, d.$$

Solving for the mass:

$$\boxed{\; \hat{m} \;=\; \dfrac{\overline{\tau_{\text{cmd}}} \;-\; \tau_{\text{ss,empty}}}{g \, d}. \;}$$

The denominator is purely geometric, computed once and never read from a sensor. The numerator is the only quantity that has to be measured at run time. The overbar denotes the time average over a one-second hold at the weigh pose.

## 3. Why disable gravity compensation on only one joint

A reasonable question is why we don't simply leave gravity compensation off at every joint. There are two reasons.

The first reason is informational. The controller's gravity feedforward is computed from MuJoCo's model of the arm's inertial parameters. If every joint is left compensated, each joint runs with essentially zero steady-state error, and the proportional term of the controller has nothing to do. The commanded torque at every joint is essentially zero, plus the gravity feedforward itself, which already contains the right answer in a form we cannot easily decompose into "arm contribution" and "payload contribution". The payload's weight gets absorbed into the model's gravity computation, and the signal we want is invisible.

Disabling gravity compensation on a single joint changes the picture. Without the feedforward, the joint cannot stay at its target with zero error. The proportional term has to develop a non-zero position error large enough that $k_p \cdot \text{error}$ produces the torque needed to hold the arm against gravity. With a cube added, the required torque grows by $m g d$, the position error grows accordingly, and the proportional torque tracks the payload weight exactly. The signal is now isolated at one joint, where we can read it directly.

The second reason is geometric. With a single uncompensated joint, the relationship between joint torque and payload mass is a one-line formula involving a single scalar moment arm. With multiple uncompensated joints, the same payload spreads its torque contribution across the kinematic chain, and recovering the mass requires solving a linear system, with each joint's moment arm acting as a different coefficient. Choosing one joint keeps the math trivial and turns the estimator into one division.

The elbow is preferred over the shoulder lift because it sits closer to the payload. Errors in the inertial model of the more proximal links (shoulder pan, shoulder lift) leak into the measurement joint through the kinematic chain. The closer the measurement joint is to the payload, the less of that chain there is to corrupt the reading.

## 4. Why the gains are softened during the weigh

There is a practical issue that the textbook formulation glosses over. Running the UR5e's tuned position-control gains ($k_p \approx 2000$, $k_d \approx 400$) with a single uncompensated joint produces a hard limit cycle. The mean commanded torque is correct, and indeed matches the gravity bias to four significant figures, but the per-sample standard deviation of the torque command climbs into the tens of newton-metres, with peak-to-peak excursions over one hundred. For a 0.2 kg cube, whose true torque contribution is just under one newton-metre, this noise floor swallows the signal completely.

The estimator therefore softens the proportional and derivative gains at the measurement joint while leaving the other five joints at their stiff base gains. The defaults in code are $k_p = 200$ and $k_d = 80$ at the elbow. With these gains the limit cycle disappears, the joint reaches a clean steady state, and the standard deviation of the torque command drops by roughly an order of magnitude. After averaging over a one-second hold, the standard error of the mean falls well below 0.2 N·m, comfortably below the signal even for the lightest cubes the project sweeps.

Softening the loop changes the static stiffness but not the static equilibrium torque. At rest, the proportional torque must still equal the disturbance torque, since that is what holds the joint still. The softening simply moves the joint farther away from its reference angle (a few degrees, instead of a few milliradians) without changing the answer. The mean torque is the signal; the soft loop just makes that mean cleaner to compute.

## 5. Calibration

The baseline torque $\tau_{\text{ss,empty}}$ is captured at initialization, before any cube is picked up. The planner moves the arm to the weigh pose with an empty gripper, applies the same gravity mask and the same softened gains that will be used at weigh time, waits for the joint to settle, and averages the commanded torque at the measurement joint over a three-second hold. The result is written to `configs/calibration.yaml` and is keyed by the weigh pose, so any change to the pose invalidates the cache automatically.

The calibration is not just a numerical convenience. It absorbs every static error the math above pretends does not exist. Small inaccuracies in the gravity bias from imperfect inertial parameters in the model, motor friction in the actuator, integration-step offsets from the discretised controller, all of these contribute a constant additive error to the steady-state torque. As long as the empty and loaded measurements share the same pose, the same mask, and the same gains, every one of those constant errors cancels exactly in the subtraction $\overline{\tau_{\text{cmd}}} - \tau_{\text{ss,empty}}$, leaving only $m g d$.

## 6. Implementation notes

The estimator is about 120 lines of Python in `software/massaware/estimators/pid_error.py`. It conforms to the project's `Estimator` interface, which has four hooks: `reset`, `update`, `estimate`, plus the calibration trio. At every physics tick during the weigh hold, `update()` appends the current commanded torque at the elbow to a buffer. After the hold completes, `estimate()` averages the buffer, subtracts the cached baseline, and divides by $g d$.

Two estimator hooks reshape the controller while the estimator is active. `gravity_comp_mask()` returns a vector with zero at the elbow index and ones at every other joint, telling the planner to disable the gravity feedforward at exactly one joint. `controller_overrides()` returns soft $k_p$ and $k_d$ at the elbow while keeping the other five joints stiff. The planner applies both before the weigh hold begins and restores everything once the hold ends.

Calibration uses the same machinery with a separate sample buffer, so the empty-arm baseline and the loaded measurement do not interfere with each other.

## 7. Sensitivities and caveats

Three sources of error matter in practice.

First, the moment arm $d$ enters the formula linearly, so a 10% error in $d$ produces a 10% error in the mass estimate. The current value (0.492 m) is hand-tuned in `configs/default.yaml` to match the forward-kinematics evaluation at the empty weigh pose. The wide-mass sweep shows a bias that grows monotonically with cube mass: a heavier cube sags the loaded pose slightly away from the empty pose, the true moment arm at the loaded pose drifts away from 0.492 m, and the formula picks up a small systematic error. Recomputing $d$ from forward kinematics at the loaded pose rather than relying on the cached empty value would flatten this drift, at the cost of one extra FK call per estimate.

Second, the formula assumes the arm reaches a static equilibrium during the hold. The non-fatal settle step that precedes the sampling window is best-effort. Some estimator configurations, including this one with softened gains, reach a low-amplitude limit cycle rather than a true rest. The averaging window is long enough that the residual oscillation drops out of the mean, but if the hold were too short relative to the closed-loop time constant, the mean would still carry transient bias.

Third, the method is sensitive to the model's gravity bias at the non-measurement joints. Those joints remain fully compensated, which is the right choice as long as the model accurately captures the gravity torques in the rest of the chain. If the arm geometry is modified without recomputing inertial parameters, the measurement joint inherits any residual disturbance from the rest of the chain through the kinematics. In the current model this effect is well below the cube-mass signal.

The method is, by construction, quasi-static. Anything that requires a dynamic signature, such as inertia, is invisible to it. That is the dividing line between this estimator and the momentum-observer method, which uses the time integral of the joint-space dynamics rather than just the steady-state torque.
