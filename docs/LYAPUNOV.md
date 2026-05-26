# Lyapunov (Spring-Sag) Mass Estimator

This note explains the spring-sag method as implemented in `software/massaware/estimators/lyapunov.py`. The geometry, the energy balance, the factor-of-two correction that distinguishes the textbook formula from the physically correct one, and the reasons for the integral and gain modifications are all covered below.

## 1. The setup

Switch off the integral term of the position controller and leave the proportional and derivative gains in place. Around any reference angle $q_{\text{ref}}$, the controller now behaves as a linear spring with stiffness $k_p$ in parallel with a viscous damper of strength $k_d$. With full gravity compensation enabled, the steady state of an empty arm is exactly $q = q_{\text{ref}}$, because the controller's gravity feedforward cancels every joint torque produced by the arm's own weight and the spring has nothing to push against.

Now attach a payload of mass $m$ at the end-effector. Gravity acts on the payload, the payload pulls the end-effector down, and every joint develops a small sag in the direction that lowers the end-effector. The arm settles at a new equilibrium $q_{\text{loaded}}$ in which the spring torque at each joint balances the joint torque produced by the payload's weight. By measuring how far the arm sagged and computing the elastic energy stored in the springs, one can recover $m$ without ever reading a torque signal directly. The estimator's only inputs are the joint angles and the end-effector height.

## 2. The naive energy balance and why it gives half the answer

A first-pass derivation goes like this. The payload moves down by $\Delta h = z_{\text{empty}} - z_{\text{loaded}}$ between the empty equilibrium and the loaded equilibrium. The work done by gravity on the payload during the quasi-static descent is $W_{\text{grav}} = m g \, \Delta h$. The spring energy stored in the controller's joint-space springs, summed over the joints, is

$$
\Delta E_{\text{spring}} \;=\; \sum_{j} \tfrac{1}{2} k_{p,j} \!\left( (q_{\text{loaded},j} - q_{\text{ref},j})^2 - (q_{\text{empty},j} - q_{\text{ref},j})^2 \right).
$$

The textbook move is to equate the two and solve:

$$
\tilde{m} \;=\; \frac{\Delta E_{\text{spring}}}{g \, \Delta h}. \quad (\text{wrong by a factor of 2})
$$

Running this formula against a 0.5 kg cube in simulation gives $\tilde{m} \approx 0.245$ kg. Running it against a 0.2 kg cube gives $\tilde{m} \approx 0.10$ kg. The ratio is right, but every estimate is exactly half the truth.

The reason is that the equation $W_{\text{grav}} = \Delta E_{\text{spring}}$ leaves out the damper. Energy conservation across the transient from the empty to the loaded equilibrium reads

$$
W_{\text{grav}} \;=\; \Delta E_{\text{spring}} + W_{\text{damp}},
$$

where $W_{\text{damp}} = \int_0^{\infty} k_d \, \dot{q}^2 \, \mathrm{d}t$ is the energy dissipated by the derivative term as the joint moves. For a step input to a damped, statically stable PD-controlled mass, this dissipated energy has a clean closed form.

Consider one joint in isolation. The new equilibrium has a sag $\delta = F / k_p$ where $F$ is the constant disturbance torque. The work done by $F$ across that sag is $W_{\text{grav}} = F \delta = F^2 / k_p$. The spring stores $\tfrac{1}{2} k_p \delta^2 = F^2 / (2 k_p)$. The remainder, $F^2 / (2 k_p)$, must be dissipated by the damper. This split holds for any positive $k_d$, whether the system is underdamped, critically damped, or overdamped, as long as the joint actually reaches its new equilibrium with zero velocity. It is purely a consequence of $W = F \delta$ being twice the integral of the linearly growing spring force over the same distance.

The correct formula is therefore

$$
\boxed{\; \hat{m} \;=\; \dfrac{2 \, \Delta E_{\text{spring}}}{g \, \Delta h}. \;}
$$

## 3. Why $k_i = 0$ matters

If the integral term is active, then at steady state the integral has wound up to provide whatever torque is needed to drive the position error to zero. The arm holds the loaded pose at the same $q$ it held the empty pose, the sag $\delta$ vanishes, and the spring energy is identically zero. The estimator loses its signal.

Zeroing $k_i$ during the weigh phase is therefore not a tuning knob, it is a structural requirement of the method. The base controller in this project happens to ship with $k_i = 0$ already, so the override is a no-op today. The estimator still requests it explicitly through `controller_overrides()` so that the requirement is documented at the point of need, and so that the system remains correct if the base controller is later tuned with an integral term.

## 4. Why $k_p$ is softened

At the UR5e's tuned position-control stiffness, $k_p$ is around 2000 N·m/rad on the proximal joints. For a 0.2 kg payload, the per-joint sag at the elbow is on the order of $\delta = m g d / k_p$, which works out to fractions of a milliradian. The corresponding end-effector drop is sub-millimetre. At those scales, floating-point noise in the equilibrium, transient leftovers from the move into the weigh pose, and quasi-static drift in the controller dominate the signal.

Softening the proportional stiffness by an order of magnitude pushes the sag into the easily measurable centimetre range. The factor-of-two relationship between $W_{\text{grav}}$ and $\Delta E_{\text{spring}}$ is independent of the stiffness, so the formula above continues to hold; only the variance of the estimate improves. The derivative gain is softened in proportion so that the closed-loop damping ratio stays in the same regime, which preserves the assumption that the joint reaches a clean rest before the sampling window.

## 5. Calibration

Calibration measures $q_{\text{empty}}$ and $z_{\text{empty}}$. The planner moves the arm to the weigh pose with an empty gripper, applies the same mask and the same softened gains that will be used during weighing, lets the arm settle, then averages the joint angles and the end-effector height over a three-second hold. The cached result is written to `configs/calibration.yaml`, keyed by the weigh pose, alongside whatever calibration the PID-error estimator produced from the same session.

Calibrating with the same configuration that will be used during the loaded measurement is important. Any systematic offsets in `qfrc_bias`, in the controller, or in the integrator drop out of the energy difference. What remains is the difference between two equilibria that differ only by the presence of the payload.

## 6. Sensitivities and caveats

The method assumes that the closed-loop step from $q_{\text{empty}}$ to $q_{\text{loaded}}$ is well-behaved: monotonic, statically stable, and reaches rest within the sampling window. The factor-of-two relationship between $W_{\text{grav}}$ and $\Delta E_{\text{spring}}$ relies on the start and end states both having zero velocity. An underdamped transient that has not finished ringing by the time the sampling window opens will bias the estimate upward (more energy in the spring at the moment of measurement than the equilibrium predicts).

The estimator is exquisitely sensitive to the value of $k_p$ used in the calibration and in the weigh. A 10% error in any $k_{p,j}$ translates directly to a 10% error in $\Delta E_{\text{spring}}$ at that joint, which in turn rolls into $\hat{m}$ in proportion to that joint's contribution to the total. This is why the estimator reads the gains from the runtime configuration rather than hardcoding them.

Finally, the method is purely quasi-static. It carries no information about the payload's inertia, and would return the same number for two payloads that have the same mass but vastly different inertia tensors. That is intentional: the comparison with the PID-error estimator, and later with the momentum observer, is meant to expose exactly this kind of structural difference between methods.
