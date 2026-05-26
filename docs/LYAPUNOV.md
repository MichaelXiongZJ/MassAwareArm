# Lyapunov (Spring-Sag) Mass Estimator

This note explains the spring-sag method as implemented in `software/massaware/estimators/lyapunov.py`. The math is the main subject. The implementation gets one short section near the end. There is also an intuition section up top for readers with no robotics background.

## 1. Intuition

Imagine a person holding a perfectly flat tray out in front of them. With nothing on the tray, the arm muscles keep it level. Now place a coin on the tray. The tray dips down by a tiny amount, just enough that the arm muscles can fight back with a force equal to the coin's weight. If you knew exactly how stiff those arm muscles were, and you measured how far the tray dipped, you could work out the coin's weight without ever putting it on a scale.

The Lyapunov estimator does this with a robot arm. The robot's joint controller, with one feature turned off and another turned down, behaves like a set of springs. With nothing in the gripper, the arm sits exactly where it is told. Once a cube is placed in the gripper, each joint sags slightly because gravity pulls on the cube. By measuring how far the gripper has dropped and how much elastic energy is now stored in the joint springs, the mass of the cube can be recovered.

It is the same idea as a bathroom scale, with the spring distributed across all six joints of the arm rather than packed into a single coil under one footplate.

## 2. The setup

The UR5e arm is held in place by a PID position controller at every joint. PID stands for proportional, integral, and derivative. The proportional term produces a torque proportional to how far the joint has drifted from its target. The derivative term produces a torque proportional to how fast the joint is moving. The integral term accumulates the position error over time and produces a torque that keeps growing until the error vanishes.

For the Lyapunov estimator, the integral term is turned off entirely. With only the proportional and derivative terms active, every joint behaves like a mechanical spring in parallel with a viscous damper:

- The proportional gain $k_p$ acts as a spring stiffness. If the joint drifts an angle $\delta$ away from its target $q_{\text{ref}}$, the controller produces a restoring torque $k_p \delta$, exactly as a Hookean spring stretched by $\delta$ would.
- The derivative gain $k_d$ acts as a damper. Any motion of the joint is opposed by a torque proportional to the joint's angular velocity.
- Gravity compensation is on at every joint. The controller adds a feedforward torque that exactly cancels the joint torque caused by the arm's own weight, so without a payload the arm settles exactly at $q = q_{\text{ref}}$.

With nothing in the gripper, every joint sits at its target and every spring is unstretched. Now attach a cube of mass $m$ to the gripper. Gravity pulls on the cube, the cube pulls on the gripper, and a torque appears at every joint along the kinematic chain. Each joint drifts away from its target by exactly the angle needed for the spring restoring torque to balance the joint torque produced by the cube. The arm settles at a new resting pose $q_{\text{loaded}}$, with the gripper a few millimetres lower than before.

This is the signal the estimator reads. It only needs two things: the six joint angles, and the height of the end-effector site in the world frame. No torque sensors, no force sensors, no contact reading from the cube.

## 3. The naive energy balance and why it is off by a factor of two

There is a natural temptation to balance energies directly. The cube has fallen by

$$\Delta h \;=\; z_{\text{empty}} \,-\, z_{\text{loaded}},$$

so gravity has done work

$$W_{\text{grav}} \;=\; m g \, \Delta h$$

on it. The joint springs have stored elastic energy

$$\Delta E_{\text{spring}} \;=\; \sum_{j=1}^{6} \tfrac{1}{2} k_{p,j} \Big[ (q_{\text{loaded},j} - q_{\text{ref},j})^2 \;-\; (q_{\text{empty},j} - q_{\text{ref},j})^2 \Big].$$

The naive move is to equate the two and solve for the mass:

$$\tilde{m} \;=\; \frac{\Delta E_{\text{spring}}}{g \, \Delta h}. \qquad (\text{wrong by a factor of 2})$$

If you actually run this formula in simulation against a 0.5 kg cube, you get about 0.245 kg. Against a 0.2 kg cube, about 0.10 kg. The ratio is right but every estimate is exactly half the truth.

The reason the naive formula falls short is that it leaves the damper out of the energy book. As the arm drifts from its empty pose to its loaded pose, the joints have a non-zero velocity for a while, and the derivative term of the controller resists that motion, dissipating energy as heat. The correct energy balance across the full transient is

$$W_{\text{grav}} \;=\; \Delta E_{\text{spring}} \;+\; W_{\text{damp}},$$

where $W_{\text{damp}} = \int_0^{\infty} k_d \dot{q}^2 \, dt$ is the energy the damper has absorbed from the moment the cube is placed in the gripper until the arm reaches rest.

For a PD-controlled step response of a spring-damper system from one rest state to another, the dissipated energy has a clean closed form, and remarkably it is equal to the stored spring energy exactly. The argument fits in a few lines.

Consider one joint in isolation. The cube produces a constant disturbance torque $F$ at this joint. The new resting position is where the spring balances $F$, so the sag is

$$\delta \;=\; F / k_p.$$

The work done by the disturbance torque across that sag is force times distance,

$$W_{\text{grav}} \;=\; F \cdot \delta \;=\; F^2 / k_p.$$

The spring, stretched from zero to $\delta$, has stored

$$\Delta E_{\text{spring}} \;=\; \tfrac{1}{2} k_p \delta^2 \;=\; F^2 / (2 k_p).$$

Energy conservation forces the damper to have absorbed everything else gravity did:

$$W_{\text{damp}} \;=\; W_{\text{grav}} - \Delta E_{\text{spring}} \;=\; F^2 / (2 k_p) \;=\; \Delta E_{\text{spring}}.$$

So half the work done by gravity ends up as elastic energy in the spring, and the other half is dissipated by the damper. This split holds for any positive damping, whether the joint response is underdamped, critically damped, or overdamped, as long as the joint actually starts at rest and finishes at rest. The argument generalises directly to a multi-joint system because the work done on each joint by its share of the gravity load splits the same way at each joint.

The correct mass formula therefore has a factor of two:

$$\boxed{\; \hat{m} \;=\; \dfrac{2 \, \Delta E_{\text{spring}}}{g \, \Delta h}. \;}$$

## 4. Why the integral gain has to be zero

If the integral term were left on, the picture changes completely. At steady state, the integral term winds up to provide exactly the torque needed to drive the position error to zero. The arm holds the loaded pose at the same joint angles it held the empty pose, every spring relaxes back to zero stretch, the gripper returns to its original height, and the elastic energy difference vanishes. There is no signal left for the estimator to read.

So setting $k_i = 0$ during the weigh phase is not a tuning choice. It is a structural requirement of the method. The base controller in this project happens to ship with $k_i = 0$ already, so the override is a no-op today, but the estimator still requests it explicitly so that the requirement is documented at the point of need, and so the system stays correct if the base controller is ever tuned with an integral term in the future.

## 5. Why the proportional gain is softened

At the UR5e's tuned position-control stiffness, $k_p$ is around 2000 N·m/rad on the three proximal joints (shoulder pan, shoulder lift, elbow). For a 0.2 kg cube the per-joint sag at the elbow is on the order of

$$\delta \;=\; \frac{m g d}{k_p},$$

where $d$ is the moment arm from the joint axis to the cube. Plugging numbers in gives sags on the order of fractions of a milliradian, and an end-effector drop of well under a millimetre. At that scale, the signal is buried under floating-point noise in the equilibrium computation, leftover oscillation from the move into the weigh pose, and quasi-static drift in the controller.

The estimator therefore softens $k_p$ by a factor of ten during weighing. The sag grows into the easily measurable centimetre range, the numerical noise floor stays the same, and the signal-to-noise ratio improves by roughly the same factor. The factor-of-two relationship between $W_{\text{grav}}$ and $\Delta E_{\text{spring}}$ is independent of the stiffness, so the formula above continues to hold. Only the variance of the estimate improves.

The derivative gain is softened by half as much (a factor of two rather than ten). This keeps the closed-loop damping ratio in roughly the same regime as before, so the joint still reaches a clean rest within the sampling window. Softening $k_d$ by the same factor as $k_p$ would push the loop into a strongly underdamped regime, which would violate the rest-to-rest assumption that the factor-of-two derivation depends on.

## 6. Calibration

The estimator needs an empty-arm baseline. At initialization, with no cube in the gripper, the planner moves the arm to the weigh pose, applies the same soft gains and the same gravity mask that will be used at weigh time, lets the arm settle, and averages the joint angles and the end-effector height over a three-second hold. The averaged values $q_{\text{empty}}$ and $z_{\text{empty}}$ are written to `configs/calibration.yaml`, keyed by the weigh pose.

Calibrating with the same configuration that will be used during the loaded measurement is important. Any systematic offset that looks the same in both cases drops out of the difference $\Delta E_{\text{spring}}$. Small modelling errors in the gravity feedforward, leftover integrator state from the controller's discretisation, residual quasi-static drift: all of them cancel as long as the empty and loaded configurations match. What remains in the energy difference is the part caused by the cube alone.

## 7. Implementation notes

The estimator is about 170 lines of Python in `software/massaware/estimators/lyapunov.py`. It conforms to the project's `Estimator` interface, which has four hooks: `reset`, `update`, `estimate`, plus the calibration trio. At every physics tick during the weigh hold, `update()` appends one sample of $(q, z_{\text{ee}})$ to a buffer. After the hold completes, `estimate()` averages the buffer and applies the formula above.

Two estimator hooks shape the controller while the estimator is active. `gravity_comp_mask()` returns a vector of ones, meaning gravity compensation stays on at every joint. `controller_overrides()` returns soft $k_p$ and $k_d$ alongside zero $k_i$, which the planner applies to the PID controller before the weigh hold and clears once the hold ends.

Calibration uses the same machinery with a separate sample buffer, so the empty-arm hold and the loaded-arm hold do not interfere with each other.

## 8. Sensitivities and caveats

The factor-of-two relationship depends on the start and end states both being at rest. If the arm has not finished settling when the sampling window opens, the spring carries more energy at the moment of measurement than the equilibrium analysis predicts, and the estimate is biased upward. The planner runs a settle step before sampling to make this unlikely, but if the closed-loop damping is too low, residual oscillation will still leak into the answer.

The estimator is sensitive to the value of $k_p$ used during calibration and weighing. A 10% error in any single $k_{p,j}$ translates directly to a 10% error in that joint's contribution to $\Delta E_{\text{spring}}$, and propagates into $\hat{m}$ in proportion to that joint's share of the total energy. The estimator therefore reads the active gains from the runtime configuration rather than hardcoding them.

Finally, the method is purely quasi-static. It carries no information about the cube's inertia. Two cubes of the same mass but very different inertia tensors would return the same estimate. That is by design. The contrast with the momentum-observer method, which does carry dynamic information, is exactly the kind of structural difference between estimators that this project's case study is meant to expose.
