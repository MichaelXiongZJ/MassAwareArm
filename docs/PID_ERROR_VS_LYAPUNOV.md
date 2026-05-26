# PID-Error vs Lyapunov: Two Quasi-Static Estimators Side by Side

The PID-error estimator and the Lyapunov (spring-sag) estimator look similar from a distance. Both are quasi-static, both wait for the arm to settle, and both rely on a softened proportional gain so that the response to a cube is measurable. But they read very different signals from the controller, and they lean on very different parts of the closed loop. This note explains where the two diverge.

For the underlying math of each estimator, see [LYAPUNOV.md](LYAPUNOV.md) and [PID_ERROR.md](PID_ERROR.md). The intent of this note is to put them next to each other.

## 1. What each one reads

The PID-error estimator reads a **torque** signal. During the weigh hold, it samples the commanded torque at one joint of the controller and asks: how much harder is the controller pushing now than it was with an empty gripper? The difference, divided by gravity and a known moment arm, is the cube's mass.

The Lyapunov estimator reads a **position** signal. During the weigh hold, it samples the joint angles and the end-effector height and asks: how far has the arm sagged compared to its empty pose? The answer goes through an energy calculation to produce the mass.

So the two methods can be thought of as two different sensors built out of the same controller. PID-error uses the controller's commanded torque as a load cell. Lyapunov uses the joint encoders as a dial gauge.

## 2. What each one does to gravity compensation

The two methods configure the arm in opposite ways.

PID-error **turns off gravity compensation at exactly one joint** (the elbow). Without the feedforward, the elbow has to develop a position error so that its proportional term generates the gravity-holding torque on its own. Adding a cube increases the torque the elbow has to produce, and the difference in commanded torque carries the payload signal. The other five joints keep their gravity feedforward intact and hold position essentially perfectly.

Lyapunov **keeps gravity compensation on at every joint**. Without a cube, the arm sits exactly at its target pose with every spring unstretched. When a cube is added, the springs at every joint stretch a little to absorb the new load, and the displacement at the end-effector encodes the mass.

This is the deepest split between the two methods. PID-error keeps the arm essentially at the empty pose (apart from the small error at the measurement joint) and reads the controller's effort. Lyapunov lets the arm move to a new pose and reads the new pose.

## 3. What each one does to the integral gain

The PID-error method relies on a stable steady-state torque at the measurement joint. If the integral term is active and wide-band, it will wind up to drive the position error to zero, which forces the proportional torque to vanish, which destroys the signal. The method therefore implicitly requires either $k_i = 0$ or an integrator slow enough that wind-up cannot finish during the weigh hold. In practice the project ships $k_i = 0$, so this is automatic.

The Lyapunov method requires $k_i = 0$ explicitly. If $k_i$ is non-zero, the arm holds the loaded pose at the same joint angles it held the empty pose, every spring relaxes back to zero stretch, the gripper returns to its original height, and the elastic energy difference is exactly zero. The signal disappears entirely. So the Lyapunov estimator declares $k_i = 0$ as part of its controller override.

Same outcome, different stakes. PID-error tolerates a small non-zero $k_i$ as long as wind-up does not finish during the hold. Lyapunov cannot tolerate non-zero $k_i$ at all, because the structure of its formula assumes pure spring behaviour at every joint.

## 4. Mathematical structure

The PID-error formula is

$$\hat{m} \;=\; \dfrac{\overline{\tau_{\text{cmd}}} \;-\; \tau_{\text{ss,empty}}}{g \, d}.$$

The numerator is a single scalar (the difference between two averaged torque readings at one joint). The denominator is a single scalar (gravity times the moment arm at the weigh pose). Both come straight from the geometry and have no closed-loop dependence: change the gains and the same formula still applies, provided the controller is stable enough to settle.

The Lyapunov formula is

$$\hat{m} \;=\; \dfrac{2 \, \Delta E_{\text{spring}}}{g \, \Delta h},$$

with

$$\Delta E_{\text{spring}} \;=\; \sum_{j=1}^{6} \tfrac{1}{2} k_{p,j} \Big[ (q_{\text{loaded},j} - q_{\text{ref},j})^2 \;-\; (q_{\text{empty},j} - q_{\text{ref},j})^2 \Big].$$

The numerator is a sum over all six joints, weighted by the proportional gain at each joint. The denominator is the drop in end-effector height. The formula depends on $k_p$ explicitly, and on the calibration $q_{\text{empty}}$, $z_{\text{empty}}$.

The factor of 2 in the Lyapunov formula is the non-obvious part. The naive energy balance would say the spring stores everything gravity does, but in a damped step response the damper dissipates exactly the same amount of energy that the spring stores. Skipping that step gives an estimate that is exactly half the truth. PID-error has no analogous trap because it reads the steady-state torque directly, not an energy.

## 5. The strengths

The PID-error method has very few moving parts. There is one measurement (the torque at one joint), one geometric constant (the moment arm), and one calibration scalar (the empty-arm baseline). The formula does not contain the proportional gain at all. The gain only affects how quickly and noisily the joint settles into the steady state, not the value of the steady state itself. So PID-error is robust to gain tuning errors, as long as the controller is stable.

The Lyapunov method needs no estimate of any moment arm. The geometry is already baked into how the multi-dimensional spring interacts through the kinematic chain. The end-effector vertical drop $\Delta h$ is a single forward-kinematics output that the simulator provides directly, with no model inversion needed. If the moment arm at the weigh pose is hard to compute reliably (for example because the payload's centre of mass is offset from the gripper site), Lyapunov sidesteps the problem.

## 6. The weaknesses

The PID-error method is sensitive to the moment arm. A 10% error in $d$ produces a 10% error in $\hat{m}$. In the project's wide-mass sweep, the loaded pose for a heavy cube is slightly different from the empty pose, so the true moment arm at the loaded pose is not exactly the cached value, and the formula picks up a small bias that grows with payload. Recomputing $d$ from forward kinematics at the loaded pose would fix this but adds a step.

The Lyapunov method is sensitive to the proportional gain. A 10% gain calibration error produces a 10% mass error in proportion to that joint's share of the total spring energy. The estimator reads the active gain from the controller config at estimate time to keep this in sync, but any drift between the assumed and the actual gain leaks into the answer. The PID-error formula does not contain $k_p$, so it has no comparable sensitivity.

The Lyapunov method also depends on the rest-to-rest assumption that the factor-of-two derivation relies on. If the arm has not finished settling when the sampling window opens, the spring carries more energy at the moment of measurement than the equilibrium predicts, and the estimate is biased upward. PID-error's averaging works around mild oscillation more cleanly, because mean torque is unbiased by symmetric ringing.

## 7. Two views of the same closed loop

The deepest way to see the two methods is that they read two different signals from a single closed-loop system. PID-error reads the controller's torque output. Lyapunov reads the controller's position error. In a perfectly linear system, these two signals are related by the proportional gain: $\tau = k_p \cdot (\text{position error})$. So in some sense the two methods are reading the same physical quantity at different points in the control loop.

The reason they give different answers in practice is that they are sensitive to different error sources. The torque signal is corrupted by feedforward errors at the other joints and by limit-cycle noise on the measurement joint. The position signal is corrupted by integrator effects, residual oscillation, and uncertainty in the spring constants themselves. Whether one is more accurate than the other depends on which set of error sources dominates in a given configuration, which is exactly what the project's comparison sweep is designed to expose.

## 8. Summary table

| Aspect | PID-Error | Lyapunov |
|---|---|---|
| Signal read | Commanded torque at one joint | Joint angles and end-effector height |
| Gravity compensation | Off at the measurement joint, on elsewhere | On at every joint |
| $k_i$ requirement | Must not wind up during the hold | Strictly zero |
| $k_p$ requirement | Softened only for noise reduction; not in formula | Softened to make sag visible; appears in formula |
| Formula dependence | One moment arm, one baseline torque | Six joint gains, six baseline angles, one baseline height |
| Failure mode | Wrong moment arm at loaded pose | Wrong $k_p$, or arm not at rest when sampled |
| Robust to gain tuning? | Yes (gain absent from formula) | No (10% gain error gives 10% mass error) |
| Carries inertia info? | No | No |
