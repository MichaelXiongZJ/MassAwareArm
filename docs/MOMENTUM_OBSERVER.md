# Momentum Observer Mass Estimator

This note explains the momentum-observer method as implemented in `software/massaware/estimators/momentum_observer.py`. The math is the main subject. The implementation gets one short section near the end. There is also an intuition section up top for readers with no robotics background.

## 1. Intuition

The two static estimators in this project, PID-error and Lyapunov, both wait for the arm to come to rest and then read off something about the rest state. The momentum observer takes a different angle. It watches how much momentum the arm has built up over time, compares that against how much momentum the gravity model and the controller's commands would have produced on their own, and attributes the discrepancy to an unmodelled load on the gripper.

A loose analogy. Imagine pushing a shopping cart along a flat aisle. You know how hard you are pushing. You know how much the cart should accelerate per unit push, because you know roughly how heavy an empty cart is. If you push for a few seconds and the cart accelerates less than expected, there must be something extra in the cart that you did not account for. The "extra" is the payload. The momentum observer formalises this idea for a robot arm: it computes how much momentum the arm should have built up given the commanded torques and the gravity model, compares that with the momentum it actually has, and the gap is the joint-space signature of the cube.

The non-obvious part is that this measurement does not require knowing the arm's acceleration. Acceleration is hard to compute cleanly from sensor data because differentiating velocity amplifies noise. By tracking momentum, which depends only on positions and velocities, the math stays in terms of clean signals throughout.

## 2. Setting up the residual

The equations of motion of the arm with an unknown payload contributing a joint-space disturbance $\tau_{\text{ext}}$ are

$$M(q) \, \ddot{q} \;+\; C(q, \dot{q}) \, \dot{q} \;+\; g(q) \;=\; \tau \;+\; \tau_{\text{ext}}.$$

The terms, one at a time:
- $q$, $\dot{q}$, $\ddot{q}$ are the six-component vectors of joint angle, velocity, and acceleration.
- $M(q)$ is the joint-space mass matrix. It is the rotational analogue of mass for an articulated body, and it depends on the current pose because the moment of inertia about each joint axis changes as the geometry of the chain changes.
- $C(q, \dot{q}) \dot{q}$ is the Coriolis and centrifugal term. It accounts for the fact that when several joints rotate at once, each one feels an apparent torque from the others.
- $g(q)$ is the joint-space gravity torque. It is the torque the controller has to provide just to hold the arm in place against gravity.
- $\tau$ is the torque the controller is actually commanding.
- $\tau_{\text{ext}}$ is the unknown joint-space torque produced by the payload. The cube's mass is hidden inside this term.

Define the generalized momentum:

$$p(t) \;=\; M(q(t)) \, \dot{q}(t).$$

This is a six-vector quantity, one component per joint. Differentiating with respect to time and using the equations of motion to eliminate $\ddot{q}$:

$$\dot{p} \;=\; \dot{M} \dot{q} + M \ddot{q} \;=\; \dot{M} \dot{q} + \tau + \tau_{\text{ext}} - C \dot{q} - g.$$

A central identity of rigid-body dynamics is that the matrix $\dot{M} - 2C$ is skew-symmetric, for the standard convention on how $C$ is constructed from $M$. This is equivalent to writing $\dot{M} = C + C^{\top}$. Substituting:

$$\dot{p} \;=\; \tau + \tau_{\text{ext}} + C^{\top} \dot{q} - g.$$

The crucial feature of this expression is that the acceleration $\ddot{q}$ has disappeared. The right-hand side depends only on $q$, $\dot{q}$, the commanded torque $\tau$, and the model quantities $C$ and $g$. This is what makes the observer noise-tolerant: at no point does anyone have to differentiate a velocity signal.

The observer's job is to estimate $\tau_{\text{ext}}$ given everything else. The trick is to build a predicted momentum $\hat{p}$ whose dynamics mirror the real ones, with the unknown $\tau_{\text{ext}}$ replaced by a feedback signal $r$ that the observer constructs:

$$\dot{\hat{p}} \;=\; \tau + r + C^{\top} \dot{q} - g, \qquad \hat{p}(0) = p(0).$$

The feedback signal is just a constant gain times the gap between true and predicted momentum:

$$r(t) \;=\; K_O \bigl( p(t) - \hat{p}(t) \bigr).$$

Subtracting the equation for $\dot{\hat{p}}$ from the equation for $\dot{p}$ kills every term except the disturbance and the feedback:

$$\dot{p} \;-\; \dot{\hat{p}} \;=\; \tau_{\text{ext}} - r.$$

Differentiating $r$ in time and substituting:

$$\dot{r} \;=\; K_O \bigl( \dot{p} - \dot{\hat{p}} \bigr) \;=\; K_O \bigl( \tau_{\text{ext}} - r \bigr).$$

This is the central result. The residual $r$ is a first-order low-pass filter on the unknown disturbance with cutoff $K_O$. For a constant external torque, which is what a static payload produces, $r$ converges exponentially to $\tau_{\text{ext}}$ with time constant $1/K_O$. After three time constants $r$ is within 5% of the true disturbance; after five, within 1%.

The construction is sometimes called the De Luca observer after Alessandro De Luca, who developed this form. It has been the workhorse of disturbance estimation in robotics for two decades precisely because it never asks for $\ddot{q}$.

## 3. Working with MuJoCo's `qfrc_bias` (no explicit $C$ needed)

The clean derivation above requires the Coriolis-transpose product $C^{\top} \dot{q}$. The MuJoCo simulator does not expose the matrix $C$ directly. It does report the combined quantity `qfrc_bias`, which equals $C \dot{q} + g$. The Coriolis term and the gravity term are mixed together in this single vector.

Fortunately the observer never needs $C$ in isolation. Substituting $\dot{M} = C + C^{\top}$ into $\dot{p} = \tau + \tau_{\text{ext}} + C^{\top}\dot{q} - g$ gives

$$\dot{p} \;=\; \tau + \tau_{\text{ext}} + \dot{M}\dot{q} - \bigl(C\dot{q} + g\bigr) \;=\; \tau + \tau_{\text{ext}} + \dot{M}\dot{q} - \text{qfrc\_bias},$$

so the observer becomes

$$\dot{\hat{p}} \;=\; \tau + r + \dot{M}\dot{q} - \text{qfrc\_bias}, \qquad r \;=\; K_O \, (p - \hat{p}).$$

The $\dot{M}\dot{q}$ term is recovered per tick by finite-differencing the mass matrix, which the observation interface already provides every tick — no extra simulation queries. This makes the recursion exact during motion, which is what enables weighing during the lift trajectory instead of only at a settled hold. At rest $M$ is constant, the term vanishes, and the recursion reduces to the simpler static-pose form $\dot{\hat{p}} = \tau + r - \text{qfrc\_bias}$ that earlier versions of this estimator used exclusively.

## 4. Discrete-time recursion

In code, time advances in discrete steps of length $\Delta t$. Applying forward Euler integration to $\hat{p}$, followed by the algebraic evaluation of $r$:

$$\hat{p}_{k+1} \;=\; \hat{p}_k + \Delta t \, \bigl( \tau_k + r_k - \text{qfrc\_bias}_k \bigr) + \bigl( M_k - M_{k-1} \bigr)\dot{q}_k,$$

$$r_{k+1} \;=\; K_O \, \bigl( p_{k+1} - \hat{p}_{k+1} \bigr).$$

The last term of the $\hat{p}$ update is the finite-difference form of $\dot{M}\dot{q}\,\Delta t$ (the $\Delta t$ cancels); at a stationary hold it is identically zero.

The seeds are $\hat{p}_0 = p_0$ and $r_0 = 0$. Starting with $\hat{p}$ exactly equal to $p$ means the prediction error is zero at the first sample, so the residual builds up only as the dynamics evolve.

With the project's default time step $\Delta t = 2$ ms and observer gain $K_O = 50$ s$^{-1}$, the dimensionless step is $K_O \Delta t = 0.1$. Forward-Euler stability for a first-order low-pass requires this dimensionless step to stay below 2, so the current choice has a generous margin. The continuous time constant of the loop is $1/K_O = 20$ ms. After three time constants (60 ms), the residual has reached about 95% of its steady-state value. After five (100 ms), 99%. The default burn-in window of 500 ms therefore discards roughly 25 time constants of transient before averaging starts, far more than enough.

## 5. From the joint-space residual to a scalar mass

The converged residual $\hat{r}$ is an estimate of the joint-space disturbance produced by the payload. For a point mass at the end-effector, the only Cartesian force exerted on the arm by the payload is gravity acting on the cube:

$$F_{\text{ext}} \;=\; -m g \hat{z},$$

where $\hat{z}$ is the world vertical unit vector. The joint-space torque corresponding to this Cartesian force is the pull-back through the linear Jacobian:

$$\tau_{\text{ext}} \;=\; J_v^{\top}(q) \, F_{\text{ext}} \;=\; -m g \, J_{v,z}(q),$$

where $J_{v,z}(q)$ is the third row of the linear Jacobian, that is, the six-vector of partial derivatives of end-effector height with respect to each joint angle:

$$J_{v,z} \;=\; \left( \dfrac{\partial z_{\text{ee}}}{\partial q_1}, \dfrac{\partial z_{\text{ee}}}{\partial q_2}, \ldots, \dfrac{\partial z_{\text{ee}}}{\partial q_6} \right).$$

In words, each component of $J_{v,z}$ tells you how much the end-effector rises or falls per unit rotation of the corresponding joint. The converged observer therefore approximately satisfies

$$\hat{r} \;\approx\; -m g \, J_{v,z}(q_{\text{weigh}}).$$

This is six equations in one unknown. The natural way to extract $m$ is by least squares: project $\hat{r}$ onto the known direction $-g J_{v,z}$. After subtracting the empty-arm baseline $\hat{r}_{\text{empty}}$ to cancel any constant model error and any small disturbance bias, the closed form is

$$\boxed{\; \hat{m} \;=\; -\dfrac{\bigl\langle \hat{r} - \hat{r}_{\text{empty}}, \; J_{v,z} \bigr\rangle}{g \, \|J_{v,z}\|^2}. \;}$$

The angle brackets denote the dot product, summed over the six joint components. The denominator is just the squared norm of $J_{v,z}$, a single scalar.

When the pose changes during sampling (weighing during the lift motion), $J_{v,z}$ is no longer a constant, so the code solves the equivalent one-parameter least squares **sample-wise** rather than projecting the window means:

$$\hat{m} \;=\; -\dfrac{\sum_k \bigl\langle r_k - \hat{r}_{\text{empty}}, \; \Phi_k \bigr\rangle}{g \sum_k \|\Phi_k\|^2},$$

where $\Phi_k$ is $J_{v,z}$ passed through the same first-order low-pass as $r$ (gain $K_O$), so the regressor carries the same lag as the residual it is fitted against. Samples with negligible leverage $\|\Phi_k\|^2$ are skipped. At a fixed pose $\Phi \to J_{v,z}$ and the sum collapses to the boxed projection above.

The per-tick scatter of the per-sample mass solutions yields an uncertainty estimate $\sigma_{\hat{m}}$, which the code reports alongside the point estimate.

## 6. Why calibration is still required

The momentum observer is often described as "controller-agnostic" because it does not depend on the PD gains, on the integral state, or on any other controller-side knob. That is true of the structural derivation, but in practice the static-pose approximation leaves a residual error in the empty arm that does not vanish on its own. Three sources matter:

1. **Model error in $g(q)$.** The gravity bias `qfrc_bias` is computed from the model's inertial parameters, which approximate the real link masses and centres of mass. Any mismatch shows up as a constant additive offset in $\hat{r}$ at the weigh pose.

2. **Discretisation drift.** Forward Euler on $\hat{p}$ introduces an $O(\Delta t)$ error per step that does not average to zero over time. With $\Delta t = 2$ ms and a settled residual, the cumulative effect is small but non-zero, and it appears as a constant bias.

3. **Residual Coriolis term.** Even after settling, the joint velocities are not exactly zero. The dropped $C \dot{q}$ term contributes a small offset to the bias.

The empty-arm calibration captures all three of these in a single vector $\hat{r}_{\text{empty}}$ and subtracts them at estimate time. The result is a clean separation between the systematic offset the observer cannot avoid and the payload-induced part it is supposed to measure.

## 7. Implementation notes

The estimator is about 200 lines of Python in `software/massaware/estimators/momentum_observer.py`. Unlike the other two estimators, it carries internal state across ticks: the predicted momentum $\hat{p}$ and the residual $r$. `reset()` clears this state at the start of each weigh hold. `update()` advances the discrete recursion by one tick and, after the burn-in window has elapsed, appends one sample of $(\hat{r}, J_{v,z})$ to the buffer. `estimate()` averages the buffer and applies the projection above.

The observation interface gives the estimator everything it needs each tick: $q$, $\dot{q}$, the commanded torque, the gravity bias, the mass matrix $M$, and the Jacobian at the end-effector. Forming $p$ is a single matrix-vector product. The projection step at the end requires no extra simulation calls.

Unlike PID-error and Lyapunov, this estimator does not need any controller overrides. The base PID is fine. `gravity_comp_mask()` returns all ones (full gravity compensation everywhere), and `controller_overrides()` returns an empty dict. The cube simply hangs in the gripper while the base controller holds the weigh pose, and the observer integrates the dynamics in parallel.

## 8. Sensitivities and caveats

Three things deserve attention.

**Choice of $K_O$.** A smaller $K_O$ gives slower convergence but more aggressive low-pass filtering, which suppresses tick-by-tick noise in $\hat{r}$. A larger $K_O$ converges within a handful of milliseconds but exposes the projection to high-frequency content in $p$ and $\hat{p}$. The default of 50 s$^{-1}$ is a compromise that produces a clean residual within the project's one-second weigh hold. A future tuning pass might select per-joint gains. The shoulder lift and the elbow carry most of the signal at the current weigh pose, so they could afford a smaller $K_O$ than the wrist joints if wrist-channel noise ever became the dominant error source.

**Degenerate poses.** The projection denominator $\|J_{v,z}\|^2$ is the sum of the squared partial derivatives of end-effector height with respect to each joint angle. If the weigh pose were chosen so that no single-joint perturbation moved the end-effector vertically (for example, with the arm fully extended along a horizontal line), this denominator would collapse and the estimator would be structurally blind to the payload. The current weigh pose has $\|J_{v,z}\|^2$ on the order of 0.4, well away from the degenerate region. The code guards against the edge case anyway: if the denominator drops below numerical tolerance, the estimator returns $\hat{m} = 0$ with $\sigma = \infty$ rather than dividing by something close to zero.

**Weighing in motion.** With the $\dot{M}\dot{q}$ correction (section 3) and the sample-wise fit (section 5), the observer supports mid-trajectory weighing: in the tracking pipeline (`mission_tracking.py --collect-lift-samples`) it recovers the payload to sub-gram accuracy while sampling only during the lift segment. Two caveats remain. First, the empty-arm baseline $\hat{r}_{\text{empty}}$ is captured at a fixed pose, and the gripper-linkage bias it absorbs is mildly pose-dependent — see [MOMENTUM_PAUSE_FREE_PLAN.md](MOMENTUM_PAUSE_FREE_PLAN.md). Second, the tracking pipeline models the payload as a pure gravity force at the end-effector; for a *physically grasped* payload in motion, the regressor must be extended with the payload's inertial reaction, $J_v^{\top}(a_{\text{ee}} + g\hat{z})$, which is a planned follow-up rather than a current property.

The momentum observer's distinctive contribution to the case-study comparison is that the time profile of $\hat{r}(t)$ is itself a diagnostic that the other two methods do not produce. Even when all three estimators give compatible mass estimates, watching the residual converge during the weigh hold provides an independent check on the observer's internal consistency. The static methods, which only read the final equilibrium, cannot offer the same.
