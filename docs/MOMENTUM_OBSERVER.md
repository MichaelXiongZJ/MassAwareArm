# Momentum Observer Mass Estimator

This note explains the momentum-observer method as implemented in `software/massaware/estimators/momentum_observer.py`. The structure of the derivation, the role of the gain $K_O$, the projection from joint-space residual to scalar mass, and the conditions under which the static-pose approximation in the code is acceptable are all covered below.

## 1. Setting up the residual

Write the equations of motion of the arm with an unknown joint-space disturbance $\tau_{\text{ext}}$ produced by the payload:

$$
M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q} + g(q) \;=\; \tau \;+\; \tau_{\text{ext}}.
$$

Define the generalized momentum $p(t) = M(q)\,\dot{q}$. Differentiating and substituting the dynamics gives

$$
\dot{p} \;=\; \dot{M}\,\dot{q} + M\,\ddot{q} \;=\; \dot{M}\,\dot{q} + \tau + \tau_{\text{ext}} - C\,\dot{q} - g.
$$

The key identity, which carries the geometry of rigid-body dynamics, is $\dot{M} = C + C^{\top}$ (the matrix $\dot{M} - 2C$ is skew-symmetric for the standard choice of $C$). Substituting and simplifying:

$$
\dot{p} \;=\; \tau + \tau_{\text{ext}} + C^{\top}\,\dot{q} - g.
$$

The observer's job is to estimate $\tau_{\text{ext}}$ given only $\tau$, $q$, $\dot{q}$, $M$, $C$, and $g$. The trick is to build a predicted momentum $\hat{p}$ whose dynamics mirror the real ones but with the unknown $\tau_{\text{ext}}$ replaced by a feedback signal $r$. Choose

$$
\dot{\hat{p}} \;=\; \tau + r + C^{\top}\,\dot{q} - g, \qquad \hat{p}(0) = p(0),
$$

and let

$$
r(t) \;=\; K_O \bigl( p(t) - \hat{p}(t) \bigr).
$$

Differentiating $r$ and using the two expressions for $\dot{p}$ and $\dot{\hat{p}}$ kills every term except the disturbance and the feedback:

$$
\dot{r} \;=\; K_O \bigl( \dot{p} - \dot{\hat{p}} \bigr) \;=\; K_O \bigl( \tau_{\text{ext}} - r \bigr).
$$

This is the central result. The residual $r$ is a first-order low-pass filter on $\tau_{\text{ext}}$ with cutoff $K_O$. There is no algebraic relation that requires knowing $\dot{q}$ or the inertia matrix outside of forming $p$ and $\hat{p}$. There is no acceleration $\ddot{q}$ anywhere, which is the whole point of the construction: $\ddot{q}$ is noisy and would need to be differentiated from $\dot{q}$ if used directly.

## 2. The static-pose approximation

The clean derivation above requires the Coriolis-transpose product $C^{\top}\,\dot{q}$. MuJoCo does not expose $C$ directly. It does report $q_{\text{frc,bias}} = C\,\dot{q} + g$, which mixes the Coriolis term with gravity.

During the WEIGH hold the arm has reached a settled equilibrium. The joint velocity $\dot{q}$ is on the order of a few millirad/s after settle, so $C\,\dot{q}$ is at least three orders of magnitude smaller than $g$ and effectively negligible. Under this approximation $C^{\top}\dot{q} \approx 0$ and $g \approx q_{\text{frc,bias}}$, which reduces the observer to

$$
\dot{\hat{p}} \;=\; \tau + r - q_{\text{frc,bias}}, \qquad r = K_O\,(p - \hat{p}).
$$

This is the form implemented in `update()`. The static-pose approximation is what makes the implementation simple. It is also what limits the observer to weighing at rest, and is the first thing to revisit if a later phase wants to weigh during the lift trajectory. In that case the full $C^{\top}\dot{q}$ has to be computed, either by extracting $C$ via a finite difference on the bias term or by switching to a formulation that does not need $C$ at all.

## 3. Discrete-time recursion

Forward Euler on $\hat{p}$, followed by an algebraic evaluation of $r$, gives the loop in code:

$$
\hat{p}_{k+1} \;=\; \hat{p}_k + \Delta t\,\bigl(\tau_k + r_k - q_{\text{frc,bias},k}\bigr),
\qquad
r_{k+1} \;=\; K_O\,\bigl(p_{k+1} - \hat{p}_{k+1}\bigr).
$$

Seeds: $\hat{p}_0 = p_0$ (so the observer starts with zero prediction error) and $r_0 = 0$.

With the project's default $\Delta t = 2$ ms and $K_O = 50$ s$^{-1}$, the dimensionless step $K_O\Delta t = 0.1$. Forward Euler stability for a first-order low-pass requires $K_O\Delta t < 2$, which leaves a generous margin. The continuous time constant of the loop is $1/K_O = 20$ ms. After roughly three time constants, about 60 ms, the residual has reached $1 - e^{-3} \approx 95\%$ of its steady-state value. Five time constants, 100 ms, brings it within $1\%$. The default `burn_in_s = 0.5` accordingly throws away the first 250 time constants of transient, which is more than enough.

## 4. From joint-space residual to scalar mass

The converged residual $\hat{r}$ is an estimate of the joint-space disturbance produced by the payload. For a point mass attached at the end-effector, the only Cartesian wrench is the gravity force $F_{\text{ext}} = -m g \hat{z}$, and the joint-space pull-back through the linear Jacobian $J_v(q)$ is

$$
\tau_{\text{ext}} \;=\; J_v^{\top}(q)\,F_{\text{ext}} \;=\; -m\,g\,J_{v,z}^{\top}(q),
$$

where $J_{v,z}$ is the third row of $J_v$, namely the vector of partial derivatives $\partial z_{\text{EE}}/\partial q_j$. The converged observer therefore satisfies

$$
\hat{r} \;\approx\; -m\,g\,J_{v,z}^{\top}(q_{\text{weigh}}).
$$

This is six equations in one unknown. The natural extraction is a least-squares projection of $\hat{r}$ onto the known direction $-g\,J_{v,z}^{\top}$. Subtracting the empty-arm baseline $\hat{r}_{\text{empty}}$ to cancel constant model error and any disturbance bias, the closed form is

$$
\boxed{\;\hat{m} \;=\; -\frac{\bigl\langle \hat{r} - \hat{r}_{\text{empty}},\; J_{v,z} \bigr\rangle}{g\,\|J_{v,z}\|^2}.\;}
$$

Per-tick sample scatter in $\hat{r}$ propagates through the projection to give an uncertainty estimate $\sigma_{\hat{m}}$, which the code reports alongside the point estimate.

## 5. Why calibration is still required

Although the observer construction is "controller-agnostic" in the sense that it does not depend on the PD gains, in practice the static-pose approximation leaves residual error in the empty arm that does not vanish. Three sources matter:

1. **Model error in $g(q)$.** MuJoCo's `qfrc_bias` is computed from the model's inertial parameters, which approximate the real link masses and centers of mass. Any mismatch shows up as a constant bias in $\hat{r}$ at the weigh pose.
2. **Discretization and integrator drift.** The Forward Euler step introduces $O(\Delta t)$ error in $\hat{p}$ that does not average to zero.
3. **Residual $C\,\dot{q}$.** Even after settling, the joint velocity is not exactly zero. The dropped Coriolis term contributes a tiny offset.

The empty-arm calibration captures all three of these in $\hat{r}_{\text{empty}}$ and subtracts them at estimate time. The result is a clean separation between the systematic offset that the observer cannot avoid and the payload-induced part that it is supposed to measure.

## 6. Sensitivities and caveats

Three things deserve attention.

**Choice of $K_O$.** Smaller $K_O$ gives slower convergence but more aggressive low-pass filtering, which suppresses tick-by-tick noise in $\hat{r}$. Larger $K_O$ converges within a handful of milliseconds but exposes the projection to high-frequency content in $p$ and $\hat{p}$. The default of $50$ s$^{-1}$ is a compromise that produces a clean residual within the project's 1 s weigh hold. A future tuning pass might select per-joint gains: the shoulder lift and the elbow carry most of the signal here, so they could afford a smaller $K_O$ than the wrist joints if noise becomes the dominant error source.

**Degenerate poses.** The projection denominator $\|J_{v,z}\|^2$ is the sum of squared partials of EE height with respect to each joint angle. If the weigh pose happens to be such that vertical motion of the end-effector is impossible for any single-joint perturbation, this denominator collapses and the estimator is structurally blind. The current weigh pose has $\|J_{v,z}\|^2$ of order $0.4$, well away from the degenerate region. A guard in the code returns $\hat{m} = 0$ with $\sigma = \infty$ if the denominator ever drops below numerical tolerance.

**Quasi-static assumption.** As noted in Section 2, the observer in this project's static form is meaningful only when the arm has settled. It is not the right tool for an arm in motion. The De Luca form does support mid-trajectory weighing in principle, but doing so requires the full $C^{\top}\dot{q}$ term and an awareness of which time windows along a trajectory have well-conditioned Jacobians. That is a stretch goal, not a property of the current implementation.

The momentum observer rounds out the set of estimators by carrying information that the static methods cannot: the time profile of the residual is itself a diagnostic that the other two methods do not produce. Even when all three give compatible mass estimates, the convergence profile of $\hat{r}(t)$ during the WEIGH hold gives an independent check on the observer's internal consistency.
