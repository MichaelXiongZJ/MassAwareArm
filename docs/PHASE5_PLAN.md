# Phase 5 — Calibration + Baseline Estimators (PID-Error & Lyapunov)

> **Goal:** Add the PID-error and Lyapunov mass estimators, proving the estimator-swap architecture works. After this phase, changing `estimator: pid_error` or `estimator: lyapunov` in config is the *only* step needed to switch estimation methods.

## Current State

| What | Status |
|---|---|
| Phases 0–3 (sim, controller, perception, bare FSM) | ✅ Done |
| Sub-phase A (Config migration) | ✅ Done |
| Phase 4 shared infra (`base.py`, `registry.py`, WEIGH states) | ❌ Not started |
| Phase 4 inverse-dynamics estimator | ❌ Not started (another team member) |
| Estimator hook in `tick_loop.py` | Commented-out stub (line 86–87) |
| FSM flow | `SEARCH → GRASP → PLACE → HOME` (no weighing) |

Because the shared estimator infrastructure doesn't exist yet, this plan includes it as a prerequisite sub-phase. Each sub-phase is an independent, reviewable code push.

---

## Sub-phases Overview

```
Sub-phase A   Config migration (HOME_QPOS, WEIGH_POSE, bin drops → YAML) [DONE]
     │
Sub-phase B   Shared estimator infrastructure (base, registry, FSM states,
              tick_loop hook, per-estimator controller config, auto-calibration in INIT)
     │
Sub-phase C   PID-error estimator (measurement joint = elbow)
     │
Sub-phase D   Lyapunov (Spring-Sag) estimator (uses per-estimator ki=0 override)
     │
Sub-phase E   Integration verification
```

> **See also:** [`PINNED_ISSUES.md`](PINNED_ISSUES.md) for items deliberately deferred
> (currently: `moment_arm` derivation, WEIGH transient handling).

---

## Sub-phase B — Shared Estimator Infrastructure

**Goal:** Build everything that *any* estimator needs — the ABC (with controller-config and gravity-mask hooks), registry, classifier, FSM states (`INIT`, `WEIGH`, `CLASSIFY`), the tick_loop hook, and auto-calibration triggered from `INIT`. After this, a teammate can drop in `inverse_dynamics.py`, register it, and the pipeline works.

### Key design decisions captured in this sub-phase

1. **Controller config is per-estimator.** The active estimator declares the PID gains overrides it needs (currently only Lyapunov overrides `ki → 0`). The base controller config in YAML is the default; estimator-specific overrides are applied at estimator-build time.
2. **Gravity-comp mask is per-estimator.** The active estimator declares which joints should be uncompensated during WEIGH. `WeighState` reads this from the estimator on `enter()` and restores all-ones on `exit()`. No `weighing` boolean is needed.
3. **Calibration is automatic, run inside `INIT`.** No standalone `scripts/calibrate.py`. If the active estimator requires calibration (`estimator.requires_calibration` is True) and no cached `configs/calibration.yaml` exists (or it is stale), `InitState` runs the calibration routine in-sim before transitioning to `SEARCH`. Cached calibration is reused across runs.
4. **Fail-fast on missing calibration:** if the estimator requires calibration and calibration cannot be produced (e.g. sim error), `INIT` transitions to `ERROR`, not `SEARCH`.

### [NEW] `massaware/estimators/base.py`

Exactly as specified in ARCHITECTURE.md §3:

```python
class Estimator(ABC):
    name: str

    # --- Declarations the planner / controller read ---
    requires_calibration: bool = False
    # Per-joint mask applied during WEIGH; 1.0 = gravity comp ON, 0.0 = OFF.
    # Default = all-ones (full gravity comp). Estimators override as needed.
    def gravity_comp_mask(self, n_joints: int) -> np.ndarray:
        return np.ones(n_joints)
    # PID gain overrides applied while this estimator is active.
    # Return {} for "no override". Lyapunov returns {"ki": np.zeros(n_joints)}.
    def controller_overrides(self) -> dict:
        return {}

    # --- Lifecycle ---
    def reset(self) -> None: ...
    def update(self, obs: EstimatorObs) -> None: ...
    def estimate(self) -> EstimateResult: ...

    # --- Optional: estimators with requires_calibration=True implement this ---
    def calibrate(self, ctx: "PlannerContext") -> dict:
        """Run an in-sim calibration routine and return a dict to be saved as
        configs/calibration.yaml. Called by InitState if needed."""
        raise NotImplementedError

@dataclass
class EstimatorObs:
    t: float
    q: np.ndarray
    q_dot: np.ndarray
    tau_cmd: np.ndarray
    tau_meas: np.ndarray
    qfrc_bias: np.ndarray
    jacobian_ee: np.ndarray    # 3×6
    q_ref: np.ndarray

@dataclass
class EstimateResult:
    m_hat: float
    sigma: float | None
    diagnostics: dict
```

### [NEW] `massaware/estimators/registry.py`

```python
_REGISTRY: dict[str, type[Estimator]] = {}

def register(name: str, cls: type[Estimator]) -> None: ...
def build(name: str, cfg: dict) -> Estimator: ...
```

Estimator files self-register at import time (e.g. `register("pid_error", PIDErrorEstimator)`).

### [NEW] `massaware/classify.py`

```python
def classify_mass(m_hat: float, threshold: float) -> str:
    """Return 'heavy' if m_hat >= threshold, else 'light'."""
```

### [MODIFY] `massaware/planner.py`

Add three new FSM states. Existing states (`SEARCH`, `GRASP`, `PLACE`, `HOME`) remain unchanged except for one routing tweak in `GraspState`.

#### `InitState` (now responsible for calibration)
- If `ctx.estimator is None` → transition straight to `SEARCH` (backward compat with Phase 3).
- If `ctx.estimator.requires_calibration is False` → transition to `SEARCH`.
- Otherwise:
  1. Try to load `configs/calibration.yaml`.
  2. If the file exists and its `weigh_qpos_used` matches the current `ctx.weigh_qpos` (within tolerance), reuse it. Store on `ctx.calibration`.
  3. Otherwise call `ctx.estimator.calibrate(ctx)`, save the returned dict to `configs/calibration.yaml`, and store on `ctx.calibration`.
  4. On any failure during calibration, transition to `ERROR`.
  5. On success, transition to `SEARCH`.

> **Why move calibration into INIT:** removes the manual `scripts/calibrate.py` step from the workflow. Calibration is regenerated whenever `weigh_qpos` changes. Estimators that don't need calibration (e.g. inverse-dynamics) skip the step automatically via the `requires_calibration` flag.

#### `WeighState`
- On `enter()`: apply the estimator's gravity-comp mask via `ctx.gravity_comp_mask = ctx.estimator.gravity_comp_mask(n_joints)`. Apply controller overrides (e.g. zero out `ki`) via `controller.apply_overrides(ctx.estimator.controller_overrides())`.
- Moves arm to `ctx.weigh_qpos`.
- Calls `settle()` to wait for velocity to drop.
- After a configurable hold duration (`ctx.weigh_hold_s`), calls `estimator.estimate()` and stores the result on `ctx.estimate_result`.
- On `exit()`: restore `ctx.gravity_comp_mask = np.ones(n_joints)` and revert controller overrides (`controller.clear_overrides()`).
- Transitions to `CLASSIFY`.

> **Transient handling:** see [`PINNED_ISSUES.md` P2](PINNED_ISSUES.md). For now we rely on `settle()` alone; if variance is high we will add a post-settle skip window.

#### `ClassifyState`
- Reads the `EstimateResult` from context.
- Calls `classify_mass(m_hat, threshold)` to determine the bin.
- Sets the drop target (`ctx.light_bin_drop` or `ctx.heavy_bin_drop`).
- Transitions to `PLACE`.

#### Updated `PlannerContext`

New fields (additive):

```python
estimator: Estimator | None = None
weigh_qpos: np.ndarray = ...
weigh_hold_s: float = 1.0             # how long WEIGH samples after settle
calibration: dict | None = None       # loaded or produced inside InitState
estimate_result: EstimateResult | None = None
bin_label: str | None = None          # "light" | "heavy"
drop_target: np.ndarray | None = None # set by CLASSIFY, read by PLACE
gravity_comp_mask: np.ndarray = np.ones(6)  # 1.0 = gravity comp ON for that joint
                                            # Set by WeighState.enter() from estimator;
                                            # restored to all-ones by WeighState.exit().
```

#### Updated routing

- When estimator is set: `INIT → SEARCH → GRASP → WEIGH → CLASSIFY → PLACE → HOME`
- When estimator is `None` (backward compat): `SEARCH → GRASP → PLACE → HOME` (unchanged)

> **Backward compatibility:** if no estimator is configured, the FSM skips `INIT`/`WEIGH`/`CLASSIFY` and behaves exactly as Phase 3.

#### `GraspState` routing tweak

`on_done` changes from `"PLACE"` to `"WEIGH"` when `ctx.estimator is not None`.

#### `PlaceState` update

Uses `ctx.drop_target` instead of hardcoded `LIGHT_BIN_DROP`. Falls back to `ctx.light_bin_drop` when no estimator is set (backward compat).

### [MODIFY] `massaware/controller.py`

Add lightweight override hooks so estimators can temporarily change gains during WEIGH without modifying base config:

```python
def apply_overrides(self, overrides: dict) -> None:
    """Stash current gains, apply overrides (e.g. {'ki': np.zeros(6)})."""

def clear_overrides(self) -> None:
    """Restore the stashed gains."""
```

Only Lyapunov uses this today (it sets `ki → 0` for the spring-energy model). The base controller config in `default.yaml` is unchanged. Integral state is reset whenever overrides change.

### [MODIFY] `massaware/tick_loop.py`

Two additive changes:

**1. Per-joint gravity comp mask** (keeps controller code untouched):

```python
# After controller.compute() with use_gravity_comp=False:
tau = controller.compute(q, q_dot, q_ref, qfrc_bias, dt, use_gravity_comp=False)
tau += qfrc_bias * ctx.gravity_comp_mask
```

Default mask is all-ones (full gravity comp, mathematically identical to today). `WeighState.enter()` sets it from the active estimator; `WeighState.exit()` restores all-ones. No `weighing` flag is needed — the mask itself carries the state.

**2. Estimator hook** (uncomment + implement lines 86–87):

```python
if self.estimator is not None:
    obs = _build_obs(env, robot, q_ref=ctx.arm_target, tau_cmd=tau)
    self.estimator.update(obs)
```

`update()` is called every tick unconditionally. Each estimator decides internally whether to accumulate samples.

#### `_build_obs()` helper

```python
def _build_obs(env: MujocoEnv, robot: Robot, q_ref: np.ndarray, tau_cmd: np.ndarray) -> EstimatorObs:
    """Assemble an EstimatorObs from current sim state."""
```

### [MODIFY] `massaware/estimators/__init__.py`

Re-export `Estimator`, `EstimatorObs`, `EstimateResult`, and `build` from sub-modules.

---

## Sub-phase C — PID-Error Estimator (Auto-Calibrated)

**Goal:** Implement the PID-error estimator. Calibration is run automatically by `InitState` (see Sub-phase B); there is **no standalone `scripts/calibrate.py`**.

### Design: Single-Joint Measurement (Elbow)

The PID-error method measures mass from a **single measurement joint = `elbow` (index 2)** while all other joints remain stiff with full gravity compensation ON. Elbow is chosen over shoulder_lift because it is closer to the payload, which reduces contamination from upstream gravity-comp model error.

The integral term is **kept active** on the measurement joint (it winds up to hold the payload — this is exactly the signal PID-error reads). No controller override is requested by this estimator.

**Formula:**
```
m_hat = (mean(tau_cmd[meas_joint]) - tau_ss_empty[meas_joint]) / (g * moment_arm)
```

> **`moment_arm` is currently a hand-tuned scalar in YAML. Pinned for later derivation from FK — see [PINNED_ISSUES.md P1](PINNED_ISSUES.md).**

### [NEW] `massaware/estimators/pid_error.py`

```python
class PIDErrorEstimator(Estimator):
    name = "pid_error"
    requires_calibration = True
    # gravity_comp_mask: ones everywhere EXCEPT the measurement joint (elbow=2).
    # controller_overrides: {} — keep integral term active so it carries the load.

    def __init__(self, measurement_joint: int, moment_arm: float, tau_ss_empty: np.ndarray):
        ...

    def update(self, obs: EstimatorObs) -> None:
        """Accumulate obs.tau_cmd[measurement_joint]."""

    def estimate(self) -> EstimateResult:
        """m_hat = (mean(tau_cmd[j]) - tau_ss_empty[j]) / (g * moment_arm)"""

    def calibrate(self, ctx) -> dict:
        """Move empty arm to ctx.weigh_qpos with gravity_comp_mask
        (measurement joint uncompensated). Hold 3 s. Record mean tau_cmd
        at measurement joint. Return {'tau_ss_empty': [...]}."""
```

### Calibration file format

`configs/calibration.yaml` is produced by the active estimator's `calibrate()` method and merged across estimators (each estimator owns its own keys):

```yaml
# Produced by InitState the first time an estimator with requires_calibration=True
# runs against a given weigh_qpos. Cached and reused across runs.
tau_ss_empty: [0.0, 0.0, 12.3, 0.0, 0.0, 0.0]   # PID-error: elbow only
q_empty: [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]  # Lyapunov
z_empty: 0.405                                  # Lyapunov
weigh_qpos_used: [0, -90, -90, -90, 90, 0]      # degrees, for cache invalidation
timestamp: "2026-05-25T12:00:00"
```

If a second estimator is selected later, `InitState` augments the file with that estimator's keys (it does not overwrite unrelated keys).

### Config additions to `default.yaml`

```yaml
estimator:
  name: "pid_error"
  pid_error:
    measurement_joint: 2          # elbow (scalar, not a list)
    moment_arm: 0.5               # hand-tuned; see PINNED_ISSUES.md P1
```

---

## Sub-phase D — Lyapunov (Spring-Sag) Estimator

**Goal:** Implement the energy-based estimator using joint displacement.

### Design: Energy-based Mass Estimation
- **Lyapunov is the only estimator that sets `ki = 0`**, and it does so via the per-estimator `controller_overrides()` hook introduced in Sub-phase B. The base PID config in `default.yaml` is **unchanged**; PID-error and inverse-dynamics keep their normal `ki` values.
- With `ki = 0`, the proportional term acts as a multi-dimensional spring.
- The mass is calculated by equating the loss of gravitational potential energy ($\Delta E_{grav} = m \cdot g \cdot \Delta h$) with the gain in controller elastic energy ($\Delta E_{spring}$).
- Requires full gravity compensation (so the arm only sags due to the cube's weight): `gravity_comp_mask` returns all-ones. Reads `q_empty` and `z_empty` from calibration.

### [NEW] `massaware/estimators/lyapunov.py`

```python
class LyapunovEstimator(Estimator):
    name = "lyapunov"
    requires_calibration = True
    # gravity_comp_mask: all-ones (full gravity comp; arm sags purely from payload).
    # controller_overrides: {"ki": np.zeros(n_joints)}  ← only Lyapunov does this.

    def __init__(self, q_empty: np.ndarray, z_empty: float, kp: np.ndarray):
        """
        q_empty: calibrated joint positions with empty gripper.
        z_empty: calibrated EE Z-height with empty gripper.
        kp: Proportional gains of the controller (the "spring stiffness").
        """

    def reset(self) -> None: ...
    def update(self, obs: EstimatorObs) -> None:
        """Accumulate joint positions (obs.q) and Cartesian Z-height (via FK)"""

    def estimate(self) -> EstimateResult:
        """
        q_loaded = mean(accumulated_q)
        z_loaded = mean(accumulated_z)
        
        ΔE_spring = sum(0.5 * kp * (q_loaded - q_ref)**2 - 0.5 * kp * (q_empty - q_ref)**2)
        Δh = z_empty - z_loaded
        
        m_hat = ΔE_spring / (g * Δh)
        """

    def calibrate(self, ctx) -> dict:
        """Move empty arm to ctx.weigh_qpos with full gravity comp and ki=0.
        Hold 3 s. Record mean q (q_empty) and mean EE z-height via FK (z_empty).
        Return {'q_empty': [...], 'z_empty': float}."""
```

### Config additions to `default.yaml`

```yaml
estimator:
  name: "lyapunov"
  # No extra config needed.
  # ki=0 is injected via controller_overrides() at WEIGH enter, not via YAML.
  # kp comes from the base controller config; q_empty/z_empty come from calibration.
```

### [MODIFY] `scripts/run.py`

- Wire in `LyapunovEstimator` when selected in config. **No changes to base PID gains** — the `ki=0` behavior is local to WEIGH via the controller-override hook.

---

## Sub-phase E — Integration Verification

**Goal:** Prove correctness and the swap architecture for both methods.

### Tests

| Test | Expected |
|---|---|
| `run.py` with `estimator: pid_error` sorts all 3 cubes correctly | `m_hat` within ±20% of true mass |
| `run.py` with `estimator: lyapunov` sorts all 3 cubes correctly | `m_hat` within ±20% of true mass |
| Change only `estimator:` in YAML → pipeline works with no code changes | Zero code changes |

### Tune during this phase

- Hand-tune `moment_arm` for PID-error to get reasonable `m_hat`; defer FK-derivation per [PINNED_ISSUES.md P1](PINNED_ISSUES.md).
- Confirm `mass_threshold` cleanly separates the three cube masses.
- If `m_hat` shows high variance, revisit the WEIGH transient handling per [PINNED_ISSUES.md P2](PINNED_ISSUES.md).

---

## Summary of Decisions Applied to This Plan (2026-05-25)

| Decision | Resolution |
|---|---|
| `ki = 0` is **Lyapunov-only**, applied via per-estimator `controller_overrides()` | Sub-phase B (interface) + Sub-phase D (uses it) |
| `moment_arm` derivation deferred | [PINNED_ISSUES.md P1](PINNED_ISSUES.md) |
| `measurement_joint` is **scalar**, not a list | Sub-phase C config |
| PID-error measurement joint switched from `shoulder_lift` → `elbow` (index 2) | Sub-phase C |
| WEIGH transient handling deferred | [PINNED_ISSUES.md P2](PINNED_ISSUES.md) |
| Calibration moved from standalone `scripts/calibrate.py` → automatic inside `InitState` | Sub-phase B (InitState) + Sub-phase C/D (each estimator implements `calibrate()`) |
