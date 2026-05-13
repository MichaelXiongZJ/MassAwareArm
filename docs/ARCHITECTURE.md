# MassAwareArm — Architecture & Planner Recommendation

## Context

A robot arm in a MuJoCo simulation picks colored cubes, **infers each cube's mass without a force sensor or scale**, and sorts them into heavy/light bins. The deliverable is a case-study comparison of mass estimators (baseline: PID-error and inverse-dynamics residual; extended: momentum observer, more later). Today the repo only has a stub `scene.xml`.

**Design goals (in priority order):**
1. **Simple** — match the FSM pseudocode already drafted in `Control Loop Ideation.pdf`.
2. **Expandable in one place** — adding a new mass estimator is a one-file change; nothing else moves.
3. **Clean layer boundaries** — perception, control, estimation, and planning don't know about each other's internals.

**Confirmed decisions:**
- **Arm:** UR5e with all 6 joints actively controlled. (We initially explored a 4-active-joint subset, but the locked-joint configuration removed too much gripper-orientation freedom for reliable top-down grasps across the workspace. Full 6-DOF gives orientation-aware IK and a cleaner reach.)
- **Perception:** Pluggable; first impl reads MuJoCo ground truth, camera+CV swappable later.
- **Planner:** Finite State Machine.
- **Extended estimator in scope:** Momentum / disturbance observer (time-integrating; the interface must support stateful estimators).

---

## 1. Architecture

Five layers, top to bottom. Each layer only talks to the layer immediately below it.

```
┌─────────────────────────────────────────────────────────┐
│  Planner (FSM)                                          │
│   SEARCH → GRASP → WEIGH → CLASSIFY → PLACE → HOME      │
├─────────────────────────────────────────────────────────┤
│  Primitives                                             │
│   move_to · grasp · release · settle · weigh_once       │
├─────────────────────────────────────────────────────────┤
│  Perception  │  Controller  │  Estimator   │ Classifier │
│  (interface) │  (PID 500Hz) │  (interface) │  (thresh)  │
├─────────────────────────────────────────────────────────┤
│  Robot model: FK / IK / Jacobian / dynamics queries     │
├─────────────────────────────────────────────────────────┤
│  MuJoCo sim:  scene.xml + actuators + sensors + camera  │
└─────────────────────────────────────────────────────────┘
                  Logger spans all layers
```

| Layer | Owns | Does not own |
|---|---|---|
| Planner | Task sequencing, transitions | Setpoints, torques |
| Primitives | Goal → setpoint, completion check | Mission logic |
| Perception | Detect cubes, return pose+color | Picking a target |
| Controller | `q_ref` → joint torques @500 Hz | Choosing `q_ref` |
| Estimator | `obs` → `m_hat` | Choosing a bin |
| Classifier | `m_hat` → bin label | State |
| Robot model | FK/IK/Jacobian/`qfrc_bias` | Stepping the sim |
| Sim | Step, reset, sensor reads | Anything task-related |

The three modules behind `interface` arrows (Perception, Estimator, plus Controller and Classifier) are the swap points — everything else stays put when you change them.

---

## 2. Planner (FSM)

Table-driven. Each state has `enter()`, `update() → signal`, `exit()`. The driver ticks the active state, reads its signal, looks up the next state.

| State | What it does | Exits to |
|---|---|---|
| `INIT` | Calibrate empty-arm reference torques at `WEIGH_POSE` | `SEARCH` |
| `SEARCH` | Ask perception for a target cube | `GRASP` (found) / `ERROR` (timeout) |
| `GRASP` | Approach, descend, close gripper, lift | `WEIGH` |
| `WEIGH` | Move to `WEIGH_POSE`, settle, run active estimator | `CLASSIFY` |
| `CLASSIFY` | `bin = Classifier(m_hat)` | `PLACE` |
| `PLACE` | Move above bin, descend, open gripper | `HOME` |
| `HOME` | Return to rest pose | `SEARCH` |
| `ERROR` | Release if holding, return home, log | `SEARCH` / halt |

**Rates:**
- FSM: ~50 Hz (decisions)
- Controller: 500 Hz inner loop on a shared `q_ref` (primitives write, controller reads)

**Expansion:** New state = new class + one row in the transition table. Example future addition: `RE_WEIGH` when the estimator reports high uncertainty.

---

## 3. Estimator Interface (the expansion hot-spot)

```python
class Estimator(ABC):
    name: str

    def reset(self) -> None: ...
    def update(self, obs: EstimatorObs) -> None:   # called every controller tick
        ...
    def estimate(self) -> EstimateResult:          # called at end of WEIGH
        ...

@dataclass
class EstimatorObs:
    t: float
    q: np.ndarray;       q_dot: np.ndarray
    tau_cmd: np.ndarray; tau_meas: np.ndarray
    qfrc_bias: np.ndarray         # gravity+Coriolis (MuJoCo)
    jacobian_ee: np.ndarray       # 3×N linear Jacobian at EE
    q_ref: np.ndarray             # current PID setpoint

@dataclass
class EstimateResult:
    m_hat: float
    sigma: float | None
    diagnostics: dict             # per-estimator extras for the logger
```

This shape works for every method on the table:
- **PID-error** — read `q_ref - q`, `tau_cmd`; subtract calibrated `tau_ss_empty`.
- **Inverse-dynamics residual** — read `tau_meas - qfrc_bias`; project onto `Jᵀg`.
- **Momentum observer** — uses `update()` every tick to integrate; `estimate()` returns the converged value.

All three share the same PID controller — they only differ in which signals they read during the `WEIGH` state. Note: PID-error assumes a *pure* PID (no gravity compensation); the controller exposes a `use_gravity_comp` flag that must be **off** during `WEIGH` so the steady-state command carries the payload signal. Inverse-dynamics and momentum observer are controller-agnostic.

**Registration:** `estimators/registry.py` exposes `register(name, cls)` and `build(name, cfg)`. Config picks the active estimator by name — no other file changes.

---

## 4. Repository Layout

Folders only where multiple swappable implementations live (`estimators/`, `perception/`). Everything else is a single file.

```
software/
├── assets/
│   ├── scene.xml             # modify the existing stub
│   ├── ur5e.xml              # from mujoco_menagerie
│   └── gripper/              # umi_gripper
│
├── massaware/
│   ├── mujoco_env.py         # sim wrapper (load, step, reset, sensors)
│   ├── robot.py              # FK / IK / Jacobian (6-DOF)
│   ├── controller.py         # joint-space PID
│   ├── planner.py            # FSM + motion primitives
│   ├── classify.py           # threshold classifier
│   ├── log.py                # structured CSV logger
│   ├── perception/
│   │   ├── base.py
│   │   └── groundtruth.py    # camera_cv.py added later
│   └── estimators/
│       ├── base.py           # Estimator ABC, EstimatorObs, EstimateResult
│       ├── registry.py
│       ├── pid_error.py
│       ├── inverse_dynamics.py
│       └── momentum_observer.py
│
├── configs/
│   └── default.yaml          # gains, poses, threshold, estimator name
│
└── scripts/
    ├── run.py                # one full pipeline run
    ├── calibrate.py          # capture empty-arm reference torques
    └── compare_estimators.py # case-study driver, produces slide-8 plot
```

Adding a new estimator = drop one file in `estimators/` and register it. Adding a new perception backend = drop one file in `perception/`. Nothing else moves.

---

## 5. `scene.xml` changes anticipated

Existing stub already has the table, three cubes, and two bin sites. To make it run:

- Confirm `<actuator>` entries cover all 6 UR5e joints (the vendor `ur5e.xml` already provides these).
- Attach the chosen gripper to the UR5e wrist.
- Add an overhead `<camera>` (for the future perception swap).
- Add `<sensor>` entries: joint pos/vel, actuator force.

Exact XML belongs to implementation; called out only to confirm the architecture covers it.

---

## 6. Verification

1. Sim loads, gravity behaves, no NaNs after 1 s.
2. PID holds `WEIGH_POSE` to `<1e-3 rad` with an empty gripper.
3. FSM dry run with ground-truth perception and `PIDError` estimator — confirm state trace in the log.
4. Each estimator recovers a known cube mass within target accuracy on ≥5 trials.
5. Swap estimator via config alone — repeat (4) with no other code changes.
6. `scripts/compare_estimators.py` produces the slide-8 estimator-comparison plot.

---

## 7. Out of Scope

- Specific PID gains, IK solver choice, exact `WEIGH_POSE` values.
- Real-hardware deployment.
- Multi-cube / conveyor handling.
- Learning-based estimator (interface already supports it).
