"""Non-blocking motion steps and mission FSM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from massaware.controller import PIDController
from massaware.estimators.base import Estimator, EstimateResult, EstimatorObs
from massaware.mujoco_env import MujocoEnv
from massaware.perception.base import CubeDetection, Perception
from massaware.robot import Robot
from massaware.tick_loop import GripperCmd

# Calibration cache lives next to default.yaml.
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "configs" / "calibration.yaml"
# Tolerance for matching cached weigh_qpos against the active one (radians, per joint).
CALIBRATION_QPOS_TOL = 1e-4

# --- Constants ---

APPROACH_DZ = 0.10  # m above target before descending
LIFT_DZ = 0.15      # m above grasp after closing the gripper


@dataclass
class PlannerContext:
    env: MujocoEnv
    robot: Robot
    perception: Perception
    controller: PIDController
    target_color: str = "grey"
    target_cube: CubeDetection | None = None
    trace: list[str] = field(default_factory=list)

    # Configured poses
    home_qpos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    weigh_qpos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    light_bin_drop: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heavy_bin_drop: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Control outputs
    arm_target: np.ndarray | None = None
    gripper_cmd: GripperCmd = GripperCmd.OPEN
    reset_controller: bool = False

    # --- Estimator-related fields (additive, populated only when an estimator is set) ---
    estimator: Estimator | None = None
    mass_threshold: float = 0.35               # kg — see configs/default.yaml
    weigh_hold_s: float = 1.0
    calibration: dict | None = None            # loaded/produced inside InitState
    estimate_result: EstimateResult | None = None
    bin_label: str | None = None               # "light" | "heavy"
    drop_target: np.ndarray | None = None      # set by CLASSIFY, read by PLACE
    # Per-joint multiplier on qfrc_bias in tick_loop. WeighState / CalibrationHoldStep
    # set this from the active estimator; all-ones means full gravity compensation.
    gravity_comp_mask: np.ndarray = field(default_factory=lambda: np.ones(6))
    # Per-tick callback invoked by TickLoop with an EstimatorObs. Set by the
    # active step (WEIGH or InitState's calibration step), cleared on exit.
    # The callable type is intentionally loose to avoid a hard tick_loop import here.
    estimator_sink: Callable[[EstimatorObs], None] | None = None


# --- Step Abstraction ---


class Step(ABC):
    """Atomic action within a state."""

    @abstractmethod
    def start(self, ctx: PlannerContext) -> None:
        """Called once when step becomes active."""
        ...

    @abstractmethod
    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        """Advance one physics tick. Returns (finished, succeeded)."""
        ...


class MoveToJointStep(Step):
    """Drive arm to joint target."""

    def __init__(self, q_target: np.ndarray, *, tol: float = 3e-2, timeout: float = 5.0):
        self.q_target = np.asarray(q_target, dtype=float)
        self.tol = tol
        self.timeout = timeout
        self._t_start: float | None = None

    def start(self, ctx: PlannerContext) -> None:
        self._t_start = ctx.env.data.time
        ctx.arm_target = self.q_target
        ctx.gripper_cmd = GripperCmd.HOLD

    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        ctx.arm_target = self.q_target
        err = np.linalg.norm(ctx.env.get_arm_qpos() - self.q_target)
        if err < self.tol:
            return True, True
        if ctx.env.data.time - self._t_start > self.timeout:
            return True, False
        return False, False


class MoveToCartesianStep(Step):
    """IK then joint drive."""

    def __init__(self, xyz: np.ndarray, *, tol: float = 3e-2, timeout: float = 5.0):
        self.xyz = np.asarray(xyz, dtype=float)
        self.tol = tol
        self.timeout = timeout
        self._joint_step: MoveToJointStep | None = None
        self._failed = False

    def start(self, ctx: PlannerContext) -> None:
        q_target, ok = ctx.robot.ik(self.xyz, q_seed=ctx.env.get_arm_qpos())
        if not ok:
            self._failed = True
            return
        self._failed = False
        self._joint_step = MoveToJointStep(q_target, tol=self.tol, timeout=self.timeout)
        self._joint_step.start(ctx)

    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        if self._failed:
            return True, False
        return self._joint_step.tick(ctx)


class GripperStep(Step):
    """Command gripper and hold for duration."""

    def __init__(self, cmd: GripperCmd, *, hold_seconds: float = 0.3):
        self.cmd = cmd
        self.hold_seconds = hold_seconds
        self._t_start: float | None = None

    def start(self, ctx: PlannerContext) -> None:
        self._t_start = ctx.env.data.time
        ctx.gripper_cmd = self.cmd

    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        ctx.gripper_cmd = self.cmd
        if ctx.env.data.time - self._t_start >= self.hold_seconds:
            return True, True
        return False, False


class SettleStep(Step):
    """Wait for velocities to drop."""

    def __init__(self, *, tol: float = 1e-3, timeout: float = 2.0):
        self.tol = tol
        self.timeout = timeout
        self._t_start: float | None = None

    def start(self, ctx: PlannerContext) -> None:
        self._t_start = ctx.env.data.time
        ctx.gripper_cmd = GripperCmd.HOLD

    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        if np.linalg.norm(ctx.env.get_arm_qvel()) < self.tol:
            return True, True
        if ctx.env.data.time - self._t_start > self.timeout:
            return True, False
        return False, False


class _EnterEstimatorModeStep(Step):
    """Flip on the estimator's gravity-comp mask and controller overrides.

    Run *after* the arm has reached weigh_qpos under base gains; running it
    earlier would force the move-to-pose loop to fight an uncompensated joint
    while a payload is attached, which the soft P+D loop can't overcome.
    Paired teardown is the owning State's exit().
    """

    def start(self, ctx: PlannerContext) -> None:
        est = ctx.estimator
        assert est is not None
        n_joints = ctx.weigh_qpos.shape[0]
        ctx.gravity_comp_mask = est.gravity_comp_mask(n_joints)
        overrides = est.controller_overrides()
        if overrides:
            ctx.controller.apply_overrides(overrides)
        ctx.gripper_cmd = GripperCmd.HOLD

    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        return True, True


class _EstimatorHoldStep(Step):
    """Wait `hold_seconds` while routing per-tick obs to a sink on the estimator.

    Used by both WeighState (sink = estimator.update) and InitState's calibration
    (sink = estimator.update_calibration). The estimator's mask + controller
    overrides must already be applied by the owning state.
    """

    def __init__(self, sink: Callable[[EstimatorObs], None], *, hold_seconds: float):
        self._sink = sink
        self.hold_seconds = hold_seconds
        self._t_start: float | None = None

    def start(self, ctx: PlannerContext) -> None:
        self._t_start = ctx.env.data.time
        ctx.gripper_cmd = GripperCmd.HOLD
        ctx.estimator_sink = self._sink

    def tick(self, ctx: PlannerContext) -> tuple[bool, bool]:
        if ctx.env.data.time - self._t_start >= self.hold_seconds:
            ctx.estimator_sink = None
            return True, True
        return False, False


# --- FSM ---


class State(ABC):
    name: str

    def enter(self, ctx: PlannerContext) -> None:
        pass

    @abstractmethod
    def tick(self, ctx: PlannerContext) -> str | None:
        """Returns next state name or None."""
        ...

    def exit(self, ctx: PlannerContext) -> None:
        pass


class _SequenceState(State):
    """Executes sequence of Steps."""

    _steps: list[Step]
    _idx: int

    def _init_steps(self, ctx: PlannerContext, steps: list[Step]) -> None:
        self._steps = steps
        self._idx = 0
        if self._steps:
            self._steps[0].start(ctx)

    def _tick_steps(self, ctx: PlannerContext, *, on_done: str, on_fail: str = "HOME") -> str | None:
        if self._idx >= len(self._steps):
            return on_done
        
        step = self._steps[self._idx]
        done, ok = step.tick(ctx)
        if not done:
            return None
        
        if not ok:
            print(f"  [{self.name}] step {self._idx} ({type(step).__name__}) failed")
            return on_fail
            
        self._idx += 1
        if self._idx >= len(self._steps):
            return on_done
            
        self._steps[self._idx].start(ctx)
        return None


class SearchState(State):
    name = "SEARCH"

    def enter(self, ctx: PlannerContext) -> None:
        self._next = "DONE"
        for det in ctx.perception.detect():
            if det.color == ctx.target_color:
                ctx.target_cube = det
                print(f"  [SEARCH] target '{ctx.target_color}' at xyz={np.round(det.xyz, 3)}")
                self._next = "GRASP"
                return
        print(f"  [SEARCH] no '{ctx.target_color}' cube found")

    def tick(self, ctx: PlannerContext) -> str | None:
        return self._next


class GraspState(_SequenceState):
    name = "GRASP"

    def enter(self, ctx: PlannerContext) -> None:
        cube = ctx.target_cube
        assert cube is not None
        approach = cube.xyz + np.array([0.0, 0.0, APPROACH_DZ])
        lift = cube.xyz + np.array([0.0, 0.0, LIFT_DZ])
        self._init_steps(ctx, [
            GripperStep(GripperCmd.OPEN),
            MoveToCartesianStep(approach),
            MoveToCartesianStep(cube.xyz),
            GripperStep(GripperCmd.CLOSE),
            MoveToCartesianStep(lift),
            # SettleStep(tol=0.2, timeout=3.0), # optional, but (should?) stablize arm before estimating
        ])

    def tick(self, ctx: PlannerContext) -> str | None:
        # If an estimator is configured, weigh before placing.
        on_done = "WEIGH" if ctx.estimator is not None else "PLACE"
        return self._tick_steps(ctx, on_done=on_done, on_fail="HOME")


class PlaceState(_SequenceState):
    name = "PLACE"

    def enter(self, ctx: PlannerContext) -> None:
        # CLASSIFY sets ctx.drop_target; fall back to light bin when weighing is skipped.
        target = ctx.drop_target if ctx.drop_target is not None else ctx.light_bin_drop
        self._init_steps(ctx, [
            MoveToCartesianStep(target),
            GripperStep(GripperCmd.OPEN),
        ])

    def tick(self, ctx: PlannerContext) -> str | None:
        return self._tick_steps(ctx, on_done="HOME", on_fail="HOME")


class HomeState(_SequenceState):
    name = "HOME"

    def enter(self, ctx: PlannerContext) -> None:
        self._init_steps(ctx, [MoveToJointStep(ctx.home_qpos)])

    def tick(self, ctx: PlannerContext) -> str | None:
        result = self._tick_steps(ctx, on_done="DONE")
        if result == "DONE":
            print("  [HOME] returned to home pose")
        return result


# --- Estimator-dependent states (INIT / WEIGH / CLASSIFY) ---


def _load_cached_calibration(weigh_qpos: np.ndarray) -> dict | None:
    """Return cached calibration if the stored weigh_qpos matches, else None."""
    if not CALIBRATION_PATH.exists():
        return None
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        cached = yaml.safe_load(f) or {}
    stored = cached.get("weigh_qpos_used")
    if stored is None:
        return None
    if np.max(np.abs(np.asarray(stored, dtype=float) - weigh_qpos)) > CALIBRATION_QPOS_TOL:
        return None
    return cached


def _save_calibration(weigh_qpos: np.ndarray, new_keys: dict) -> dict:
    """Merge `new_keys` into configs/calibration.yaml. Keeps unrelated keys intact
    when the cached `weigh_qpos_used` matches the active one. Returns the full dict."""
    existing: dict = {}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        stored = existing.get("weigh_qpos_used")
        if stored is None or np.max(np.abs(np.asarray(stored, dtype=float) - weigh_qpos)) > CALIBRATION_QPOS_TOL:
            # weigh_qpos changed -> stale; start fresh.
            existing = {}

    merged = {**existing, **new_keys}
    merged["weigh_qpos_used"] = [float(x) for x in weigh_qpos]
    merged["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return merged


class InitState(State):
    """Loads or produces calibration for the active estimator, then exits to SEARCH.

    Flow:
      - No estimator OR estimator doesn't need calibration -> straight to SEARCH.
      - Cached calibration matches current weigh_qpos -> load it -> SEARCH.
      - Otherwise: move to weigh_qpos, apply mask + controller overrides, settle,
        run a tick-driven calibration hold, write configs/calibration.yaml, SEARCH.
      - Any failure during calibration -> ERROR.
    """

    name = "INIT"

    def __init__(self, hold_seconds: float = 3.0, settle_tol: float = 1e-3, settle_timeout: float = 3.0):
        self.hold_seconds = hold_seconds
        self.settle_tol = settle_tol
        self.settle_timeout = settle_timeout
        self._mode: str = "skip"          # "skip" | "load" | "run"
        self._steps: list[Step] = []
        self._idx: int = 0
        self._overrides_applied: bool = False
        self._mask_applied: bool = False

    def enter(self, ctx: PlannerContext) -> None:
        est = ctx.estimator
        if est is None or not est.requires_calibration:
            self._mode = "skip"
            return

        cached = _load_cached_calibration(ctx.weigh_qpos)
        if cached is not None:
            try:
                est.load_calibration(cached)
            except KeyError as exc:
                # File matches the pose but doesn't carry this estimator's
                # keys (e.g. another estimator wrote it). Fall through to a
                # fresh calibration; the merged save will keep both sets.
                print(f"  [INIT] cached file present but missing keys ({exc}); running calibration")
            else:
                ctx.calibration = cached
                print(f"  [INIT] loaded cached calibration from {CALIBRATION_PATH.name}")
                self._mode = "load"
                return

        # Need to run calibration in-sim.
        print(f"  [INIT] running calibration for estimator '{est.name}'")
        # NOTE: estimator gravity-comp mask + controller overrides are applied
        # by _EnterEstimatorModeStep, which runs *after* the move-to-pose so
        # the move happens under base gains.
        self._mask_applied = True       # exit() must restore mask + overrides
        self._overrides_applied = bool(est.controller_overrides())

        est.start_calibration(ctx)
        self._steps = [
            MoveToJointStep(ctx.weigh_qpos, timeout=5.0),
            _EnterEstimatorModeStep(),
            SettleStep(tol=self.settle_tol, timeout=self.settle_timeout),
            _EstimatorHoldStep(est.update_calibration, hold_seconds=self.hold_seconds),
        ]
        self._idx = 0
        self._steps[0].start(ctx)
        self._mode = "run"

    def tick(self, ctx: PlannerContext) -> str | None:
        if self._mode in ("skip", "load"):
            return "SEARCH"

        # mode == "run"
        if self._idx >= len(self._steps):
            return "SEARCH"
        step = self._steps[self._idx]
        done, ok = step.tick(ctx)
        if not done:
            return None
        if not ok:
            if isinstance(step, SettleStep):
                # Settle is best-effort. Some estimators (PID-error, softened
                # gains) reach a low-amplitude limit cycle rather than a true
                # rest; the hold-average still produces an unbiased estimate.
                print(f"  [INIT] step {self._idx} (SettleStep) did not converge; proceeding")
                ok = True
            else:
                print(f"  [INIT] step {self._idx} ({type(step).__name__}) failed during calibration")
                return "ERROR"
        self._idx += 1
        if self._idx >= len(self._steps):
            # Calibration hold complete -> persist.
            est = ctx.estimator
            assert est is not None
            new_keys = est.finish_calibration()
            ctx.calibration = _save_calibration(ctx.weigh_qpos, new_keys)
            est.load_calibration(ctx.calibration)
            print(f"  [INIT] wrote calibration -> {CALIBRATION_PATH.name}")
            return "SEARCH"
        self._steps[self._idx].start(ctx)
        return None

    def exit(self, ctx: PlannerContext) -> None:
        if self._mask_applied:
            ctx.gravity_comp_mask = np.ones_like(ctx.gravity_comp_mask)
            self._mask_applied = False
        if self._overrides_applied:
            ctx.controller.clear_overrides()
            self._overrides_applied = False
        ctx.estimator_sink = None


class WeighState(State):
    """Move to weigh_qpos, settle, sample the estimator, call estimate()."""

    name = "WEIGH"

    def __init__(self, settle_tol: float = 1e-3, settle_timeout: float = 3.0):
        self.settle_tol = settle_tol
        self.settle_timeout = settle_timeout
        self._steps: list[Step] = []
        self._idx: int = 0
        self._mask_applied: bool = False
        self._overrides_applied: bool = False

    def enter(self, ctx: PlannerContext) -> None:
        est = ctx.estimator
        assert est is not None, "WeighState entered with no estimator"
        # Mask + overrides are applied by _EnterEstimatorModeStep AFTER the
        # move-to-pose, so the arm reaches weigh_qpos with full stiffness even
        # when carrying a payload.
        self._mask_applied = True
        self._overrides_applied = bool(est.controller_overrides())

        est.reset()
        self._steps = [
            MoveToJointStep(ctx.weigh_qpos, timeout=5.0),
            _EnterEstimatorModeStep(),
            SettleStep(tol=self.settle_tol, timeout=self.settle_timeout),
            _EstimatorHoldStep(est.update, hold_seconds=ctx.weigh_hold_s),
        ]
        self._idx = 0
        self._steps[0].start(ctx)

    def tick(self, ctx: PlannerContext) -> str | None:
        if self._idx >= len(self._steps):
            return "CLASSIFY"
        step = self._steps[self._idx]
        done, ok = step.tick(ctx)
        if not done:
            return None
        if not ok:
            if isinstance(step, SettleStep):
                print(f"  [WEIGH] step {self._idx} (SettleStep) did not converge; proceeding")
                ok = True
            else:
                print(f"  [WEIGH] step {self._idx} ({type(step).__name__}) failed")
                return "HOME"
        self._idx += 1
        if self._idx >= len(self._steps):
            est = ctx.estimator
            assert est is not None
            ctx.estimate_result = est.estimate()
            print(f"  [WEIGH] m_hat = {ctx.estimate_result.m_hat:.4f} kg")
            return "CLASSIFY"
        self._steps[self._idx].start(ctx)
        return None

    def exit(self, ctx: PlannerContext) -> None:
        if self._mask_applied:
            ctx.gravity_comp_mask = np.ones_like(ctx.gravity_comp_mask)
            self._mask_applied = False
        if self._overrides_applied:
            ctx.controller.clear_overrides()
            self._overrides_applied = False
        ctx.estimator_sink = None


class ClassifyState(State):
    """Reads ctx.estimate_result and picks a drop target."""

    name = "CLASSIFY"

    def enter(self, ctx: PlannerContext) -> None:
        from massaware.classify import classify_mass  # local import to avoid cycles
        result = ctx.estimate_result
        assert result is not None, "ClassifyState entered with no estimate"
        threshold = ctx.mass_threshold
        ctx.bin_label = classify_mass(result.m_hat, threshold)
        ctx.drop_target = ctx.heavy_bin_drop if ctx.bin_label == "heavy" else ctx.light_bin_drop
        print(f"  [CLASSIFY] m_hat={result.m_hat:.4f} kg -> '{ctx.bin_label}' bin")

    def tick(self, ctx: PlannerContext) -> str | None:
        return "PLACE"


class ErrorState(State):
    """Terminal failure state. Releases gripper, returns home."""

    name = "ERROR"

    def enter(self, ctx: PlannerContext) -> None:
        print("  [ERROR] entering error state; releasing and returning home")
        ctx.gripper_cmd = GripperCmd.OPEN

    def tick(self, ctx: PlannerContext) -> str | None:
        return "HOME"


DEFAULT_STATES = {
    s.name: s
    for s in [
        InitState(),
        SearchState(),
        GraspState(),
        WeighState(),
        ClassifyState(),
        PlaceState(),
        HomeState(),
        ErrorState(),
    ]
}


class FSM:
    """Finite State Machine driven by TickLoop."""

    def __init__(self, ctx: PlannerContext, states: dict[str, State] | None = None, initial: str | None = None):
        self.ctx = ctx
        self.states = states or DEFAULT_STATES
        if initial is None:
            # Auto-pick: skip INIT entirely when no estimator is configured
            # (backward compatibility with the Phase 3 SEARCH -> ... flow).
            initial = "INIT" if ctx.estimator is not None else "SEARCH"
        self._current = initial
        self.done = False
        self.ctx.trace.append(self._current)
        print(f"[FSM] state = {self._current}")
        self.states[self._current].enter(self.ctx)

    def tick(self) -> None:
        """Advance one tick."""
        next_name = self.states[self._current].tick(self.ctx)
        if next_name is None or next_name == self._current:
            return
        
        self.states[self._current].exit(self.ctx)
        if next_name == "DONE":
            self.done = True
            self.ctx.trace.append("DONE")
            return
        
        self._current = next_name
        self.ctx.trace.append(next_name)
        self.ctx.reset_controller = True
        print(f"[FSM] state = {next_name}")
        self.states[self._current].enter(self.ctx)
