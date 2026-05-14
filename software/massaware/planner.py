"""Non-blocking motion steps and mission FSM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from massaware.mujoco_env import MujocoEnv
from massaware.perception.base import CubeDetection, Perception
from massaware.robot import Robot
from massaware.tick_loop import GripperCmd

# --- Constants ---

HOME_QPOS = np.deg2rad([0, -90, -90, -90, 90, 0])

APPROACH_DZ = 0.10  # m above target before descending
LIFT_DZ = 0.15      # m above grasp after closing the gripper

# Drop inside the light bin footprint (bin spans y ∈ 0.54..0.82, top at
# z=0.365 with walls to z≈0.42).
LIGHT_BIN_DROP = np.array([0.0, 0.60, 0.55])


@dataclass
class PlannerContext:
    env: MujocoEnv
    robot: Robot
    perception: Perception
    target_color: str = "green"
    target_cube: CubeDetection | None = None
    trace: list[str] = field(default_factory=list)

    # Control outputs
    arm_target: np.ndarray | None = None
    gripper_cmd: GripperCmd = GripperCmd.OPEN
    reset_controller: bool = False


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
        return self._tick_steps(ctx, on_done="PLACE", on_fail="HOME")


class PlaceState(_SequenceState):
    name = "PLACE"

    def enter(self, ctx: PlannerContext) -> None:
        self._init_steps(ctx, [
            MoveToCartesianStep(LIGHT_BIN_DROP),
            GripperStep(GripperCmd.OPEN),
        ])

    def tick(self, ctx: PlannerContext) -> str | None:
        return self._tick_steps(ctx, on_done="HOME", on_fail="HOME")


class HomeState(_SequenceState):
    name = "HOME"

    def enter(self, ctx: PlannerContext) -> None:
        self._init_steps(ctx, [MoveToJointStep(HOME_QPOS)])

    def tick(self, ctx: PlannerContext) -> str | None:
        result = self._tick_steps(ctx, on_done="DONE")
        if result == "DONE":
            print("  [HOME] returned to home pose")
        return result


DEFAULT_STATES = {s.name: s for s in [SearchState(), GraspState(), PlaceState(), HomeState()]}


class FSM:
    """Finite State Machine driven by TickLoop."""

    def __init__(self, ctx: PlannerContext, states: dict[str, State] | None = None, initial: str = "SEARCH"):
        self.ctx = ctx
        self.states = states or DEFAULT_STATES
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
