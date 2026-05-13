"""Motion primitives and the mission FSM.

Phase 3 scope: pick-and-place a single cube into the light bin, no weighing.
The class-based FSM matches ARCHITECTURE.md §2 — each state exposes
``enter`` / ``update`` / ``exit`` so Phase 4's ``WEIGH``/``CLASSIFY`` states
can slot in without a refactor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from massaware.mujoco_env import MujocoEnv
from massaware.perception.base import CubeDetection, Perception
from massaware.robot import Robot

# --- Mission constants -------------------------------------------------------

#            pan     lift    elbow   wrist1  wrist2  wrist3
HOME_QPOS = np.deg2rad([0, -90, -90, -90, 90, 0])

GRIPPER_CTRL_IDX = 6
# Robotiq 2f85 menagerie convention: 0 = open, 255 = closed.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0

APPROACH_DZ = 0.10  # m above target before descending
LIFT_DZ = 0.15      # m above grasp after closing the gripper

# Drop inside the light bin footprint (bin spans y ∈ 0.54..0.82, top at
# z=0.365 with walls to z≈0.42).
LIGHT_BIN_DROP = np.array([0.0, 0.60, 0.55])


# --- Primitives --------------------------------------------------------------


class Primitives:
    """Blocking motion primitives over the built-in position actuator."""

    def __init__(self, env: MujocoEnv, robot: Robot):
        self.env = env
        self.robot = robot

    def _ctrl_with_gripper(self, q_arm: np.ndarray) -> np.ndarray:
        ctrl = np.empty(self.env.model.nu)
        ctrl[:6] = q_arm
        ctrl[GRIPPER_CTRL_IDX] = self.env.data.ctrl[GRIPPER_CTRL_IDX]
        return ctrl

    def move_to_joint(
        self,
        q_target: np.ndarray,
        *,
        timeout: float = 5.0,
        tol: float = 3e-2,
    ) -> bool:
        ctrl = self._ctrl_with_gripper(q_target)
        t0 = self.env.data.time
        while True:
            self.env.step(ctrl=ctrl, n=10)
            if np.linalg.norm(self.env.get_arm_qpos() - q_target) < tol:
                return True
            if self.env.data.time - t0 > timeout:
                return False

    def move_to_cartesian(
        self,
        xyz: np.ndarray,
        *,
        timeout: float = 5.0,
        tol: float = 3e-2,
    ) -> bool:
        q_seed = self.env.get_arm_qpos()
        q_target, ok = self.robot.ik(np.asarray(xyz), q_seed=q_seed)
        if not ok:
            return False
        return self.move_to_joint(q_target, timeout=timeout, tol=tol)

    def grasp(self, *, hold_steps: int = 300) -> None:
        self.env.data.ctrl[GRIPPER_CTRL_IDX] = GRIPPER_CLOSE
        self.env.step(n=hold_steps)

    def release(self, *, hold_steps: int = 200) -> None:
        self.env.data.ctrl[GRIPPER_CTRL_IDX] = GRIPPER_OPEN
        self.env.step(n=hold_steps)

    def settle(self, *, tol: float = 1e-3, timeout: float = 2.0) -> bool:
        t0 = self.env.data.time
        while True:
            self.env.step(n=10)
            if np.linalg.norm(self.env.get_arm_qvel()) < tol:
                return True
            if self.env.data.time - t0 > timeout:
                return False


# --- FSM ---------------------------------------------------------------------


@dataclass
class PlannerContext:
    env: MujocoEnv
    robot: Robot
    perception: Perception
    prim: Primitives
    target_color: str = "green"
    target_cube: CubeDetection | None = None
    trace: list[str] = field(default_factory=list)


class State(ABC):
    name: str

    def enter(self, ctx: PlannerContext) -> None:
        pass

    @abstractmethod
    def update(self, ctx: PlannerContext) -> str:
        ...

    def exit(self, ctx: PlannerContext) -> None:
        pass


class SearchState(State):
    name = "SEARCH"

    def update(self, ctx: PlannerContext) -> str:
        for det in ctx.perception.detect():
            if det.color == ctx.target_color:
                ctx.target_cube = det
                print(f"  [SEARCH] target '{ctx.target_color}' at xyz={np.round(det.xyz, 3)}")
                return "GRASP"
        print(f"  [SEARCH] no '{ctx.target_color}' cube found")
        return "DONE"


class GraspState(State):
    name = "GRASP"

    def update(self, ctx: PlannerContext) -> str:
        cube = ctx.target_cube
        assert cube is not None, "GRASP entered without a target cube"

        approach = cube.xyz + np.array([0.0, 0.0, APPROACH_DZ])
        lift = cube.xyz + np.array([0.0, 0.0, LIFT_DZ])

        ctx.prim.release()
        if not ctx.prim.move_to_cartesian(approach):
            print("  [GRASP] approach failed")
            return "HOME"
        if not ctx.prim.move_to_cartesian(cube.xyz):
            print("  [GRASP] descend failed")
            return "HOME"
        ctx.prim.grasp()
        if not ctx.prim.move_to_cartesian(lift):
            print("  [GRASP] lift failed")
            return "HOME"
        print(f"  [GRASP] picked up {cube.name}")
        return "PLACE"


class PlaceState(State):
    name = "PLACE"

    def update(self, ctx: PlannerContext) -> str:
        if not ctx.prim.move_to_cartesian(LIGHT_BIN_DROP):
            print("  [PLACE] move-to-bin failed")
            return "HOME"
        ctx.prim.release()
        print(f"  [PLACE] released over light bin at {np.round(LIGHT_BIN_DROP, 3)}")
        return "HOME"


class HomeState(State):
    name = "HOME"

    def update(self, ctx: PlannerContext) -> str:
        ctx.prim.move_to_joint(HOME_QPOS)
        print("  [HOME] returned to home pose")
        return "DONE"


DEFAULT_STATES: dict[str, State] = {
    SearchState.name: SearchState(),
    GraspState.name:  GraspState(),
    PlaceState.name:  PlaceState(),
    HomeState.name:   HomeState(),
}


class FSM:
    def __init__(self, states: dict[str, State] | None = None, initial: str = "SEARCH"):
        self.states = states if states is not None else DEFAULT_STATES
        self.initial = initial

    def run(self, ctx: PlannerContext) -> list[str]:
        name = self.initial
        while name != "DONE":
            state = self.states[name]
            ctx.trace.append(name)
            print(f"[FSM] state = {name}")
            state.enter(ctx)
            next_name = state.update(ctx)
            state.exit(ctx)
            name = next_name
        ctx.trace.append("DONE")
        return ctx.trace
