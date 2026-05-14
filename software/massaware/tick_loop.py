"""Single-owner tick loop and gripper abstraction."""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import TYPE_CHECKING

import mujoco

from massaware.mujoco_env import MujocoEnv

if TYPE_CHECKING:
    from massaware.controller import PIDController
    from massaware.planner import FSM


class GripperCmd(Enum):
    """Semantic gripper command."""
    OPEN = auto()
    CLOSE = auto()
    HOLD = auto()


class Gripper:
    """Translates GripperCmd to raw ctrl value."""
    CTRL_OPEN = 0.0
    CTRL_CLOSE = 255.0

    def __init__(self, env: MujocoEnv):
        self._env = env
        self._actuator_id = env.model.actuator("gripper_fingers_actuator").id

    def apply(self, cmd: GripperCmd) -> None:
        if cmd is GripperCmd.OPEN:
            self._env.data.ctrl[self._actuator_id] = self.CTRL_OPEN
        elif cmd is GripperCmd.CLOSE:
            self._env.data.ctrl[self._actuator_id] = self.CTRL_CLOSE


class TickLoop:
    """Single-owner loop for physics stepping."""

    def __init__(self, env: MujocoEnv, fsm: FSM, gripper: Gripper, controller: PIDController, *, viewer=None):
        self.env = env
        self.fsm = fsm
        self.gripper = gripper
        self.controller = controller
        self._viewer = viewer
        self._tick = 0

    def run(self) -> None:
        """Run until FSM reaches DONE or viewer is closed."""
        while not self.fsm.done:
            if self._viewer and not self._viewer.is_running():
                break

            # 1. Decision (FSM)
            self.fsm.tick()
            ctx = self.fsm.ctx

            # 2. Control assembly
            if ctx.reset_controller:
                self.controller.reset()
                ctx.reset_controller = False

            # Fail-safe: if FSM hasn't provided a target, hold current position
            if ctx.arm_target is None:
                ctx.arm_target = self.env.get_arm_qpos()

            tau = self.controller.compute(
                q=self.env.get_arm_qpos(),
                q_dot=self.env.get_arm_qvel(),
                q_ref=ctx.arm_target,
                qfrc_bias=self.env.qfrc_bias,
                dt=self.env.dt,
            )
            self.env.set_arm_ctrl(tau)
            self.gripper.apply(ctx.gripper_cmd)

            # 3. Physics step
            t0 = time.perf_counter() if self._viewer else 0.0
            mujoco.mj_step(self.env.model, self.env.data)

            # 4. (Future) Estimator hook — called after every physics step
            # if self.estimator is not None:
            #     self.estimator.update(build_obs(self.env))

            # 5. Sync viewer and pace to real time
            if self._viewer:
                self._viewer.sync()
                elapsed = time.perf_counter() - t0
                remaining = self.env.dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            self._tick += 1
