"""Mission runner: pick the blue cube and drop it in the light bin."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.controller import PIDController
from massaware.mujoco_env import MujocoEnv
from massaware.perception import GroundTruthPerception
from massaware.planner import FSM, PlannerContext
from massaware.robot import Robot
from massaware.tick_loop import Gripper, TickLoop
from massaware.config import load_config

TARGET_COLOR = "blue"


def build_controller() -> PIDController:
    """Initialize PID controller with baseline UR5e gains."""
    kp = [2000, 2000, 2000, 500, 500, 500]
    kd = [400, 400, 400, 100, 100, 100]
    ki = [0.0] * 6
    return PIDController(kp=kp, ki=ki, kd=kd)


def build() -> tuple[MujocoEnv, PlannerContext, PIDController]:
    """Initialize environment, controller, and planner context."""
    cfg = load_config()

    env = MujocoEnv()
    env.reset(arm_qpos=cfg["poses"]["home_qpos"])
    robot = Robot(env)
    controller = build_controller()

    ctx = PlannerContext(
        env=env,
        robot=robot,
        perception=GroundTruthPerception(env),
        controller=controller,
        target_color=TARGET_COLOR,
        home_qpos=cfg["poses"]["home_qpos"],
        weigh_qpos=cfg["poses"]["weigh_qpos"],
        light_bin_drop=cfg["poses"]["light_bin_drop"],
        heavy_bin_drop=cfg["poses"]["heavy_bin_drop"],
    )
    return env, ctx, controller


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    env, ctx, controller = build()
    target_body = f"{TARGET_COLOR}_cube"
    initial_pos = env.data.xpos[env.model.body(target_body).id].copy()

    gripper = Gripper(env)
    fsm = FSM(ctx)

    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat = np.array([0.25, 0.0, 0.4])
            viewer.cam.distance = 2
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -20
            loop = TickLoop(env, fsm, gripper, controller, ctx.robot, viewer=viewer)
            loop.run()
            _print_summary(env, target_body, initial_pos, ctx.trace)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    else:
        loop = TickLoop(env, fsm, gripper, controller, ctx.robot)
        loop.run()
        _print_summary(env, target_body, initial_pos, ctx.trace)
    return 0


def _print_summary(env: MujocoEnv, target_body: str, initial_pos: np.ndarray, trace: list[str]) -> None:
    final_pos = env.data.xpos[env.model.body(target_body).id].copy()
    print("\n=== Mission summary ===")
    print(f"target cube     : {target_body}")
    print(f"initial position: {np.round(initial_pos, 3)}")
    print(f"final position  : {np.round(final_pos, 3)}")
    print(f"state trace     : {' -> '.join(trace)}")


if __name__ == "__main__":
    raise SystemExit(main())
