"""Phase 3 mission runner: pick the blue cube and drop it in the light bin."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.mujoco_env import MujocoEnv
from massaware.perception import GroundTruthPerception
from massaware.planner import FSM, HOME_QPOS, PlannerContext, Primitives
from massaware.robot import Robot

# Phase 3 targets a single cube to keep the demo focused. Blue sits at y=0, so
# shoulder_pan stays near zero and the gripper opening axis aligns with world
# +x — the most kinematically robust of the three cubes. Multi-cube sorting
# comes back in Phase 4 once orientation-aware moves are in.
TARGET_COLOR = "blue"


def build() -> tuple[MujocoEnv, PlannerContext]:
    env = MujocoEnv()
    env.reset(arm_qpos=HOME_QPOS)
    robot = Robot(env)
    return env, PlannerContext(
        env=env,
        robot=robot,
        perception=GroundTruthPerception(env),
        prim=Primitives(env, robot),
        target_color=TARGET_COLOR,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    env, ctx = build()
    target_body = f"{TARGET_COLOR}_cube"
    initial_pos = env.data.xpos[env.model.body(target_body).id].copy()

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            env.bind_viewer(viewer)
            FSM().run(ctx)
            _print_summary(env, target_body, initial_pos, ctx.trace)
            # Keep the window responsive after the FSM finishes so the user can
            # inspect the final state. Closes when the user closes the viewer.
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    else:
        FSM().run(ctx)
        _print_summary(env, target_body, initial_pos, ctx.trace)
    return 0


def _print_summary(env: MujocoEnv, target_body: str, initial_pos: np.ndarray, trace: list[str]) -> None:
    final_pos = env.data.xpos[env.model.body(target_body).id].copy()
    print("\n=== Mission summary ===")
    print(f"target cube     : {target_body}")
    print(f"initial position: {np.round(initial_pos, 3)}")
    print(f"final position  : {np.round(final_pos, 3)}")
    print(f"displacement    : {np.linalg.norm(final_pos - initial_pos):.3f} m")
    print(f"state trace     : {' -> '.join(trace)}")


if __name__ == "__main__":
    raise SystemExit(main())
