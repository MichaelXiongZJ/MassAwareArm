"""Comprehensive verification of robot model, kinematics, and basic control."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.controller import PIDController
from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot

#            pan     lift    elbow   wrist1  wrist2  wrist3
HOME_QPOS = np.deg2rad([0, -90, -90, -90, 90, 0])


def main() -> int:
    env = MujocoEnv()
    robot = Robot(env)

    print(f"Model loaded: nq={env.model.nq}, nu={env.model.nu}, nsensor={env.model.nsensor}")
    
    # 1. Component Verification
    assert env.model.nsensor > 0, "Missing sensors"
    assert env.model.ncam > 0, "Missing camera"
    assert "gripper_fingers_actuator" in [env.model.actuator(i).name for i in range(env.model.nu)]
    
    # 2. Kinematics Verification
    env.reset(arm_qpos=HOME_QPOS)
    xyz, _ = robot.fk(HOME_QPOS)
    print(f"FK(home) -> EE xyz = {np.round(xyz, 4)}")

    J = robot.jacobian_ee(HOME_QPOS)
    print(f"Jacobian rank = {np.linalg.matrix_rank(J)}")
    assert J.shape == (3, 6)

    target = xyz + np.array([0.05, 0.0, -0.05])
    q_ik, ok = robot.ik(target, q_seed=HOME_QPOS)
    xyz_ik, _ = robot.fk(q_ik)
    print(f"IK convergence: {ok}, residual: {np.linalg.norm(target - xyz_ik):.2e}")
    assert ok, "IK failed to converge"

    # 3. Control & Stability Verification
    # Using run.py baseline gains
    controller = PIDController(
        kp=[2000, 2000, 2000, 500, 500, 500],
        kd=[400, 400, 400, 100, 100, 100],
        ki=[0.0] * 6,
    )

    print("Settling at HOME_QPOS (3000 steps)...")
    for i in range(3000):
        tau = controller.compute(
            q=env.get_arm_qpos(),
            q_dot=env.get_arm_qvel(),
            q_ref=HOME_QPOS,
            qfrc_bias=env.qfrc_bias,
            dt=env.dt,
        )
        env.set_arm_ctrl(tau)
        mujoco.mj_step(env.model, env.data)
        if i % 500 == 0:
            err = np.linalg.norm(env.get_arm_qpos() - HOME_QPOS)
            print(f"  T+{i*env.dt:.1f}s error: {err:.3e}")
    
    settle_err = np.linalg.norm(env.get_arm_qpos() - HOME_QPOS)
    qdot = np.linalg.norm(env.get_arm_qvel())
    print(f"Final settle error: {settle_err:.3e}, Velocity: {qdot:.3e}")
    
    assert settle_err < 1e-2, f"Arm failed to settle (err={settle_err:.2e})"
    assert not np.any(np.isnan(env.data.qpos))

    print("\nVerification OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
