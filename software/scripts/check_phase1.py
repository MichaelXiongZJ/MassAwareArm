"""Phase 1 done check: scene loads, sensors/camera present, FK / Jacobian /
IK exercised on the full 6-DOF arm, and the position actuator settles near
home under gravity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot

#            pan     lift    elbow   wrist1  wrist2  wrist3
HOME_QPOS = np.deg2rad([0, -90, -90, -90, 90, 0])


def main() -> int:
    env = MujocoEnv()
    robot = Robot(env)

    print(f"Model loaded: nq={env.model.nq}, nu={env.model.nu}, "
          f"nsensor={env.model.nsensor}, ncam={env.model.ncam}")
    assert env.model.nsensor > 0, "scene.xml is missing the <sensor> block"
    assert env.model.ncam > 0, "scene.xml is missing the overhead camera"

    env.reset(arm_qpos=HOME_QPOS)
    print(f"reset -> arm qpos = {np.round(env.get_arm_qpos(), 3)}")

    xyz, _ = robot.fk(HOME_QPOS)
    print(f"FK(home) -> EE xyz = {np.round(xyz, 4)}")

    J = robot.jacobian_ee(HOME_QPOS)
    print(f"Jacobian shape = {J.shape}, rank = {np.linalg.matrix_rank(J)}")
    assert J.shape == (3, 6)

    target = xyz + np.array([0.05, 0.0, -0.05])
    q_ik, ok = robot.ik(target, q_seed=HOME_QPOS)
    xyz_ik, _ = robot.fk(q_ik)
    print(f"IK target={np.round(target, 4)} -> solved={np.round(q_ik, 3)} "
          f"({'converged' if ok else 'did NOT converge'}); "
          f"residual={np.linalg.norm(target - xyz_ik):.2e}")

    env.step(ctrl=HOME_QPOS, n=1000)
    settle_err = np.linalg.norm(env.get_arm_qpos() - HOME_QPOS)
    qdot = np.linalg.norm(env.get_arm_qvel())
    print(f"after 1000 steps: ||q_err||={settle_err:.3e}, ||qdot||={qdot:.3e}")
    # Built-in position actuator has finite gain; ~3 deg under gravity is
    # expected. Phase 2's custom PID with integral will tighten this.
    assert settle_err < 5e-2, "arm failed to settle near home pose"
    assert not np.any(np.isnan(env.data.qpos))

    print("\n--- sensor reads (sample) ---")
    for name in ("q_shoulder_pan", "qdot_shoulder_pan", "tau_shoulder_pan"):
        print(f"  {name} = {env.get_sensor(name)}")
    print(f"qfrc_bias (UR5e dofs) = {np.round(env.qfrc_bias, 3)}")

    print("\nPhase 1 done-check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
