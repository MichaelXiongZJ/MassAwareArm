"""Comprehensive verification of PID controller performance and tracking."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massaware.mujoco_env import MujocoEnv
from massaware.controller import PIDController
from massaware.planner import HOME_QPOS

def main() -> int:
    env = MujocoEnv()
    env.reset(arm_qpos=HOME_QPOS)
    
    # Matching run.py gains
    kp = [2000, 2000, 2000, 500, 500, 500]
    kd = [400, 400, 400, 100, 100, 100]
    ki = [0.0] * 6
    controller = PIDController(kp, ki, kd)
    
    # 1. Static Hold Test
    print("Test 1: Static Hold (HOME_QPOS)")
    errors = []
    for _ in range(1000):
        tau = controller.compute(
            q=env.get_arm_qpos(),
            q_dot=env.get_arm_qvel(),
            q_ref=HOME_QPOS,
            qfrc_bias=env.qfrc_bias,
            dt=env.dt,
        )
        env.set_arm_ctrl(tau)
        mujoco.mj_step(env.model, env.data)
        errors.append(np.linalg.norm(env.get_arm_qpos() - HOME_QPOS))
    
    final_err = errors[-1]
    print(f"  Final hold error: {final_err:.3e} rad")
    assert final_err < 5e-3, "Static hold failed"

    # 2. Step Response Test
    print("\nTest 2: Step Response (joint 0 + 0.1 rad)")
    target_q = HOME_QPOS.copy()
    target_q[0] += 0.1
    
    for i in range(1000):
        tau = controller.compute(
            q=env.get_arm_qpos(),
            q_dot=env.get_arm_qvel(),
            q_ref=target_q,
            qfrc_bias=env.qfrc_bias,
            dt=env.dt,
        )
        env.set_arm_ctrl(tau)
        mujoco.mj_step(env.model, env.data)
        if i % 200 == 0:
            err = np.linalg.norm(env.get_arm_qpos() - target_q)
            print(f"  T+{i*env.dt:.2f}s error: {err:.3e}")
    
    final_err = np.linalg.norm(env.get_arm_qpos() - target_q)
    print(f"  Final step error: {final_err:.3e} rad")
    assert final_err < 5e-3, "Step response failed to settle"

    # 3. Dynamic Tracking Test (Sine wave)
    print("\nTest 3: Dynamic Tracking (Sine wave on shoulder_lift)")
    t_span = 2.0
    steps = int(t_span / env.dt)
    tracking_errors = []
    
    for i in range(steps):
        t = i * env.dt
        ref = HOME_QPOS.copy()
        ref[1] += 0.05 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz oscillation
        
        tau = controller.compute(
            q=env.get_arm_qpos(),
            q_dot=env.get_arm_qvel(),
            q_ref=ref,
            qfrc_bias=env.qfrc_bias,
            dt=env.dt,
        )
        env.set_arm_ctrl(tau)
        mujoco.mj_step(env.model, env.data)
        tracking_errors.append(np.linalg.norm(env.get_arm_qpos() - ref))
    
    mean_tracking_err = np.mean(tracking_errors)
    print(f"  Mean tracking error: {mean_tracking_err:.3e} rad")
    assert mean_tracking_err < 5e-2, "Dynamic tracking performance too poor"

    print("\nAll controller tests PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
