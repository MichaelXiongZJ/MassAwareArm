"""Forward / inverse kinematics for the UR5e."""

from __future__ import annotations

from contextlib import contextmanager

import mujoco
import numpy as np

from massaware.mujoco_env import EE_SITE, MujocoEnv


class Robot:
    """IK and FK utilities."""

    def __init__(self, env: MujocoEnv):
        self.env = env
        self._ee_site_id = env.model.site(EE_SITE).id
        self._ur5e_dof_adr = env._ur5e_dof_adr

    @contextmanager
    def _at(self, q: np.ndarray):
        """Temporarily set arm state for kinematics; restore on exit."""
        q_saved = self.env.data.qpos.copy()
        self.env.set_arm_qpos(q)
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_comPos(self.env.model, self.env.data)
        try:
            yield
        finally:
            self.env.data.qpos[:] = q_saved
            mujoco.mj_kinematics(self.env.model, self.env.data)
            mujoco.mj_comPos(self.env.model, self.env.data)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: return (xyz, 3x3 rot)."""
        with self._at(q):
            xyz = self.env.data.site_xpos[self._ee_site_id].copy()
            rot = self.env.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return xyz, rot

    def jacobian_ee(self, q: np.ndarray) -> np.ndarray:
        """Linear Jacobian of end-effector site."""
        with self._at(q):
            jacp = np.zeros((3, self.env.model.nv))
            mujoco.mj_jacSite(self.env.model, self.env.data, jacp, None, self._ee_site_id)
            return jacp[:, self._ur5e_dof_adr].copy()

    def ik(
        self,
        target_xyz: np.ndarray,
        q_seed: np.ndarray,
        *,
        max_iters: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-2,
        step_scale: float = 0.5,
        max_step: float = 0.1,
    ) -> tuple[np.ndarray, bool]:
        """Position-only DLS inverse kinematics."""
        q = q_seed.astype(float).copy()
        target_xyz = np.asarray(target_xyz, dtype=float)
        I_n = np.eye(len(q))
        for _ in range(max_iters):
            xyz, _ = self.fk(q)
            err = target_xyz - xyz
            if np.linalg.norm(err) < tol:
                return q_seed + ((q - q_seed + np.pi) % (2 * np.pi)) - np.pi, True
            J = self.jacobian_ee(q)
            dq = np.linalg.solve(J.T @ J + (damping ** 2) * I_n, J.T @ err)
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q = q + step_scale * dq
        return q, False
