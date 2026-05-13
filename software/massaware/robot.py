"""Forward / inverse kinematics and the 6-DOF → 4-DOF joint subset.

The architecture locks two of the six UR5e joints (default: ``wrist_2`` and
``wrist_3``) so the planner only commands four active joints. ``JointSubset``
expands a 4-vector into the full 6-vector MuJoCo expects, and projects in the
reverse direction.

``Robot`` queries kinematics from a live ``MujocoEnv``: FK by setting qpos and
calling ``mj_forward``, Jacobian via ``mj_jacSite`` at the gripper pinch site,
IK via damped least-squares.
"""

from __future__ import annotations

import mujoco
import numpy as np

from massaware.mujoco_env import EE_SITE, UR5E_JOINTS, MujocoEnv

ACTIVE_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
)
LOCKED_VALUES: dict[str, float] = {
    "wrist_2_joint": np.pi / 2,
    "wrist_3_joint": 0.0,
}


class JointSubset:
    """4-active-joint ↔ 6-DOF UR5e adapter."""

    def __init__(
        self,
        active: tuple[str, ...] = ACTIVE_JOINTS,
        locked: dict[str, float] = LOCKED_VALUES,
    ):
        self.active = active
        self.locked = locked
        self._active_idx = np.array([UR5E_JOINTS.index(n) for n in active])
        self._locked_idx = np.array([UR5E_JOINTS.index(n) for n in locked])
        self._locked_values = np.array([locked[UR5E_JOINTS[i]] for i in self._locked_idx])

    def expand(self, q_active: np.ndarray) -> np.ndarray:
        """(4,) active → (6,) full arm qpos."""
        q_full = np.empty(len(UR5E_JOINTS))
        q_full[self._active_idx] = q_active
        q_full[self._locked_idx] = self._locked_values
        return q_full

    def project(self, q_full: np.ndarray) -> np.ndarray:
        """(6,) full → (4,) active."""
        return q_full[self._active_idx]

    @property
    def active_dof_idx(self) -> np.ndarray:
        """Indices of active joints inside the UR5e dof block (for Jacobian slicing)."""
        return self._active_idx


class Robot:
    def __init__(self, env: MujocoEnv, subset: JointSubset | None = None):
        self.env = env
        self.subset = subset or JointSubset()
        self._ee_site_id = env.model.site(EE_SITE).id
        self._ur5e_dof_adr = env._ur5e_dof_adr

    # --- forward kinematics ---------------------------------------------------

    def fk(self, q_active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyz, 3x3 rot) of the EE for the given 4-active-joint vector.

        Side-effect free: snapshots qpos, runs mj_kinematics, restores qpos.
        """
        q_saved = self.env.data.qpos.copy()
        self.env.set_arm_qpos(self.subset.expand(q_active))
        mujoco.mj_kinematics(self.env.model, self.env.data)
        xyz = self.env.data.site_xpos[self._ee_site_id].copy()
        rot = self.env.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        self.env.data.qpos[:] = q_saved
        mujoco.mj_kinematics(self.env.model, self.env.data)
        return xyz, rot

    # --- Jacobian -------------------------------------------------------------

    def jacobian_ee(self, q_active: np.ndarray | None = None) -> np.ndarray:
        """3x4 linear Jacobian of the EE site w.r.t. the 4 active joints.

        If ``q_active`` is given, evaluate at that pose (with a snapshot/restore);
        otherwise evaluate at the current state.
        """
        if q_active is not None:
            q_saved = self.env.data.qpos.copy()
            self.env.set_arm_qpos(self.subset.expand(q_active))
            mujoco.mj_kinematics(self.env.model, self.env.data)
            mujoco.mj_comPos(self.env.model, self.env.data)

        jacp = np.zeros((3, self.env.model.nv))
        mujoco.mj_jacSite(self.env.model, self.env.data, jacp, None, self._ee_site_id)
        active_dofs = self._ur5e_dof_adr[self.subset.active_dof_idx]
        j_active = jacp[:, active_dofs].copy()

        if q_active is not None:
            self.env.data.qpos[:] = q_saved
            mujoco.mj_kinematics(self.env.model, self.env.data)
            mujoco.mj_comPos(self.env.model, self.env.data)
        return j_active

    # --- inverse kinematics ---------------------------------------------------

    def ik(
        self,
        target_xyz: np.ndarray,
        q_seed: np.ndarray,
        *,
        max_iters: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-2,
        step_scale: float = 0.5,
    ) -> tuple[np.ndarray, bool]:
        """Damped least-squares position IK for the 4 active joints.

        Returns ``(q_active, converged)``. ``q_active`` is the best estimate
        even when the iteration cap is hit; check the bool before using it.
        """
        q = q_seed.astype(float).copy()
        target_xyz = np.asarray(target_xyz, dtype=float)
        I4 = np.eye(4)
        for _ in range(max_iters):
            xyz, _ = self.fk(q)
            err = target_xyz - xyz
            if np.linalg.norm(err) < tol:
                return q, True
            J = self.jacobian_ee(q)
            # DLS: dq = Jᵀ (J Jᵀ + λ² I)⁻¹ err  →  rewrite via the 4x4 normal eq
            dq = np.linalg.solve(J.T @ J + (damping ** 2) * I4, J.T @ err)
            q = q + step_scale * dq
        return q, False
