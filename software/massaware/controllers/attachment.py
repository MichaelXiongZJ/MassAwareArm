"""Demo attachment model for tracking-controller experiments."""

from __future__ import annotations

import mujoco
import numpy as np

from massaware.mujoco_env import EE_SITE, MujocoEnv
from massaware.robot import Robot


class TrackingAttachment:
    """Kinematically pin a free-joint payload to the end-effector while carried."""

    def __init__(
        self,
        env: MujocoEnv,
        robot: Robot,
        body_name: str,
        gravity: float = 9.81,
    ) -> None:
        self.env = env
        self.robot = robot
        self.body_name = body_name
        self.gravity = float(gravity)
        self.attached = False
        self.collision_enabled = True
        self._ee_site_id = env.model.site(EE_SITE).id

    def set_attached(self, attached: bool) -> None:
        self.attached = attached
        if attached and self.collision_enabled:
            self.set_collision(False)
        elif not attached and not self.collision_enabled:
            self.set_collision(True)

    def update(self, payload_mass: float) -> None:
        if not self.attached:
            return
        self.pin_body_to_end_effector()
        self.apply_payload_force(payload_mass)

    def restore(self) -> None:
        self.set_attached(False)

    def pin_body_to_end_effector(self) -> None:
        qpos_addr, qvel_addr = self.freejoint_addresses()
        ee_xyz, _ = self.env.ee_pose()
        self.env.data.qpos[qpos_addr : qpos_addr + 3] = ee_xyz
        self.env.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.env.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def apply_payload_force(self, payload_mass: float) -> None:
        body_id = int(self.env.model.site_bodyid[self._ee_site_id])
        ee_xyz, _ = self.env.ee_pose()
        mujoco.mj_applyFT(
            self.env.model,
            self.env.data,
            np.array([0.0, 0.0, -payload_mass * self.gravity]),
            np.zeros(3),
            ee_xyz,
            body_id,
            self.env.data.qfrc_applied,
        )

    def set_collision(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        for geom_id in self.geom_ids():
            self.env.model.geom_contype[geom_id] = value
            self.env.model.geom_conaffinity[geom_id] = value
        self.collision_enabled = enabled

    def freejoint_addresses(self) -> tuple[int, int]:
        body_id = self.env.model.body(self.body_name).id
        joint_count = int(self.env.model.body_jntnum[body_id])
        joint_start = int(self.env.model.body_jntadr[body_id])
        for offset in range(joint_count):
            joint_id = joint_start + offset
            if self.env.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return (
                    int(self.env.model.jnt_qposadr[joint_id]),
                    int(self.env.model.jnt_dofadr[joint_id]),
                )
        raise ValueError(f"Body '{self.body_name}' does not have a free joint")

    def geom_ids(self) -> list[int]:
        body_id = self.env.model.body(self.body_name).id
        geom_start = int(self.env.model.body_geomadr[body_id])
        geom_count = int(self.env.model.body_geomnum[body_id])
        return list(range(geom_start, geom_start + geom_count))
