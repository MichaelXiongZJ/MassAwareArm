"""Thin MuJoCo wrapper.

Owns the (MjModel, MjData) lifecycle and exposes the minimum surface area the
rest of the package needs: step, reset, joint state, sensor reads, end-effector
pose, and the gravity+Coriolis bias term used by the inverse-dynamics estimator.

If a passive viewer is bound via :meth:`bind_viewer`, ``step`` syncs the viewer
and paces the inner loop to real time, so the FSM renders smoothly without a
separate viewer thread (and without the MjData races that come with one).
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "assets" / "scene.xml"

UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
EE_SITE = "gripper_pinch"


class MujocoEnv:
    def __init__(self, xml_path: str | Path = DEFAULT_SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self._ur5e_qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in UR5E_JOINTS]
        )
        self._ur5e_dof_adr = np.array(
            [self.model.joint(n).dofadr[0] for n in UR5E_JOINTS]
        )
        self._ee_site_id = self.model.site(EE_SITE).id
        self._viewer = None

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def bind_viewer(self, viewer) -> None:
        """Attach a passive MuJoCo viewer; ``step`` will sync and pace to real time."""
        self._viewer = viewer

    def reset(self, arm_qpos: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if arm_qpos is not None:
            self.set_arm_qpos(arm_qpos)
            self.data.ctrl[: len(arm_qpos)] = arm_qpos
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray | None = None, n: int = 1) -> None:
        if ctrl is not None:
            self.data.ctrl[: len(ctrl)] = ctrl
        for _ in range(n):
            t0 = time.perf_counter() if self._viewer is not None else 0.0
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
                remaining = self.dt - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)

    def get_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._ur5e_qpos_adr].copy()

    def get_arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self._ur5e_dof_adr].copy()

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self._ur5e_qpos_adr] = q

    def get_sensor(self, name: str) -> np.ndarray:
        sensor = self.model.sensor(name)
        start = sensor.adr[0]
        return self.data.sensordata[start : start + sensor.dim[0]].copy()

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyz, 3x3 rotation matrix) of the gripper pinch point."""
        xyz = self.data.site_xpos[self._ee_site_id].copy()
        rot = self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return xyz, rot

    @property
    def qfrc_bias(self) -> np.ndarray:
        """Generalised gravity + Coriolis term restricted to UR5e dofs."""
        return self.data.qfrc_bias[self._ur5e_dof_adr].copy()

    @property
    def actuator_force(self) -> np.ndarray:
        """Last applied actuator force on UR5e joints (read via sensor)."""
        return np.array(
            [self.get_sensor(f"tau_{n.removesuffix('_joint')}")[0] for n in UR5E_JOINTS]
        )
