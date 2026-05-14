"""Thin MuJoCo wrapper for robot state and sensors."""

from __future__ import annotations

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
    """Interface for MuJoCo physics state."""

    def __init__(self, xml_path: str | Path = DEFAULT_SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self._ur5e_qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in UR5E_JOINTS]
        )
        self._ur5e_dof_adr = np.array(
            [self.model.joint(n).dofadr[0] for n in UR5E_JOINTS]
        )
        
        actuator_names = [n.removesuffix("_joint") for n in UR5E_JOINTS]
        self._ur5e_ctrl_adr = np.array(
            [self.model.actuator(n).id for n in actuator_names]
        )
        self._ee_site_id = self.model.site(EE_SITE).id

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def reset(self, arm_qpos: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if arm_qpos is not None:
            self.set_arm_qpos(arm_qpos)
        mujoco.mj_forward(self.model, self.data)

    def get_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._ur5e_qpos_adr].copy()

    def get_arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self._ur5e_dof_adr].copy()

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self._ur5e_qpos_adr] = q

    def set_arm_ctrl(self, tau: np.ndarray) -> None:
        """Inject control torques."""
        self.data.ctrl[self._ur5e_ctrl_adr] = tau

    def get_sensor(self, name: str) -> np.ndarray:
        sensor = self.model.sensor(name)
        start = sensor.adr[0]
        return self.data.sensordata[start : start + sensor.dim[0]].copy()

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyz, 3x3 rotation matrix) of end-effector."""
        xyz = self.data.site_xpos[self._ee_site_id].copy()
        rot = self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return xyz, rot

    @property
    def qfrc_bias(self) -> np.ndarray:
        """Generalized gravity + Coriolis terms."""
        return self.data.qfrc_bias[self._ur5e_dof_adr].copy()

    @property
    def actuator_force(self) -> np.ndarray:
        """Actuator forces read via sensors."""
        return np.array(
            [self.get_sensor(f"tau_{n.removesuffix('_joint')}")[0] for n in UR5E_JOINTS]
        )
