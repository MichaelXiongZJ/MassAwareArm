"""Ground-truth perception: read cube poses directly from MuJoCo."""

from __future__ import annotations

from massaware.mujoco_env import MujocoEnv
from massaware.perception.base import CubeDetection, Perception

# Single payload cube. The color attribute is nominal — it labels the cube
# for the (now mass-only) target filter but does not affect physics.
CUBES: tuple[tuple[str, str], ...] = (
    ("cube", "grey"),
)


class GroundTruthPerception(Perception):
    def __init__(self, env: MujocoEnv):
        self.env = env
        self._body_ids = {name: env.model.body(name).id for name, _ in CUBES}

    def detect(self) -> list[CubeDetection]:
        return [
            CubeDetection(
                name=name,
                color=color,
                xyz=self.env.data.xpos[self._body_ids[name]].copy(),
                quat=self.env.data.xquat[self._body_ids[name]].copy(),
            )
            for name, color in CUBES
        ]
