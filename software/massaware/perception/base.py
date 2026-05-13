"""Perception interface and the ``CubeDetection`` data carrier.

Two backends are planned: ``GroundTruthPerception`` (Phase 3) reads the cube
body poses directly from MuJoCo; ``CameraCVPerception`` (Phase 8) will do HSV
blob detection on the overhead camera frame. Both return the same shape so
the planner doesn't care which is plugged in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class CubeDetection:
    name: str          # MuJoCo body name, e.g. "green_cube"
    color: str         # "green" | "blue" | "red"
    xyz: np.ndarray    # world position, shape (3,)
    quat: np.ndarray   # world orientation, wxyz, shape (4,)


class Perception(ABC):
    @abstractmethod
    def detect(self) -> list[CubeDetection]:
        """Return every cube currently visible to this backend."""
        ...
