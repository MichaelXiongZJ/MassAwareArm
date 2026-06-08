"""Shared controller reference and output dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointReference:
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray


@dataclass(frozen=True)
class ControlOutput:
    tau_cmd: np.ndarray
    tau_feedforward: np.ndarray
    tau_nominal: np.ndarray
    tau_payload: np.ndarray
    q_error: np.ndarray
    q_dot_error: np.ndarray
