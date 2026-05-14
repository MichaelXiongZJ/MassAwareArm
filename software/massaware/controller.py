"""Joint-space PID controller with gravity compensation."""

from __future__ import annotations

import numpy as np


class PIDController:
    """PID controller for 6-DOF arm."""

    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        self.kp = np.asarray(kp)
        self.ki = np.asarray(ki)
        self.kd = np.asarray(kd)
        self._integral_err = np.zeros(6)

    def reset(self) -> None:
        """Clear integral state."""
        self._integral_err.fill(0.0)

    def compute(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        qfrc_bias: np.ndarray,
        *,
        use_gravity_comp: bool = True,
    ) -> np.ndarray:
        """Compute control torques (Nm)."""
        err = q_ref - q
        self._integral_err += err
        
        tau = (self.kp * err) + (self.ki * self._integral_err) - (self.kd * q_dot)
        
        if use_gravity_comp:
            tau += qfrc_bias
            
        return tau
