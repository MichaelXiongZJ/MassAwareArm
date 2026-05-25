"""Joint-space PID controller with gravity compensation."""

from __future__ import annotations

import numpy as np


class PIDController:
    """PID controller for 6-DOF arm."""

    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self._integral_err = np.zeros(6)
        # Stash for apply_overrides / clear_overrides
        self._base_gains: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        """Clear integral state."""
        self._integral_err.fill(0.0)

    def apply_overrides(self, overrides: dict) -> None:
        """Temporarily replace gains. Keys may be 'kp', 'ki', 'kd'.

        The current gains are stashed and restored by `clear_overrides()`.
        Integral state is reset so a stale wind-up does not bleed into the
        new regime. Calling apply_overrides() while overrides are already
        active is an error.
        """
        if self._base_gains is not None:
            raise RuntimeError("PIDController overrides already active; call clear_overrides() first")
        if not overrides:
            return
        self._base_gains = {"kp": self.kp, "ki": self.ki, "kd": self.kd}
        for key, value in overrides.items():
            if key not in ("kp", "ki", "kd"):
                raise KeyError(f"Unknown PID gain '{key}' in overrides")
            setattr(self, key, np.asarray(value, dtype=float))
        self.reset()

    def clear_overrides(self) -> None:
        """Restore the stashed gains. No-op if no overrides are active."""
        if self._base_gains is None:
            return
        self.kp = self._base_gains["kp"]
        self.ki = self._base_gains["ki"]
        self.kd = self._base_gains["kd"]
        self._base_gains = None
        self.reset()

    def compute(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        qfrc_bias: np.ndarray,
        dt: float,
        *,
        use_gravity_comp: bool = True,
    ) -> np.ndarray:
        """Compute control torques (Nm)."""
        err = q_ref - q
        self._integral_err += err * dt
        
        tau = (self.kp * err) + (self.ki * self._integral_err) - (self.kd * q_dot)
        
        if use_gravity_comp:
            tau += qfrc_bias
            
        return tau
