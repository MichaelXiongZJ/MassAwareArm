"""Estimator ABC and observation/result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from massaware.planner import PlannerContext


@dataclass
class EstimatorObs:
    """Per-tick observation handed to every Estimator."""
    t: float
    q: np.ndarray
    q_dot: np.ndarray
    tau_cmd: np.ndarray
    tau_meas: np.ndarray
    qfrc_bias: np.ndarray
    jacobian_ee: np.ndarray   # 3x6 linear Jacobian at the EE site
    q_ref: np.ndarray
    ee_xyz: np.ndarray        # world-frame EE site position


@dataclass
class EstimateResult:
    m_hat: float
    sigma: float | None = None
    diagnostics: dict = field(default_factory=dict)


class Estimator(ABC):
    """Base class for all mass estimators.

    Subclasses set `name` and may override the controller/gravity hooks and
    the calibration hooks. See `requires_calibration` for the contract.
    """

    name: str = "base"

    # Whether InitState must produce calibration data before WEIGH runs.
    requires_calibration: bool = False

    # ---- Controller / gravity hooks (read by WeighState and InitState) ----

    def gravity_comp_mask(self, n_joints: int) -> np.ndarray:
        """Per-joint multiplier on qfrc_bias during WEIGH (and calibration).

        Default is full gravity compensation (all ones). Override to leave
        specific joints uncompensated (e.g. PID-error returns 0 at the
        measurement joint).
        """
        return np.ones(n_joints)

    def controller_overrides(self) -> dict:
        """PID gain overrides applied while this estimator is active.

        Return {} for no override. Keys may be "kp", "ki", "kd"; values are
        np.ndarray of shape (n_joints,). Currently only Lyapunov uses this
        (returns {"ki": np.zeros(n_joints)}).
        """
        return {}

    # ---- Estimation lifecycle (called from TickLoop and WeighState) ----

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state. Called before each WEIGH hold."""
        ...

    @abstractmethod
    def update(self, obs: EstimatorObs) -> None:
        """Called every tick by TickLoop. Estimator decides what to record."""
        ...

    @abstractmethod
    def estimate(self) -> EstimateResult:
        """Called once at the end of WEIGH. Returns the mass estimate."""
        ...

    # ---- Calibration hooks (only required if requires_calibration is True) ----
    # Driven tick-by-tick by InitState's CalibrationHoldStep (see planner.py).
    # Default implementations raise so misuse is loud.

    def start_calibration(self, ctx: "PlannerContext") -> None:
        raise NotImplementedError(
            f"{self.name}.requires_calibration=True but start_calibration() is not implemented"
        )

    def update_calibration(self, obs: EstimatorObs) -> None:
        raise NotImplementedError(
            f"{self.name}.requires_calibration=True but update_calibration() is not implemented"
        )

    def finish_calibration(self) -> dict:
        """Return a dict of calibration keys to merge into configs/calibration.yaml.

        Called by InitState after the calibration hold completes; the same dict
        is then handed back via `load_calibration()` so the cached-load path
        and the fresh-calibration path produce identical estimator state.
        """
        raise NotImplementedError(
            f"{self.name}.requires_calibration=True but finish_calibration() is not implemented"
        )

    def load_calibration(self, data: dict) -> None:
        """Restore calibration state from a cached dict (configs/calibration.yaml).

        Default raises; estimators with requires_calibration=True must implement.
        """
        raise NotImplementedError(
            f"{self.name}.requires_calibration=True but load_calibration() is not implemented"
        )
