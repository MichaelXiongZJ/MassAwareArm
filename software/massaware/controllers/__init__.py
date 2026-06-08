"""Trajectory-tracking controllers for comparison experiments."""

from massaware.controllers.references import ControlOutput, JointReference
from massaware.controllers.pid_tracking import TrackingPIDController
from massaware.controllers.inverse_dynamics_tracking import (
    InverseDynamicsTrackingController,
)

__all__ = [
    "ControlOutput",
    "InverseDynamicsTrackingController",
    "JointReference",
    "TrackingPIDController",
]
