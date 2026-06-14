"""Trajectory-tracking controllers for comparison experiments."""

from massaware.controllers.references import ControlOutput, JointReference
from massaware.controllers.pd_with_gravity_tracking import TrackingPIDController
from massaware.controllers.inverse_dynamics_tracking import (
    InverseDynamicsTrackingController,
)

__all__ = [
    "ControlOutput",
    "InverseDynamicsTrackingController",
    "JointReference",
    "TrackingPIDController",
]
