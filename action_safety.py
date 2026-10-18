from dataclasses import dataclass

import numpy as np


class UnsafeActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationStatus:
    arm_mapping_verified: bool = False
    coordinate_frame_verified: bool = False
    action_limits_verified: bool = False
    gripper_mapping_verified: bool = False

    @property
    def ready(self):
        return all((
            self.arm_mapping_verified,
            self.coordinate_frame_verified,
            self.action_limits_verified,
            self.gripper_mapping_verified,
        ))


class ActionSafetyGate:
    def __init__(self, calibration=None):
        self.calibration = calibration or CalibrationStatus()

    def validate(self, prediction):
        action = np.asarray(prediction.action)

        if action.shape != (7,):
            raise UnsafeActionError(
                "OpenVLA action must contain exactly 7 values"
            )

        if not np.issubdtype(action.dtype, np.number):
            raise UnsafeActionError(
                "OpenVLA action must be numeric"
            )

        if not np.all(np.isfinite(action)):
            raise UnsafeActionError(
                "OpenVLA action contains NaN or infinity"
            )

        if prediction.backend == "mock":
            raise UnsafeActionError(
                "Mock predictions cannot control a robot"
            )

        if prediction.action_space != "7d_bridge_orig":
            raise UnsafeActionError(
                "Unsupported OpenVLA action space"
            )

        if not self.calibration.ready:
            raise UnsafeActionError(
                "Robot action mapping is not verified. "
                "Execution is blocked."
            )

        return action.copy()
