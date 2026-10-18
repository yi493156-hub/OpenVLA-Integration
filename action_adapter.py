from dataclasses import dataclass
import numpy as np


@dataclass
class RobotAction:
    arm_delta: np.ndarray
    gripper_command: float
    source: str


class ActionAdapter:
    """
    OpenVLA 7D action interface.

    Assumed layout:
    [dx, dy, dz, dRx, dRy, dRz, gripper]

    This is an interface representation only.
    It is NOT an ORCA 17D actuator mapping.
    """

    def convert(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (7,):
            raise ValueError(
                f"Expected OpenVLA 7D action, got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf")

        return RobotAction(
            arm_delta=action[:6].copy(),
            gripper_command=float(action[6]),
            source="openvla",
        )


class RobotInterface:
    """
    Robot integration contract.

    No arm is currently available in MuJoCo.
    """

    def __init__(self, arm_available=False):
        self.arm_available = arm_available

    def execute(self, robot_action):
        if not self.arm_available:
            raise RuntimeError(
                "Robot arm is not integrated yet. "
                "OpenVLA action cannot be executed."
            )

        raise NotImplementedError(
            "Arm kinematics, coordinate frame mapping, "
            "control limits, and gripper mapping "
            "must be implemented before execution."
        )
