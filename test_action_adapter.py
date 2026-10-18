import sys
from pathlib import Path
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from action_adapter import ActionAdapter, RobotInterface


print("===== OPENVLA ACTION ADAPTER TEST =====")

adapter = ActionAdapter()

sample_action = np.array(
    [0.01, -0.02, 0.03, 0.0, 0.1, 0.0, 1.0],
    dtype=np.float32,
)

robot_action = adapter.convert(sample_action)

assert robot_action.arm_delta.shape == (6,)
assert robot_action.gripper_command == 1.0

print("OpenVLA input:", sample_action)
print("Arm delta:", robot_action.arm_delta)
print("Gripper command:", robot_action.gripper_command)

try:
    adapter.convert(np.zeros(17))
    raise AssertionError("Invalid 17D action accepted")
except ValueError:
    print("17D mismatch rejected: PASS")

try:
    adapter.convert(np.array([0, 0, 0, 0, 0, 0, np.nan]))
    raise AssertionError("NaN accepted")
except ValueError:
    print("NaN rejected: PASS")

robot = RobotInterface(arm_available=False)

try:
    robot.execute(robot_action)
    raise AssertionError("Executed without robot arm")
except RuntimeError:
    print("Missing robot arm safely blocked: PASS")

print("\nALL ACTION ADAPTER TESTS PASSED")
print("Physical action execution: NOT CONNECTED")
