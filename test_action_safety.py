import numpy as np

from backend import Prediction
from action_safety import (
    ActionSafetyGate,
    CalibrationStatus,
    UnsafeActionError,
)


def prediction(backend, action_space, action=None):
    return Prediction(
        action=np.zeros(7, dtype=np.float32)
        if action is None else action,
        latency_ms=10.0,
        backend=backend,
        action_space=action_space,
    )


gate = ActionSafetyGate()


def expect_block(label, item):
    try:
        gate.validate(item)
    except UnsafeActionError:
        print("[PASS]", label)
    else:
        raise AssertionError(label + " was not blocked")


expect_block(
    "Mock action blocked",
    prediction("mock", "7d_placeholder"),
)

expect_block(
    "Uncalibrated real action blocked",
    prediction("openvla-7b", "7d_bridge_orig"),
)

expect_block(
    "Invalid action shape blocked",
    prediction(
        "openvla-7b",
        "7d_bridge_orig",
        np.zeros(5),
    ),
)

expect_block(
    "NaN action blocked",
    prediction(
        "openvla-7b",
        "7d_bridge_orig",
        np.array([0, 0, 0, 0, 0, 0, np.nan]),
    ),
)

assert not CalibrationStatus().ready
print("[PASS] Calibration defaults to unverified")

print("\nALL ACTION SAFETY TESTS PASSED")
