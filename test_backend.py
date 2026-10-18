import sys
from pathlib import Path
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from backend import (
    MockOpenVLA,
    build_prompt,
    validate_image,
)

print("===== OPENVLA INTERFACE TEST =====")

image = np.zeros(
    (240, 320, 3),
    dtype=np.uint8,
)

image[:, :, 0] = 120

instruction = (
    "pick up the cube, rotate it so the "
    "red face points upward, and place "
    "it at the target"
)

backend = MockOpenVLA()

result = backend.predict(
    image=image,
    instruction=instruction,
)

assert result.action.shape == (7,)
assert np.all(np.isfinite(result.action))
assert result.latency_ms >= 0

print("RGB image:", image.shape)
print("Instruction:", instruction)
print("Prompt:", repr(build_prompt(instruction)))
print("Backend:", result.backend)
print("Action:", result.action)
print("Action shape:", result.action.shape)
print("Latency (ms):", round(result.latency_ms, 3))

try:
    validate_image(np.zeros((240, 320)))
    raise AssertionError("Invalid image accepted")
except ValueError:
    print("Invalid image validation: PASS")

try:
    build_prompt("  ")
    raise AssertionError("Empty instruction accepted")
except ValueError:
    print("Empty instruction validation: PASS")

print("\nALL INTERFACE TESTS PASSED")
print("Real OpenVLA inference: NOT YET TESTED")
print("Robot action execution: NOT YET CONNECTED")
