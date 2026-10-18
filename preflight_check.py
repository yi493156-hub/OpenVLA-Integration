import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend import (
    MODEL_ID,
    DEFAULT_UNNORM_KEY,
    MockOpenVLA,
    build_prompt,
    validate_image,
)
from action_adapter import ActionAdapter


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main():
    print("===== OPENVLA DEPLOYMENT PREFLIGHT =====")

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    instruction = "pick up the cube"

    validated = validate_image(image)
    check("RGB image shape", validated.shape == (240, 320, 3))
    check("RGB image dtype", validated.dtype == np.uint8)

    prompt = build_prompt(instruction)
    check("Prompt contains instruction", instruction in prompt)
    check("Prompt ends with Out:", prompt.endswith("Out:"))

    backend = MockOpenVLA()
    prediction = backend.predict(image, instruction)

    check("Mock returns 7D", prediction.action.shape == (7,))
    check("Mock output is finite", np.isfinite(prediction.action).all())
    check("Mock correctly identified", prediction.backend == "mock")

    adapter = ActionAdapter()
    robot_action = adapter.convert(prediction.action)

    check("Arm action interface is 6D", robot_action.arm_delta.shape == (6,))
    check("Gripper command is scalar", np.isscalar(robot_action.gripper_command))

    check("Model ID configured", MODEL_ID == "openvla/openvla-7b")
    check("Normalization key configured", DEFAULT_UNNORM_KEY == "bridge_orig")

    print("\n===== CURRENT ENVIRONMENT =====")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    try:
        import transformers
        print("Transformers:", transformers.__version__)
    except ImportError:
        print("Transformers: NOT INSTALLED")

    print("\n===== DEPLOYMENT STATUS =====")
    print("Mock inference: READY")
    print("7D software interface: READY")
    print("Robot action calibration: NOT VERIFIED")
    print("Real OpenVLA-7B inference: NOT TESTED")
    print("GPU deployment: PENDING")

    print("\nALL CPU PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
