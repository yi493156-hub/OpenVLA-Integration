import time
from dataclasses import dataclass
import numpy as np


MODEL_ID = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"


@dataclass
class Prediction:
    action: np.ndarray
    latency_ms: float
    backend: str
    action_space: str


def build_prompt(instruction):
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Instruction cannot be empty")
    return (
        f"In: What action should the robot take to "
        f"{instruction}?\nOut:"
    )


def validate_image(image):
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected RGB image with shape (H, W, 3)"
        )

    if image.dtype != np.uint8:
        raise ValueError("RGB image must use uint8")

    return image


class MockOpenVLA:
    """
    Tests the software interface only.
    Does NOT represent learned OpenVLA behavior.
    """

    def predict(self, image, instruction):
        start = time.perf_counter()

        validate_image(image)
        build_prompt(instruction)

        action = np.zeros(7, dtype=np.float32)

        latency = (
            time.perf_counter() - start
        ) * 1000

        return Prediction(
            action=action,
            latency_ms=latency,
            backend="mock",
            action_space="7d_placeholder",
        )


class RealOpenVLA:
    """
    Requires compatible transformers, torch,
    model weights and an NVIDIA GPU.
    """

    def __init__(
        self,
        model_id=MODEL_ID,
        unnorm_key=DEFAULT_UNNORM_KEY,
    ):
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForVision2Seq,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Real OpenVLA requires a CUDA GPU "
                "in this implementation"
            )

        self.torch = torch
        self.device = "cuda:0"
        self.dtype = torch.bfloat16
        self.unnorm_key = unnorm_key

        print("Loading processor:", model_id)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        print("Loading OpenVLA model:", model_id)

        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    def predict(self, image, instruction):
        from PIL import Image

        rgb = validate_image(image)
        prompt = build_prompt(instruction)

        pil_image = Image.fromarray(rgb)

        self.torch.cuda.synchronize()
        start = time.perf_counter()

        inputs = self.processor(
            prompt, pil_image
        ).to(
            self.device,
            dtype=self.dtype,
        )

        with self.torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )

        self.torch.cuda.synchronize()

        latency = (
            time.perf_counter() - start
        ) * 1000

        action = np.asarray(
            action, dtype=np.float32
        ).reshape(-1)

        if action.shape != (7,):
            raise RuntimeError(
                f"Expected 7 actions, got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                "OpenVLA returned non-finite actions"
            )

        return Prediction(
            action=action,
            latency_ms=latency,
            backend="openvla-7b",
            action_space="7d_bridge_orig",
        )
