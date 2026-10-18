import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from backend import MockOpenVLA, RealOpenVLA


HERE = Path(__file__).resolve().parent


def check_environment():
    print("===== GPU ENVIRONMENT CHECK =====")
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())

    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print(
                "GPU memory (GB):",
                round(
                    torch.cuda.get_device_properties(0).total_memory
                    / 1024**3,
                    2,
                ),
            )
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    try:
        import transformers
        print("Transformers:", transformers.__version__)
    except ImportError:
        print("Transformers: NOT INSTALLED")


def percentile(values, p):
    return float(np.percentile(values, p))


def evaluate(backend, image, instruction, warmup, runs):
    for _ in range(warmup):
        backend.predict(image, instruction)

    latencies = []
    actions = []

    for i in range(runs):
        result = backend.predict(image, instruction)

        action = np.asarray(result.action, dtype=np.float32)

        if action.shape != (7,):
            raise RuntimeError(
                f"Run {i}: expected 7D action, got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"Run {i}: non-finite action"
            )

        latencies.append(float(result.latency_ms))
        actions.append(action.tolist())

    return {
        "backend": result.backend,
        "runs": runs,
        "warmup": warmup,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "latencies_ms": latencies,
        "actions": actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["mock", "real"],
        default="mock",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--image", default=None)
    parser.add_argument(
        "--instruction",
        default="rotate the cube so the red face points upward",
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "outputs" / "evaluation.json"),
    )

    args = parser.parse_args()

    if args.runs <= 0 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup nonnegative")

    check_environment()

    if args.image:
        with Image.open(args.image) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)
    else:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :, 0] = 120

    print("\n===== OPENVLA EVALUATION =====")
    print("Backend:", args.backend)
    print("Image shape:", image.shape)
    print("Instruction:", args.instruction)

    if args.backend == "real":
        backend = RealOpenVLA()
    else:
        backend = MockOpenVLA()

    result = evaluate(
        backend,
        image,
        args.instruction,
        args.warmup,
        args.runs,
    )

    result["timestamp_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    result["model_id"] = (
        "openvla/openvla-7b"
        if args.backend == "real"
        else None
    )

    result["is_real_model"] = args.backend == "real"
    result["instruction"] = args.instruction
    result["image_shape"] = list(image.shape)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\n===== RESULTS =====")
    print("Runs:", result["runs"])
    print(
        "Mean latency (ms):",
        round(result["mean_latency_ms"], 3),
    )
    print(
        "Median latency (ms):",
        round(result["median_latency_ms"], 3),
    )
    print(
        "P95 latency (ms):",
        round(result["p95_latency_ms"], 3),
    )
    print("Results saved:", output)

    if args.backend == "mock":
        print("\nMOCK ONLY — NOT REAL OPENVLA PERFORMANCE")
    else:
        print("\nREAL MODEL INFERENCE COMPLETED")
        print("Robot task success: NOT EVALUATED")


if __name__ == "__main__":
    main()
