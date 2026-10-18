import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from reliable_inference_client import (
    ReliableRemoteOpenVLA,
    InferenceError,
)


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * p / 100
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765/predict",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--image",
        default="vla/openvla_integration/outputs/mujoco_rgb.png",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the cube",
    )
    parser.add_argument(
        "--output",
        default="vla/openvla_integration/outputs/openvla_inference_evaluation.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    with Image.open(args.image) as img:
        image = np.asarray(img.convert("RGB"), dtype=np.uint8)

    client = ReliableRemoteOpenVLA(
        url=args.url,
        timeout=args.timeout,
    )

    records = []

    print("===== OPENVLA REMOTE INFERENCE EVALUATION =====")
    print("URL:", args.url)
    print("Runs:", args.runs)

    for index in range(args.runs):
        start = time.perf_counter()

        record = {
            "run": index + 1,
            "success": False,
            "backend": "",
            "action_space": "",
            "inference_latency_ms": "",
            "round_trip_latency_ms": "",
            "error_type": "",
            "error_message": "",
        }

        try:
            prediction, round_trip = client.predict(
                image,
                args.instruction,
            )

            record["success"] = True
            record["backend"] = prediction.backend
            record["action_space"] = prediction.action_space
            record["inference_latency_ms"] = round(
                prediction.latency_ms, 4
            )
            record["round_trip_latency_ms"] = round(
                round_trip, 4
            )

        except (InferenceError, ValueError, OSError) as exc:
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)

        records.append(record)

        print(
            "Run {}/{}: {}".format(
                index + 1,
                args.runs,
                "OK" if record["success"] else record["error_type"],
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    successful = [r for r in records if r["success"]]
    inference = [
        float(r["inference_latency_ms"]) for r in successful
    ]
    round_trip = [
        float(r["round_trip_latency_ms"]) for r in successful
    ]

    print("\n===== EVALUATION SUMMARY =====")
    print("Total requests:", len(records))
    print("Successful requests:", len(successful))
    print("Request success rate: {:.1%}".format(
        len(successful) / len(records)
    ))

    if successful:
        print("Backend:", successful[0]["backend"])
        print("Mean inference latency: {:.3f} ms".format(
            statistics.mean(inference)
        ))
        print("P95 inference latency: {:.3f} ms".format(
            percentile(inference, 95)
        ))
        print("Mean round-trip latency: {:.3f} ms".format(
            statistics.mean(round_trip)
        ))
        print("P95 round-trip latency: {:.3f} ms".format(
            percentile(round_trip, 95)
        ))

    print("CSV saved:", output)
    print("Robot task success: NOT MEASURED")


if __name__ == "__main__":
    main()
