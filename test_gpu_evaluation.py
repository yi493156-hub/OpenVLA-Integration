import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / "evaluation.json"

    command = [
        sys.executable,
        str(HERE / "gpu_evaluation.py"),
        "--backend", "mock",
        "--runs", "10",
        "--warmup", "2",
        "--output", str(output),
    ]

    subprocess.run(command, check=True)

    assert output.exists()

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["runs"] == 10
    assert result["warmup"] == 2
    assert result["is_real_model"] is False
    assert len(result["latencies_ms"]) == 10
    assert len(result["actions"]) == 10
    assert all(len(action) == 7 for action in result["actions"])
    assert result["mean_latency_ms"] >= 0
    assert result["p95_latency_ms"] >= 0

print("\nGPU EVALUATION SCRIPT TEST PASSED")
print("Real GPU inference: PENDING")
