import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

FIELDS = [
    "timestamp_utc", "episode_id", "step_id", "backend",
    "instruction", "action", "inference_latency_ms",
    "round_trip_latency_ms", "execution_status",
    "task_success", "failure_reason",
]

class ExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "inference_records.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, episode_id, step_id, backend, instruction,
            action, inference_latency_ms, round_trip_latency_ms=None,
            execution_status="not_executed", task_success=None,
            failure_reason=""):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected finite 7D action")
        if backend == "mock" and execution_status == "executed":
            raise ValueError("Mock cannot be real execution")
        if task_success is not None and execution_status != "executed":
            raise ValueError("Task success requires execution")
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": episode_id,
            "step_id": step_id,
            "backend": backend,
            "instruction": instruction,
            "action": json.dumps(action.tolist()),
            "inference_latency_ms": float(inference_latency_ms),
            "round_trip_latency_ms": "" if round_trip_latency_ms is None else float(round_trip_latency_ms),
            "execution_status": execution_status,
            "task_success": "" if task_success is None else bool(task_success),
            "failure_reason": failure_reason,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(record)
        return record
