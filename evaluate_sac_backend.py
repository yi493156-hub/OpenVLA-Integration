import argparse
import csv
import time
from pathlib import Path

from hybrid_controller import Skill
from sac_hand_backend import SACHandBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    backend = SACHandBackend(max_steps=200)
    records = []

    for index in range(args.episodes):
        seed = args.seed + index
        start = time.perf_counter()
        result = backend.execute(Skill.REORIENT_RED_UP, seed=seed)
        elapsed = time.perf_counter() - start

        fields = {}
        for part in result.message.split(", "):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value

        record = {
            "seed": seed,
            "success": result.success,
            "dropped": fields.get("dropped") == "True",
            "steps": int(fields.get("steps", 0)),
            "angle_rad": float(fields.get("angle_rad", "nan")),
            "elapsed_seconds": round(elapsed, 4),
        }
        records.append(record)

        print(
            f"Episode {index + 1}/{args.episodes}: "
            f"success={record[chr(115)+chr(117)+chr(99)+chr(99)+chr(101)+chr(115)+chr(115)]}, "
            f"dropped={record[chr(100)+chr(114)+chr(111)+chr(112)+chr(112)+chr(101)+chr(100)]}, "
            f"steps={record[chr(115)+chr(116)+chr(101)+chr(112)+chr(115)]}"
        )

    total = len(records)
    success_rate = sum(r["success"] for r in records) / total
    drop_rate = sum(r["dropped"] for r in records) / total
    mean_steps = sum(r["steps"] for r in records) / total

    output = Path(
        "vla/openvla_integration/outputs/sac_v3_evaluation.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\n===== SAC V3 EVALUATION SUMMARY =====")
    print(f"Episodes: {total}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Drop rate: {drop_rate:.1%}")
    print(f"Mean steps: {mean_steps:.1f}")
    print(f"Results saved: {output}")


if __name__ == "__main__":
    main()
