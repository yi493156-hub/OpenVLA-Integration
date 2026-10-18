import csv
from collections import Counter
from enum import Enum
from pathlib import Path


class Outcome(str, Enum):
    NETWORK_FAILURE = "NETWORK_FAILURE"
    MODEL_FAILURE = "MODEL_FAILURE"
    INVALID_ACTION = "INVALID_ACTION"
    ACTION_MAPPING_FAILURE = "ACTION_MAPPING_FAILURE"
    PICK_FAILURE = "PICK_FAILURE"
    REORIENTATION_FAILURE = "REORIENTATION_FAILURE"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    PLACE_FAILURE = "PLACE_FAILURE"
    TASK_SUCCESS = "TASK_SUCCESS"


def classify_sac(success, dropped):
    if dropped:
        return Outcome.OBJECT_DROPPED
    if success:
        return None
    return Outcome.REORIENTATION_FAILURE


def analyze_sac_csv(input_path, output_path=None):
    input_path = Path(input_path)

    with input_path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if not records:
        raise ValueError("Evaluation CSV is empty")

    outcomes = []

    for record in records:
        success = record["success"].strip().lower() == "true"
        dropped = record["dropped"].strip().lower() == "true"

        outcome = classify_sac(success, dropped)

        if outcome is None:
            outcomes.append("REORIENTATION_SUCCESS")
        else:
            outcomes.append(outcome.value)

    counts = Counter(outcomes)

    print("===== SAC V3 FAILURE ANALYSIS =====")
    print("Episodes:", len(records))

    for name in (
        "REORIENTATION_SUCCESS",
        "OBJECT_DROPPED",
        "REORIENTATION_FAILURE",
    ):
        count = counts[name]
        print(
            "{}: {} ({:.1%})".format(
                name, count, count / len(records)
            )
        )

    print("Full task success: NOT MEASURED")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["outcome", "count", "rate"])

            for name in (
                "REORIENTATION_SUCCESS",
                "OBJECT_DROPPED",
                "REORIENTATION_FAILURE",
            ):
                writer.writerow([
                    name,
                    counts[name],
                    counts[name] / len(records),
                ])

        print("Summary saved:", output_path)

    return counts


if __name__ == "__main__":
    analyze_sac_csv(
        "vla/openvla_integration/outputs/sac_v3_evaluation.csv",
        "vla/openvla_integration/outputs/sac_v3_failure_summary.csv",
    )
