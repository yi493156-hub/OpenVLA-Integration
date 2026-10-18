import csv
import tempfile
import numpy as np
from experiment_logger import ExperimentLogger

with tempfile.TemporaryDirectory() as directory:
    logger = ExperimentLogger(directory)
    logger.log(1, 0, "mock", "rotate the cube", np.zeros(7), 0.01)
    try:
        logger.log(2, 0, "mock", "rotate", np.zeros(7), 0.01,
                   execution_status="executed", task_success=True)
        raise AssertionError("Invalid mock execution accepted")
    except ValueError:
        print("Mock execution correctly rejected: PASS")
    try:
        logger.log(3, 0, "mock", "rotate", np.zeros(7), 0.01,
                   task_success=True)
        raise AssertionError("Invalid success accepted")
    except ValueError:
        print("Unexecuted success correctly rejected: PASS")
    with logger.csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    print("Experiment CSV logging: PASS")

print("EXPERIMENT LOGGER TEST PASSED")
