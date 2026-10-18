import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inference_client import RemoteOpenVLA


server = subprocess.Popen(
    [
        sys.executable,
        str(HERE / "inference_server.py"),
        "--backend", "mock",
        "--port", "8765",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)

try:
    client = RemoteOpenVLA()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 0] = 120

    for attempt in range(30):
        try:
            prediction, round_trip_ms = client.predict(
                image,
                "rotate the cube so the red face points upward",
            )
            break
        except URLError:
            if server.poll() is not None:
                raise RuntimeError("Server exited unexpectedly")
            time.sleep(0.2)
    else:
        raise RuntimeError("Server did not start")

    assert prediction.action.shape == (7,)
    assert prediction.backend == "mock"

    print("===== REMOTE OPENVLA TEST =====")
    print("Backend:", prediction.backend)
    print("Action:", prediction.action)
    print("Action shape:", prediction.action.shape)
    print("Server latency (ms):", round(prediction.latency_ms, 3))
    print("Round-trip latency (ms):", round(round_trip_ms, 3))
    print("\nREMOTE INFERENCE TEST PASSED")
    print("Real GPU inference: NOT YET TESTED")

finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
