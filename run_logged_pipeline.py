import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from orca_sim import OrcaHandRightCubeOrientation
from inference_client import RemoteOpenVLA
from action_adapter import ActionAdapter
from experiment_logger import ExperimentLogger


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    print("===== LOGGED OPENVLA PIPELINE =====")

    output = HERE / "outputs" / "logged_pipeline"
    logger = ExperimentLogger(output)

    env = None
    server = None

    try:
        env = OrcaHandRightCubeOrientation(
            version="v2",
            render_mode="rgb_array",
            max_episode_steps=200,
        )

        options = env.sample_randomized_reset_options(
            seed=42,
            initial_red_face="random",
        )
        env.reset(seed=42, options=options)
        image = env.render()

        port = free_port()
        server = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "inference_server.py"),
                "--backend", "mock",
                "--host", "127.0.0.1",
                "--port", str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        client = RemoteOpenVLA(
            url=f"http://127.0.0.1:{port}/predict"
        )

        instruction = (
            "rotate the cube so the red face points upward"
        )

        for attempt in range(30):
            try:
                prediction, round_trip = client.predict(
                    image, instruction
                )
                break
            except URLError:
                if server.poll() is not None:
                    raise RuntimeError("Server exited")
                time.sleep(0.2)
        else:
            raise RuntimeError("Server startup timeout")

        action = ActionAdapter().convert(prediction.action)

        record = logger.log(
            episode_id=1,
            step_id=0,
            backend=prediction.backend,
            instruction=instruction,
            action=prediction.action,
            inference_latency_ms=prediction.latency_ms,
            round_trip_latency_ms=round_trip,
            execution_status="not_executed",
            task_success=None,
        )

        assert record["backend"] == "mock"
        assert record["task_success"] == ""
        assert action.arm_delta.shape == (6,)

        print("Image shape:", np.asarray(image).shape)
        print("Backend:", prediction.backend)
        print("OpenVLA action:", prediction.action)
        print("Arm delta:", action.arm_delta)
        print("Gripper:", action.gripper_command)
        print("Round-trip (ms):", round(round_trip, 3))
        print("CSV saved:", logger.csv_path)
        print("\nLOGGED PIPELINE TEST PASSED")
        print("Real OpenVLA: NOT TESTED")
        print("Robot execution: NOT CONNECTED")

    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
