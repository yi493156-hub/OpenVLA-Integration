import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from orca_sim import OrcaHandRightCubeOrientation
from inference_client import RemoteOpenVLA
from action_adapter import ActionAdapter, RobotInterface
from backend import validate_image


INSTRUCTION = (
    "rotate the cube so the red face points upward"
)


def find_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    print("===== FULL OPENVLA PIPELINE TEST =====")

    output_dir = HERE / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None
    server = None

    try:
        # STEP 1: Initialize MuJoCo
        print("\n[1/6] Initializing MuJoCo...")

        env = OrcaHandRightCubeOrientation(
            version="v2",
            render_mode="rgb_array",
            max_episode_steps=200,
        )

        options = env.sample_randomized_reset_options(
            seed=42,
            initial_red_face="random",
        )

        observation, info = env.reset(
            seed=42,
            options=options,
        )

        print("Observation:", np.asarray(observation).shape)

        # STEP 2: Capture real MuJoCo RGB image
        print("\n[2/6] Capturing RGB image...")

        image = validate_image(env.render())

        image_path = output_dir / "full_pipeline_rgb.png"
        Image.fromarray(image).save(image_path)

        print("RGB shape:", image.shape)
        print("Image saved:", image_path)

        # STEP 3: Start local Mock inference server
        print("\n[3/6] Starting Mock OpenVLA server...")

        port = find_free_port()

        server = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "inference_server.py"),
                "--backend", "mock",
                "--host", "127.0.0.1",
                "--port", str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        client = RemoteOpenVLA(
            url=f"http://127.0.0.1:{port}/predict"
        )

        # STEP 4: Send MuJoCo image and language instruction
        print("\n[4/6] Sending image and instruction...")

        for attempt in range(30):
            try:
                prediction, round_trip_ms = client.predict(
                    image=image,
                    instruction=INSTRUCTION,
                )
                break
            except (URLError, ConnectionError):
                if server.poll() is not None:
                    raise RuntimeError(
                        "Inference server exited unexpectedly"
                    )
                time.sleep(0.2)
        else:
            raise RuntimeError(
                "Inference server did not become ready"
            )

        assert prediction.backend == "mock"
        assert prediction.action.shape == (7,)

        print("Instruction:", INSTRUCTION)
        print("Backend:", prediction.backend)
        print("Predicted action:", prediction.action)
        print(
            "Server latency (ms):",
            round(prediction.latency_ms, 3),
        )
        print(
            "Round-trip latency (ms):",
            round(round_trip_ms, 3),
        )

        # STEP 5: Parse action safely
        print("\n[5/6] Converting OpenVLA action...")

        adapter = ActionAdapter()
        robot_action = adapter.convert(prediction.action)

        print("Arm delta:", robot_action.arm_delta)
        print("Gripper command:", robot_action.gripper_command)

        assert robot_action.arm_delta.shape == (6,)

        # STEP 6: Confirm unavailable robot arm blocks execution
        print("\n[6/6] Checking robot execution safety...")

        robot = RobotInterface(arm_available=False)

        try:
            robot.execute(robot_action)
        except RuntimeError as exc:
            print("Execution safely blocked:", exc)
        else:
            raise AssertionError(
                "Robot action executed without an arm!"
            )

        print("\n===================================")
        print("FULL DATA PIPELINE TEST PASSED")
        print("===================================")
        print("MuJoCo RGB capture: PASS")
        print("Language input: PASS")
        print("HTTP inference: PASS (MOCK)")
        print("7D action validation: PASS")
        print("Robot execution safety: PASS")
        print("Real OpenVLA inference: NOT TESTED")
        print("Robot control: NOT CONNECTED")

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
