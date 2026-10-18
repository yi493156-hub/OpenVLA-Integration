import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from orca_sim import OrcaHandRightCubeOrientation
from backend import MockOpenVLA, validate_image


def main():
    output_dir = ROOT / "vla" / "openvla_integration" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== MUJOCO RGB CAMERA TEST =====")

    env = OrcaHandRightCubeOrientation(
        version="v2",
        render_mode="rgb_array",
        max_episode_steps=200,
    )

    try:
        options = env.sample_randomized_reset_options(
            seed=42,
            initial_red_face="random",
        )

        obs, info = env.reset(
            seed=42,
            options=options,
        )

        image = env.render()

        if image is None:
            raise RuntimeError("MuJoCo render returned None")

        image = validate_image(image)

        print("Observation shape:", np.asarray(obs).shape)
        print("RGB image shape:", image.shape)
        print("RGB image dtype:", image.dtype)

        from PIL import Image

        image_path = output_dir / "mujoco_rgb.png"
        Image.fromarray(image).save(image_path)

        print("Image saved:", image_path)

        instruction = (
            "rotate the cube so the red face "
            "points upward"
        )

        backend = MockOpenVLA()
        prediction = backend.predict(image, instruction)

        print("\n===== OPENVLA INPUT PIPELINE =====")
        print("Instruction:", instruction)
        print("Backend:", prediction.backend)
        print("Action shape:", prediction.action.shape)
        print("Action:", prediction.action)
        print("Latency (ms):", round(prediction.latency_ms, 3))

        assert prediction.action.shape == (7,)

        print("\nCAMERA PIPELINE TEST PASSED")
        print("Real OpenVLA inference: NOT YET TESTED")
        print("Robot action execution: NOT YET CONNECTED")

    finally:
        env.close()


if __name__ == "__main__":
    main()
