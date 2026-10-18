import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

from rl_env_v3 import ReorientationRLEnvV3
from hybrid_controller import Skill, SkillResult


class SACHandBackend:
    def __init__(self, model_path=None, max_steps=200):
        if model_path is None:
            model_path = (
                ROOT / "rl" / "models"
                / "sac_reorientation_v3.zip"
            )

        self.model_path = Path(model_path)
        self.max_steps = max_steps

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        print("Loading SAC V3:", self.model_path)
        self.model = SAC.load(
            str(self.model_path),
            device="cpu",
        )

    def execute(self, skill, seed=42):
        if skill != Skill.REORIENT_RED_UP:
            return SkillResult(
                False,
                "SAC backend only supports REORIENT_RED_UP",
            )

        env = ReorientationRLEnvV3()

        try:
            obs, info = env.reset(seed=seed)

            success = False
            dropped = False
            steps = 0

            for step in range(self.max_steps):
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                steps = step + 1
                success = bool(info.get("is_success", False))
                dropped = bool(info.get("dropped", False))

                if terminated or truncated:
                    break

            angle = float(
                info.get("red_face_up_angle_rad", np.nan)
            )

            message = (
                f"SAC V3 reorientation: "
                f"success={success}, "
                f"dropped={dropped}, "
                f"steps={steps}, "
                f"angle_rad={angle:.4f}"
            )

            return SkillResult(
                success=success and not dropped,
                message=message,
                simulated=False,
            )

        finally:
            env.close()
