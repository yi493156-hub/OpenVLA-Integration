import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from orca_sim import OrcaHandRightCubeOrientation
from robot_capability import inspect_robot, report

env = OrcaHandRightCubeOrientation(
    version="v2",
    render_mode=None,
    max_episode_steps=200,
)

try:
    model = env.unwrapped.model
    capabilities = inspect_robot(model)
    report(capabilities)

    assert capabilities.actuator_count == 17
    assert capabilities.has_hand
    assert not capabilities.has_arm

    print("\nROBOT CAPABILITY TEST PASSED")
finally:
    env.close()
