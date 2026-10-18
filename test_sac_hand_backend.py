from hybrid_controller import Skill
from sac_hand_backend import SACHandBackend

print("===== REAL SAC V3 BACKEND TEST =====")

backend = SACHandBackend(max_steps=200)

result = backend.execute(
    Skill.REORIENT_RED_UP,
    seed=42,
)

print("Result:", result.message)
print("Orientation success:", result.success)
print("Simulated backend:", result.simulated)

assert result.simulated is False
assert "steps=" in result.message

unsupported = backend.execute(Skill.PICK)

assert unsupported.success is False
print("Unsupported PICK correctly rejected: PASS")

print("\nSAC V3 BACKEND EXECUTION TEST PASSED")
print("Note: orientation success is not guaranteed.")
print("Full PICK-PLACE task: NOT CONNECTED")
