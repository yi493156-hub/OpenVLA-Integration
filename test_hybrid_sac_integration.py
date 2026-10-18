from hybrid_controller import (
    HybridController,
    RealSkillBackend,
    Skill,
)
from sac_hand_backend import SACHandBackend


print("===== HYBRID + REAL SAC INTEGRATION =====")

hand = SACHandBackend(max_steps=200)

print("\nTEST 1: Real SAC hand execution")
result = hand.execute(
    Skill.REORIENT_RED_UP,
    seed=42,
)
print(result.message)

assert result.simulated is False
print("Real trained SAC policy executed: PASS")


print("\nTEST 2: Hybrid controller without arm")
backend = RealSkillBackend(
    arm=None,
    hand=hand,
)

controller = HybridController(backend)
task_result = controller.run()

assert task_result is False
assert controller.real_task_success is False
assert len(controller.history) == 1
assert controller.history[0][0] == Skill.PICK

print("PICK correctly blocked without arm: PASS")
print("Full task success correctly rejected: PASS")


print("\nTEST 3: SAC rejects arm skills")
assert hand.execute(Skill.PICK).success is False
assert hand.execute(Skill.PLACE_TARGET).success is False

print("SAC only controls reorientation: PASS")

print("\nALL HYBRID + SAC INTEGRATION TESTS PASSED")
print("Real SAC: CONNECTED")
print("Real OpenVLA arm control: NOT CONNECTED")
print("Full task execution: NOT AVAILABLE")
