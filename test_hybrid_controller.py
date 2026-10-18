from hybrid_controller import (
    HybridController,
    MockSkillBackend,
    RealSkillBackend,
    Skill,
    SkillResult,
    Status,
)


print("===== HYBRID CONTROLLER TEST =====")

mock = HybridController(MockSkillBackend())
assert mock.run()
assert mock.status == Status.COMPLETED
assert len(mock.history) == 3
assert not mock.real_task_success

for skill, result in mock.history:
    print(skill.value, ":", result.message)

print("Mock pipeline: PASS")
print("Mock cannot count as real task success: PASS")


real = HybridController(RealSkillBackend())
assert not real.run()
assert real.status == Status.FAILED
assert len(real.history) == 1
assert real.history[0][0] == Skill.PICK
assert not real.real_task_success

print("Missing arm correctly blocks PICK: PASS")


class FailedPick:
    def execute(self, skill):
        return SkillResult(False, "Pick failed")


failed = HybridController(
    RealSkillBackend(arm=FailedPick())
)
assert not failed.run()
assert len(failed.history) == 1

print("Failed skill stops pipeline: PASS")


class FakeArm:
    def execute(self, skill):
        return SkillResult(
            True, "Test double completed", simulated=True
        )


class FakeHand:
    def execute(self, skill):
        return SkillResult(
            True, "Test double completed", simulated=True
        )


integrated = HybridController(
    RealSkillBackend(arm=FakeArm(), hand=FakeHand())
)
assert integrated.run()
assert [s for s, _ in integrated.history] == [
    Skill.PICK,
    Skill.REORIENT_RED_UP,
    Skill.PLACE_TARGET,
]
assert not integrated.real_task_success

print("Skill execution order: PASS")
print("Test doubles cannot count as real success: PASS")

try:
    integrated.run()
    raise AssertionError("Repeated run accepted")
except RuntimeError:
    print("Repeated execution blocked: PASS")

print("\nALL HYBRID CONTROLLER TESTS PASSED")
print("Actual OpenVLA/SAC robot control: NOT CONNECTED")
