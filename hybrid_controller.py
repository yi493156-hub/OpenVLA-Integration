from dataclasses import dataclass
from enum import Enum


class Skill(Enum):
    PICK = "pick"
    REORIENT_RED_UP = "reorient_red_up"
    PLACE_TARGET = "place_target"


class Status(Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SkillResult:
    success: bool
    message: str
    simulated: bool = False


class MockSkillBackend:
    """Software integration test only; no robot motion."""

    def execute(self, skill):
        return SkillResult(
            success=True,
            message=f"Mock {skill.value} completed",
            simulated=True,
        )


class RealSkillBackend:
    """Strict interface for future OpenVLA and SAC integration."""

    def __init__(self, arm=None, hand=None):
        self.arm = arm
        self.hand = hand

    def execute(self, skill):
        if skill in (Skill.PICK, Skill.PLACE_TARGET):
            if self.arm is None:
                return SkillResult(
                    False,
                    "Robot arm is not integrated",
                )
            return self.arm.execute(skill)

        if skill == Skill.REORIENT_RED_UP:
            if self.hand is None:
                return SkillResult(
                    False,
                    "SAC hand controller is not integrated",
                )
            return self.hand.execute(skill)

        return SkillResult(False, "Unknown skill")


class HybridController:
    def __init__(self, backend):
        self.backend = backend
        self.skills = [
            Skill.PICK,
            Skill.REORIENT_RED_UP,
            Skill.PLACE_TARGET,
        ]
        self.status = Status.READY
        self.history = []

    def run(self):
        if self.status != Status.READY:
            raise RuntimeError("Controller can only run once")

        self.status = Status.RUNNING

        for skill in self.skills:
            try:
                result = self.backend.execute(skill)
            except Exception as exc:
                result = SkillResult(
                    False,
                    f"Backend exception: {exc}",
                )

            self.history.append((skill, result))

            if not result.success:
                self.status = Status.FAILED
                return False

        self.status = Status.COMPLETED
        return True

    @property
    def real_task_success(self):
        return (
            self.status == Status.COMPLETED
            and len(self.history) == len(self.skills)
            and all(
                result.success and not result.simulated
                for _, result in self.history
            )
        )
