from dataclasses import dataclass
import mujoco


@dataclass
class RobotCapabilities:
    actuator_count: int
    joint_count: int
    arm_joints: list
    hand_joints: list
    has_arm: bool
    has_hand: bool


def inspect_robot(model):
    joints = [
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, i
        )
        for i in range(model.njnt)
    ]

    hand_joints = [
        name for name in joints
        if name and name.startswith("right_")
    ]

    arm_joints = [
        name for name in joints
        if name and (
            "shoulder" in name.lower()
            or "elbow" in name.lower()
            or "arm_joint" in name.lower()
            or "arm_axis" in name.lower()
        )
    ]

    return RobotCapabilities(
        actuator_count=model.nu,
        joint_count=model.njnt,
        arm_joints=arm_joints,
        hand_joints=hand_joints,
        has_arm=len(arm_joints) > 0,
        has_hand=len(hand_joints) > 0,
    )


def report(capabilities):
    print("===== ROBOT CAPABILITY REPORT =====")
    print("Actuators:", capabilities.actuator_count)
    print("Joints:", capabilities.joint_count)
    print("Hand joints:", len(capabilities.hand_joints))
    print("Arm joints (name heuristic):", len(capabilities.arm_joints))
    print("Hand detected:", capabilities.has_hand)
    print("Arm detected:", capabilities.has_arm)

    if not capabilities.has_arm:
        print("STATUS: ARM INTEGRATION PENDING")
    else:
        print("STATUS: ARM CANDIDATE DETECTED")
        print("Kinematics and control mapping still require verification")
