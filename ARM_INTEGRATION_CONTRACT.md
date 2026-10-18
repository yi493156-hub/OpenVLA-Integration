# OpenVLA–Arm Integration Contract

## Project task
PICK -> REORIENT_RED_UP -> PLACE_TARGET

## OpenVLA module (current implementation)
Input:
- RGB image: uint8, shape (H, W, 3)
- Natural-language instruction: string

Output:
- Action: float array, shape (7,)
- Backend identifier
- Action-space identifier
- Inference latency in milliseconds

Current normalization:
- OpenVLA unnorm_key: bridge_orig
- NOT calibrated to the ORCA arm

## Required arm-controller interface (proposal)

1. get_observation()
   Returns:
   - RGB camera image
   - Robot state
   - Object state if available for evaluation

2. execute_arm_action(action)
   Accepts:
   - Calibrated arm action, format to be agreed
   Returns:
   - Execution success/failure
   - Updated robot state

3. execute_grasp(command)
   Accepts:
   - Hand/grasp command, format to be agreed
   Returns:
   - Grasp execution status

4. get_task_feedback()
   Returns:
   - Object grasped?
   - Object dropped?
   - Red face oriented upward?
   - Object placed in target?

## Required decisions from arm teammate
- Arm model and degrees of freedom
- Arm control type (joint / Cartesian)
- Reference coordinate frame
- Position and orientation units
- Control frequency and simulation steps per command
- Workspace and velocity limits
- ORCA hand grasp interface
- Camera-to-robot coordinate transformation
- Collision handling and emergency stop behavior
- How grasp and placement success are measured

## Integration safety
- Never send raw bridge_orig actions directly to robot actuators.
- Reject invalid or nonfinite actions.
- Keep calibration unverified until experimentally confirmed.
- Validate workspace and actuator limits at execution time.
- Mock predictions must never count as real robot task success.

## Evaluation
Track separately:
- Model inference success
- Action mapping validity
- PICK success
- REORIENT_RED_UP success
- PLACE_TARGET success
- Full task success

## Current blockers
1. NVIDIA GPU for real OpenVLA inference.
2. Arm-controller implementation and agreed interface.
3. Action calibration and end-to-end validation.
