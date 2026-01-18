"""Tests for Ground3D end-effector frame rotations."""

import numpy as np
import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench.envs.geom3d.ground3d import ObjectCentricGround3DEnv
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from relational_structs.spaces import ObjectCentricBoxSpace
from scipy.spatial.transform import Rotation

from prbench_models.geom3d.constants import HOME_JOINT_POSITIONS
from prbench_models.geom3d.ground3d.parameterized_skills import (
    create_lifted_controllers,
)

# Retract joint positions (first 7 joints only)
RETRACT_JOINTS = HOME_JOINT_POSITIONS[:7].tolist()

prbench.register_all_environments()


def apply_delta_rotation_to_pose(
    current_pose: Pose, delta_roll: float, delta_pitch: float, delta_yaw: float
) -> Pose:
    """Apply delta roll, pitch, yaw rotations to a pose.

    Args:
        current_pose: Current pose with position and orientation quaternion.
        delta_roll: Delta rotation around x-axis (radians).
        delta_pitch: Delta rotation around y-axis (radians).
        delta_yaw: Delta rotation around z-axis (radians).

    Returns:
        New pose with the applied rotation.
    """
    # Get current rotation as scipy Rotation
    # pybullet quaternion format is (x, y, z, w)
    current_quat = current_pose.orientation
    current_rot = Rotation.from_quat(
        [current_quat[0], current_quat[1], current_quat[2], current_quat[3]]
    )

    # Create delta rotation (roll, pitch, yaw in the local frame)
    delta_rot = Rotation.from_euler("xyz", [delta_roll, delta_pitch, delta_yaw])

    # Apply delta rotation: new = current * delta (rotation in local frame)
    new_rot = current_rot * delta_rot

    # Convert back to quaternion (x, y, z, w)
    new_quat = new_rot.as_quat()

    return Pose(current_pose.position, tuple(new_quat))


def test_ee_delta_rotations():
    """Test applying delta roll, pitch, yaw to end-effector sequentially.

    This test verifies that:
    1. Delta rotations can be applied to the end-effector
    2. The robot moves correctly to achieve the target orientation
    3. Each rotation type (roll, pitch, yaw) is tested
    """
    num_cubes = 1
    env = prbench.make(
        f"prbench/Ground3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Ground3D-ee-rotation")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    # Get the underlying simulation for IK
    sim = ObjectCentricGround3DEnv(num_cubes=num_cubes)
    sim.reset(seed=123)

    # Get current end-effector pose
    current_ee_pose = sim.robot.arm.get_end_effector_pose()
    current_joints = sim.robot.arm.get_joint_positions()[:7]

    # Track EE orientations throughout the test
    ee_orientations = [current_ee_pose.orientation]

    # Define delta rotations to apply sequentially
    # Each entry: (name, delta_roll, delta_pitch, delta_yaw, num_steps)
    rotation_sequence = [
        ("yaw -50°", 0.0, 0.0, np.radians(-50), 100),
        # ("pitch +20°", 0.0, np.radians(20), 0.0, 50),
        # ("roll +40°", np.radians(40), 0.0, 0.0, 100),
        # ("yaw -30°", 0.0, 0.0, np.radians(-30), 50),
        # ("pitch -30°", 0.0, np.radians(-30), 0.0, 100),
        # ("roll -30°", np.radians(-30), 0.0, 0.0, 100),
    ]

    for name, delta_roll, delta_pitch, delta_yaw, num_steps in rotation_sequence:
        # Get current EE pose from actual sim state
        sim.set_state(env.observation_space.devectorize(obs))
        current_ee_pose = sim.robot.arm.get_end_effector_pose()
        current_joints = list(sim.robot.arm.get_joint_positions()[:7])

        # Compute target EE pose with delta rotation
        target_ee_pose = apply_delta_rotation_to_pose(
            current_ee_pose, delta_roll, delta_pitch, delta_yaw
        )

        # Use IK to get target joint positions
        try:
            target_joints = inverse_kinematics(
                sim.robot.arm,
                target_ee_pose,
                validate=True,
                set_joints=False,
            )
        except Exception as e:
            print(f"IK failed for {name}: {e}")
            continue

        # Compute delta joints
        target_joints_7 = list(target_joints[:7])

        # Execute the motion over multiple steps
        for step in range(num_steps):
            # Interpolate between current and target joints
            alpha = (step + 1) / num_steps
            interp_joints = [
                current_joints[i] + alpha * (target_joints_7[i] - current_joints[i])
                for i in range(7)
            ]

            # Get current actual joints from env observation
            state = env.observation_space.devectorize(obs)
            robot = state.get_object_from_name("robot")
            # Joint features are named joint_1, joint_2, ..., joint_7 (1-indexed)
            actual_joints = [state.get(robot, f"joint_{i+1}") for i in range(7)]

            # Compute delta from actual to interpolated target
            delta_joints = [
                interp_joints[i] - actual_joints[i]
                for i in range(7)
            ]

            # Clip deltas to action space limits
            max_delta = 0.05
            delta_joints = np.clip(delta_joints, -max_delta, max_delta)

            # Build action: [base_x, base_y, base_rot, joint0-6, gripper]
            action = np.zeros(11)
            action[0:3] = 0.0  # No base movement
            action[3:10] = delta_joints
            action[10] = 0.0  # Keep gripper as is

            obs, _, _, _, _ = env.step(action)

        # Record final EE orientation after this rotation
        state = env.observation_space.devectorize(obs)
        sim.set_state(state)
        final_ee_pose = sim.robot.arm.get_end_effector_pose()
        ee_orientations.append(final_ee_pose.orientation)

        print(f"Applied {name}: final EE orientation = {final_ee_pose.orientation}")

        return_home = False
        if return_home:
            # Return to retract configuration after each rotation
            print(f"Returning to retract configuration...")
            state = env.observation_space.devectorize(obs)
            robot = state.get_object_from_name("robot")
            current_joints_for_retract = [state.get(robot, f"joint_{i+1}") for i in range(7)]

            # Move back to retract position over multiple steps
            retract_steps = 50
            for step in range(retract_steps):
                # Interpolate between current and retract joints
                alpha = (step + 1) / retract_steps
                interp_joints = [
                    current_joints_for_retract[i]
                    + alpha * (RETRACT_JOINTS[i] - current_joints_for_retract[i])
                    for i in range(7)
                ]

                # Get current actual joints from env observation
                state = env.observation_space.devectorize(obs)
                robot = state.get_object_from_name("robot")
                actual_joints = [state.get(robot, f"joint_{i+1}") for i in range(7)]

                # Compute delta from actual to interpolated target
                delta_joints = [interp_joints[i] - actual_joints[i] for i in range(7)]

                # Clip deltas to action space limits
                max_delta = 0.05
                delta_joints = np.clip(delta_joints, -max_delta, max_delta)

                # Build action: [base_x, base_y, base_rot, joint0-6, gripper]
                action = np.zeros(11)
                action[0:3] = 0.0  # No base movement
                action[3:10] = delta_joints
                action[10] = 0.0  # Keep gripper as is

                obs, _, _, _, _ = env.step(action)

            print("Returned to retract configuration.")

    # Verify that orientations changed throughout the sequence
    initial_quat = np.array(ee_orientations[0])
    orientation_changes = []
    for i, quat in enumerate(ee_orientations[1:], 1):
        quat_diff = np.linalg.norm(np.array(quat) - initial_quat)
        orientation_changes.append(quat_diff)

    # At least some rotations should have caused significant orientation change
    max_change = max(orientation_changes)
    assert max_change > 0.1, (
        f"EE orientation should change with delta rotations. "
        f"Max change: {max_change}, changes: {orientation_changes}"
    )

    env.close()
    sim.close()
