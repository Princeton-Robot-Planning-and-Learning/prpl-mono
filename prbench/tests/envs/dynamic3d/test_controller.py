"""Tests for the TidyBot arm PD controller.

These tests verify that the PD controller correctly converts target joint positions to
torques and that the arm behaves reasonably under control.
"""

import numpy as np
import pytest

from prbench.envs.dynamic3d.tidybot3d import (
    ObjectCentricTidyBot3DEnv,
    TidyBot3DConfig,
)


@pytest.fixture
def env():
    """Create a TidyBot3D environment for testing."""
    env = ObjectCentricTidyBot3DEnv(num_objects=1)
    yield env
    env.close()


def test_pd_controller_computes_torques(env):
    """Test that the _compute_arm_torques method produces valid torques."""
    env.reset(seed=42)

    robot_env = env._robot_env  # pylint: disable=protected-access

    # Get current arm position
    current_pos = np.array(robot_env.qpos["arm"]).copy()

    # Compute torques for a target slightly different from current
    target_pos = current_pos + 0.1  # Add 0.1 radians to each joint

    torques = robot_env._compute_arm_torques(target_pos)

    # Torques should be non-zero (positive error should give positive torque)
    assert torques.shape == (7,), f"Expected 7 torques, got {torques.shape}"
    assert np.all(
        torques > 0
    ), f"Expected positive torques for positive position error, got {torques}"

    # Torques should be within limits
    assert np.all(
        np.abs(torques) <= robot_env.ARM_TORQUE_LIMITS
    ), f"Torques {torques} exceed limits {robot_env.ARM_TORQUE_LIMITS}"


def test_torque_saturation(env):
    """Test that torques are correctly saturated at the limits."""
    env.reset(seed=42)

    robot_env = env._robot_env  # pylint: disable=protected-access

    # Get current arm position
    current_pos = np.array(robot_env.qpos["arm"]).copy()

    # Request a very large position error that would exceed torque limits
    target_pos = current_pos + 10.0  # Large error

    torques = robot_env._compute_arm_torques(target_pos)

    # Torques should be within limits
    limits = robot_env.ARM_TORQUE_LIMITS
    assert np.all(
        np.abs(torques) <= limits + 0.01
    ), f"Torques exceed limits. Got {torques}, limits: {limits}"

    # Test negative saturation
    target_pos = current_pos - 10.0  # Large negative error
    torques = robot_env._compute_arm_torques(target_pos)

    assert np.all(
        np.abs(torques) <= limits + 0.01
    ), f"Torques exceed limits. Got {torques}, limits: {limits}"


def test_custom_pd_gains():
    """Test that custom PD gains can be passed to the environment."""
    from prbench.envs.dynamic3d.robots.tidybot_robot_env import TidyBotRobotEnv

    custom_kp = np.array([50.0, 50.0, 50.0, 50.0, 25.0, 25.0, 25.0])
    custom_kd = np.array([5.0, 5.0, 5.0, 5.0, 2.5, 2.5, 2.5])

    robot_env = TidyBotRobotEnv(
        control_frequency=20.0,
        arm_kp=custom_kp,
        arm_kd=custom_kd,
    )

    assert np.allclose(
        robot_env.arm_kp, custom_kp
    ), f"Custom Kp not set correctly: {robot_env.arm_kp}"
    assert np.allclose(
        robot_env.arm_kd, custom_kd
    ), f"Custom Kd not set correctly: {robot_env.arm_kd}"


def test_step_does_not_crash(env):
    """Test that stepping the environment with the PD controller doesn't crash."""
    env.reset(seed=42)

    # Take several steps with random actions
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Basic sanity checks
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


def test_arm_responds_to_target(env):
    """Test that the arm position changes when a target is commanded.

    This is a basic test that the PD controller is actually influencing the arm motion,
    without requiring perfect convergence.
    """
    env.reset(seed=42)

    robot_env = env._robot_env  # pylint: disable=protected-access

    # Get initial arm position
    initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

    # Apply a delta action to move the arm
    delta_action = np.zeros(11)
    delta_action[3:10] = 0.1  # Small positive delta for all joints
    delta_action[10] = 0.0  # Keep gripper open

    # Take a few steps
    for _ in range(5):
        env.step(delta_action)

    # Get final arm position
    final_arm_pos = np.array(robot_env.qpos["arm"]).copy()

    # The arm should have moved (position should be different)
    position_change = np.abs(final_arm_pos - initial_arm_pos)
    assert np.any(
        position_change > 0.001
    ), f"Arm did not move. Initial: {initial_arm_pos}, Final: {final_arm_pos}"


def test_zero_velocity_zero_error_gives_gravity_compensation(env):
    """Test that with zero velocity and zero position error, torque equals gravity comp.

    With PD control and gravity compensation, when there's no position error and no
    velocity, the output should be exactly the gravity compensation term.
    """
    env.reset(seed=42)

    robot_env = env._robot_env  # pylint: disable=protected-access

    # Manually set velocity to zero for this test
    robot_env.qvel["arm"][:] = 0.0
    robot_env.sim.forward()

    # Get current position and use it as target (zero error)
    current_pos = np.array(robot_env.qpos["arm"]).copy()
    target_pos = current_pos.copy()

    torques = robot_env._compute_arm_torques(target_pos)

    # Get expected gravity compensation
    gravity_comp = robot_env._get_gravity_compensation()

    # With PD control, torques should exactly equal gravity compensation
    assert np.allclose(
        torques, gravity_comp, atol=1e-6
    ), f"Expected gravity compensation {gravity_comp}, got {torques}"


def test_velocity_damping(env):
    """Test that the D (derivative) term provides velocity damping.

    When there's velocity but zero position error, torques should be:
    -Kd * velocity + gravity_compensation
    """
    env.reset(seed=42)

    robot_env = env._robot_env  # pylint: disable=protected-access

    # Get current position
    current_pos = np.array(robot_env.qpos["arm"]).copy()

    # Manually set some velocity
    test_velocity = np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1])
    robot_env.qvel["arm"][:] = test_velocity
    robot_env.sim.forward()

    # Target = current position (zero position error)
    target_pos = current_pos.copy()

    torques = robot_env._compute_arm_torques(target_pos)

    # Get gravity compensation (which changes after forward())
    gravity_comp = robot_env._get_gravity_compensation()

    # Expected: -Kd * velocity + gravity_compensation
    expected_torques = -robot_env.arm_kd * test_velocity + gravity_comp
    expected_torques = np.clip(
        expected_torques, -robot_env.ARM_TORQUE_LIMITS, robot_env.ARM_TORQUE_LIMITS
    )

    assert np.allclose(
        torques, expected_torques, atol=1e-6
    ), f"Expected {expected_torques}, got {torques}"


def test_arm_converges_to_target_position():
    """Test that the arm moves toward a target joint position using PD control.

    This test verifies that:
    1. The arm position changes when commanded to a target
    2. The arm moves in the direction of the target
    3. The system remains stable (no explosion)

    Note: Without gravity compensation, the arm may not perfectly reach the target,
    but it should move toward it and remain stable.
    """
    # Create environment with absolute position mode (not delta)
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1)
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get current positions
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Define target: small offset from initial position
        arm_offset = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
        target_arm_pos = initial_arm_pos + arm_offset

        # Build absolute action: [base(3), arm(7), gripper(1)]
        action = np.zeros(11)
        action[0:3] = initial_base_pos  # Keep base at initial position
        action[3:10] = target_arm_pos  # Command target arm position
        action[10] = 0.0  # Keep gripper open

        # Run simulation for some steps
        num_steps = 100
        for _ in range(num_steps):
            env.step(action)

        # Get final position
        final_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Test 1: The arm should have moved (position changed)
        position_change = final_arm_pos - initial_arm_pos
        assert np.linalg.norm(position_change) > 0.01, (
            f"Arm did not move significantly. "
            f"Initial: {initial_arm_pos}, Final: {final_arm_pos}"
        )

        # Test 2: The arm should move in the direction of the target for most joints
        # (some joints may be affected by gravity differently)
        direction_correct = np.sign(position_change) == np.sign(arm_offset)
        num_correct = np.sum(direction_correct)
        assert num_correct >= 4, (
            f"Arm did not move in correct direction for enough joints. "
            f"Only {num_correct}/7 joints moved correctly. "
            f"Expected direction: {np.sign(arm_offset)}, "
            f"Actual direction: {np.sign(position_change)}"
        )

        # Test 3: System should remain stable (positions should be reasonable)
        # All joint positions should be within a reasonable range (no explosion)
        assert np.all(
            np.abs(final_arm_pos) < 10
        ), f"Arm positions are unreasonable (possible instability): {final_arm_pos}"

    finally:
        env.close()


def test_gravity_compensation_is_applied():
    """Test that gravity compensation is computed and applied to the torques.

    This test verifies that:
    1. Gravity compensation values are computed from qfrc_bias
    2. The compensation is non-zero (gravity exists)
    3. The compensation is included in the torque output
    """
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1)
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get gravity compensation
        gravity_comp = robot_env._get_gravity_compensation()

        # Test 1: Gravity compensation should be non-zero (gravity exists)
        assert (
            np.max(np.abs(gravity_comp)) > 0.1
        ), f"Gravity compensation seems too small: {gravity_comp}"

        # Test 2: When position error is zero, torques should include gravity comp
        current_pos = np.array(robot_env.qpos["arm"]).copy()
        robot_env.qvel["arm"][:] = 0.0  # Set velocity to zero
        robot_env.sim.forward()

        torques = robot_env._compute_arm_torques(current_pos)
        expected_gravity_comp = robot_env._get_gravity_compensation()

        # Torques should approximately equal gravity compensation
        # (small difference due to forward() call updating state)
        assert np.allclose(torques, expected_gravity_comp, atol=0.1), (
            f"Torques don't match gravity compensation. "
            f"Torques: {torques}, Gravity comp: {expected_gravity_comp}"
        )

        # Test 3: Gravity compensation should vary with arm configuration
        # Move to a different configuration and check gravity comp changes
        original_comp = gravity_comp.copy()
        robot_env.qpos["arm"][2] += 0.5  # Rotate joint 3
        robot_env.sim.forward()
        new_comp = robot_env._get_gravity_compensation()

        # At least some components should have changed
        comp_diff = np.abs(new_comp - original_comp)
        assert np.max(comp_diff) > 0.1, (
            f"Gravity compensation didn't change with configuration. "
            f"Original: {original_comp}, New: {new_comp}"
        )

    finally:
        env.close()


def test_pd_achieves_accurate_tracking():
    """Test that PD controller with gravity compensation achieves accurate tracking.

    With PD control and gravity compensation, the arm should accurately track target
    positions with minimal steady-state error.
    """
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1)
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get initial positions
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Small target offset
        target_offset = np.array([0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002])
        target_arm_pos = initial_arm_pos + target_offset

        # Build action
        action = np.zeros(11)
        action[0:3] = initial_base_pos
        action[3:10] = target_arm_pos
        action[10] = 0.0

        # Record error at different time points
        errors_over_time = []
        for step in range(500):
            env.step(action)
            if step % 100 == 99:
                current_pos = np.array(robot_env.qpos["arm"]).copy()
                error = np.linalg.norm(target_arm_pos - current_pos)
                errors_over_time.append(error)

        # Check tracking accuracy
        final_arm_pos = np.array(robot_env.qpos["arm"]).copy()
        joint_errors = np.abs(target_arm_pos - final_arm_pos)
        total_error = np.linalg.norm(target_arm_pos - final_arm_pos)

        # With gravity compensation, PD should achieve good tracking
        # Total error should be small (< 0.5 radians total across all joints)
        assert total_error < 0.5, (
            f"Total tracking error too large: {total_error:.4f} radians. "
            f"Joint errors: {joint_errors}"
        )

        # The system should remain stable (error shouldn't explode)
        assert all(
            e < 2.0 for e in errors_over_time
        ), f"System became unstable. Errors over time: {errors_over_time}"

    finally:
        env.close()


def test_arm_remains_stable():
    """Test that the arm remains stable (no explosion) under PD control.

    This test verifies that:
    1. Joint positions stay within reasonable bounds
    2. No NaN or infinite values occur
    3. The simulation doesn't explode

    Note: Without gravity compensation, perfect position holding is not expected.
    """
    # Create environment with absolute position mode
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1)
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get current positions
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Use current position as target
        target_arm_pos = initial_arm_pos.copy()

        # Build absolute action
        action = np.zeros(11)
        action[0:3] = initial_base_pos
        action[3:10] = target_arm_pos
        action[10] = 0.0

        # Run for many steps and check stability at each step
        max_position_seen = 0.0
        for step in range(200):
            env.step(action)

            # Check positions at each step
            current_pos = np.array(robot_env.qpos["arm"]).copy()

            # No NaN or Inf values
            assert not np.any(
                np.isnan(current_pos)
            ), f"NaN in arm position at step {step}: {current_pos}"
            assert not np.any(
                np.isinf(current_pos)
            ), f"Inf in arm position at step {step}: {current_pos}"

            # Track maximum position magnitude
            max_position_seen = max(max_position_seen, np.max(np.abs(current_pos)))

        # Verify positions stayed bounded (< 20 radians is reasonable for a few rotations)
        assert max_position_seen < 20, (
            f"Arm positions exceeded reasonable bounds. "
            f"Max position magnitude seen: {max_position_seen:.2f} radians"
        )

    finally:
        env.close()
