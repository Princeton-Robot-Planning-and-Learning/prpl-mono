"""Tests for the TidyBot3D NAMO (Navigation Among Movable Objects) environment."""

from pathlib import Path

import numpy as np
import pytest

from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv

# Path to tasks directory
TASKS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "prbench"
    / "envs"
    / "dynamic3d"
    / "tasks"
)

# Path to mimiclabs scenes for skip condition
MIMICLABS_SCENES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "prbench"
    / "envs"
    / "dynamic3d"
    / "models"
    / "assets"
    / "mimiclabs_scenes"
    / "meshes"
)


def test_namo_env_loads():
    """Test that the NAMO environment loads correctly."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    # Verify we have the obstacle block
    obstacle = obs.get_object_from_name("obstacle_block")
    assert obstacle is not None

    env.close()


def test_namo_goal_not_satisfied_initially():
    """Test that goal is not satisfied after reset (block not in goal region)."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    env.reset(seed=42)

    # Goal should not be satisfied initially
    assert not env._check_goals(), (  # pylint: disable=protected-access
        "Goal should not be satisfied after reset - "
        "obstacle_block should not be in goal region initially"
    )

    env.close()


def test_namo_goal_satisfied_when_block_in_region():
    """Test that goal is satisfied when obstacle block is moved to goal region."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    env.reset(seed=42)

    # Get current state
    current_state = env._get_current_state()  # pylint: disable=protected-access

    # Get the obstacle block object
    obstacle = current_state.get_object_from_name("obstacle_block")

    # Move obstacle to the goal region (center of goal region is at x=1.0, y=0.0)
    modified_state = current_state.copy()
    modified_state.set(obstacle, "x", 1.0)
    modified_state.set(obstacle, "y", 0.0)
    modified_state.set(obstacle, "z", 0.06)  # size/2 above ground

    # Set the modified state
    env.set_state(modified_state)

    # Now goal should be satisfied
    assert env._check_goals(), (  # pylint: disable=protected-access
        "Goal should be satisfied after moving obstacle_block to goal region"
    )

    env.close()


def test_namo_robot_can_approach_block():
    """Test that robot can approach the obstacle block.

    This is a smoke test that verifies the environment handles robot navigation
    towards the obstacle block. Full pushing behavior depends on physics tuning.
    """
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    obs, _ = env.reset(seed=123)

    # Get initial positions
    obstacle = obs.get_object_from_name("obstacle_block")
    robot = obs.get_object_from_name("robot")
    initial_obstacle_x = obs.get(obstacle, "x")
    initial_obstacle_y = obs.get(obstacle, "y")
    robot_x = obs.get(robot, "pos_base_x")
    robot_y = obs.get(robot, "pos_base_y")
    robot_rot = obs.get(robot, "pos_base_rot")

    # Verify initial positions make sense for NAMO task
    # Robot should start behind the obstacle (smaller x)
    assert robot_x < initial_obstacle_x, (
        f"Robot should start behind obstacle. "
        f"Robot x: {robot_x}, Obstacle x: {initial_obstacle_x}"
    )

    # Goal region is at x=0.8 to x=1.2, obstacle starts around x=0.5
    # This verifies the obstacle is between robot and goal
    goal_region_start_x = 0.8
    assert initial_obstacle_x < goal_region_start_x, (
        f"Obstacle should be between robot and goal region. "
        f"Obstacle x: {initial_obstacle_x}, Goal region starts at: {goal_region_start_x}"
    )

    # Move robot towards the block
    approach_x = initial_obstacle_x - 0.1  # Position near the block
    approach_y = initial_obstacle_y

    max_magnitude = 1e-2
    dx = approach_x - robot_x
    dy = approach_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = max(int(distance / max_magnitude), 1)

    for i in range(1, steps + 1):
        frac = i / steps
        action = np.array(
            [robot_x + frac * dx, robot_y + frac * dy, robot_rot] + [0.0] * 8
        )
        obs, _, done, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if done:
            break

    # Verify robot moved towards the obstacle
    robot = obs.get_object_from_name("robot")
    final_robot_x = obs.get(robot, "pos_base_x")

    # Robot x should have increased (moved forward)
    assert final_robot_x > robot_x, (
        f"Robot should have moved forward. "
        f"Initial x: {robot_x}, Final x: {final_robot_x}"
    )

    env.close()


def test_namo_action_space():
    """Test that action space is valid."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    env.reset(seed=42)
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    env.close()


def test_namo_step():
    """Test that step returns valid outputs."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. "
    "Run: python scripts/download_mimiclabs_assets.py",
)
def test_namo_with_mimiclabs_scene():
    """Test NAMO environment with MimicLabs background scene."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
        scene_bg=True,
    )

    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)

    # Verify scene configuration
    active_scene = env.task_config.get("_active_scene", {})
    assert active_scene.get("type") == "mimiclabs"
    assert active_scene.get("lab") == 5

    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    env.close()
