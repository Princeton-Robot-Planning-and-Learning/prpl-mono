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
    obstacle = obs.get_object_from_name("obstacle_chair")
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
        "obstacle_chair should not be in goal region initially"
    )

    env.close()


def test_namo_goal_satisfied_when_robot_in_region():
    """Test that goal is satisfied when robot reaches the goal region.

    In this NAMO task, the goal is for the robot (tidybot) to navigate to the goal
    region, potentially by pushing the obstacle out of the way.
    """
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    env.reset(seed=42)

    # Get current state
    current_state = env._get_current_state()  # pylint: disable=protected-access

    # Get the robot object
    robot = current_state.get_object_from_name("robot")

    # Move robot to the goal region (center of goal region is at x=1.0, y=0.0)
    modified_state = current_state.copy()
    modified_state.set(robot, "pos_base_x", 1.0)
    modified_state.set(robot, "pos_base_y", 0.0)

    # Set the modified state
    env.set_state(modified_state)

    # Now goal should be satisfied (robot is in the goal region)
    assert (
        env._check_goals()
    ), (  # pylint: disable=protected-access
        "Goal should be satisfied after moving robot to goal region"
    )

    env.close()


def test_namo_robot_can_navigate_to_goal():
    """Test that robot can navigate towards the goal region.

    This is a smoke test that verifies the environment handles robot navigation. The
    robot should be able to move towards the goal region.
    """
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "tidybot-namo-o1.json"),
    )

    obs, _ = env.reset(seed=123)

    # Get initial positions
    obstacle = obs.get_object_from_name("obstacle_chair")
    robot = obs.get_object_from_name("robot")
    initial_obstacle_x = obs.get(obstacle, "x")
    initial_obstacle_y = obs.get(obstacle, "y")
    robot_x = obs.get(robot, "pos_base_x")
    robot_y = obs.get(robot, "pos_base_y")
    robot_rot = obs.get(robot, "pos_base_rot")

    # Verify obstacle chair exists and has valid position
    assert obstacle is not None, "Obstacle chair should exist in the scene"
    assert (
        initial_obstacle_x > 0
    ), f"Obstacle should have positive x: {initial_obstacle_x}"

    # Goal region is at x=0.8 to x=1.2, y=-0.2 to 0.2
    # Move robot towards the goal region center
    goal_x = 1.0
    goal_y = 0.0

    max_magnitude = 1e-2
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = max(int(distance / max_magnitude), 1)

    for i in range(1, min(steps + 1, 50)):  # Limit steps for test speed
        frac = i / steps
        action = np.array(
            [robot_x + frac * dx, robot_y + frac * dy, robot_rot] + [0.0] * 8
        )
        obs, _, done, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if done:
            break

    # Verify robot moved towards the goal
    robot = obs.get_object_from_name("robot")
    final_robot_x = obs.get(robot, "pos_base_x")

    # Robot x should have increased (moved towards goal)
    assert final_robot_x > robot_x, (
        f"Robot should have moved towards goal. "
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
    assert active_scene.get("lab") == 2

    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    env.close()
