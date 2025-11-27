"""Tests for the TidyBot3D cupboard scene: observation/action spaces, reset, and step."""

from pathlib import Path

import prbench
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv


def test_tidybot3d_cupboard_observation_space():
    """Reset should return an observation within the observation space."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard", num_objects=8, render_images=False
    )
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_cupboard_action_space():
    """A sampled action should be valid for the action space."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard", num_objects=8, render_images=False
    )
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    env.close()


def test_tidybot3d_cupboard_step():
    """Step should return a valid obs, float reward, bool done flags, and info dict."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard", num_objects=8, render_images=False
    )
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_cupboard_reset_seed_reproducible():
    """Reset with the same seed should produce identical observations."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard", num_objects=8, render_images=False
    )
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert obs1.allclose(obs2, atol=1e-3)
    env.close()


def test_tidybot3d_cupboard_reset_changes_with_different_seeds():
    """Resets with different seeds should produce different observations."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard", num_objects=8, render_images=False
    )
    obs1, _ = env.reset(seed=10)
    obs2, _ = env.reset(seed=20)
    if len(obs1.data) != len(obs2.data):
        raise AssertionError("Observations have different number of objects")
    if len(obs1.data) > 0:
        assert not obs1.allclose(obs2, atol=1e-4)
    env.close()


def test_tidybot3d_cupboard_has_eight_objects():
    """Cupboard environment should be configured with 8 objects."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard", num_objects=8, render_images=False
    )
    assert env.num_objects == 8
    assert env.scene_type == "cupboard"
    env.close()


def test_tidybot_cupboard_constrained_fitting_goals():
    """Test that tidybot-cupboard-o12-ConstrainedFitting env correctly checks goals."""

    tasks_root = (
        Path(prbench.__path__[0]).parent / "prbench" / "envs" / "dynamic3d" / "tasks"
    )
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=12,
        task_config_path=str(
            tasks_root / "tidybot-cupboard-o12-ConstrainedFitting.json"
        ),
        render_images=False,
    )

    # Reset the environment
    env.reset()

    # After reset, goals should not be satisfied
    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied after reset"

    # Get the current state
    current_state = env._get_current_state()  # pylint: disable=protected-access

    # Get the cupboard fixture
    cupboard = None
    for fixture in env._fixtures_dict.values():  # pylint: disable=protected-access
        if fixture.name == "cupboard_1":
            cupboard = fixture
            break

    assert cupboard is not None, "Cupboard fixture not found"

    # Get all objects
    objects_dict = env._objects_dict  # pylint: disable=protected-access

    # Create a modified state with objects in their goal regions
    modified_state = current_state.copy()

    # Place red cuboid in shelf 3 red partition
    red_cuboid = objects_dict.get("red_cuboid1")
    if red_cuboid:
        goal_pos = cupboard.sample_pose_in_region(
            "cupboard_1_shelf_3_red_partition_goal", env.np_random
        )
        modified_state.set(red_cuboid.symbolic_object, "x", goal_pos[0])
        modified_state.set(red_cuboid.symbolic_object, "y", goal_pos[1])
        modified_state.set(red_cuboid.symbolic_object, "z", goal_pos[2])

    # Place green cuboid in shelf 3 green partition
    green_cuboid = objects_dict.get("green_cuboid1")
    if green_cuboid:
        goal_pos = cupboard.sample_pose_in_region(
            "cupboard_1_shelf_3_green_partition_goal", env.np_random
        )
        modified_state.set(green_cuboid.symbolic_object, "x", goal_pos[0])
        modified_state.set(green_cuboid.symbolic_object, "y", goal_pos[1])
        modified_state.set(green_cuboid.symbolic_object, "z", goal_pos[2])

    # Place blue cuboid in shelf 3 blue partition
    blue_cuboid = objects_dict.get("blue_cuboid1")
    if blue_cuboid:
        goal_pos = cupboard.sample_pose_in_region(
            "cupboard_1_shelf_3_blue_partition_goal", env.np_random
        )
        modified_state.set(blue_cuboid.symbolic_object, "x", goal_pos[0])
        modified_state.set(blue_cuboid.symbolic_object, "y", goal_pos[1])
        modified_state.set(blue_cuboid.symbolic_object, "z", goal_pos[2])

    # Place red cubes on shelf 0
    for i in range(1, 4):
        red_cube = objects_dict.get(f"red_cube{i}")
        if red_cube:
            goal_pos = cupboard.sample_pose_in_region(
                "cupboard_1_shelf_0_red_goal", env.np_random
            )
            modified_state.set(red_cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(red_cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(red_cube.symbolic_object, "z", goal_pos[2])

    # Place green cubes on shelf 1
    for i in range(1, 4):
        green_cube = objects_dict.get(f"green_cube{i}")
        if green_cube:
            goal_pos = cupboard.sample_pose_in_region(
                "cupboard_1_shelf_1_green_goal", env.np_random
            )
            modified_state.set(green_cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(green_cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(green_cube.symbolic_object, "z", goal_pos[2])

    # Place blue cubes on shelf 2
    for i in range(1, 4):
        blue_cube = objects_dict.get(f"blue_cube{i}")
        if blue_cube:
            goal_pos = cupboard.sample_pose_in_region(
                "cupboard_1_shelf_2_blue_goal", env.np_random
            )
            modified_state.set(blue_cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(blue_cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(blue_cube.symbolic_object, "z", goal_pos[2])

    # Set the modified state in the environment
    env.set_state(modified_state)

    # Now goals should be satisfied
    assert (
        env._check_goals()  # pylint: disable=protected-access
    ), "Goals should be satisfied after placing objects in goal regions"

    env.close()
