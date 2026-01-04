"""Tests for the TidyBot3D base motion environment."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

import prbench
from tests.conftest import MAKE_VIDEOS

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
)


def test_straight_base_motion():
    """This environment is really simple: moving directly towards the target works."""

    prbench.register_all_environments()
    env = prbench.make("prbench/TidyBot3D-base_motion-o1-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    # Extract the positions of the target and robot.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    target = state.get_object_from_name("cube1")
    robot = state.get_object_from_name("robot")
    target_x = state.get(target, "x")
    target_y = state.get(target, "y")
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")
    robot_rot = state.get(robot, "pos_base_rot")

    # Actions are delta positions.
    max_magnitude = 1e-2
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1
    plan = []
    for i in range(1, steps + 1):
        frac = i / steps
        plan.append(np.array([frac * dx, frac * dy, robot_rot] + [0.0] * 8))

    # Execute the plan.
    for action in plan:
        _, _, done, _, _ = env.step(action)
        if done:  # success
            break
    else:
        assert False, "Failed to reach target"

    env.close()


@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. Run: python scripts/download_mimiclabs_assets.py",
)
@pytest.mark.parametrize("lab_num", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(
    "view", ["frontview_image", "sideview_image", "birdview_image", "agentview_image"]
)
def test_straight_base_motion_mimiclabs(lab_num, view):
    """Test base motion with MimicLabs background scenes (lab2-lab8)."""

    prbench.register_all_environments()
    env = prbench.make(
        "prbench/TidyBot3D-base_motion-o1-v0",
        render_mode="rgb_array",
        scene_bg=f"mimiclabs-lab{lab_num}",
        scene_render_camera=f"{view}",
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, f"unit_test_videos_lab{lab_num}_view_{view}")

    # Extract the positions of the target and robot.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    target = state.get_object_from_name("cube1")
    robot = state.get_object_from_name("robot")
    target_x = state.get(target, "x")
    target_y = state.get(target, "y")
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")
    robot_rot = state.get(robot, "pos_base_rot")

    # Actions are delta positions.
    max_magnitude = 1e-2
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1
    plan = []
    for i in range(1, steps + 1):
        frac = i / steps
        plan.append(np.array([frac * dx, frac * dy, robot_rot] + [0.0] * 8))

    # Execute the plan.
    for action in plan:
        _, _, done, _, _ = env.step(action)
        if done:  # success
            break
    else:
        assert False, f"Failed to reach target with mimiclabs-lab{lab_num} background"

    env.close()
