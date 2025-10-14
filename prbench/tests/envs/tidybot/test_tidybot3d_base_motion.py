"""Tests for the TidyBot3D base motion environment."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

import prbench


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

    # Actions are absolute positions, but we don't want to move too much at once.
    max_magnitude = 1e-2
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1
    plan = []
    for i in range(1, steps + 1):
        frac = i / steps
        next_x = robot_x + frac * dx
        next_y = robot_y + frac * dy
        plan.append(np.array([next_x, next_y, robot_rot] + [0.0] * 8))

    # Execute the plan.
    for action in plan:
        _, _, done, _, _ = env.step(action)
        if done:  # success
            break
    else:
        assert False, "Failed to reach target"

    env.close()
