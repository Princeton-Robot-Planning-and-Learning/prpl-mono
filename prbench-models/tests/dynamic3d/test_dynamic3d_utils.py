"""Test utils for dynamic3d models."""

import numpy as np
import prbench
from matplotlib import pyplot as plt
from relational_structs.spaces import ObjectCentricBoxSpace
from tomsgeoms2d.structs import Rectangle

from prbench_models.dynamic3d.utils import (
    get_overhead_geom2ds,
    get_overhead_object_se2_pose,
    get_overhead_robot_se2_pose,
    plot_overhead_scene,
)

prbench.register_all_environments()


def test_get_overhead_object_se2_pose():
    """Tests for get_overhead_object_se2_pose()."""

    # Get a real object-centric state.
    env = prbench.make("prbench/TidyBot3D-ground-o1-v0")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state1 = env.observation_space.devectorize(obs)
    cube = state1.get_object_from_name("cube1")

    # Extract the initial SE2 pose.
    pose1 = get_overhead_object_se2_pose(state1, cube)

    # Moving the object z shouldn't change anything.
    state2 = state1.copy()
    state2.set(cube, "z", 1000)
    pose2 = get_overhead_object_se2_pose(state2, cube)
    assert np.allclose(pose1.A, pose2.A, atol=1e-5)

    # Move the object x should have an effect.
    state3 = state1.copy()
    state3.set(cube, "x", state1.get(cube, "x") + 1.0)
    pose3 = get_overhead_object_se2_pose(state3, cube)
    assert np.isclose(pose1.x + 1, pose3.x)
    assert np.isclose(pose1.y, pose3.y)
    assert np.isclose(pose1.theta(), pose3.theta())


def test_get_overhead_robot_se2_pose():
    """Tests for get_overhead_robot_se2_pose()."""

    # Get a real object-centric state.
    env = prbench.make("prbench/TidyBot3D-ground-o1-v0")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state1 = env.observation_space.devectorize(obs)
    robot = state1.get_object_from_name("robot")

    # Extract the initial SE2 pose.
    pose1 = get_overhead_robot_se2_pose(state1, robot)

    # Move the object x should have an effect.
    state2 = state1.copy()
    state2.set(robot, "pos_base_x", state1.get(robot, "pos_base_x") + 1.0)
    pose2 = get_overhead_robot_se2_pose(state2, robot)
    assert np.isclose(pose1.x + 1, pose2.x)
    assert np.isclose(pose1.y, pose2.y)
    assert np.isclose(pose1.theta(), pose2.theta())


def test_get_overhead_geom2ds():
    """Tests for get_overhead_geom2ds()."""
    env = prbench.make("prbench/TidyBot3D-ground-o1-v0")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    geoms = get_overhead_geom2ds(state)
    assert len(geoms) == 2
    robot_geom = geoms["robot"]
    assert isinstance(robot_geom, Rectangle)
    cube_geom = geoms["cube1"]
    assert isinstance(cube_geom, Rectangle)


def test_plot_overhead_scene():
    """Tests for plot_overhead_scene()."""

    env = prbench.make("prbench/TidyBot3D-ground-o3-v0", render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    fig, ax = plot_overhead_scene(state, min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Uncomment to debug.
    # from prpl_utils.utils import fig2data
    # import imageio.v2 as iio
    # ax.set_title("Overhead Scene Example")
    # plt.tight_layout()
    # img = fig2data(fig)
    # outfile = "out_plot_overhead_scene.png"
    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")
    # img = env.render()
    # outfile = "actual_scene.png"
    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")
