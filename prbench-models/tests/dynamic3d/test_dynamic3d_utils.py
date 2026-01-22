"""Test utils for dynamic3d models."""

import numpy as np
import prbench
from matplotlib import pyplot as plt
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2
from tomsgeoms2d.structs import Rectangle

from prbench_models.dynamic3d.utils import (
    get_bounding_box,
    get_overhead_geom2ds,
    get_overhead_object_se2_pose,
    get_overhead_robot_se2_pose,
    plot_overhead_scene,
    run_base_motion_planning,
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


def test_run_base_motion_planning():
    """Tests for run_base_motion_planning()."""

    env = prbench.make("prbench/TidyBot3D-cupboard_real-o1-v0", render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)

    target_base_pose = SE2(-1, 1, 0.0)
    x_bounds = (-1.5, 1.5)
    y_bounds = (-1.5, 1.5)
    seed = 123
    base_motion_plan = run_base_motion_planning(
        state,
        target_base_pose,
        x_bounds,
        y_bounds,
        seed,
        extend_xy_magnitude=0.5,
        extend_rot_magnitude=np.pi / 2,
    )
    assert base_motion_plan is not None

    fig, ax = plot_overhead_scene(
        state,
        min_x=x_bounds[0],
        max_x=x_bounds[1],
        min_y=y_bounds[0],
        max_y=y_bounds[1],
    )
    assert isinstance(fig, plt.Figure)
    robot = state.get_object_from_name("robot")
    robot_width, robot_height, _ = get_bounding_box(state, robot)
    for pose in base_motion_plan:
        robot_geom = Rectangle.from_center(
            pose.x,
            pose.y,
            robot_width,
            robot_height,
            rotation_about_center=pose.theta(),
        )
        robot_geom.plot(ax, fc="none", ec="gray", linestyle="dashed")

    # Uncomment to debug.
    # from prpl_utils.utils import fig2data, get_signed_angle_distance

    # ax.set_title("Motion Planning Example")
    # plt.tight_layout()
    # img = fig2data(fig)
    # outfile = "base_motion_planning.png"
    # import imageio.v2 as iio

    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")
    # print("Number of steps:", len(base_motion_plan))

    # import time
    # imgs = []
    # for t in range(1, len(base_motion_plan)):
    #     pose = base_motion_plan[t]
    #     max_control_steps = 10
    #     tolerance = 1e-2
    #     control_period = 0.1  # 10hz
    #     for control_step in range(max_control_steps):
    #         previous_pose = SE2(
    #             state.get(robot, "pos_base_x"),
    #             state.get(robot, "pos_base_y"),
    #             state.get(robot, "pos_base_rot"),
    #         )
    #         dx = pose.x - previous_pose.x
    #         dy = pose.y - previous_pose.y
    #         drot = get_signed_angle_distance(pose.theta(), previous_pose.theta())
    #         action = np.zeros(11, dtype=np.float32)
    #         action[0] = dx
    #         action[1] = dy
    #         action[2] = drot
    #         # assert env.action_space.contains(action)

    #         obs, _, _, _, _ = env.step(action)
    #         state = env.observation_space.devectorize(obs)
    #         print("Expected x, y, rot:", pose.x, pose.y, pose.theta())
    #         print(
    #             "Actual x, y, rot:",
    #             state.get(robot, "pos_base_x"),
    #             state.get(robot, "pos_base_y"),
    #             state.get(robot, "pos_base_rot"),
    #         )
    #         time.sleep(
    #             control_period
    #         )  # sleep for 100ms to allow the action to be executed
    #         if (
    #             np.isclose(state.get(robot, "pos_base_x"), pose.x, atol=tolerance)
    #             and np.isclose(state.get(robot, "pos_base_y"), pose.y, atol=tolerance)
    #             and np.isclose(
    #                 state.get(robot, "pos_base_rot"), pose.theta(), atol=tolerance
    #             )
    #         ):
    #             print(
    #                 f"Reached target pose {pose.x}, {pose.y}, {pose.theta()} "
    #                 f"in {control_step + 1} steps"
    #             )
    #             break
    #         img = env.render()
    #         imgs.append(img)
    # outfile = "base_motion_planning.mp4"
    # iio.mimsave(outfile, imgs)
    # print(f"Wrote out to {outfile}")
