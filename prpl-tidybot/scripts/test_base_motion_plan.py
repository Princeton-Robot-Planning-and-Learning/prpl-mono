"""Tests for real world base motion planning."""

import time

import prbench
from matplotlib import pyplot as plt
from prbench_models.dynamic3d.utils import (
    get_bounding_box,
    plot_overhead_scene,
    run_base_motion_planning,
)
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2
from tomsgeoms2d.structs import Rectangle

from prpl_tidybot.base_movement import reach_target_pose
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface

prbench.register_all_environments()


def test_run_base_motion_planning() -> None:
    """Tests for real world base motion planning."""

    env = prbench.make("prbench/TidyBot3D-ground-o3-v0", render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)

    target_base_pose = SE2(0.5, 0.5, 0.0)
    x_bounds = (-1.5, 1.5)
    y_bounds = (-1.5, 1.5)
    seed = 123
    base_motion_plan = run_base_motion_planning(
        state, target_base_pose, x_bounds, y_bounds, seed
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

    ### real interface

    interface = RealInterface()
    # initialization
    pose_map = SE2(0, 0, 0)
    pose_odom = SE2(0, 0, 0)
    map_to_odom_converter = CoordFrameConverter(pose_map, pose_odom)
    odom_to_map_converter = CoordFrameConverter(pose_odom, pose_map)

    # get initial pose
    observation = interface.get_observation()
    map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
    odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

    for t in range(1, len(base_motion_plan)):
        pose = base_motion_plan[t]
        print(f"Target pose: {pose.x}, {pose.y}, {pose.theta()}")
        reach_target_pose(interface, pose, map_to_odom_converter, odom_to_map_converter)
        time.sleep(0.1)

    time.sleep(1)
    interface.close()
    print("Interface closed")


if __name__ == "__main__":
    test_run_base_motion_planning()
