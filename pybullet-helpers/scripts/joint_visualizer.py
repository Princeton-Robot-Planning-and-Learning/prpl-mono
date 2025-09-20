"""Interactively visualize the joint space of a robot in pybullet."""

import pybullet as p

from pybullet_helpers.gui import create_gui_connection, run_interactive_joint_gui
from pybullet_helpers.robots import create_pybullet_robot


def _main(robot_name: str) -> None:
    physics_client_id = create_gui_connection()
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, True, physicsClientId=physics_client_id
    )
    robot = create_pybullet_robot(
        robot_name,
        physics_client_id,
        control_mode="reset",
    )
    run_interactive_joint_gui(robot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", type=str)
    args = parser.parse_args()
    _main(args.robot)
