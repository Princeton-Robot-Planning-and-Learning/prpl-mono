"""Fetch Robotics Mobile Manipulator (Fetch)."""

from pathlib import Path

import numpy as np

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class FetchPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """A Fetch robot with a fixed base and only one arm in use.

    The fingers are symmetric, so the finger state is just a float.
    """

    @classmethod
    def get_name(cls) -> str:
        return "fetch"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "fetch_description" / "robots" / "fetch.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [
            -0.5591804653688366,
            -0.5948112040931831,
            1.6380556206810288,
            1.2637851140067282,
            2.1300614898498007,
            -1.794984465148684,
            0.7899035789605409,
            0.0,
            0.0,
        ]

    @property
    def end_effector_name(self) -> str:
        return "gripper_axis"

    @property
    def tool_link_name(self) -> str:
        return "gripper_link"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["l_gripper_finger_joint", "r_gripper_finger_joint"]

    @property
    def open_fingers_state(self) -> float:
        return 0.04

    @property
    def closed_fingers_state(self) -> float:
        return 0.01

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 2
        assert np.isclose(joint_positions[0], joint_positions[1])
        return joint_positions[0]
