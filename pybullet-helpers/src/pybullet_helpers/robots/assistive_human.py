"""Human from assistive gym with right arm animated."""

from pathlib import Path

import numpy as np

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class AssistiveHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human from assistive gym with right arm animated."""

    @classmethod
    def get_name(cls) -> str:
        return "assistive-human"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "assistive_human" / "human.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, -np.pi / 2]

    @property
    def end_effector_name(self) -> str:
        return "end_effector"

    @property
    def tool_link_name(self) -> str:
        return "end_effector_link"
