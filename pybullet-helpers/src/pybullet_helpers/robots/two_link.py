"""A simple two-link arm with no gripper."""

from pathlib import Path

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class TwoLinkPyBulletRobot(SingleArmPyBulletRobot):
    """A simple two-link arm with no gripper."""

    @classmethod
    def get_name(cls) -> str:
        return "two-link"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "two_link" / "two_link_robot.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "end_effector"

    @property
    def tool_link_name(self) -> str:
        return "end_effector_link"
