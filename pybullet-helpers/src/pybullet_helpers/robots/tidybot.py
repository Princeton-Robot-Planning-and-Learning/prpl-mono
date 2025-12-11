"""TidyBot mobile base and mobile manipulators."""

from pathlib import Path

from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
from pybullet_helpers.robots.mobile import (
    MobilePyBulletBase,
    SingleArmPyBulletMobileManipulator,
)
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class TidyBotMobileBase(MobilePyBulletBase):
    """The TidyBot mobile base."""

    @classmethod
    def get_name(cls) -> str:
        return "tidybot-base"

    @property
    def urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "tidybot" / "tidybot_base.urdf"


class TidyBotKinova(SingleArmPyBulletMobileManipulator):
    """TidyBot with a Kinova gen-3."""

    @classmethod
    def create_arm(
        cls,
        physics_client_id: int,
        base_pose: Pose,
    ) -> SingleArmPyBulletRobot:
        return KinovaGen3RobotiqGripperPyBulletRobot(
            physics_client_id,
            base_pose,
            fixed_base=False,
            control_mode="reset",
        )

    @classmethod
    def create_base(
        cls,
        physics_client_id: int,
        z: float,
        pose_lower_bound: SE2Pose,
        pose_upper_bound: SE2Pose,
        home_pose: SE2Pose,
    ) -> MobilePyBulletBase:
        return TidyBotMobileBase(
            physics_client_id, z, pose_lower_bound, pose_upper_bound, home_pose
        )

    @property
    def base_to_arm_transform(self) -> Pose:
        return Pose((0.1199, 0, 0.3948), (0, 0, 0, 1))

    @classmethod
    def get_name(cls) -> str:
        return "tidybot-kinova"
