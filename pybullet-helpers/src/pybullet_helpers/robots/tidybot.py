"""TidyBot mobile base and mobile manipulators."""

from pybullet_helpers.robots.mobile import (
    MobilePyBulletBase,
    SingleArmPyBulletMobileManipulator,
)
from pathlib import Path
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


# class TidyBotKinova(SingleArmPyBulletMobileManipulator):
#     """TidyBot with a Kinova gen-3."""
