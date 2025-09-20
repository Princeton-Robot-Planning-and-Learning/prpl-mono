"""Franka Emika Panda robot."""

from pathlib import Path
from typing import Optional

import numpy as np

from pybullet_helpers.ikfast import IKFastInfo
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class PandaPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """Franka Emika Panda which we assume is fixed on some base.

    The fingers are symmetric, so the finger state is just a float.
    """

    @classmethod
    def get_name(cls) -> str:
        return "panda"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "franka_description" / "robots" / "panda_arm_hand.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [
            -1.6760817784086874,
            -0.8633617886115512,
            1.0820023618960484,
            -1.7862427129376002,
            0.7563762599673787,
            1.3595324116603988,
            1.7604148617061273,
            0.04,
            0.04,
        ]

    @property
    def end_effector_name(self) -> str:
        """The tool joint is offset from the final arm joint such that it
        represents the point in the center of the two fingertips of the gripper
        (fingertips, NOT the entire fingers).

        This differs from the "panda_hand" joint which represents the
        center of the gripper itself including parts of the gripper
        body.
        """
        return "tool_joint"

    @property
    def tool_link_name(self) -> str:
        return "tool_link"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["panda_finger_joint1", "panda_finger_joint2"]

    @property
    def open_fingers_state(self) -> float:
        return 0.04

    @property
    def closed_fingers_state(self) -> float:
        return 0.03

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 2
        assert np.isclose(joint_positions[0], joint_positions[1], atol=1e-5)
        return joint_positions[0]

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="panda_arm",
            module_name="ikfast_panda_arm",
            base_link="panda_link0",
            ee_link="panda_link8",
            free_joints=["panda_joint7"],
        )
