"""Humans."""

from pathlib import Path
from typing import Any

import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class RightArmHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with right arm animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-right-arm"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "right_arm_7dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.1, 0.1, -1.08786023, 0.0, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class LeftArmHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with left arm animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-left-arm"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "left_arm_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class RightLegHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with right leg animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-right-leg"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "right_leg_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class LeftLegHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with left leg animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-left-leg"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "left_leg_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class Human:
    """A wrapper around the four limbs that also includes the head and torso.

    A typical use case would be to create this human and then access
    just one of the limbs to manipulate.
    """

    def __init__(
        self,
        physics_client_id: int,
        base_pose: Pose = Pose.identity(),
        right_arm_kwargs: dict[str, Any] | None = None,
        left_arm_kwargs: dict[str, Any] | None = None,
        right_leg_kwargs: dict[str, Any] | None = None,
        left_leg_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.physics_client_id = physics_client_id
        self.base_pose = base_pose

        # Create the static torso and head.
        self.torso = p.loadURDF(
            str(self.head_torso_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.torso,
            self.base_pose.position,
            self.base_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create each limb with its respective kwargs.
        right_arm_kwargs = right_arm_kwargs or {}
        left_arm_kwargs = left_arm_kwargs or {}
        right_leg_kwargs = right_leg_kwargs or {}
        left_leg_kwargs = left_leg_kwargs or {}
        right_arm_base_pose = multiply_poses(
            base_pose, self.right_arm_relative_base_pose
        )
        self.right_arm = RightArmHumanPyBulletRobot(
            physics_client_id=self.physics_client_id,
            base_pose=right_arm_base_pose,
            **right_arm_kwargs,
        )
        left_arm_base_pose = multiply_poses(base_pose, self.left_arm_relative_base_pose)
        self.left_arm = LeftArmHumanPyBulletRobot(
            physics_client_id=self.physics_client_id,
            base_pose=left_arm_base_pose,
            **left_arm_kwargs,
        )
        right_leg_base_pose = multiply_poses(
            base_pose, self.right_leg_relative_base_pose
        )
        self.right_leg = RightLegHumanPyBulletRobot(
            physics_client_id=self.physics_client_id,
            base_pose=right_leg_base_pose,
            **right_leg_kwargs,
        )
        left_leg_base_pose = multiply_poses(base_pose, self.left_leg_relative_base_pose)
        self.left_leg = LeftLegHumanPyBulletRobot(
            physics_client_id=self.physics_client_id,
            base_pose=left_leg_base_pose,
            **left_leg_kwargs,
        )

    @property
    def head_torso_urdf_path(self) -> Path:
        """URDF path for the head and torso."""
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "torso_and_head.urdf"

    @property
    def right_arm_relative_base_pose(self) -> Pose:
        """Relative base pose for the right arm."""
        return Pose.from_rpy((-0.24, 0.048, 0.45), (0.0, 0.0, 0.0))

    @property
    def left_arm_relative_base_pose(self) -> Pose:
        """Relative base pose for the left arm."""
        return Pose.from_rpy((0.24, 0.048, 0.45), (0.0, 0.0, 0.0))

    @property
    def right_leg_relative_base_pose(self) -> Pose:
        """Relative base pose for the right leg."""
        return Pose.from_rpy((-0.11, 0.0, 0.0), (0.0, 0.0, 0.0))

    @property
    def left_leg_relative_base_pose(self) -> Pose:
        """Relative base pose for the leg leg."""
        return Pose.from_rpy((0.11, 0.0, 0.0), (0.0, 0.0, 0.0))
