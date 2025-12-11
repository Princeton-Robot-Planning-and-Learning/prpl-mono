"""Base classes for mobile bases and mobile manipulators."""

import abc
from pathlib import Path

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


class MobilePyBulletBase(abc.ABC):
    """Base class for a mobile base."""

    def __init__(
        self,
        physics_client_id: int,
        z: float,  # mobile bases have a fixed z position in the 3D world
        pose_lower_bound: SE2Pose = SE2Pose(-np.inf, -np.inf, -np.pi),
        pose_upper_bound: SE2Pose = SE2Pose(np.inf, np.inf, np.pi),
        home_pose: SE2Pose = SE2Pose.identity(),
    ) -> None:
        self.physics_client_id = physics_client_id
        self.z = z
        self.pose_lower_bound = pose_lower_bound
        self.pose_upper_bound = pose_upper_bound
        self.home_pose = home_pose

        se3_pose = home_pose.to_se3(z=z)
        self.robot_id = p.loadURDF(
            str(self.urdf_path),
            basePosition=se3_pose.position,
            baseOrientation=se3_pose.orientation,
            # We always use reset() rather than let PyBullet physics act on the base.
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the base."""

    @property
    @abc.abstractmethod
    def urdf_path(self) -> Path:
        """Get the path to the URDF file for the robot."""

    def get_pose(self) -> SE2Pose:
        """Get the SE2Pose of the base."""
        se3_pose = get_pose(self.robot_id, self.physics_client_id)
        assert np.isclose(se3_pose.position[2], self.z)
        return se3_pose.to_se2()

    def set_pose(self, pose: SE2Pose) -> None:
        """Set the pose of the base."""
        se3_pose = pose.to_se3(self.z)
        set_pose(self.robot_id, se3_pose, self.physics_client_id)


class SingleArmPyBulletMobileManipulator(abc.ABC):
    """A single arm mounted on a mobile base."""

    def __init__(
        self,
        physics_client_id: int,
        base_z: float,
        base_pose_lower_bound: SE2Pose = SE2Pose(-np.inf, -np.inf, -np.pi),
        base_pose_upper_bound: SE2Pose = SE2Pose(np.inf, np.inf, np.pi),
        base_home_pose: SE2Pose = SE2Pose.identity(),
    ) -> None:
        base = self.create_base(
            physics_client_id,
            base_z,
            pose_lower_bound=base_pose_lower_bound,
            pose_upper_bound=base_pose_upper_bound,
            home_pose=base_home_pose,
        )
        base_se3_pose = base_home_pose.to_se3(base_z)
        arm_base_pose = multiply_poses(base_se3_pose, self.base_to_arm_transform)
        arm = self.create_arm(physics_client_id, base_pose=arm_base_pose)
        assert not arm.fixed_base, "Set fixed_base=False in arm"
        assert arm.physics_client_id == base.physics_client_id
        self.physics_client_id = arm.physics_client_id

        self.arm = arm
        self.base = base

    @property
    @abc.abstractmethod
    def base_to_arm_transform(self) -> Pose:
        """Pose from base to arm."""

    @classmethod
    @abc.abstractmethod
    def create_arm(
        cls,
        physics_client_id: int,
        base_pose: Pose,
    ) -> SingleArmPyBulletRobot:
        """Create the arm."""

    @classmethod
    @abc.abstractmethod
    def create_base(
        cls,
        physics_client_id: int,
        z: float,  # mobile bases have a fixed z position in the 3D world
        pose_lower_bound: SE2Pose,
        pose_upper_bound: SE2Pose,
        home_pose: SE2Pose,
    ) -> MobilePyBulletBase:
        """Create the base."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the base."""

    def set_base(self, pose: SE2Pose) -> None:
        """Set the base pose."""
        self.base.set_pose(pose)
        arm_base_pose = multiply_poses(
            pose.to_se3(self.base.z), self.base_to_arm_transform
        )
        self.arm.set_base(arm_base_pose)

    def get_base(self) -> SE2Pose:
        """Get the base pose."""
        return self.base.get_pose()
