"""Base class for bimanual (two-arm) robots that share a single PyBullet body."""

import abc
from functools import cached_property
from pathlib import Path

import numpy as np
import pybullet as p
from gymnasium.spaces import Box

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import (
    JointPositions,
    get_joint_infos,
    get_joint_lower_limits,
    get_joint_positions,
    get_joint_upper_limits,
    get_joints,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot


class BimanualPyBulletRobot(abc.ABC):
    """A fixed-base robot with two arms exposed over one shared PyBullet body.

    The two arms are exposed as SingleArmPyBulletRobot views bound to the same body (see
    SingleArmPyBulletRobot's robot_id binding), so existing single-arm tools --
    inverse_kinematics(robot.left_arm, ...), run_motion_planning(robot.left_arm, ...) --
    work unchanged on each arm. The shared "torso" and "head" joints, which are not part
    of either arm's kinematic chain, are owned by this class.

    The whole-robot joint ordering is: torso, left arm (incl. fingers), right arm (incl.
    fingers), head.
    """

    def __init__(
        self,
        physics_client_id: int,
        base_pose: Pose = Pose.identity(),
        control_mode: str = "reset",
        home_torso_positions: JointPositions | None = None,
        home_head_positions: JointPositions | None = None,
    ) -> None:
        self.physics_client_id = physics_client_id
        self._base_pose = base_pose
        self._control_mode = control_mode

        # Load the single shared body. Self-collision flags are a body-level decision
        # made here (the arm views bind to this body and cannot change them).
        flags = p.URDF_USE_INERTIA_FROM_FILE
        if self.self_collision_link_names:
            flags |= p.URDF_USE_SELF_COLLISION
            flags |= p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self.robot_id = p.loadURDF(
            str(self.urdf_path),
            basePosition=base_pose.position,
            baseOrientation=base_pose.orientation,
            useFixedBase=True,
            physicsClientId=physics_client_id,
            flags=flags,
        )

        # Expose each arm as a SingleArmPyBulletRobot bound to the shared body.
        self.left_arm = self.create_left_arm(
            physics_client_id, self.robot_id, base_pose, control_mode
        )
        self.right_arm = self.create_right_arm(
            physics_client_id, self.robot_id, base_pose, control_mode
        )
        assert self.left_arm.robot_id == self.robot_id == self.right_arm.robot_id

        self._torso_joint_ids = [
            self._joint_from_name(n) for n in self.torso_joint_names
        ]
        self._head_joint_ids = [self._joint_from_name(n) for n in self.head_joint_names]

        # Set the initial posture: torso, head, and both arms to their home values.
        if home_torso_positions is None:
            home_torso_positions = [0.0] * len(self._torso_joint_ids)
        if home_head_positions is None:
            home_head_positions = [0.0] * len(self._head_joint_ids)
        self.set_torso_joints(home_torso_positions)
        self.set_head_joints(home_head_positions)
        self.left_arm.set_joints(self.left_arm.home_joint_positions)
        self.right_arm.set_joints(self.right_arm.home_joint_positions)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the robot."""

    @property
    @abc.abstractmethod
    def urdf_path(self) -> Path:
        """Path to the URDF file for the shared body."""

    @property
    @abc.abstractmethod
    def torso_joint_names(self) -> list[str]:
        """Names of the shared torso joints (actuated, owned by this class)."""

    @property
    @abc.abstractmethod
    def head_joint_names(self) -> list[str]:
        """Names of the head joints (actuated, owned by this class)."""

    @classmethod
    @abc.abstractmethod
    def create_left_arm(
        cls,
        physics_client_id: int,
        robot_id: int,
        base_pose: Pose,
        control_mode: str,
    ) -> FingeredSingleArmPyBulletRobot:
        """Create the left arm bound to the shared body."""

    @classmethod
    @abc.abstractmethod
    def create_right_arm(
        cls,
        physics_client_id: int,
        robot_id: int,
        base_pose: Pose,
        control_mode: str,
    ) -> FingeredSingleArmPyBulletRobot:
        """Create the right arm bound to the shared body."""

    @property
    def self_collision_link_names(self) -> list[tuple[str, str]]:
        """Link name pairs for self-collision checking."""
        # TODO(bimanual): enroll cross-arm and arm-vs-torso/head link pairs so motion
        # planning detects left-vs-right-arm collisions. Empty for now.
        return []

    @cached_property
    def _name_to_joint_id(self) -> dict[str, int]:
        infos = get_joint_infos(
            self.robot_id,
            get_joints(self.robot_id, self.physics_client_id),
            self.physics_client_id,
        )
        return {info.jointName: info.jointIndex for info in infos}

    def _joint_from_name(self, name: str) -> int:
        return self._name_to_joint_id[name]

    def _set_joints(self, joint_ids: list[int], positions: JointPositions) -> None:
        for joint_id, position in zip(joint_ids, positions, strict=True):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=position,
                targetVelocity=0.0,
                physicsClientId=self.physics_client_id,
            )

    def get_torso_joints(self) -> JointPositions:
        """Get the shared torso joint positions."""
        return get_joint_positions(
            self.robot_id, self._torso_joint_ids, self.physics_client_id
        )

    def set_torso_joints(self, positions: JointPositions) -> None:
        """Set the shared torso joint positions (this moves both arms)."""
        self._set_joints(self._torso_joint_ids, positions)

    def get_head_joints(self) -> JointPositions:
        """Get the head joint positions."""
        return get_joint_positions(
            self.robot_id, self._head_joint_ids, self.physics_client_id
        )

    def set_head_joints(self, positions: JointPositions) -> None:
        """Set the head joint positions."""
        self._set_joints(self._head_joint_ids, positions)

    def get_joint_positions(self) -> JointPositions:
        """Get all actuated joint positions: torso, left arm, right arm, head."""
        return (
            list(self.get_torso_joints())
            + list(self.left_arm.get_joint_positions())
            + list(self.right_arm.get_joint_positions())
            + list(self.get_head_joints())
        )

    def set_joints(self, positions: JointPositions) -> None:
        """Set all actuated joints from a concatenated vector (see get_joint_positions
        for the ordering)."""
        num_torso = len(self._torso_joint_ids)
        num_left = len(self.left_arm.arm_joints)
        num_right = len(self.right_arm.arm_joints)
        num_head = len(self._head_joint_ids)
        assert len(positions) == num_torso + num_left + num_right + num_head
        i = 0
        self.set_torso_joints(positions[i : i + num_torso])
        i += num_torso
        self.left_arm.set_joints(positions[i : i + num_left])
        i += num_left
        self.right_arm.set_joints(positions[i : i + num_right])
        i += num_right
        self.set_head_joints(positions[i : i + num_head])

    @property
    def action_space(self) -> Box:
        """Position-control action space over all actuated joints, in the same order as
        get_joint_positions."""
        lower = (
            list(
                get_joint_lower_limits(
                    self.robot_id, self._torso_joint_ids, self.physics_client_id
                )
            )
            + list(self.left_arm.joint_lower_limits)
            + list(self.right_arm.joint_lower_limits)
            + list(
                get_joint_lower_limits(
                    self.robot_id, self._head_joint_ids, self.physics_client_id
                )
            )
        )
        upper = (
            list(
                get_joint_upper_limits(
                    self.robot_id, self._torso_joint_ids, self.physics_client_id
                )
            )
            + list(self.left_arm.joint_upper_limits)
            + list(self.right_arm.joint_upper_limits)
            + list(
                get_joint_upper_limits(
                    self.robot_id, self._head_joint_ids, self.physics_client_id
                )
            )
        )
        return Box(np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32))
