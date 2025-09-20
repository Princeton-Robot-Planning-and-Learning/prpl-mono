"""Abstract class for single armed manipulators with PyBullet helper
functions."""

import abc
from functools import cached_property
from pathlib import Path
from typing import Generic, Optional, TypeVar

import numpy as np
import pybullet as p
from gymnasium.spaces import Box

from pybullet_helpers.geometry import Pose, get_pose
from pybullet_helpers.ikfast import IKFastInfo
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    JointVelocities,
    get_joint_infos,
    get_joint_lower_limits,
    get_joint_positions,
    get_joint_upper_limits,
    get_joint_velocities,
    get_joints,
    get_kinematic_chain,
)
from pybullet_helpers.link import BASE_LINK, get_link_state


class SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot."""

    def __init__(
        self,
        physics_client_id: int,
        base_pose: Pose = Pose.identity(),
        control_mode: str = "position",
        home_joint_positions: JointPositions | None = None,
        fixed_base: bool = True,
        custom_urdf_path: Path | None = None,
    ) -> None:
        self.physics_client_id = physics_client_id

        # Pose of base of robot.
        self._base_pose = base_pose

        # Whether the base is fixed.
        self.fixed_base = fixed_base

        # Allow user to use custom urdf.
        self._custom_urdf_path = custom_urdf_path

        # Control mode for the robot.
        self._control_mode = control_mode

        # Home joint positions.
        self._home_joint_positions = (
            home_joint_positions or self.default_home_joint_positions
        )

        # Load the robot and set base position and orientation.
        flags = p.URDF_USE_INERTIA_FROM_FILE
        if self.self_collision_link_names:
            flags |= p.URDF_USE_SELF_COLLISION
            flags |= p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self.robot_id = p.loadURDF(
            str(self.urdf_path),
            basePosition=self._base_pose.position,
            baseOrientation=self._base_pose.orientation,
            # Even if the robot has a mobile base, we treat it as static in
            # pybullet for now and just update the position directly. Otherwise
            # physics starts to affect the robot base.
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
            flags=flags,
        )

        # Make sure the home joint positions are within limits.
        assert len(self._home_joint_positions) == len(self.joint_lower_limits)
        assert self.check_joint_limits(
            self._home_joint_positions
        ), "Home joint positions are out of the limit range"

        # Robot initially at home pose.
        self.set_joints(self.home_joint_positions)

        # Give a one-time warning about using IKFast with custom URDFs.
        if custom_urdf_path is not None and self.ikfast_info() is not None:
            print(
                "WARNING: running IKFast with a custom URDF file may not work "
                "if the URDF is importantly different from what was used to "
                "create the IKFast model."
            )

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the robot."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def default_urdf_path(self) -> Path:
        """Get the default path to the URDF file for the robot."""
        raise NotImplementedError("Override me!")

    @property
    def urdf_path(self) -> Path:
        """Get the path to the URDF file for the robot."""
        if self._custom_urdf_path is not None:
            return self._custom_urdf_path
        return self.default_urdf_path

    @property
    @abc.abstractmethod
    def default_home_joint_positions(self) -> JointPositions:
        """The default joint values for the robot."""
        raise NotImplementedError("Override me!")

    @property
    def home_joint_positions(self) -> JointPositions:
        """The home joint positions for this robot."""
        return self._home_joint_positions

    @property
    def action_space(self) -> Box:
        """The action space for the robot.

        Represents position control of the arm and finger joints.
        """
        return Box(
            np.array(self.joint_lower_limits, dtype=np.float32),
            np.array(self.joint_upper_limits, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    @abc.abstractmethod
    def end_effector_name(self) -> str:
        """The name of the end effector."""
        raise NotImplementedError("Override me!")

    @property
    def end_effector_id(self) -> int:
        """The PyBullet joint ID for the end effector."""
        return self.joint_from_name(self.end_effector_name)

    @property
    @abc.abstractmethod
    def tool_link_name(self) -> str:
        """The name of the end effector link (i.e., the tool link)."""
        raise NotImplementedError("Override me!")

    @cached_property
    def tool_link_id(self) -> int:
        """The PyBullet link ID for the tool link."""
        return self.link_from_name(self.tool_link_name)

    @cached_property
    def base_link_name(self) -> str:
        """Name of the base link for the robot."""
        base_info = p.getBodyInfo(self.robot_id, physicsClientId=self.physics_client_id)
        base_name = base_info[0].decode(encoding="UTF-8")
        return base_name

    @cached_property
    def arm_joints(self) -> list[int]:
        """The PyBullet joint IDs of the joints of the robot arm as determined
        by the kinematic chain.

        Note these are joint indices not body IDs, and that the arm
        joints may be a subset of all the robot joints.
        """
        joint_ids = get_kinematic_chain(
            self.robot_id, self.end_effector_id, self.physics_client_id
        )
        # NOTE: pybullet tools assumes sorted arm joints.
        joint_ids = sorted(joint_ids)
        return joint_ids

    @cached_property
    def arm_joint_names(self) -> list[str]:
        """The names of the arm joints."""
        return [
            info.jointName
            for info in get_joint_infos(
                self.robot_id, self.arm_joints, self.physics_client_id
            )
        ]

    @property
    def self_collision_link_names(self) -> list[tuple[str, str]]:
        """Link names for self-collision checking."""
        # By default, robots do not do self-collision checking.
        return []

    @cached_property
    def self_collision_link_ids(self) -> list[tuple[int, int]]:
        """Link IDs for self-collision checking."""
        return [
            (self.link_from_name(n1), self.link_from_name(n2))
            for n1, n2 in self.self_collision_link_names
        ]

    @cached_property
    def joint_infos(self) -> list[JointInfo]:
        """Get the joint info for each joint of the robot.

        This may be a superset of the arm joints.
        """
        all_joint_ids = get_joints(self.robot_id, self.physics_client_id)
        return get_joint_infos(self.robot_id, all_joint_ids, self.physics_client_id)

    @cached_property
    def joint_names(self) -> list[str]:
        """Get the names of all the joints in the robot."""
        joint_names = [info.jointName for info in self.joint_infos]
        return joint_names

    def joint_from_name(self, joint_name: str) -> int:
        """Get the joint index for a joint name."""
        return self.joint_names.index(joint_name)

    def joint_info_from_name(self, joint_name: str) -> JointInfo:
        """Get the joint info for a joint name."""
        return self.joint_infos[self.joint_from_name(joint_name)]

    def link_from_name(self, link_name: str) -> int:
        """Get the link index for a given link name."""
        if link_name == self.base_link_name:
            return BASE_LINK

        # In PyBullet, each joint has an associated link.
        for joint_info in self.joint_infos:
            if joint_info.linkName == link_name:
                return joint_info.jointIndex
        raise ValueError(f"Could not find link {link_name}")

    def link_name_from_link(self, link: int) -> str:
        """Get the link name for a given link name."""
        if link == BASE_LINK:
            return self.base_link_name

        # In PyBullet, each joint has an associated link.
        for joint_info in self.joint_infos:
            if joint_info.jointIndex == link:
                return joint_info.linkName
        raise ValueError(f"Could not find link {link}")

    @cached_property
    def joint_lower_limits(self) -> JointPositions:
        """Lower bound on the arm joint limits."""
        return get_joint_lower_limits(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    @cached_property
    def joint_upper_limits(self) -> JointPositions:
        """Upper bound on the arm joint limits."""
        return get_joint_upper_limits(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    def get_joint_limits_from_name(self, joint_name: str) -> tuple[float, float]:
        """Get the lower and upper limits for a given joint."""
        assert (
            joint_name in self.arm_joint_names
        ), f"Unrecognized joint name {joint_name}"
        idx = self.arm_joint_names.index(joint_name)
        lower = self.joint_lower_limits[idx]
        upper = self.joint_upper_limits[idx]
        return (lower, upper)

    def get_joint_positions(self) -> JointPositions:
        """Get the joint positions from the current PyBullet state."""
        return get_joint_positions(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    def set_joints(
        self,
        joint_positions: JointPositions,
        joint_velocities: JointVelocities | None = None,
    ) -> None:
        """Directly set the joint positions.

        Outside of resetting to an initial state, this should not be
        used with the robot that uses stepSimulation(); it should only
        be used for motion planning, collision checks, etc., in a robot
        that does not maintain state.
        """
        if joint_velocities is None:
            joint_velocities = [0] * len(joint_positions)
        for joint_id, joint_pos, joint_vel in zip(
            self.arm_joints, joint_positions, joint_velocities, strict=True
        ):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=joint_pos,
                targetVelocity=joint_vel,
                physicsClientId=self.physics_client_id,
            )

    def get_joint_velocities(self) -> JointVelocities:
        """Get the joint velocities from the current PyBullet state."""
        return get_joint_velocities(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        """Check if the given joint positions are within limits."""
        return all(
            l <= v <= u
            for l, v, u in zip(
                self.joint_lower_limits,
                joint_positions,
                self.joint_upper_limits,
                strict=True,
            )
        )

    def get_end_effector_pose(self) -> Pose:
        """Get the robot end-effector pose based on the current PyBullet
        state."""
        ee_link_state = get_link_state(
            self.robot_id,
            self.end_effector_id,
            physics_client_id=self.physics_client_id,
        )
        return Pose(
            ee_link_state.worldLinkFramePosition,
            ee_link_state.worldLinkFrameOrientation,
        )

    def get_jacobian(self, joint_positions: JointPositions | None = None) -> np.ndarray:
        """Get the Jacobian matrix for the end effector."""

        if joint_positions is None:
            joint_positions = self.get_joint_positions()

        jac_t, jac_r = p.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.end_effector_id,
            localPosition=[0, 0, 0],
            objPositions=joint_positions,
            objVelocities=[0] * len(joint_positions),
            objAccelerations=[0] * len(joint_positions),
            physicsClientId=self.physics_client_id,
        )

        return np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)

    def set_base(self, pose: Pose) -> None:
        """Reset the robot base position and orientation."""
        assert not self.fixed_base, "Cannot set base for fixed-base robot"
        p.resetBasePositionAndOrientation(
            self.robot_id,
            pose.position,
            pose.orientation,
            physicsClientId=self.physics_client_id,
        )

    def get_base_pose(self) -> Pose:
        """Get the current base pose."""
        return get_pose(self.robot_id, self.physics_client_id)

    def set_motors(self, joint_positions: JointPositions) -> None:
        """Update the motors to move toward the given joint positions."""
        assert len(joint_positions) == len(self.arm_joints)

        # Set arm joint motors.
        if self._control_mode == "position":
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.arm_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_positions,
                physicsClientId=self.physics_client_id,
            )
        elif self._control_mode == "reset":
            self.set_joints(joint_positions)
        else:
            raise NotImplementedError(
                "Unrecognized pybullet_control_mode: " f"{self._control_mode }"
            )

    def go_home(self) -> None:
        """Move the robot to its home end-effector pose."""
        self.set_motors(self.default_home_joint_positions)

    def forward_kinematics(self, joint_positions: JointPositions) -> Pose:
        """Compute the end effector pose that would result if the robot arm
        joint positions was equal to the input joint_positions.

        WARNING: This method will make use of resetJointState(), and so it
        should NOT be used during simulation.
        """
        self.set_joints(joint_positions)
        ee_link_state = get_link_state(
            self.robot_id,
            self.end_effector_id,
            physics_client_id=self.physics_client_id,
        )
        position = ee_link_state.worldLinkFramePosition
        orientation = ee_link_state.worldLinkFrameOrientation
        return Pose(position, orientation)

    @property
    def default_inverse_kinematics_method(self) -> str:
        """The default inverse kinematics used with inverse_kinematics()."""
        if self.ikfast_info() is not None:
            return "ikfast"
        return "pybullet"

    def custom_inverse_kinematics(
        self,
        end_effector_pose: Pose,
        validate: bool = True,
        best_effort: bool = False,
        validation_atol: float = 1e-3,
    ) -> JointPositions | None:
        """Robots can implement custom IK; see inverse_kinematics()."""
        raise NotImplementedError

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        """IKFastInfo for this robot.

        If this is specified, then IK will use IKFast.
        """
        return None


FingerState = TypeVar("FingerState")  # see docstring below


class FingeredSingleArmPyBulletRobot(SingleArmPyBulletRobot, Generic[FingerState]):
    """A single-arm fixed-base PyBullet robot with one or more fingers.

    NOTE: the fingers are determined by a state of FingerState type, which is
    usually a float or list of floats. For example, if all of the fingers mimic
    each other, then a single value defines their state. The conversion between
    finger states and finger joint positions is defined per robot.
    """

    @property
    @abc.abstractmethod
    def finger_joint_names(self) -> list[str]:
        """The names of the finger joints."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def open_fingers_state(self) -> FingerState:
        """The values at which the finger joints should be open."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def closed_fingers_state(self) -> FingerState:
        """The value at which the finger joints should be closed."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def finger_state_to_joints(self, state: FingerState) -> list[float]:
        """Convert a FingerState into joint values."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def joints_to_finger_state(self, joint_positions: list[float]) -> FingerState:
        """Convert joint values into a FingerState."""
        raise NotImplementedError("Override me!")

    @cached_property
    def arm_joints(self) -> list[int]:
        """Add the fingers to the arm joints."""
        joint_ids = super().arm_joints
        joint_ids.extend(self.finger_ids)
        return joint_ids

    @cached_property
    def finger_ids(self) -> list[int]:
        """The PyBullet joint IDs for the fingers."""
        return [self.joint_from_name(n) for n in self.finger_joint_names]

    @cached_property
    def finger_joint_idxs(self) -> list[int]:
        """The indices into the joints corresponding to the fingers.

        Note this is not the joint ID, but the index of the joint within
        the list of arm joints.
        """
        return [self.arm_joints.index(i) for i in self.finger_ids]

    def open_fingers(self) -> None:
        """Execute opening the fingers."""
        self.set_finger_state(self.open_fingers_state)

    def close_fingers(self) -> None:
        """Execute closing the fingers."""
        self.set_finger_state(self.closed_fingers_state)

    def set_finger_state(self, state: FingerState) -> None:
        """Change the fingers to the given joint values."""
        current_joints = self.get_joint_positions()
        new_values = self.finger_state_to_joints(state)
        for v, idx in zip(new_values, self.finger_joint_idxs, strict=True):
            current_joints[idx] = v
        self.set_motors(current_joints)

    def get_finger_state(self) -> FingerState:
        """Get the state of the fingers."""
        joint_values = [
            p.getJointState(
                self.robot_id,
                i,
                physicsClientId=self.physics_client_id,
            )[0]
            for i in self.finger_ids
        ]
        return self.joints_to_finger_state(joint_values)
