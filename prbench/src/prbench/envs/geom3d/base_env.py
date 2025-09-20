"""Base environment class for all Geom3D environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Generic
from typing import Type as TypingType
from typing import TypeVar

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    check_collisions_with_held_object,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from relational_structs import (
    Array,
    Object,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict

from prbench.core import ObjectCentricPRBenchEnv, PRBenchEnvConfig, RobotActionSpace
from prbench.envs.geom3d.object_types import (
    Geom3DCuboidType,
    Geom3DEnvTypeFeatures,
    Geom3DPointType,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import (
    Geom3DObjectCentricState,
    Geom3DRobotActionSpace,
    extend_joints_to_include_fingers,
    remove_fingers_from_extended_joints,
)


@dataclass(frozen=True)
class Geom3DEnvConfig(PRBenchEnvConfig):
    """Config for Geom3DEnv()."""

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_base_pose: Pose = Pose.identity()
    initial_joints: JointPositions = field(
        # This is a retract position.
        default_factory=lambda: [
            0.0,  # "joint_1", starting at the robot base and going up to the gripper
            -0.35,  # "joint_2"
            -np.pi,  # "joint_3"
            -2.5,  # "joint_4"
            0.0,  # "joint_5"
            -0.87,  # "joint_6"
            np.pi / 2,  # "joint_7"
        ]
    )
    initial_finger_state: float = 0.0
    end_effector_viz_radius: float = 0.01
    end_effector_viz_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    max_action_mag: float = 0.05

    # This is used to check whether a grasped object can be placed on a surface.
    min_placement_dist: float = 1e-3

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20
    render_image_width: int = 836
    render_image_height: int = 450

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": self.robot_base_pose.position,
            "camera_yaw": 90,
            "camera_distance": 1.5,
            "camera_pitch": -20,
        }


# Subclasses may extend the state.
_ObsType = TypeVar("_ObsType", bound=Geom3DObjectCentricState)
_ConfigType = TypeVar("_ConfigType", bound=Geom3DEnvConfig)


class ObjectCentricGeom3DRobotEnv(
    ObjectCentricPRBenchEnv[_ObsType, Array, _ConfigType],
    Generic[_ObsType, _ConfigType],
):
    """Base class for Geom3D environments."""

    def __init__(self, *args, use_gui: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_gui = use_gui

        # Create the PyBullet client.
        if use_gui:
            camera_info = self.config.get_camera_kwargs()
            self.physics_client_id = create_gui_connection(**camera_info)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            self.config.robot_name,
            self.physics_client_id,
            base_pose=self.config.robot_base_pose,
            control_mode="reset",
            home_joint_positions=extend_joints_to_include_fingers(
                self.config.initial_joints
            ),
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        self.robot = robot

        # Show a visualization of the end effector.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.config.end_effector_viz_radius,
            rgbaColor=self.config.end_effector_viz_color,
            physicsClientId=self.physics_client_id,
        )

        # Also create a collision body because we use it for grasp detection.
        collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.config.end_effector_viz_radius,
            physicsClientId=self.physics_client_id,
        )

        # Create the body for the end effector.
        end_effector_pose = self.robot.get_end_effector_pose()
        self.end_effector_viz_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=end_effector_pose.position,
            baseOrientation=end_effector_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Track current held object.
        self._grasped_object: str | None = None
        self._grasped_object_transform: Pose | None = None

    @property
    @abc.abstractmethod
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        """The type of states in this environment."""

    @abc.abstractmethod
    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        """Create the constant initial state dict."""

    @abc.abstractmethod
    def _get_obs(self) -> _ObsType:
        """Get the current observation."""

    @abc.abstractmethod
    def _goal_reached(self) -> bool:
        """Check if the goal is currently reached."""

    @abc.abstractmethod
    def _reset_objects(self) -> None:
        """Reset objects."""

    @abc.abstractmethod
    def _set_object_states(self, obs: _ObsType) -> None:
        """Reset the state of objects; helper for set_state()."""

    @abc.abstractmethod
    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        """Look up the PyBullet ID for a given object name."""

    @abc.abstractmethod
    def _get_collision_object_ids(self) -> set[int]:
        """Get the collision object IDs."""

    @abc.abstractmethod
    def _get_movable_object_names(self) -> set[str]:
        """The names of objects that can be moved by the robot (grasped and placed)."""

    @abc.abstractmethod
    def _get_surface_object_names(self) -> set[str]:
        """The names of objects that can be used as surfaces for other objects.

        Note that surfaces might be movable, for example, consider block stacking.
        """

    @abc.abstractmethod
    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        """Get the half extents for a cuboid object."""

    @property
    def _grasped_object_id(self) -> int | None:
        if self._grasped_object is not None:
            return self._object_name_to_pybullet_id(self._grasped_object)
        return None

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Geom3DEnvTypeFeatures

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        types = set(self.type_features)
        return ObjectCentricStateSpace(types, state_cls=self.state_cls)

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return Geom3DRobotActionSpace(max_magnitude=config.max_action_mag)

    def _create_constant_initial_state(self) -> _ObsType:
        initial_state_dict = self._create_constant_initial_state_dict()
        state = create_state_from_dict(
            initial_state_dict, Geom3DEnvTypeFeatures, state_cls=self.state_cls
        )
        # This is tricky for type annotation because we are dynamically setting the
        # class to be self.state_cls, which should be _ObsType.
        return state  # type: ignore

    def reset(
        self,
        *args,
        **kwargs,
    ) -> tuple[_ObsType, dict]:
        # Reset the random seed.
        gymnasium.Env.reset(self, *args, **kwargs)

        # Reset the robot. In the future, we may want to allow randomizing the initial
        # robot joint positions.
        self._set_robot_and_held_object(
            self.config.initial_joints, self.config.initial_finger_state
        )

        # Reset objects.
        self._reset_objects()

        return self._get_obs(), {}

    def set_state(self, obs: _ObsType) -> None:
        """Set the state of the environment to the given one.

        This is useful when treating the environment as a simulator.
        """
        self._set_robot_and_held_object(obs.joint_positions, obs.finger_state)
        self._grasped_object = obs.grasped_object
        self._grasped_object_transform = obs.grasped_object_transform
        self._set_object_states(obs)

    def step(self, action: Array) -> tuple[_ObsType, float, bool, bool, dict]:
        # Store the current robot joints because we may need to revert in collision.
        current_joints = remove_fingers_from_extended_joints(
            self.robot.get_joint_positions()
        )
        current_finger_state = self.robot.get_finger_state()

        # Tentatively apply robot action.
        delta_arm_joints = action[:7]
        # Clip the action to be within the allowed limits.
        delta_joints = np.clip(
            delta_arm_joints,
            -self.config.max_action_mag,
            self.config.max_action_mag,
        )
        next_joints = np.clip(
            current_joints + delta_joints,
            self.robot.joint_lower_limits[:7],
            self.robot.joint_upper_limits[:7],
        ).tolist()
        self._set_robot_and_held_object(next_joints, current_finger_state)

        # Check for collisions.
        if self._robot_or_held_object_collision_exists():
            # Revert!
            self._set_robot_and_held_object(current_joints, current_finger_state)

        # Check for grasping.
        if action[7] < -0.5:
            gripper_action = "close"
        elif action[7] > 0.5:
            gripper_action = "open"
        else:
            gripper_action = "none"

        if gripper_action == "close" and self._grasped_object is None:
            # Check if an object is in collision with the end effector marker.
            # If multiple objects are in collision, treat this as a failed grasp.
            objects_in_grasp_zone: set[str] = set()
            # Perform collision detection one-time rather than once per check.
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for obj in sorted(self._get_movable_object_names()):
                obj_id = self._object_name_to_pybullet_id(obj)
                if check_body_collisions(
                    obj_id,
                    self.end_effector_viz_id,
                    self.physics_client_id,
                    perform_collision_detection=False,
                ):
                    objects_in_grasp_zone.add(obj)
            # There must be exactly one object in the grasp zone to succeed.
            if len(objects_in_grasp_zone) == 1:
                self._grasped_object = next(iter(objects_in_grasp_zone))
                assert self._grasped_object_id is not None
                # Create grasp transform.
                world_to_robot = self.robot.get_end_effector_pose()
                world_to_object = get_pose(
                    self._grasped_object_id, self.physics_client_id
                )
                self._grasped_object_transform = multiply_poses(
                    world_to_robot.invert(), world_to_object
                )
                # Close the fingers until they are touching the object.
                while not check_body_collisions(
                    self._grasped_object_id, self.robot.robot_id, self.physics_client_id
                ):
                    # If the fingers are fully closed, stop.
                    current_finger_state = self.robot.get_finger_state()
                    closed_finger_state = self.robot.closed_fingers_state
                    assert isinstance(current_finger_state, float)
                    assert isinstance(closed_finger_state, float)
                    if current_finger_state >= closed_finger_state - 1e-2:
                        break
                    next_finger_state = current_finger_state + 1e-2
                    self.robot.set_finger_state(next_finger_state)

        # Check for ungrasping.
        elif gripper_action == "open" and self._grasped_object_id is not None:
            # Check if the held object is being placed on a surface. The rule is that
            # the distance between the object and the surface must be less than thresh.
            surface_supports = self._get_surfaces_supporting_object(
                self._grasped_object_id
            )
            # Placement is successful.
            if surface_supports:
                self._grasped_object = None
                self._grasped_object_transform = None
                self.robot.open_fingers()

        reward = -1
        terminated = self._goal_reached()
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(
            self.physics_client_id,
            image_width=self.config.render_image_width,
            image_height=self.config.render_image_height,
            **self.config.get_camera_kwargs(),
        )

    def _set_robot_and_held_object(
        self, joints: JointPositions, finger_state: float
    ) -> None:
        # First handle the robot arm joints.
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            self._grasped_object_id,
            self._grasped_object_transform,
            extend_joints_to_include_fingers(joints),
        )
        # Now handle the fingers.
        self.robot.set_finger_state(finger_state)
        # Update the end effector visualization.
        end_effector_pose = self.robot.get_end_effector_pose()
        set_pose(self.end_effector_viz_id, end_effector_pose, self.physics_client_id)

    def _robot_or_held_object_collision_exists(self) -> bool:
        collision_bodies = self._get_collision_object_ids()
        if self._grasped_object_id is not None:
            collision_bodies.discard(self._grasped_object_id)
        return check_collisions_with_held_object(
            self.robot,
            collision_bodies,
            self.physics_client_id,
            self._grasped_object_id,
            self._grasped_object_transform,
            self.robot.get_joint_positions(),
        )

    def _get_surfaces_supporting_object(self, object_id: int) -> set[int]:
        thresh = self.config.min_placement_dist
        supporting_surface_ids: set[int] = set()
        for surface in self._get_surface_object_names():
            surface_id = self._object_name_to_pybullet_id(surface)
            if check_body_collisions(
                object_id, surface_id, self.physics_client_id, distance_threshold=thresh
            ):
                supporting_surface_ids.add(surface_id)
        return supporting_surface_ids

    def _create_state_dict(
        self, objects: list[tuple[str, Type]]
    ) -> dict[Object, dict[str, float]]:
        state_dict: dict[Object, dict[str, float]] = {}
        for object_name, object_type in objects:
            obj = Object(object_name, object_type)
            feats: dict[str, float] = {}
            # Handle robots.
            if object_type == Geom3DRobotType:
                # Add joints.
                joints = remove_fingers_from_extended_joints(
                    self.robot.get_joint_positions()
                )
                for i, v in enumerate(joints):
                    feats[f"joint_{i+1}"] = v
                # Add finger state.
                feats["finger_state"] = self.robot.get_finger_state()
                # Add grasp.
                grasp_tf_feat_names = [
                    "grasp_tf_x",
                    "grasp_tf_y",
                    "grasp_tf_z",
                    "grasp_tf_qx",
                    "grasp_tf_qy",
                    "grasp_tf_qz",
                    "grasp_tf_qw",
                ]
                if self._grasped_object_transform is None:
                    feats["grasp_active"] = 0
                    for feat_name in grasp_tf_feat_names:
                        feats[feat_name] = 0
                else:
                    feats["grasp_active"] = 1
                    grasp_tf_feats = list(
                        self._grasped_object_transform.position
                    ) + list(self._grasped_object_transform.orientation)
                    for feat_name, feat in zip(
                        grasp_tf_feat_names, grasp_tf_feats, strict=True
                    ):
                        feats[feat_name] = feat
            # Handle cuboids.
            elif object_type == Geom3DCuboidType:
                # Add pose.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                pose_feat_names = [
                    "pose_x",
                    "pose_y",
                    "pose_z",
                    "pose_qx",
                    "pose_qy",
                    "pose_qz",
                    "pose_qw",
                ]
                pose_feats = list(pose.position) + list(pose.orientation)
                for feat_name, feat in zip(pose_feat_names, pose_feats, strict=True):
                    feats[feat_name] = feat
                # Add grasp active.
                if self._grasped_object == object_name:
                    feats["grasp_active"] = 1
                else:
                    feats["grasp_active"] = 0
                # Add half extents.
                half_extent_names = ["half_extent_x", "half_extent_y", "half_extent_z"]
                half_extents = self._get_half_extents(object_name)
                for feat_name, feat in zip(
                    half_extent_names, half_extents, strict=True
                ):
                    feats[feat_name] = feat
            # Handle points.
            elif object_type == Geom3DPointType:
                # Add position.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                feats["x"] = pose.position[0]
                feats["y"] = pose.position[1]
                feats["z"] = pose.position[2]
            else:
                raise NotImplementedError(f"Unsupported object type: {object_type}")
            # Add feats to state dict.
            state_dict[obj] = feats
        return state_dict
