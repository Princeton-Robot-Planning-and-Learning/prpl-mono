"""Environment where obstructions must be cleared to place a target on a region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prbench.core import ConstantObjectPRBenchEnv
from prbench.envs.geom3d.base_env import (
    Geom3DEnvConfig,
    ObjectCentricGeom3DRobotEnv,
)
from prbench.envs.geom3d.object_types import (
    Geom3DCuboidType,
    Geom3DEnvTypeFeatures,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState
from prbench.envs.utils import PURPLE


@dataclass(frozen=True)
class Obstruction3DEnvConfig(Geom3DEnvConfig):
    """Config for Obstruction3DEnv()."""

    # Table.
    table_pose: Pose = Pose((0.3, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.25)

    # Target region.
    target_region_half_extents_lb: tuple[float, float, float] = (0.02, 0.02, 0.005)
    target_region_half_extents_ub: tuple[float, float, float] = (0.05, 0.05, 0.005)
    target_region_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)

    # Target block.
    target_block_size_scale: float = 0.8  # x, y -- relative to target region
    target_block_height: float = 0.025
    target_block_rgba: tuple[float, float, float, float] = target_region_rgba

    # Obstructions.
    obstruction_half_extents_lb: tuple[float, float, float] = (0.01, 0.01, 0.01)
    obstruction_half_extents_ub: tuple[float, float, float] = (0.02, 0.02, 0.03)
    obstruction_rgba: tuple[float, float, float, float] = (0.75, 0.1, 0.1, 1.0)
    # NOTE: this is not the "real" probability, but rather, the probability
    # that we will attempt to sample the obstruction somewhere on the target
    # surface during each round of rejection sampling during reset().
    obstruction_init_on_target_prob: float = 0.9

    def _sample_block_on_block_pose(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample one block pose on top of another one, with no hanging allowed."""
        assert np.allclose(
            bottom_block_pose.orientation, (0, 0, 0, 1)
        ), "Not implemented"

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            + top_block_half_extents[0],
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            + top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            - top_block_half_extents[0],
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            - top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))

    def _sample_block_on_block_pose_with_overhang(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample one block pose on top of another one, where hanging is allowed."""
        assert np.allclose(
            bottom_block_pose.orientation, (0, 0, 0, 1)
        ), "Not implemented"

        overhang_pad = 1e-3

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            - top_block_half_extents[0]
            + overhang_pad,
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            - top_block_half_extents[1]
            + overhang_pad,
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            + top_block_half_extents[0]
            - overhang_pad,
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            + top_block_half_extents[1]
            - overhang_pad,
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )

    def get_target_block_half_extents(
        self, target_region_half_extents: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Calculate the target block half extents based on the target region."""
        return (
            self.target_block_size_scale * target_region_half_extents[0],
            self.target_block_size_scale * target_region_half_extents[1],
            self.target_block_height,
        )

    def sample_obstruction_pose_on_target(
        self,
        obstruction_half_extents: tuple[float, float, float],
        target_region_half_extents: tuple[float, float, float],
        target_region_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample a pose for the obstruction on top of the target region."""
        return self._sample_block_on_block_pose_with_overhang(
            obstruction_half_extents,
            target_region_half_extents,
            target_region_pose,
            rng,
        )


class Obstruction3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the Obstruction3DEnv().

    Adds convenience methods on top of Geom3DObjectCentricState().
    """

    def get_cuboid_half_extents(self, name: str) -> tuple[float, float, float]:
        """The half extents of the cuboid."""
        obj = self.get_object_from_name(name)
        return (
            self.get(obj, "half_extent_x"),
            self.get(obj, "half_extent_y"),
            self.get(obj, "half_extent_z"),
        )

    def get_cuboid_pose(self, name: str) -> Pose:
        """The pose of the cuboid."""
        obj = self.get_object_from_name(name)
        position = (
            self.get(obj, "pose_x"),
            self.get(obj, "pose_y"),
            self.get(obj, "pose_z"),
        )
        orientation = (
            self.get(obj, "pose_qx"),
            self.get(obj, "pose_qy"),
            self.get(obj, "pose_qz"),
            self.get(obj, "pose_qw"),
        )
        return Pose(position, orientation)

    @property
    def target_region_half_extents(self) -> tuple[float, float, float]:
        """The half extents of the target region, assuming the name "target_region"."""
        return self.get_cuboid_half_extents("target_region")

    @property
    def target_block_half_extents(self) -> tuple[float, float, float]:
        """The half extents of the target block, assuming the name "target_block"."""
        return self.get_cuboid_half_extents("target_block")

    @property
    def target_region_pose(self) -> Pose:
        """The pose of the target region, assuming the name "target_region"."""
        return self.get_cuboid_pose("target_region")

    @property
    def target_block_pose(self) -> Pose:
        """The pose of the target block, assuming the name "target_block"."""
        return self.get_cuboid_pose("target_block")


class ObjectCentricObstruction3DEnv(
    ObjectCentricGeom3DRobotEnv[Obstruction3DObjectCentricState, Obstruction3DEnvConfig]
):
    """Environment where obstructions must be cleared to place a target on a region."""

    def __init__(
        self,
        num_obstructions: int = 2,
        config: Obstruction3DEnvConfig = Obstruction3DEnvConfig(),
        **kwargs,
    ) -> None:
        self._num_obstructions = num_obstructions
        super().__init__(config=config, **kwargs)

        # Create table.
        self.table_id = create_pybullet_block(
            self.config.table_rgba,
            half_extents=self.config.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self.config.table_pose, self.physics_client_id)

        # The objects are created in reset() because they have geometries that change
        # in each episode.
        self._target_region_id: int | None = None
        self._target_region_half_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._target_block_id: int | None = None
        self._target_block_half_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._obstruction_ids: dict[str, int] = {}
        self._obstruction_id_to_half_extents: dict[int, tuple[float, float, float]] = {}

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return Obstruction3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("table", Geom3DCuboidType)])

    def _reset_objects(self) -> None:

        # Destroy old objects that have varying geometries.
        for old_id in {
            self._target_region_id,
            self._target_block_id,
        } | set(self._obstruction_ids.values()):
            if old_id is not None:
                p.removeBody(old_id, physicsClientId=self.physics_client_id)

        # Recreate the target region.
        self._target_region_half_extents = tuple(
            self.np_random.uniform(
                self.config.target_region_half_extents_lb,
                self.config.target_region_half_extents_ub,
            )
        )
        target_region_pose = self.config.sample_block_on_table_pose(
            self._target_region_half_extents, self.np_random
        )
        self._target_region_id = create_pybullet_block(
            self.config.target_region_rgba,
            half_extents=self._target_region_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self._target_region_id, target_region_pose, self.physics_client_id)

        # Recreate the target block.
        self._target_block_half_extents = self.config.get_target_block_half_extents(
            self._target_region_half_extents
        )
        self._target_block_id = create_pybullet_block(
            self.config.target_block_rgba,
            half_extents=self._target_block_half_extents,
            physics_client_id=self.physics_client_id,
        )
        for _ in range(100_000):
            target_block_pose = self.config.sample_block_on_table_pose(
                self._target_block_half_extents, self.np_random
            )
            set_pose(self._target_block_id, target_block_pose, self.physics_client_id)
            # Make sure the target block is not touching the target region at all.
            if not check_body_collisions(
                self._target_block_id,
                self._target_region_id,
                self.physics_client_id,
            ):
                break
        else:
            raise RuntimeError("Failed to sample target block pose")

        # Recreate the obstructions.
        self._obstruction_ids = {}
        self._obstruction_id_to_half_extents = {}
        for obstruction_idx in range(self._num_obstructions):
            obstruction_name = f"obstruction{obstruction_idx}"
            obstruction_half_extents: tuple[float, float, float] = tuple(
                self.np_random.uniform(
                    self.config.obstruction_half_extents_lb,
                    self.config.obstruction_half_extents_ub,
                )
            )
            obstruction_id = create_pybullet_block(
                self.config.obstruction_rgba,
                half_extents=obstruction_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self._obstruction_id_to_half_extents[obstruction_id] = (
                obstruction_half_extents
            )
            self._obstruction_ids[obstruction_name] = obstruction_id
            for _ in range(100_000):
                obstruction_init_on_target = (
                    self.np_random.uniform()
                    < self.config.obstruction_init_on_target_prob
                )
                collision_ids = (
                    {self._target_block_id} | set(self._obstruction_ids.values())
                ) - {obstruction_id}
                if obstruction_init_on_target:
                    obstruction_pose = self.config.sample_obstruction_pose_on_target(
                        obstruction_half_extents,
                        self._target_region_half_extents,
                        target_region_pose,
                        self.np_random,
                    )
                else:
                    obstruction_pose = self.config.sample_block_on_table_pose(
                        obstruction_half_extents, self.np_random
                    )
                    collision_ids.add(self._target_region_id)
                set_pose(obstruction_id, obstruction_pose, self.physics_client_id)
                # Make sure the target block is not touching the target region at all.
                collision_exists = False
                for collision_id in collision_ids:
                    if check_body_collisions(
                        obstruction_id,
                        collision_id,
                        self.physics_client_id,
                    ):
                        collision_exists = True
                        break
                if not collision_exists:
                    break
            else:
                raise RuntimeError("Failed to sample target block pose")

    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
        assert isinstance(obs, Obstruction3DObjectCentricState)
        # Check if target region needs to be recreated.
        if self._target_region_half_extents != obs.target_region_half_extents:
            # Recreate the target region.
            if self._target_region_id is not None:
                p.removeBody(
                    self._target_region_id, physicsClientId=self.physics_client_id
                )
            self._target_region_id = create_pybullet_block(
                self.config.target_region_rgba,
                half_extents=obs.target_region_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self._target_region_half_extents = obs.target_region_half_extents
        # Update target region pose.
        assert self._target_region_id is not None
        set_pose(self._target_region_id, obs.target_region_pose, self.physics_client_id)

        # Check if target block needs to be recreated.
        if self._target_block_half_extents != obs.target_block_half_extents:
            # Recreate the target block.
            if self._target_block_id is not None:
                p.removeBody(
                    self._target_block_id, physicsClientId=self.physics_client_id
                )
            self._target_block_id = create_pybullet_block(
                self.config.target_block_rgba,
                half_extents=obs.target_block_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self._target_block_half_extents = obs.target_block_half_extents
        # Update target block pose.
        assert self._target_block_id is not None
        set_pose(self._target_block_id, obs.target_block_pose, self.physics_client_id)

        # Handle obstructions.
        for obstruction_idx in range(self._num_obstructions):
            obstruction_name = f"obstruction{obstruction_idx}"
            obstruction_half_extents = obs.get_cuboid_half_extents(obstruction_name)
            obstruction_pose = obs.get_cuboid_pose(obstruction_name)
            # Check if the block needs to be recreated.
            need_recreate = False
            need_destroy = False
            if not self._obstruction_ids:
                need_recreate = True
            else:
                obstruction_id = self._object_name_to_pybullet_id(obstruction_name)
                current_half_extents = self._obstruction_id_to_half_extents[
                    obstruction_id
                ]
                need_recreate = current_half_extents != obstruction_half_extents
                need_destroy = need_recreate
            if need_recreate:
                # Recreate the obstruction.
                if need_destroy:
                    p.removeBody(obstruction_id, physicsClientId=self.physics_client_id)
                obstruction_id = create_pybullet_block(
                    self.config.obstruction_rgba,
                    half_extents=obstruction_half_extents,
                    physics_client_id=self.physics_client_id,
                )
                self._obstruction_ids[obstruction_name] = obstruction_id
                self._obstruction_id_to_half_extents[obstruction_id] = (
                    obstruction_half_extents
                )
            # Update obstruction block pose.
            set_pose(obstruction_id, obstruction_pose, self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "target_region":
            assert self._target_region_id is not None
            return self._target_region_id
        if object_name == "target_block":
            assert self._target_block_id is not None
            return self._target_block_id
        if object_name == "table":
            return self.table_id
        if object_name.startswith("obstruction"):
            return self._obstruction_ids[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        assert self._target_block_id is not None
        assert self._target_region_id is not None
        return {self._target_block_id, self._target_region_id, self.table_id} | set(
            self._obstruction_ids.values()
        )

    def _get_movable_object_names(self) -> set[str]:
        return {"target_block"} | set(self._obstruction_ids)

    def _get_surface_object_names(self) -> set[str]:
        return {"target_region", "table"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name == "target_region":
            return self._target_region_half_extents
        if object_name == "target_block":
            return self._target_block_half_extents
        assert object_name.startswith("obstruction")
        obj_id = self._object_name_to_pybullet_id(object_name)
        return self._obstruction_id_to_half_extents[obj_id]

    def _get_obs(self) -> Obstruction3DObjectCentricState:
        state_dict = self._create_state_dict(
            [
                ("robot", Geom3DRobotType),
                ("target_region", Geom3DCuboidType),
                ("target_block", Geom3DCuboidType),
            ]
            + [
                (f"obstruction{i}", Geom3DCuboidType)
                for i in range(self._num_obstructions)
            ]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Obstruction3DObjectCentricState
        )
        assert isinstance(state, Obstruction3DObjectCentricState)
        return state

    def _goal_reached(self) -> bool:
        if self._grasped_object is not None:
            return False
        assert self._target_block_id is not None
        target_supports = self._get_surfaces_supporting_object(self._target_block_id)
        return self._target_region_id in target_supports


class Obstruction3DEnv(ConstantObjectPRBenchEnv):
    """Obstruction 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricObstruction3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_region", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruct"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Obstruction3DEnvConfig)
        return f"""A 3D obstruction clearance environment where the goal is to place a target block on a designated target region by first clearing obstructions.

The robot is a Kinova Gen-3 with 7 degrees of freedom that can grasp and manipulate objects. The environment consists of:
- A **table** with dimensions {config.table_half_extents[0]*2:.3f}m × {config.table_half_extents[1]*2:.3f}m × {config.table_half_extents[2]*2:.3f}m
- A **target region** (purple block) with random dimensions between {config.target_region_half_extents_lb} and {config.target_region_half_extents_ub} half-extents
- A **target block** that must be placed on the target region, sized at {config.target_block_size_scale}× the target region's x,y dimensions
- **Obstruction(s)** (red blocks) that may be placed on or near the target region, blocking access

Obstructions have random dimensions between {config.obstruction_half_extents_lb} and {config.obstruction_half_extents_ub} half-extents. During initialization, there's a {config.obstruction_init_on_target_prob} probability that each obstruction will be placed on the target region, requiring clearance.

The task requires planning to grasp and move obstructions out of the way, then place the target block on the target region.
"""

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Obstruction3DEnvConfig)
        return f"""Observations consist of:
- **joint_positions**: Current joint positions of the {len(config.initial_joints)}-DOF robot arm (list of floats)
- **grasped_object**: Name of currently grasped object, or None if not grasping anything (string or None)
- **grasped_object_transform**: Relative transform of grasped object to gripper, or None if not grasping (transform or None)
- **target_region**: State of the target region including:
  - pose: 3D position and orientation (Pose object)
  - geometry: Half-extents (width/2, height/2, depth/2) of the region (tuple of 3 floats)
- **target_block**: State of the target block including:
  - pose: 3D position and orientation (Pose object)
  - geometry: Half-extents of the block (tuple of 3 floats)
- **obstructions**: Dictionary of obstruction states, keyed by obstruction name (e.g., "obstruction0"), each containing:
  - pose: 3D position and orientation (Pose object)
  - geometry: Half-extents of the obstruction (tuple of 3 floats)

The observation is returned as an Obstruction3DState dataclass with these fields.
"""

    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Obstruction3DEnvConfig)
        return f"""Actions control the change in joint positions:
- **delta_arm_joints**: Change in joint positions for all {len(config.initial_joints)} joints (list of floats)

The action is an Obstruction3DAction dataclass with delta_arm_joints field. Each delta is clipped to the range [-{config.max_action_mag:.3f}, {config.max_action_mag:.3f}].

The resulting joint positions are clipped to the robot's joint limits before being applied. The robot can automatically grasp objects when the gripper is close enough and release them with appropriate actions.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when the target block is placed on the target region (while not being grasped)

The goal is considered reached when:
1. The robot is not currently grasping the target block
2. The target block is resting on (supported by) the target region

Support is determined based on contact between the target block and target region, within a small distance threshold (1e-4).

This encourages the robot to efficiently clear obstructions and place the target block while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """Similar environments have been used many times, especially in the task and motion planning literature. We took inspiration especially from the "1D Continuous TAMP" environment in [PDDLStream](https://github.com/caelan/pddlstream).
"""
