"""Environment where multiple objects must be packed into a rack without collisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_triangle
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prbench.core import ConstantObjectPRBenchEnv, FinalConfigMeta
from prbench.envs.geom3d.base_env import (
    Geom3DEnvConfig,
    ObjectCentricGeom3DRobotEnv,
)
from prbench.envs.geom3d.object_types import (
    Geom3DCuboidType,
    Geom3DEnvTypeFeatures,
    Geom3DRobotType,
    Geom3DTriangleType,
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState
from prbench.envs.utils import PURPLE

@dataclass(frozen=True)
class Packing3DEnvConfig(Geom3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Packing3DEnv()."""

    # Table.
    table_pose: Pose = Pose((0.3, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.25)

    # rack (target) region.
    rack_half_extents: tuple[float, float, float] = (0.05, 0.1, 0.05)
    rack_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)

    # Parts.
    part_half_extents_lb: tuple[float, float, float] = (0.03, 0.03, 0.01)
    part_half_extents_ub: tuple[float, float, float] = (0.05, 0.05, 0.01)
    part_rgba: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 1.0)

    # Triangle parts.
    part_triangle_depth: float = 0.01  # fixed depth for triangle parts
    part_triangle_side_lb: float = 0.06  # min side length for triangle parts
    part_triangle_side_ub: float = 0.1  # max side length for triangle parts

    # Probability a part is triangular
    part_triangular_prob: float = 0.5

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

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )

    def sample_part_half_extents(self, rng: np.random.Generator) -> tuple[float, float, float]:
        return tuple(rng.uniform(self.part_half_extents_lb, self.part_half_extents_ub))
    
    def sample_part_triangle_features(self, rng: np.random.Generator) -> tuple[float, float, float, float]:
        """Sample triangle features (side_a, side_b, depth) of a triangle object.

        triangle_type is encoded as:
        0 = equilateral
        1 = isosceles
        2 = right
        """
        triangle_type = rng.choice([0, 1, 2])
        if triangle_type == 0:  # equilateral
            side = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            return side, side, self.part_triangle_depth, float(triangle_type)
        elif triangle_type == 1:  # isosceles
            base = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            height = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            return base, height, self.part_triangle_depth, float(triangle_type)
        else:  # right
            base = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            height = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            return base, height, self.part_triangle_depth, float(triangle_type)

class Packing3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the Packing3DEnv().

    Adds convenience methods on top of Geom3DObjectCentricState().
    """

    def get_object_pose(self, name: str) -> Pose:
        """The pose of the object."""
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

    def get_object_half_extents(self, name: str) -> tuple[float, float, float]:
        """Get the half extents of a cuboid object."""
        obj = self.get_object_from_name(name)
        return (
            self.get(obj, "half_extent_x"),
            self.get(obj, "half_extent_y"),
            self.get(obj, "half_extent_z"),
        )
    
    def get_object_triangle_features(self, name: str) -> tuple[float, float, float]:
        """Get the triangle features (side_a, side_b, side_c, depth) of a triangle object."""
        obj = self.get_object_from_name(name)
        return (
            self.get(obj, "side_a"),
            self.get(obj, "side_b"),
            self.get(obj, "depth")
        )
    
    @property
    def rack_half_extents(self) -> tuple[float, float, float]:
        """Get the half extents of the rack."""
        return self.get_object_half_extents("rack")

    @property
    def rack_pose(self) -> Pose:
        """Get the pose of the rack."""
        return self.get_object_pose("rack")
    
    @property
    def part_poses(self) -> dict[str, Pose]:
        """Get the poses of all parts."""
        poses = {}
        for obj in self.objects:
            if obj.name.startswith("part"):
                poses[obj.name] = self.get_object_pose(obj.name)
        return poses
    
    @property
    def part_types(self) -> dict[str, TypingType]:
        """Get the types of all parts."""
        types = {}
        for obj in self.objects:
            if obj.name.startswith("part"):
                types[obj.name] = obj.type
        return types

    @property
    def part_features(self) -> dict[str, tuple[float, float, float] | tuple[float, float, float]]:
        """Get the features of all parts."""
        features = {}
        for obj in self.objects:
            if obj.name.startswith("part"):
                if obj.type == Geom3DCuboidType:
                    features[obj.name] = self.get_object_half_extents(obj.name)
                elif obj.type == Geom3DTriangleType:
                    features[obj.name] = self.get_object_triangle_features(obj.name)
                else:
                    raise ValueError(f"Unsupported part type: {obj.type}")
        return features

class ObjectCentricPacking3DEnv(
    ObjectCentricGeom3DRobotEnv[Packing3DObjectCentricState, Packing3DEnvConfig]
):
    """Environment where small parts must be packed into a rack without collisions."""

    def __init__(
        self,
        num_parts: int = 2,
        config: Packing3DEnvConfig = Packing3DEnvConfig(),
        **kwargs,
    ) -> None:
        self._num_parts = num_parts
        super().__init__(config=config, **kwargs)

        # Create table.
        self.table_id = create_pybullet_block(
            self.config.table_rgba,
            half_extents=self.config.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self.config.table_pose, self.physics_client_id)

        # rack (created in reset because geometry could be randomized later)
        self._rack_id = create_pybullet_block(
            self.config.rack_rgba,
            half_extents=self.config.rack_half_extents,
            physics_client_id=self.physics_client_id,
        )
        rack_pose = Pose(
            (
                self.config.table_pose.position[0],
                self.config.table_pose.position[1],
                self.config.table_pose.position[2] + self.config.table_half_extents[2] + self.config.rack_half_extents[2],
            )
        )
        set_pose(self._rack_id, rack_pose, self.physics_client_id)

        # Parts
        self._part_ids = {}
        self._part_id_to_half_extents = {}

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return Packing3DObjectCentricState
    
    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("table", Geom3DCuboidType)])
    
    def _reset_objects(self) -> None:

        # Destroy previous parts.
        for old_id in set(self._part_ids.values()):
            if old_id is not None:
                p.removeBody(old_id, physicsClientId=self.physics_client_id)

        # Create parts and place them on the table with rejection sampling to avoid
        # initial collisions. Parts are modeled as cuboids with fixed z-depth.
        self._part_ids = {}
        self._part_ids_to_type = {}
        self._part_id_to_half_extents = {}
        self._part_ids_to_triangle_features = {}
        part_z_half_extent = 0.02  # fixed z-depth for all parts
        for i in range(self._num_parts):
            name = f"part{i}"
            # Type could be extended to support triangles in future.
            part_type = (Geom3DCuboidType if self.np_random.uniform() > self.config.part_triangular_prob
                         else Geom3DTriangleType)
            print(f"Creating {part_type} named {name}")
            # Sample part half extents from config.
            if part_type == Geom3DCuboidType:
                sampled = self.config.sample_part_half_extents(self.np_random)
                half_extents = (sampled[0], sampled[1], sampled[2])
                part_id = create_pybullet_block(
                    self.config.part_rgba,
                    half_extents=half_extents,
                    physics_client_id=self.physics_client_id,
                )
                self._part_id_to_half_extents[part_id] = half_extents
                self._part_ids[name] = part_id
                self._part_ids_to_type[part_id] = Geom3DCuboidType

                # Place part on table while avoiding collisions with other parts and
                # the rack (we allow parts to start outside the rack)
                for _ in range(100_000):
                    # Sample a pose on the table surface.
                    x = self.np_random.uniform(
                        self.config.table_pose.position[0] - self.config.table_half_extents[0] + half_extents[0],
                        self.config.table_pose.position[0] + self.config.table_half_extents[0] - half_extents[0],
                    )
                    y = self.np_random.uniform(
                        self.config.table_pose.position[1] - self.config.table_half_extents[1] + half_extents[1],
                        self.config.table_pose.position[1] + self.config.table_half_extents[1] - half_extents[1],
                    )
                    z = self.config.table_pose.position[2] + self.config.table_half_extents[2] + part_z_half_extent
                    set_pose(part_id, Pose((x, y, z)), self.physics_client_id)

                    collision_exists = False
                    for other_id in ({self._rack_id} | set(self._part_ids.values())) - {part_id}:
                        if check_body_collisions(part_id, other_id, self.physics_client_id):
                            collision_exists = True
                            break
                    if not collision_exists:
                        break
                else:
                    raise RuntimeError("Failed to sample part pose")
                
            elif part_type == Geom3DTriangleType:
                side_a, side_b, depth, triangle_type = self.config.sample_part_triangle_features(self.np_random)
                half_extents = (max(side_a, side_b) / 2, max(side_a, side_b) / 2, depth / 2)
                part_id = create_pybullet_triangle(
                    self.config.part_rgba,
                    type={0: 'equilateral', 1: 'isosceles', 2: 'right'}[int(triangle_type)],
                    side_lengths=(side_a, side_b),
                    depth=depth,
                    physics_client_id=self.physics_client_id,
                )
                self._part_id_to_half_extents[part_id] = half_extents
                self._part_ids[name] = part_id
                self._part_ids_to_type[part_id] = Geom3DTriangleType
                self._part_ids_to_triangle_features[part_id] = (side_a, side_b, depth, triangle_type)

                # Place part on table while avoiding collisions with other parts and
                # the rack (we allow parts to start outside the rack)
                for _ in range(100_000):
                    # Sample a pose on the table surface.
                    x = self.np_random.uniform(
                        self.config.table_pose.position[0] - self.config.table_half_extents[0] + half_extents[0],
                        self.config.table_pose.position[0] + self.config.table_half_extents[0] - half_extents[0],
                    )
                    y = self.np_random.uniform(
                        self.config.table_pose.position[1] - self.config.table_half_extents[1] + half_extents[1],
                        self.config.table_pose.position[1] + self.config.table_half_extents[1] - half_extents[1],
                    )
                    z = self.config.table_pose.position[2] + self.config.table_half_extents[2] + part_z_half_extent
                    set_pose(part_id, Pose((x, y, z)), self.physics_client_id)

                    collision_exists = False
                    for other_id in ({self._rack_id} | set(self._part_ids.values())) - {part_id}:
                        if check_body_collisions(part_id, other_id, self.physics_client_id):
                            collision_exists = True
                            break
                    if not collision_exists:
                        break
                else:
                    raise RuntimeError("Failed to sample part pose")
                
            else:
                raise ValueError(f"Unsupported part type: {part_type}")


    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
        assert isinstance(obs, Packing3DObjectCentricState)
        # Update rack (recreate if half extents changed)
        if self._rack_half_extents != getattr(obs, "rack_half_extents", self._rack_half_extents):
            if self._rack_id is not None:
                p.removeBody(self._rack_id, physicsClientId=self.physics_client_id)
            self._rack_half_extents = getattr(obs, "rack_half_extents", self._rack_half_extents)
            self._rack_id = create_pybullet_block(
                PURPLE + (0.8,),
                half_extents=self._rack_half_extents,
                physics_client_id=self.physics_client_id,
            )
        if self._rack_id is not None:
            # rack pose expected as a cuboid in the state
            set_pose(self._rack_id, obs.get_cuboid_pose("rack"), self.physics_client_id)

        parts = obs.part_poses
        assert len(parts) == self._num_parts, f"Expected {self._num_parts} parts, got {len(parts)}"

        # Update parts
        for i in range(self._num_parts):
            name = parts.keys()[i]
            half_extents = obs.get_cuboid_half_extents(name)
            obj_type = obs.part_types[name]
            pose = obs.get_cuboid_pose(name)
            need_recreate = False
            need_destroy = False
            if not self._part_ids:
                need_recreate = True
            else:
                part_id = self._object_name_to_pybullet_id(name)
                current_half_extents = self._part_id_to_half_extents[part_id]
                need_recreate = current_half_extents != half_extents
                need_destroy = need_recreate
            if need_recreate:
                if need_destroy:
                    p.removeBody(part_id, physicsClientId=self.physics_client_id)
                part_id = create_pybullet_block(
                    (0.2, 0.6, 0.2, 1.0),
                    half_extents=half_extents,
                    physics_client_id=self.physics_client_id,
                ) if obj_type == Geom3DCuboidType else create_pybullet_triangle(
                    (0.2, 0.6, 0.2, 1.0),
                    type={0: 'equilateral', 1: 'isosceles', 2: 'right'}[int(obs._part_ids_to_triangle_features[name][3])],
                    side_lengths=(obs._part_ids_to_triangle_features[name][0], obs._part_ids_to_triangle_features[name][1]),
                    depth=obs._part_ids_to_triangle_features[name][2],
                    physics_client_id=self.physics_client_id,
                )
                self._part_ids[name] = part_id
                self._part_id_to_half_extents[part_id] = half_extents
                self._part_id_to_type[part_id] = obj_type
                if obj_type == Geom3DTriangleType:
                    self._part_ids_to_triangle_features[part_id] = obs._part_ids_to_triangle_features[name]
            part_id = self._object_name_to_pybullet_id(name)
            set_pose(part_id, pose, self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "rack":
            assert self._rack_id is not None
            return self._rack_id
        if object_name == "table":
            return self.table_id
        if object_name.startswith("part"):
            return self._part_ids[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        ids = {self.table_id}
        if self._rack_id is not None:
            ids.add(self._rack_id)
        ids |= set(self._part_ids.values())
        return ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._part_ids.keys())

    def _get_surface_object_names(self) -> set[str]:
        # The rack and table are surfaces.
        names = {"table"}
        if self._rack_id is not None:
            names.add("rack")
        return names

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name == "rack":
            return self.config.rack_half_extents
        if object_name == "table":
            return self.config.table_half_extents
        assert object_name.startswith("part")
        part_id = self._object_name_to_pybullet_id(object_name)
        return self._part_id_to_half_extents[part_id]
    
    def _get_triangle_features(self, object_name: str) -> tuple[float, float, float, float]:
        if not object_name.startswith("part"):
            raise ValueError(f"Object {object_name} is not a part")
        part_id = self._object_name_to_pybullet_id(object_name)
        if part_id not in self._part_ids_to_triangle_features:
            raise ValueError(f"Object {object_name} is not a triangle")
        return self._part_ids_to_triangle_features[part_id]

    def _get_obs(self) -> Packing3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType), ("rack", Geom3DCuboidType)]
            + [(f"part{i}", self._part_ids_to_type[self._part_ids[f"part{i}"]]) for i in range(self._num_parts)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Packing3DObjectCentricState
        )
        assert isinstance(state, Packing3DObjectCentricState)
        return state

    def _goal_reached(self) -> bool:
        # Goal: no parts are grasped and all parts are supported by the rack.
        if self._grasped_object is not None:
            return False
        if self._rack_id is None:
            return False
        for part_id in self._part_ids.values():
            supports = self._get_surfaces_supporting_object(part_id)
            if self._rack_id not in supports:
                return False
        return True


class Packing3DEnv(ConstantObjectPRBenchEnv):
    """Packing 3D env with a constant number of objects."""

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricPacking3DEnv(*args, **kwargs)

    def _get_constant_object_names(self, exemplar_state: ObjectCentricState) -> list[str]:
        constant_objects = ["robot", "rack"]
        return constant_objects
    
    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Packing3DEnvConfig)
        return f"""A 3D packing environment where the goal is to place a set of parts into a rack without collisions.

The robot is a Kinova Gen-3 with 7 degrees of freedom that can grasp and manipulate objects. The environment consists of:
- A **table** with dimensions {config.table_half_extents[0]*2:.3f}m × {config.table_half_extents[1]*2:.3f}m × {config.table_half_extents[2]*2:.3f}m
- A **rack** (purple) with half-extents {config.rack_half_extents}
- **Parts** (green) that must be packed into the rack. Parts are sampled with half-extents in {config.part_half_extents_lb} to {config.part_half_extents_ub} and a probability {config.part_triangular_prob} of being triangle-shaped (triangles are represented as triangular prisms with depth {config.part_triangle_depth:.3f}m when used).

The task requires planning to grasp and place each part into the rack while avoiding collisions and ensuring parts are supported by the rack (on the rack and not grasped) at the end.
"""

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Packing3DEnvConfig)
        return f"""Observations consist of:
- **joint_positions**: Current joint positions of the {len(config.initial_joints)}-DOF robot arm (list of floats)
- **grasped_object**: Name of currently grasped object, or None if not grasping anything (string or None)
- **grasped_object_transform**: Relative transform of grasped object to gripper, or None if not grasping (transform or None)
- **rack**: State of the rack including:
  - pose: 3D position and orientation (Pose object)
  - geometry: Half-extents (width/2, height/2, depth/2) of the rack (tuple of 3 floats)
- **parts**: Dictionary of part states, keyed by part name (e.g., "part0"), each containing:
  - pose: 3D position and orientation (Pose object)
  - geometry: For cuboids: half-extents (tuple of 3 floats); for triangles: side lengths and depth (tuple of floats)

The observation is returned as a Packing3DState dataclass with these fields.
"""

    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Packing3DEnvConfig)
        return f"""Actions control the change in joint positions:
- **delta_arm_joints**: Change in joint positions for all {len(config.initial_joints)} joints (list of floats)

The action is a Packing3DAction dataclass with delta_arm_joints field. Each delta is clipped to the range [-{config.max_action_mag:.3f}, {config.max_action_mag:.3f}].

The resulting joint positions are clipped to the robot's joint limits before being applied. The robot can automatically grasp objects when the gripper is close enough and release them with appropriate actions.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when all parts are placed in the rack and none are grasped

The goal is considered reached when:
1. The robot is not currently grasping any part
2. Every part is resting on (supported by) the rack surface

Support is determined based on contact between a part and the rack within a small distance threshold (configured by the environment).

This encourages the robot to efficiently pack the parts into the rack while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """Packing tasks are common in robotics and automated warehousing literature. This environment is inspired by standard manipulation benchmarks and simple bin-packing problems; it’s intended as a deterministic, physics-based testbed for pick-and-place planning and task-and-motion planning approaches.
"""