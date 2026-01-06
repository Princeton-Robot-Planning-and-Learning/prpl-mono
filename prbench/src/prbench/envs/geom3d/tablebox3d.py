"""PyBullet environment where a box must be picked from the table.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Type as TypingType

import numpy as np
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_hollow_box
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
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState
from prbench.envs.utils import PURPLE


@dataclass(frozen=True)
class TableBox3DEnvConfig(Geom3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for TableBox3DEnv()."""

    # Table.
    table_pose: Pose = Pose((0.6, 0.0, 0.25))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.25)

    # World bounds.
    x_lb: float = -1
    x_ub: float = 1
    y_lb: float = -1
    y_ub: float = 1

    # Blocks.
    block_size: float = 0.05  # cubes (height = width = length)
    block_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)

    # Box.
    box_half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1)
    box_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)
    box_wall_thickness: float = 0.01

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 90,
            "camera_distance": 2.0,
            "camera_pitch": -20,
        }

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )

    def sample_block_on_ground(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        lb = (
            self.x_lb,
            self.y_lb,
            block_half_extents[2],
        )

        ub = (
            self.x_ub,
            self.y_ub,
            block_half_extents[2],
        )

        for _ in range(100):
            x, y, z = rng.uniform(lb, ub)
            if (
                np.abs(x - self.table_pose.position[0]) > self.table_half_extents[0]
                and np.abs(y - self.table_pose.position[1]) > self.table_half_extents[1]
            ):
                break
        else:
            raise RuntimeError("Failed to sample collision-free block pose on ground")

        return Pose((x, y, z))

    def sample_block_in_box_pose(
        self,
        block_half_extents: tuple[float, float, float],
        box_pose: Pose,
        box_half_extents: tuple[float, float, float],
        box_wall_thickness: float,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        assert np.allclose(box_pose.orientation, (0, 0, 0, 1)), "Not implemented"

        lb = (
            box_pose.position[0] - box_half_extents[0] + block_half_extents[0],
            box_pose.position[1] - box_half_extents[1] + block_half_extents[1],
            box_pose.position[2] + block_half_extents[2] + box_wall_thickness,
        )

        ub = (
            box_pose.position[0] + box_half_extents[0] - block_half_extents[0],
            box_pose.position[1] + box_half_extents[1] - block_half_extents[1],
            box_pose.position[2] + block_half_extents[2] + box_wall_thickness,
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))


class TableBox3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the TableBox3DEnv().

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


class ObjectCentricTableBox3DEnv(
    ObjectCentricGeom3DRobotEnv[Geom3DObjectCentricState, TableBox3DEnvConfig]
):
    """PyBullet environment where a box must be picked from the table.

    There may be other obstructing objects in the environment.
    """

    def __init__(
        self,
        num_cubes: int = 2,
        num_boxes: int = 1,
        config: TableBox3DEnvConfig = TableBox3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes
        self._num_boxes = num_boxes

        # Create the cubes, but their poses will be reset (with collision checking) in
        # the reset() method.
        self._cubes: dict[str, int] = {}
        for idx in range(self._num_cubes):
            cube_id = create_pybullet_block(
                self.config.block_rgba,
                (
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                ),
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id

        # Create the boxes, but their poses will be reset (with collision checking) in
        # the reset() method.
        self._boxes: dict[str, int] = {}
        for idx in range(self._num_boxes):
            box_id = create_pybullet_hollow_box(
                self.config.box_rgba,
                self.config.box_half_extents,
                self.config.box_wall_thickness,
                physics_client_id=self.physics_client_id,
            )
            self._boxes[f"box{idx}"] = box_id

        # Create table.
        self.table_id = create_pybullet_block(
            self.config.table_rgba,
            half_extents=self.config.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self.config.table_pose, self.physics_client_id)

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return TableBox3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("table", Geom3DCuboidType)])

    def _reset_objects(self) -> None:
        # Randomly sample collision-free positions for the cubes.
        # Also ensure that they are not in collision with the robot.
        # Samples the poses of the cubes
        for _, box_id in self._boxes.items():
            box_half_extents = (
                self.config.box_half_extents[0],
                self.config.box_half_extents[1],
                self.config.box_half_extents[2],
            )
            # on the table
            # box_pose = self.config.sample_block_on_table_pose(
            #     box_half_extents, self.np_random
            # )
            # on the ground
            box_pose = self.config.sample_block_on_ground(
                box_half_extents, self.np_random
            )
            set_pose(box_id, box_pose, self.physics_client_id)
        for _ in range(100_000):

            for cube_name, cube_id in self._cubes.items():
                cube_half_extents = (
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                )
                # add orientation later
                cube_pose = self.config.sample_block_in_box_pose(
                    cube_half_extents,
                    box_pose,
                    box_half_extents,
                    self.config.box_wall_thickness,
                    self.np_random,
                )
                set_pose(cube_id, cube_pose, self.physics_client_id)

            collision_free = True
            for cube_name, cube_id in self._cubes.items():
                for other_cube_name, other_cube_id in self._cubes.items():
                    if cube_name == other_cube_name:
                        continue
                    if check_body_collisions(
                        cube_id,
                        other_cube_id,
                        self.physics_client_id,
                    ):
                        collision_free = False
                        break

            if collision_free:
                break

        else:
            raise RuntimeError("Failed to sample collision-free cube poses")

    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
        assert isinstance(obs, TableBox3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

        for box_name, box_id in self._boxes.items():
            assert box_id is not None
            set_pose(
                box_id,
                obs.get_object_pose(box_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "table":
            return self.table_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        if object_name.startswith("box"):
            return self._boxes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return {self.table_id}

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys()) | set(self._boxes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"table"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return (
                self.config.block_size / 2,
                self.config.block_size / 2,
                self.config.block_size / 2,
            )
        if object_name.startswith("box"):
            return (
                self.config.box_half_extents[0],
                self.config.box_half_extents[1],
                self.config.box_half_extents[2],
            )
        if object_name == "table":
            return self.config.table_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> TableBox3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType)]
            + [("table", Geom3DCuboidType)]
            + [("cube" + str(i), Geom3DCuboidType) for i in range(self._num_cubes)]
            + [("box" + str(i), Geom3DCuboidType) for i in range(self._num_boxes)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=TableBox3DObjectCentricState
        )
        assert isinstance(state, TableBox3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        return False


class TableBox3DEnv(ConstantObjectPRBenchEnv):
    """Table Box 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricTableBox3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "table"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
            if obj.name.startswith("box"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        return """A 3D environment where the goal is to pick up a box from the table."""

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observations consist of:
- **robot**: The pose of the robot.
- **cubes**: The poses of the cubes.
- **boxes**: The poses of the boxes.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-0.01) per timestep to encourage exploration."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment."""
