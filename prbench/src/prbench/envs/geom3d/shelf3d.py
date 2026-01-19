"""PyBullet environment where an object must be picked from the ground and placed on a
shelf.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_shelf
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
    Geom3DFixtureType,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import (
    Geom3DObjectCentricState,
    sample_collision_free_object_poses,
)


@dataclass(frozen=True)
class Shelf3DEnvConfig(Geom3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Shelf3DEnv()."""

    max_action_mag: float = 0.2

    # Shelf.
    shelf_pose: Pose = Pose((2.0, 2.4, 0.02))
    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shelf_width: float = 0.60198
    shelf_depth: float = 0.254
    shelf_height: float = 0.0127
    shelf_spacing: float = 0.254
    shelf_support_width: float = 0.0127
    shelf_num_layers: int = 4
    shelf_texture: Path = Path(__file__).parent / "assets" / "dark-wood-texture.png"

    # World bounds.
    specific_range: bool = True
    if specific_range:
        x_lb: float = 0.4
        x_ub: float = 0.5
        y_lb: float = -0.1
        y_ub: float = 0.1
    else:
        x_lb: float = -1.5
        x_ub: float = 1.5
        y_lb: float = -1.5
        y_ub: float = 1.5

    # Blocks.
    block_half_extents: tuple[float, float, float] = (0.05, 0.025, 0.025)
    block_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    # Gripper.
    gripper_open_threshold: float = 0.01

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 0,
            "camera_distance": 2.0,
            "camera_pitch": -20,
        }

    def get_cube_texture(self, idx: int) -> Path:
        """Get a texture to wrap a cube given the index."""
        asset_dir = Path(__file__).parent / "assets"
        texture_filenames = [f"book{i}.jpg" for i in range(5)]
        texture_filename = texture_filenames[idx % len(texture_filenames)]
        return asset_dir / texture_filename


class Shelf3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the Shelf3DEnv().

    Adds convenience methods on top of Geom3DObjectCentricState().
    """


class ObjectCentricShelf3DEnv(
    ObjectCentricGeom3DRobotEnv[Geom3DObjectCentricState, Shelf3DEnvConfig]
):
    """PyBullet environment where objects must be picked from the ground and placed on a
    shelf."""

    def __init__(
        self,
        num_cubes: int = 2,
        config: Shelf3DEnvConfig = Shelf3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes

        # Create the cubes, but their poses will be reset (with collision checking) in
        # the reset() method.
        self._cubes: dict[str, int] = {}
        for idx in range(self._num_cubes):
            cube_id = create_pybullet_block(
                self.config.block_rgba,
                (
                    self.config.block_half_extents[0],
                    self.config.block_half_extents[1],
                    self.config.block_half_extents[2],
                ),
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id
            cube_texture_id = p.loadTexture(
                str(self.config.get_cube_texture(idx)), self.physics_client_id
            )
            p.changeVisualShape(
                cube_id,
                -1,
                textureUniqueId=cube_texture_id,
                physicsClientId=self.physics_client_id,
            )

        # Create shelf.
        self._shelf_id, self._shelf_surface_ids = create_pybullet_shelf(
            color=self.config.shelf_rgba,
            shelf_width=self.config.shelf_width,
            shelf_depth=self.config.shelf_depth,
            shelf_height=self.config.shelf_height,
            spacing=self.config.shelf_spacing,
            support_width=self.config.shelf_support_width,
            num_layers=self.config.shelf_num_layers,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self._shelf_id, self.config.shelf_pose, self.physics_client_id)

        # NOTE: use this for repositioning the shelf visually (with GUI on).
        # from pybullet_helpers.gui import interactively_visualize_pose
        # interactively_visualize_pose(
        #     self.config.shelf_pose,
        #     self.physics_client_id,
        #     min_position=-10,
        #     max_position=10,
        #     object_id=self._shelf_id,
        # )

        shelf_texture_id = p.loadTexture(
            str(self.config.shelf_texture), self.physics_client_id
        )
        for shelf_link_id in range(
            p.getNumJoints(self._shelf_id, physicsClientId=self.physics_client_id)
        ):
            p.changeVisualShape(
                self._shelf_id,
                shelf_link_id,
                textureUniqueId=shelf_texture_id,
                physicsClientId=self.physics_client_id,
            )

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return Shelf3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("shelf", Geom3DFixtureType)])

    def _reset_objects(self) -> None:
        sample_collision_free_object_poses(
            object_ids=set(self._cubes.values()),
            lb=(self.config.x_lb, self.config.y_lb, self.config.block_half_extents[2]),
            ub=(self.config.x_ub, self.config.y_ub, self.config.block_half_extents[2]),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids={self.robot.base.robot_id},
        )

    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
        assert isinstance(obs, Shelf3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "shelf":
            return self._shelf_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = {self._shelf_id} | set(self._cubes.values())
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"shelf"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return self.config.block_half_extents
        if object_name == "shelf":
            raise NotImplementedError("TODO")
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Shelf3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType)]
            + [("shelf", Geom3DFixtureType)]
            + [("cube" + str(i), Geom3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Shelf3DObjectCentricState
        )
        assert isinstance(state, Shelf3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        robot_gripper_pose = self._robot_arm.get_finger_state()
        if robot_gripper_pose > self.config.gripper_open_threshold:
            return False
        for _, cube_id in self._cubes.items():
            cube_pose = get_pose(cube_id, self.physics_client_id)
            if cube_pose.position[2] < 0.3:
                return False

        return True


class Shelf3DEnv(ConstantObjectPRBenchEnv):
    """Table 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricShelf3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "shelf"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A 3D environment where the goal is to pick up a cube from the ground and place it on a shelf."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of cubes differs between environment variants. For example, Shelf3D-o1 has 1 cube, while Shelf3D-o10 has 10 cubes."

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observations consist of:
- **robot**: The pose of the robot.
- **shelf**: The pose of the shelf.
- **cubes**: The poses of the cubes.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-0.01) per timestep to encourage exploration."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment."""
