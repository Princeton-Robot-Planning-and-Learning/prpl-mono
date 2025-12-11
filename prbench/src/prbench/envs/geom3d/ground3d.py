"""PyBullet environment where an object must be picked from the ground.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block
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
class Ground3DEnvConfig(Geom3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Ground3DEnv()."""

    # World bounds.
    x_lb: float = -1
    x_ub: float = 1
    y_lb: float = -1
    y_ub: float = 1

    # Blocks.
    block_size: float = 0.02  # cubes (height = width = length)
    block_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)


class Ground3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the GroundMotion3DEnv().

    Adds convenience methods on top of Geom3DObjectCentricState().
    """


class ObjectCentricGround3DEnv(
    ObjectCentricGeom3DRobotEnv[Geom3DObjectCentricState, Ground3DEnvConfig]
):
    """PyBullet environment where an object must be picked from the ground.

    There may be other obstructing objects in the environment.
    """

    def __init__(
        self,
        num_cubes: int = 2,
        config: Ground3DEnvConfig = Ground3DEnvConfig(),
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
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                ),
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return Ground3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # No constant objects.
        return {}

    def _reset_objects(self) -> None:
        # Randomly sample collision-free positions for the cubes.
        # Also ensure that they are not in collision with the robot.
        # Samples the poses of the cubes
        for _ in range(100_000):
            for cube_name, cube_id in self._cubes.items():
                # add orientation later
                cube_pose = Pose(
                    (
                        np.random.uniform(self.config.x_lb, self.config.x_ub),
                        np.random.uniform(self.config.y_lb, self.config.y_ub),
                        self.config.block_size / 2,
                    )
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

            for cube_name, cube_id in self._cubes.items():
                if check_body_collisions(
                    cube_id,
                    self.robot.base.robot_id,
                    self.physics_client_id,
                ):
                    collision_free = False
                    break
            if collision_free:
                break

        else:
            raise RuntimeError("Failed to sample collision-free cube poses")

    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
        assert isinstance(obs, Ground3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return set()

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return (
                self.config.block_size / 2,
                self.config.block_size / 2,
                self.config.block_size / 2,
            )
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Ground3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType)]
            + [("cube" + str(i), Geom3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Ground3DObjectCentricState
        )
        assert isinstance(state, Ground3DObjectCentricState)
        return state

    def _goal_reached(self) -> bool:
        return False


class Ground3DEnv(ConstantObjectPRBenchEnv):
    """Ground 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricGround3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        return (
            """A 3D environment where the goal is to pick up a cube from the ground."""
        )

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observations consist of:
- **robot**: The pose of the robot.
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
