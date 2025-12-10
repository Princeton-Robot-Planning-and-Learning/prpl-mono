"""PyBullet environment where an object must be picked from the ground.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict
from pybullet_helpers.utils import create_pybullet_block

from prbench.core import ConstantObjectPRBenchEnv, FinalConfigMeta
from prbench.envs.geom3d.base_env import (
    Geom3DEnvConfig,
    ObjectCentricGeom3DRobotEnv,
)
from prbench.envs.geom3d.object_types import (
    Geom3DEnvTypeFeatures,
    Geom3DPointType,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState
from prbench.envs.utils import PURPLE


@dataclass(frozen=True)
class Ground3DEnvConfig(Geom3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Ground3DEnv()."""

    # World bounds.
    x_lb: float = -2.5
    x_ub: float = 2.5
    y_lb: float = -2.5
    y_ub: float = 2.5

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
        config: Ground3DEnvConfig = Ground3DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes

        # Create the cubes, but their poses will be reset (with collision checking) in
        # the reset() method.
        self._cubes: dict[str, int] = {}
        for idx in range(self._num_cubes):
            cube_id = create_pybullet_block(self.config.block_rgba,
                                            (self.config.block_size / 2,
                                             self.config.block_size / 2,
                                             self.config.block_size / 2),
                                             physics_client_id=self.physics_client_id)
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
        import ipdb; ipdb.set_trace()
        while True:
            p.getMouseEvents(self.physics_client_id)

    def _set_object_states(self, obs: Ground3DObjectCentricState) -> None:
        # TODO set the cube states
        import ipdb; ipdb.set_trace()

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        # TODO
        import ipdb; ipdb.set_trace()
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        # TODO
        import ipdb; ipdb.set_trace()
        return set()

    def _get_movable_object_names(self) -> set[str]:
        # TODO
        import ipdb; ipdb.set_trace()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        # TODO
        import ipdb; ipdb.set_trace()

    def _get_obs(self) -> Ground3DObjectCentricState:
        # TODO add cubes
        import ipdb; ipdb.set_trace()
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Ground3DObjectCentricState
        )
        assert isinstance(state, Ground3DObjectCentricState)
        return state

    def _goal_reached(self) -> bool:
        # TODO
        import ipdb; ipdb.set_trace()


class Ground3DEnv(ConstantObjectPRBenchEnv):
    """Ground 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricGround3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        # TODO
        import ipdb; ipdb.set_trace()
        return []

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        # TODO
        return "TODO"

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        # TODO
        return "TODO"

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        # TODO
        return "TODO"

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        # TODO
        return "TODO"
