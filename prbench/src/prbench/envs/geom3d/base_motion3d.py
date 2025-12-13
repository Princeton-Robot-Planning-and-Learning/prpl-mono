"""Environment where only base motion is required to reach some goal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

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


@dataclass(frozen=True)
class BaseMotion3DEnvConfig(Geom3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for BaseMotion3DEnv()."""

    # Robot.
    robot_name: str = "tidybot-kinova"
    check_base_collisions: bool = True

    # Target.
    target_radius: float = 0.01
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: SE2Pose = SE2Pose(-1, 1, -np.pi)
    target_upper_bound: SE2Pose = SE2Pose(-1, 1, np.pi)


class BaseMotion3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the BaseMotion3DEnv().

    Adds convenience methods on top of Geom3DObjectCentricState().
    """

    @property
    def target_base_pose(self) -> SE2Pose:
        """The pose of the base target, assuming the name "target"."""
        target = self.get_object_from_name("target")
        pose = Pose(
            (self.get(target, "x"), self.get(target, "y"), self.get(target, "z"))
        )
        se2_pose = pose.to_se2()
        return se2_pose


class ObjectCentricBaseMotion3DEnv(
    ObjectCentricGeom3DRobotEnv[BaseMotion3DObjectCentricState, BaseMotion3DEnvConfig]
):
    """Environment where only base motion planning is needed to reach a goal."""

    def __init__(
        self, config: BaseMotion3DEnvConfig = BaseMotion3DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)

        # Create target.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.config.target_radius,
            rgbaColor=self.config.target_color,
            physicsClientId=self.physics_client_id,
        )

        # Create the body.
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=(0, 0, 0),  # set in reset()
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return BaseMotion3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # Neither the target nor the robot are constant in this env.
        return {}

    def _reset_objects(self) -> None:
        # Reset the target. Sample and check that the robot has not already reachd it.
        target_pose: SE2Pose | None = None
        lb = self.config.target_lower_bound
        ub = self.config.target_upper_bound
        for _ in range(100_000):
            x, y, rot = self.np_random.uniform(
                (lb.x, lb.y, lb.rot), (ub.x, ub.y, ub.rot)
            )
            target_pose = SE2Pose(x, y, rot)
            # If the goal is already reached, keep sampling.
            if not self._goal_reached():
                break
        if target_pose is None:
            raise RuntimeError("Failed to find reachable target position")
        target_se3_pose = target_pose.to_se3(0.0)
        set_pose(self.target_id, target_se3_pose, self.physics_client_id)

    def _set_object_states(self, obs: BaseMotion3DObjectCentricState) -> None:
        assert self.target_id is not None
        target_se3_pose = obs.target_base_pose.to_se3(0.0)
        set_pose(self.target_id, target_se3_pose, self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "target":
            return self.target_id
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return set()

    def _get_movable_object_names(self) -> set[str]:
        return set()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        raise NotImplementedError("No objects have half extents")

    def _get_obs(self) -> BaseMotion3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType), ("target", Geom3DPointType)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=BaseMotion3DObjectCentricState
        )
        assert isinstance(state, BaseMotion3DObjectCentricState)
        return state

    def _goal_reached(self) -> bool:
        target_se3_pose = get_pose(self.target_id, self.physics_client_id)
        target_pose = target_se3_pose.to_se2()
        robot_base_pose = self.robot.base.get_pose()
        dist = float(
            np.linalg.norm(
                np.array(
                    [
                        target_pose.x - robot_base_pose.x,
                        target_pose.y - robot_base_pose.y,
                    ]
                )
            )
        )
        return dist < self.config.target_radius


class BaseMotion3DEnv(ConstantObjectPRBenchEnv):
    """Base motion 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricBaseMotion3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", "target"]

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return (
            """Environment where only base motion planning is needed to reach a goal."""
        )

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        return """Observations consist of:
- **base_pose**: The pose of the base.
- **joint_positions**: The joint positions of the robot arm.
- **finger_state**: The state of the fingers.
- **target**: The pose of the target.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-0.01) per timestep to encourage exploration."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """This is a very common kind of environment."""
