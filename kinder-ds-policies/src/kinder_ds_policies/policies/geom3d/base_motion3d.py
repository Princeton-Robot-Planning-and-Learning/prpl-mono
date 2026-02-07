"""Domain-specific policy for BaseMotion3D environment.

This policy implements a simple proportional controller that moves the robot base toward
the target position.
"""

import numpy as np
from kinder.envs.kinematic3d.base_motion3d import BaseMotion3DObjectCentricState
from numpy.typing import NDArray
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_ds_policies.policies.base import StatefulPolicy

__all__ = ["create_domain_specific_policy"]


class BaseMotion3DPolicy(StatefulPolicy):
    """A proportional controller policy for BaseMotion3D."""

    def __init__(
        self,
        observation_space: ObjectCentricBoxSpace,
        max_action_magnitude: float = 0.05,
        position_gain: float = 1.0,
    ) -> None:
        self._observation_space = observation_space
        self._max_action_magnitude = max_action_magnitude
        self._position_gain = position_gain

    def reset(self) -> None:
        """Reset the policy state for a new episode."""

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute action to move robot base toward target."""
        oc_obs = self._observation_space.devectorize(observation)
        state = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        base_pose = state.base_pose
        target_pose = state.target_base_pose

        delta_x = target_pose.x - base_pose.x
        delta_y = target_pose.y - base_pose.y

        delta_x = np.clip(
            delta_x * self._position_gain,
            -self._max_action_magnitude,
            self._max_action_magnitude,
        )
        delta_y = np.clip(
            delta_y * self._position_gain,
            -self._max_action_magnitude,
            self._max_action_magnitude,
        )

        delta_rot = 0.0

        # Construct action: [base_x, base_y, base_rot, joints*7, gripper]
        action = np.zeros(11, dtype=np.float32)
        action[0] = delta_x
        action[1] = delta_y
        action[2] = delta_rot

        return action


def create_domain_specific_policy(
    observation_space: ObjectCentricBoxSpace,
    max_action_magnitude: float = 0.05,
    position_gain: float = 1.0,
    action_space=None,  # pylint: disable=unused-argument
) -> StatefulPolicy:
    """Create a domain-specific policy for BaseMotion3D.

    Args:
        observation_space: The observation space used to devectorize observations.
        max_action_magnitude: Maximum magnitude for base movement actions.
        position_gain: Proportional gain for position control.
        action_space: The action space (unused, for interface consistency).

    Returns:
        A policy that maps observations to actions.
    """
    del action_space

    return BaseMotion3DPolicy(
        observation_space=observation_space,
        max_action_magnitude=max_action_magnitude,
        position_gain=position_gain,
    )
