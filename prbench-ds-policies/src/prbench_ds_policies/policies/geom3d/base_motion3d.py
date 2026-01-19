"""Domain-specific policy for BaseMotion3D environment.

This policy implements a simple proportional controller that moves the robot base toward
the target position.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from prbench.envs.geom3d.base_motion3d import BaseMotion3DObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

__all__ = ["create_domain_specific_policy"]

Policy = Callable[[NDArray], NDArray]


def create_domain_specific_policy(
    observation_space: ObjectCentricBoxSpace,
    max_action_magnitude: float = 0.05,
    position_gain: float = 1.0,
    action_space=None,  # pylint: disable=unused-argument
) -> Policy:
    """Create a domain-specific policy for BaseMotion3D.

    The policy uses a simple proportional controller that computes the delta
    from the current robot base pose to the target pose and clips it to the
    maximum action magnitude.

    Args:
        observation_space: The observation space used to devectorize observations.
        max_action_magnitude: Maximum magnitude for base movement actions.
        position_gain: Proportional gain for position control.
        action_space: The action space (unused, for interface consistency).

    Returns:
        A policy function that maps observations to actions.
    """
    del action_space  # Unused in this policy.

    def policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute action to move robot base toward target.

        Args:
            observation: Vectorized observation from the environment.

        Returns:
            Action array with base movement delta and zeros for arm/gripper.
        """
        # Devectorize the observation to get the object-centric state.
        oc_obs = observation_space.devectorize(observation)
        state = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        # Get current robot base pose.
        base_pose = state.base_pose

        # Get target pose.
        target_pose = state.target_base_pose

        # Compute delta to target.
        delta_x = target_pose.x - base_pose.x
        delta_y = target_pose.y - base_pose.y

        # Apply gain and clip to max magnitude.
        delta_x = np.clip(
            delta_x * position_gain, -max_action_magnitude, max_action_magnitude
        )
        delta_y = np.clip(
            delta_y * position_gain, -max_action_magnitude, max_action_magnitude
        )

        # For rotation, we don't need to control it for this simple task
        # since the goal only checks position distance.
        delta_rot = 0.0

        # Construct action: [base_x, base_y, base_rot, joints*7, gripper]
        # Total 11 elements: 3 for base, 7 for arm joints, 1 for gripper
        action = np.zeros(11, dtype=np.float32)
        action[0] = delta_x
        action[1] = delta_y
        action[2] = delta_rot

        return action

    return policy
