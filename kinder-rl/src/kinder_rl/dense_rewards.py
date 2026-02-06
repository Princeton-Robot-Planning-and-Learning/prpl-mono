"""Environment-specific dense reward wrappers for sparse reward environments.

To add dense rewards for a new environment:
1. Create a new class inheriting from BaseDenseRewardWrapper
2. Implement _compute_dense_reward(obs, terminated) -> float
3. Register it in ENV_DENSE_REWARD_WRAPPERS dict

Example:
    class MyEnvDenseReward(BaseDenseRewardWrapper):
        def _compute_dense_reward(self, obs, terminated):
            if terminated:
                return 10.0  # goal bonus
            state = self.env.observation_space.devectorize(obs)
            # compute distance to goal...
            return -distance
"""

from typing import Any

import gymnasium as gym
import numpy as np


class BaseDenseRewardWrapper(gym.Wrapper):
    """Base class for environment-specific dense reward wrappers.

    Subclasses must implement _compute_dense_reward().
    """

    def __init__(
        self,
        env: gym.Env,
        reward_scale: float = 1.0,
    ):
        """Initialize dense reward wrapper.

        Args:
            env: The environment to wrap.
            reward_scale: Scale factor for dense reward.
        """
        super().__init__(env)
        self.reward_scale = reward_scale

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step environment and add dense reward."""
        obs, sparse_reward, terminated, truncated, info = self.env.step(action)

        dense_reward = self._compute_dense_reward(obs, terminated)

        # Store both rewards in info
        info["sparse_reward"] = sparse_reward
        info["dense_reward"] = dense_reward

        total_reward = float(sparse_reward) + self.reward_scale * dense_reward
        return obs, total_reward, terminated, truncated, info

    def _compute_dense_reward(self, obs: Any, terminated: bool) -> float:
        """Compute dense reward.

        Must be implemented by subclass.
        """
        raise NotImplementedError


class BaseMotion3DDenseReward(BaseDenseRewardWrapper):
    """Dense reward for BaseMotion3D: negative distance from robot to target."""

    def __init__(
        self, env: gym.Env, reward_scale: float = 0.1, goal_bonus: float = 10.0
    ):
        super().__init__(env, reward_scale)
        self.goal_bonus = goal_bonus
        self.curr_dist = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.curr_dist = self.compute_distance(obs)

        return obs, info

    def compute_distance(self, obs: Any) -> float:
        """Compute distance from robot to target."""
        state = self.env.observation_space.devectorize(obs)  # type: ignore[attr-defined]
        obj_map = {o.name: o for o in state.data.keys()}

        robot = obj_map.get("robot")
        target = obj_map.get("target")
        if robot is None or target is None:
            return float("inf")

        # Robot base position
        robot_x = state.get(robot, "pos_base_x")
        robot_y = state.get(robot, "pos_base_y")

        # Target position
        target_x = state.get(target, "x")
        target_y = state.get(target, "y")

        distance = np.sqrt((robot_x - target_x) ** 2 + (robot_y - target_y) ** 2)
        return distance

    def _compute_dense_reward(self, obs: Any, terminated: bool) -> float:
        if terminated:
            return self.goal_bonus

        # Get object-centric state
        current_distance = self.compute_distance(obs)

        # 2. Distance Shaping (The "Guide")
        # Formula: (Old - New)
        # If we get closer, (Old > New), result is Positive.
        raw_shaping = self.curr_dist - current_distance

        # Scale this up!
        # Since your world is 0.1 units wide, a step might be 0.001.
        # Multiply by 100 or 1000 so the gradient is felt by the network.
        shaping_reward = raw_shaping * 100.0

        # 3. Time Penalty (The "Clock")
        # Forces the agent to not loiter.
        # Must be small enough that moving closer (shaping) > penalty.
        time_penalty = -0.1

        # Update state
        self.curr_dist = current_distance

        return shaping_reward + time_penalty


# Registry: env_id prefix -> wrapper class
ENV_DENSE_REWARD_WRAPPERS: dict[str, type[BaseDenseRewardWrapper]] = {
    "kinder/BaseMotion3D": BaseMotion3DDenseReward,
    "kinder/Motion3D": BaseMotion3DDenseReward,
}


def wrap_with_dense_reward(
    env: gym.Env,
    env_id: str,
    reward_scale: float = 0.1,
    **kwargs: Any,
) -> gym.Env:
    """Wrap environment with dense reward if available.

    Args:
        env: Environment to wrap.
        env_id: Environment ID (e.g., "kinder/BaseMotion3D-v0").
        reward_scale: Scale factor for dense reward.
        **kwargs: Additional args passed to wrapper.

    Returns:
        Wrapped environment.

    Raises:
        NotImplementedError: If dense reward not implemented for this env.
    """
    # Find matching wrapper by prefix
    for prefix, wrapper_cls in ENV_DENSE_REWARD_WRAPPERS.items():
        if env_id.startswith(prefix):
            return wrapper_cls(env, reward_scale=reward_scale, **kwargs)

    raise NotImplementedError(
        f"Dense reward not implemented for '{env_id}'. "
        f"Available: {list(ENV_DENSE_REWARD_WRAPPERS.keys())}"
    )
