"""Reward functions for TidyBot tasks."""

from typing import Any


class TidyBotRewardCalculator:
    """Base class for TidyBot task rewards."""

    completed_objects: set[int]
    episode_step: int

    def __init__(self, scene_type: str, num_objects: int):
        self.scene_type = scene_type
        self.num_objects = num_objects
        self.completed_objects = set()
        self.episode_step = 0

    def calculate_reward(self, obs: dict[str, Any]) -> float:
        """Calculate reward based on current observation."""
        self.episode_step += 1
        base_reward = -0.01  # Small negative reward per timestep

        # Add task-specific rewards
        task_reward = self._calculate_task_reward(obs)

        return base_reward + task_reward

    def _calculate_task_reward(self, obs: dict[str, Any]) -> float:
        """Calculate task-specific reward.

        Override in subclasses.
        """
        _ = obs  # Unused in base class, overridden in subclasses
        return 0

    def is_terminated(self, obs: dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        return self._is_task_completed(obs)

    def _is_task_completed(self, obs: dict[str, Any]) -> bool:
        """Check if task is completed.

        Override in subclasses.
        """
        _ = obs  # Unused in base class, overridden in subclasses
        return False


def create_reward_calculator(
    scene_type: str, num_objects: int
) -> TidyBotRewardCalculator:
    """Factory function to create appropriate reward calculator."""
    return TidyBotRewardCalculator(scene_type, num_objects)
