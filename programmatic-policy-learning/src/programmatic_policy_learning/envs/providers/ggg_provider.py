"""GGG environment provider."""

from omegaconf import DictConfig
from prpl_utils.gym_utils import GymToGymnasium


def create_ggg_env(env_config: DictConfig) -> GymToGymnasium:
    """Create GGG environment with legacy gym compatibility."""
    # Lazy import, to avoid deprecation warnings at module import time
    import generalization_grid_games  # pylint: disable=unused-import,import-outside-toplevel
    import gym as legacy_gym  # pylint: disable=import-outside-toplevel

    env_id = env_config.make_kwargs.id
    base = legacy_gym.make(env_id, disable_env_checker=True)
    return GymToGymnasium(base)
