"""PRBench environment provider."""

import prbench
from gymnasium import Env, spaces
from omegaconf import DictConfig
from prpl_utils.gym_utils import patch_box_float32


def create_prbench_env(env_config: DictConfig) -> Env:
    """Create PRBench environment with float32 Box spaces."""
    # Register environments with prbench
    prbench.register_all_environments()
    kwargs = env_config["make_kwargs"]
    # Patch Box creation to use float32 by default and avoid warnings
    original_box_init = patch_box_float32()
    try:
        # Create environment with patched Box class
        env = prbench.make(**kwargs)
    finally:
        # Restore original Box.__init__
        spaces.Box.__init__ = original_box_init  # type: ignore

    return env
