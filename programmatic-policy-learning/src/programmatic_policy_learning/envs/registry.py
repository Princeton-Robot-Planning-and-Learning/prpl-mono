"""Environment provider registry."""

from typing import Any, Callable

import gymnasium

from programmatic_policy_learning.envs.providers.ggg_provider import create_ggg_env
from programmatic_policy_learning.envs.providers.prbench_provider import (
    create_prbench_env,
)


class EnvRegistry:
    """Registry for environment providers."""

    def __init__(self) -> None:
        self._providers: dict[str, Callable[[Any], Any]] = {
            "ggg": create_ggg_env,
            "prbench": create_prbench_env,
        }

    def load(self, env_config: Any) -> Any:
        """Load environment from provider or fallback to gymnasium."""
        if "provider" in env_config:
            provider: str = env_config["provider"]
            return self._providers[provider](env_config)
        # Fall back to gymnasium if nothing was found...
        return gymnasium.make(**env_config.make_kwargs)
