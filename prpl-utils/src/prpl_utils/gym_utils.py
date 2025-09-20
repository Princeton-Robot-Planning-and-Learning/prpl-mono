"""Utilities for Gym/Gymnasium environment compatibility and wrappers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium import spaces


class GymToGymnasium:
    """Wrap a legacy Gym environment to present the Gymnasium API.

    - Maps `reset()` → `(obs, info)`; seeds via `env.seed(seed)` if available.
    - Maps `step()` → `(obs, reward, terminated, truncated, info)`; derives
      `truncated` from `info.get('TimeLimit.truncated', False)`.
    - Forwards `observation_space`, `action_space`, `spec`, `render()`, `close()`.
    """

    def __init__(self, base_env: Any) -> None:
        """Wrap a legacy gym environment."""
        self._env = base_env
        self.observation_space: spaces.Space | None = getattr(
            base_env, "observation_space", None
        )
        self.action_space: spaces.Space | None = getattr(base_env, "action_space", None)
        self.spec = getattr(base_env, "spec", None)

    @property
    def unwrapped(self) -> Any:
        """Get the underlying unwrapped environment."""
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset environment and return (obs, info)."""
        if seed is not None:
            if hasattr(self._env, "seed") and callable(self._env.seed):
                try:
                    self._env.seed(seed)
                except (AttributeError, TypeError):
                    # Legacy env has broken seed implementation, continue without seeding
                    _ = None
        res = self._env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            return res  # already (obs, info)
        return res, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step environment and return (obs, reward, terminated, truncated,
        info)."""
        result = self._env.step(action)
        if isinstance(result, tuple):
            if len(result) == 4:
                obs, reward, done, info = result
                terminated = bool(done)
                truncated = bool(info.get("TimeLimit.truncated", False))
                return obs, float(reward), terminated, truncated, info
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return obs, float(reward), bool(terminated), bool(truncated), info
        raise ValueError("Unexpected number of values returned from env.step")

    def render(self) -> Any:
        """Render the environment."""
        if hasattr(self._env, "render"):
            try:
                return self._env.render()
            except TypeError:
                return self._env.render(mode="human")
        return None

    def close(self) -> None:
        """Close the environment."""
        getattr(self._env, "close", lambda: None)()


def patch_box_float32() -> Callable[..., None]:
    """Resolves Gymnasium Box precision warnings by patching space creation to
    use float32 dtypes from the start, eliminating the need for runtime casting
    that triggers "precision lowered" warnings from environment libraries like
    PRBench (PRBench creates box spaces with float64 bounds).

    Returns:
        The original Box.__init__ method for restoration.
    """

    original_box_init = spaces.Box.__init__

    def patched_box_init(self, low, high, shape=None, dtype=np.float32, seed=None):
        # Convert bounds to float32 if they're float64 to avoid warnings
        if hasattr(low, "dtype") and low.dtype == np.float64:
            low = low.astype(np.float32)
        if hasattr(high, "dtype") and high.dtype == np.float64:
            high = high.astype(np.float32)

        # Force dtype to float32 for floating point types
        if dtype == np.float64:
            dtype = np.float32

        return original_box_init(self, low, high, shape, dtype, seed)

    spaces.Box.__init__ = patched_box_init  # type: ignore
    return original_box_init


__all__ = ["GymToGymnasium", "patch_box_float32"]
