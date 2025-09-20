"""Tests for gym_utils."""

from typing import Any

import numpy as np
from gymnasium import spaces

from prpl_utils.gym_utils import GymToGymnasium, patch_box_float32


# Mock legacy gym environment since the old 'gym' package is not a dependency
class MockLegacyGymEnv:
    """Mock legacy gym environment for testing."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = spaces.Discrete(2)
        self.spec = None

    def reset(self) -> np.ndarray:
        """Legacy reset that returns only observation."""
        return np.array([0.1, 0.2, 0.3, 0.4])

    def step(self, _action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Legacy step that returns 4-tuple."""
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        reward = 1.0
        done = False
        info: dict[str, Any] = {}
        return obs, reward, done, info

    def seed(self, _seed: int) -> None:
        """Legacy seed method."""

    def render(self, _mode: str = "human") -> None:
        """Legacy render method."""
        return None


def test_adapter_step_and_reset() -> None:
    """Test GymToGymnasium step and reset outputs."""
    env = MockLegacyGymEnv()
    wrapped = GymToGymnasium(env)
    obs, info = wrapped.reset()
    assert obs is not None
    assert isinstance(info, dict)
    obs, reward, terminated, truncated, info = wrapped.step(0)  # action 0
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_patch_box_float32() -> None:
    """Test patch_box_float32 changes Box dtype."""
    original_box_init = patch_box_float32()
    box = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
    assert box.dtype == np.float32
    spaces.Box.__init__ = original_box_init  # type: ignore
