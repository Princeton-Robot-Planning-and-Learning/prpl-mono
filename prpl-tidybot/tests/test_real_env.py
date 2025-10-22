"""Tests for real_env.py."""

import numpy as np

from prpl_tidybot.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv


def test_real_tidybot_env():
    """Tests for RealTidyBotEnv()."""
    interface = FakeInterface()
    interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    env = RealTidyBotEnv(interface)
    obs, _ = env.reset()
    assert np.allclose(obs.arm_conf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
