"""Tests for real_env.py."""

import numpy as np
import spatialmath

from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv


def test_real_tidybot_env():
    """Tests for RealTidyBotEnv()."""
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.base_interface.base_state = spatialmath.SE2(x=1.0, y=0.0, theta=0.0)
    env = RealTidyBotEnv(interface)
    obs, _ = env.reset()
    # Compare homogeneous transform matrices for the SE2 poses
    assert np.allclose(obs.base_pose.A, spatialmath.SE2(x=1.0, y=0.0, theta=0.0).A)
    assert np.allclose(obs.arm_conf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
