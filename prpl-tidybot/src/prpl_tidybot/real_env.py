"""Gymnasium environment for the real TidyBot++."""

from typing import Any, SupportsFloat

import gymnasium
<<<<<<< HEAD
from gymnasium.core import RenderFrame

from prpl_tidybot.interface import Interface
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation


class RealTidyBotEnv(gymnasium.Env[TidyBotObservation, TidyBotAction]):
    """Gymnasium environment for the real TidyBot++."""

    def __init__(self, interface: Interface) -> None:
        self._interface = interface

    def _get_obs(self) -> TidyBotObservation:
        """Get the current real observation."""
        return self._interface.get_observation()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TidyBotObservation, dict[str, Any]]:  # type: ignore
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: TidyBotAction
    ) -> tuple[TidyBotObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Coming soon!
        obs = self._get_obs()
||||||| b8b1fde
=======
import numpy as np
import spatialmath
from gymnasium.core import RenderFrame

from prpl_tidybot.structs import CAMERA_DIMS, TidyBotAction, TidyBotObservation


class RealTidyBotEnv(gymnasium.Env[TidyBotObservation, TidyBotAction]):
    """Gymnasium environment for the real TidyBot++."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TidyBotObservation, dict[str, Any]]:  # type: ignore
        # Coming soon!
        obs = TidyBotObservation(
            arm_conf=[0.0] * 7,
            base_pose=spatialmath.SE2(x=0, y=0, theta=0),
            gripper=0.0,
            wrist_camera=np.zeros(CAMERA_DIMS, dtype=np.uint8),
            base_camera=np.zeros(CAMERA_DIMS, dtype=np.uint8),
        )
        return obs, {}

    def step(
        self, action: TidyBotAction
    ) -> tuple[TidyBotObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Coming soon!
        obs = TidyBotObservation(
            arm_conf=[0.0] * 7,
            base_pose=spatialmath.SE2(x=0, y=0, theta=0),
            gripper=0.0,
            wrist_camera=np.zeros(CAMERA_DIMS, dtype=np.uint8),
            base_camera=np.zeros(CAMERA_DIMS, dtype=np.uint8),
        )
>>>>>>> 233cfb0e77b9ec83ec12173a5cb3b37e94c86545
        return obs, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Coming soon!
        return None
