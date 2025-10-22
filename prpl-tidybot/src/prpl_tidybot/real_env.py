"""Gymnasium environment for the real TidyBot++."""

from typing import Any, SupportsFloat

import gymnasium
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
        return obs, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Coming soon!
        return None
