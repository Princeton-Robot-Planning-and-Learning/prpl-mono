"""Gymnasium environment for the real TidyBot++."""

from typing import Any, SupportsFloat

import gymnasium
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
        return obs, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Coming soon!
        return None
