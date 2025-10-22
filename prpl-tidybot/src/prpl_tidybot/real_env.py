"""Gymnasium environment for the real TidyBot++."""

import gymnasium
from gymnasium.core import RenderFrame
from prpl_tidybot.structs import TidyBotObservation, TidyBotAction
from typing import Any, SupportsFloat


class RealTidyBotEnv(gymnasium.Env[TidyBotObservation, TidyBotAction]):
    """Gymnasium environment for the real TidyBot++."""

    def __init__(self) -> None:
        import ipdb; ipdb.set_trace()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TidyBotObservation, dict[str, Any]]:  # type: ignore
        import ipdb; ipdb.set_trace()

    def step(
        self, action: TidyBotAction
    ) -> tuple[TidyBotObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        import ipdb; ipdb.set_trace()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Get the current images.
        import ipdb; ipdb.set_trace()
