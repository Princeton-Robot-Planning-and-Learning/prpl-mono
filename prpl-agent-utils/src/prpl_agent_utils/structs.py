"""Data structures."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentResponse:
    """A response from one agent query.

    The text is the agent's final message. Any files the agent created or modified live
    in the agent's sandbox directory, which persists between queries.
    """

    text: str
    metadata: dict[str, Any]  # cost, tokens, turns, etc.
