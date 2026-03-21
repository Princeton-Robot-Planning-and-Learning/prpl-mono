"""A discrete-time finite-horizon trajectory optimization problem."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from prpl_utils.structs import Image

TrajOptState: TypeAlias = NDArray[np.floating]
TrajOptAction: TypeAlias = NDArray[np.floating]


@dataclass
class TrajOptTraj:
    """A trajectory of states and actions."""

    states: NDArray[np.floating]  # array of TrajOptState
    actions: NDArray[np.floating]  # array of TrajOptAction

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions) + 1

    def copy(self) -> TrajOptTraj:
        """Copy the trajectory."""
        return TrajOptTraj(self.states.copy(), self.actions.copy())


class TrajOptProblem(abc.ABC):
    """A discrete-time finite-horizon trajopt problem."""

    @property
    @abc.abstractmethod
    def horizon(self) -> int:
        """The time horizon of the problem."""

    @property
    @abc.abstractmethod
    def state_space(self) -> Box:
        """The vector state space."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Box:
        """The vector action space."""

    @property
    @abc.abstractmethod
    def initial_state(self) -> TrajOptState:
        """The initial state."""

    @abc.abstractmethod
    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        """The discrete-time dynamics: get the next state."""

    @abc.abstractmethod
    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        """Get the cost of a full trajectory."""

    @abc.abstractmethod
    def render_state(self, state: TrajOptState) -> Image:
        """Optional rendering function."""
