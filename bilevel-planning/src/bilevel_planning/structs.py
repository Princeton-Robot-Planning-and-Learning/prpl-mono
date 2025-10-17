"""Common data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np
from gymnasium.spaces import Space, Box
from prpl_utils.utils import consistent_hash
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class Goal(abc.ABC, Generic[_X]):
    """Can be checked in states and optionally in abstract states."""

    @abc.abstractmethod
    def check_state(self, state: _X) -> bool:
        """Check if the goal holds in the state."""

    def check_abstract_state(self, abstract_state: Any) -> bool:
        """Optionally check if the goal holds in the abstract state."""
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionalGoal(Goal[_X]):
    """Goal defined with given functions."""

    check_state_fn: Callable[[_X], bool]
    check_abstract_state_fn: Callable[[Any], bool] | None = None

    def check_state(self, state: _X) -> bool:
        return self.check_state_fn(state)

    def check_abstract_state(self, abstract_state: Any) -> bool:
        assert self.check_abstract_state_fn is not None
        return self.check_abstract_state_fn(abstract_state)


@dataclass(frozen=True)
class PlanningProblem(Generic[_X, _U]):
    """A deterministic, fully-observed, goal-based planning problem."""

    state_space: Space[_X]
    action_space: Space[_U]
    initial_state: _X
    transition_function: Callable[[_X, _U], _X]
    goal: Goal[_X]


@dataclass(frozen=True)
class Plan(Generic[_X, _U]):
    """Stores a trajectory of states and actions."""

    states: list[_X]
    actions: list[_U]

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions) + 1


@dataclass(frozen=True)
class RelationalAbstractState:
    """An abstract state in a relational problem."""

    atoms: set[GroundAtom]
    objects: set[Object]

    @cached_property
    def _hash(self) -> int:
        return consistent_hash((frozenset(self.atoms), frozenset(self.objects)))

    def __hash__(self) -> int:
        return self._hash


@dataclass(frozen=True)
class RelationalAbstractGoal(Goal[_X]):
    """A goal defined with reference to goal atoms."""

    atoms: set[GroundAtom]
    state_abstractor: Callable[[_X], Any]

    def check_state(self, state: _X) -> bool:
        return self.check_abstract_state(self.state_abstractor(state))

    def check_abstract_state(self, abstract_state: Any) -> bool:
        assert isinstance(abstract_state, RelationalAbstractState)
        return self.atoms.issubset(abstract_state.atoms)


class ParameterizedController(abc.ABC, Generic[_X, _U]):
    """A parameterized policy, a parameter sampler, and a termination check."""

    @abc.abstractmethod
    def sample_parameters(self, x: _X, rng: np.random.Generator) -> Any:
        """Sample parameters."""

    @abc.abstractmethod
    def reset(self, x: _X, params: Any) -> None:
        """Reset the internal state and current parameters."""

    @abc.abstractmethod
    def terminated(self) -> bool:
        """Check if the controller has terminated."""

    @abc.abstractmethod
    def step(self) -> _U:
        """Return the next action to execute."""

    @abc.abstractmethod
    def observe(self, x: _X) -> None:
        """Observe the current state."""


@dataclass(frozen=True)
class LiftedParameterizedController(Generic[_X, _U]):
    """A parameterized controller factory with placeholders for objects."""

    variables: Sequence[Variable]
    controller_cls: type[GroundParameterizedController]
    params_space: Box | None = None

    def ground(
        self, objects: Sequence[Object]
    ) -> GroundParameterizedController[_X, _U]:
        """Create a ground parameterized controller."""
        assert all(
            o.is_instance(v.type) for o, v in zip(objects, self.variables, strict=True)
        )
        return self.controller_cls(objects)

    @property
    def name(self) -> str:
        """Get the name of the controller class."""
        return self.controller_cls.__name__

    @property
    def types(self) -> Sequence[Type]:
        """Get the types of the variables."""
        return [v.type for v in self.variables]

    @property
    def name_vars_str(self) -> str:
        """Get a string representation of the variable names."""
        return f"{self.controller_cls.__name__}{self.var_str}"

    @property
    def var_str(self) -> str:
        """Get a string representation of the variable types."""
        result = "(types=[" + ", ".join(
            v.type.name for v in self.variables
        ) + "])"
        if self.params_space is not None:
            result += ", params_space=" + str(self.params_space)
        return result

class GroundParameterizedController(ParameterizedController[_X, _U], abc.ABC):
    """A parameterized controller that is object-parameterized.

    Subclasses determine how the objects should be used.
    """

    def __init__(self, objects: Sequence[Object]) -> None:
        self.objects = objects


@dataclass(frozen=True)
class LiftedSkill(Generic[_X, _U]):
    """An operator and controller that share object parameters."""

    operator: LiftedOperator
    controller: LiftedParameterizedController[_X, _U]

    def __post_init__(self) -> None:
        assert tuple(self.operator.parameters) == tuple(self.controller.variables)

    @property
    def parameters(self) -> Sequence[Variable]:
        """Get the shared object placeholders."""
        return self.operator.parameters

    def ground(self, objects: Sequence[Object]) -> GroundSkill[_X, _U]:
        """Create a GroundSkill()."""
        ground_operator = self.operator.ground(objects)
        ground_controller = self.controller.ground(objects)
        return GroundSkill(self, ground_operator, ground_controller)

    @cached_property
    def _hash(self) -> int:
        return consistent_hash((self.operator, self.controller))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, LiftedSkill)
        return self.operator == other.operator and self.controller == other.controller


@dataclass(frozen=True)
class GroundSkill(Generic[_X, _U]):
    """A ground operator and controller that share object parameters."""

    parent: LiftedSkill
    operator: GroundOperator
    controller: GroundParameterizedController[_X, _U]

    def __post_init__(self) -> None:
        assert tuple(self.operator.parameters) == tuple(self.controller.objects)


_O = TypeVar("_O")  # "raw" environment observations, possibly different from states


@dataclass(frozen=True)
class SesameModels(Generic[_O, _X, _U]):
    """Container for common models used in SeSamE (with relational abstractions)."""

    observation_space: Space[_O]
    state_space: Space[_X]
    action_space: Space[_U]
    transition_fn: Callable[[_X, _U], _X]
    types: set[Type]
    predicates: set[Predicate]
    observation_to_state: Callable[[_O], _X]
    state_abstractor: Callable[[_X], RelationalAbstractState]
    goal_deriver: Callable[[_X], RelationalAbstractGoal]
    skills: set[LiftedSkill]

    @property
    def operators(self) -> set[LiftedOperator]:
        """Access the lifted operators from the lifted skills."""
        return {s.operator for s in self.skills}


class TransitionFailure(BaseException):
    """May be raised by transition functions."""
