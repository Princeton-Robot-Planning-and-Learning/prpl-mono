"""An abstract plan generator that uses heuristic search."""

from __future__ import annotations

import heapq as hq
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Generic, Hashable, Iterable, Iterator, TypeVar

from relational_structs import (
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    Type,
)

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Goal,
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    cached_all_ground_operators,
    create_pyperplan_heuristic,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


@dataclass(frozen=True)
class _Node(Generic[_S, _A]):
    """A node for the search over abstract plans."""

    abstract_state: _S
    last_abstract_action: _A | None
    parent: _Node[_S, _A] | None
    cumulative_cost: float

    @cached_property
    def abstract_action_plan(self) -> tuple[_A, ...]:
        """Return the full abstract action plan from the root."""
        if self.last_abstract_action is None:
            return tuple()
        assert self.parent is not None
        return self.parent.abstract_action_plan + (self.last_abstract_action,)

    @cached_property
    def abstract_state_plan(self) -> tuple[_S, ...]:
        """Return the full abstract state plan from the root."""
        if self.last_abstract_action is None:
            return (self.abstract_state,)
        assert self.parent is not None
        return self.parent.abstract_state_plan + (self.abstract_state,)


class HeuristicSearchAbstractPlanGenerator(AbstractPlanGenerator[_X, _S, _A]):
    """An abstract plan generator that uses heuristic search.

    At the moment, this always runs A*, but it is easy to extend.
    """

    def __init__(
        self,
        heuristic_factory: Callable[[_S, Goal[_X]], Callable[[_S], float]],
        abstract_successor_function: Callable[[_S], Iterable[tuple[_A, _S]]],
        seed: int,
    ):
        super().__init__(abstract_successor_function, seed)
        self._heuristic_factory = heuristic_factory

    def __call__(
        self,
        x0: _X,
        s0: _S,
        goal: Goal[_X],
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> Iterator[tuple[list[_S], list[_A]]]:
        # This is a bit of a weird kind of search because we allow revisiting states.
        # That's important because we need to generate multiple abstract plans. We do,
        # however, need to avoid repeatedly outputting abstract plans.

        start_time = time.perf_counter()
        assert goal.check_abstract_state is not None

        # Create the heuristic.
        heuristic = self._heuristic_factory(s0, goal)

        # Elements are: heuristic, tiebreak, node.
        queue: list[tuple[float, float, _Node]] = []

        # Create the root.
        root: _Node[_S, _A] = _Node(
            s0, last_abstract_action=None, parent=None, cumulative_cost=0.0
        )
        hq.heappush(queue, (heuristic(s0), self._rng.uniform(), root))

        # Avoid revisiting the same abstract plans.
        visited_abstract_plans: set[tuple[_A, ...]] = set()
        visited_abstract_plans.add(root.abstract_action_plan)

        # Start search.
        while queue and (time.perf_counter() - start_time < timeout):
            _, _, node = hq.heappop(queue)

            # If the goal is achieved, yield this abstract plan.
            if goal.check_abstract_state(node.abstract_state):
                yield list(node.abstract_state_plan), list(node.abstract_action_plan)
                # Don't generate successors from goal nodes.
                continue

            # Generate successors.
            for a, ns in self._abstract_successor_function(node.abstract_state):
                # Update the bilevel planning graph.
                bpg.add_abstract_state_node(ns)
                bpg.add_abstract_action_edge(node.abstract_state, a, ns)

                # Check if we have already seen this abstract plan.
                abstract_plan = node.abstract_action_plan + (a,)
                if abstract_plan in visited_abstract_plans:
                    continue

                # This is new, so create a new node. Abstract action costs are unitary.
                child_node: _Node[_S, _A] = _Node(
                    ns,
                    last_abstract_action=a,
                    parent=node,
                    cumulative_cost=node.cumulative_cost + 1.0,
                )

                # Calculate the heuristic.
                priority = child_node.cumulative_cost + heuristic(ns)

                # Update the queue.
                hq.heappush(queue, (priority, self._rng.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break


class RelationalHeuristicSearchAbstractPlanGenerator(
    HeuristicSearchAbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """Uses relational abstractions (PDDL) and heuristic search."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        operators: set[LiftedOperator],
        heuristic_name: str,
        seed: int,
    ) -> None:
        # Cannot create the problem yet because we may want to reuse this heuristic
        # in problems that have different initial states, objects, and goals.
        self._pddl_domain = PDDLDomain("custom-domain", operators, predicates, types)
        self._heuristic_name = heuristic_name
        successor_fn = RelationalAbstractSuccessorGenerator(operators)
        super().__init__(self._relational_heuristic_factory, successor_fn, seed)

    def _relational_heuristic_factory(
        self, init_abstract_state: RelationalAbstractState, goal: Goal[_X]
    ) -> Callable[[RelationalAbstractState], float]:

        assert isinstance(init_abstract_state, RelationalAbstractState)
        assert isinstance(goal, RelationalAbstractGoal)

        # Derive heuristic from relational structs.
        pddl_problem = PDDLProblem(
            "custom-domain",
            "custom-problem",
            init_abstract_state.objects,
            init_abstract_state.atoms,
            goal.atoms,
        )
        ground_operators = cached_all_ground_operators(
            self._pddl_domain.operators, init_abstract_state.objects
        )
        pyperplan_heuristic = create_pyperplan_heuristic(
            "hff", self._pddl_domain, pddl_problem, ground_operators
        )
        heuristic = lambda s: pyperplan_heuristic(s.atoms)

        return heuristic
