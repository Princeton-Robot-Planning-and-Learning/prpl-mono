"""Convenience for running the SESAME bilevel planner from ``SesameModels``.

Constructing a SESAME planner means wiring up a trajectory sampler, an abstract
plan generator, an abstract successor function, and the planner itself -- all
derived from the same ``SesameModels``. ``run_sesame`` does that wiring for you:
pass the env models and an initial state, get back the plan and the bilevel
planning graph.
"""

from typing import Hashable, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Plan, PlanningProblem, SesameModels
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

_O = TypeVar("_O", bound=Hashable)
_X = TypeVar("_X", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


def run_sesame(
    env_models: SesameModels[_O, _X, _U],
    initial_state: _X,
    *,
    seed: int = 0,
    max_abstract_plans: int = 1,
    samples_per_step: int = 5,
    max_skill_horizon: int = 100,
    heuristic_name: str = "hff",
    timeout: float = 30.0,
) -> tuple[Plan[_X, _U] | None, BilevelPlanningGraph]:
    """Run the SESAME planner on ``initial_state`` using ``env_models``.

    The goal is derived from ``env_models.goal_deriver(initial_state)``. Returns
    ``(plan, bpg)`` where ``plan`` is ``None`` if no plan was found and ``bpg`` is
    the bilevel planning graph (useful for visualization/analysis).
    """
    problem = PlanningProblem(
        env_models.state_space,
        env_models.action_space,
        initial_state,
        env_models.transition_fn,
        env_models.goal_deriver(initial_state),
    )
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(env_models.skills),
        transition_function=env_models.transition_fn,
        state_abstractor=env_models.state_abstractor,
        max_trajectory_steps=max_skill_horizon,
    )
    abstract_plan_generator: AbstractPlanGenerator = (
        RelationalHeuristicSearchAbstractPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            heuristic_name,
            seed=seed,
            precomputed_ground_operators=env_models.ground_operators,
        )
    )
    abstract_successor_fn = RelationalAbstractSuccessorGenerator(
        env_models.operators,
        precomputed_ground_operators=env_models.ground_operators,
    )
    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        max_abstract_plans,
        samples_per_step,
        abstract_successor_fn,
        env_models.state_abstractor,
        seed=seed,
    )
    return planner.run(problem, timeout=timeout)
