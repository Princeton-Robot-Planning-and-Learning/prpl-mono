"""A general interface for an agent that runs bilevel planning."""

from typing import Any, Hashable, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import (
    PlanningProblem,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from prpl_utils.gym_agent import Agent

_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class AgentFailure(BaseException):
    """Raised when the agent fails to find a plan."""


class BilevelPlanningAgent(Agent[_O, _U]):
    """A general interface for an agent that runs bilevel planning."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        planning_timeout: float = 30.0,
    ) -> None:
        self._env_models = env_models
        self._current_plan: list[_U] | None = None
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._planning_timeout = planning_timeout
        self._seed = seed
        super().__init__(seed)

    def reset(
        self,
        obs: _O,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        self._current_plan = self._run_planning()

    def _get_action(self) -> _U:
        assert self._current_plan, "Ran out of planning steps, failure!"
        return self._current_plan.pop(0)

    def _run_planning(self) -> list[_U]:
        # Create planning problem.
        initial_state = self._env_models.observation_to_state(self._last_observation)
        goal = self._env_models.goal_deriver(initial_state)
        problem = PlanningProblem(
            self._env_models.state_space,
            self._env_models.action_space,
            initial_state,
            self._env_models.transition_fn,
            goal,
        )

        # Create the sampler.
        trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the abstract plan generator.
        abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        planner = SesamePlanner(
            abstract_plan_generator,
            trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

        # Run the planner.
        plan, _ = planner.run(problem, timeout=self._planning_timeout)
        if plan is None:
            raise AgentFailure("No plan found")

        return plan.actions
