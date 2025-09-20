"""A learning approach for skeleton generation inspired by collaborative filtering."""

from itertools import islice
from typing import Callable, Iterator, TypeAlias, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Goal,
    Plan,
    PlanningProblem,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from relational_structs import GroundOperator

from alphatamp.approaches.base_approach import BaseApproach

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]


class CollaborativeFilteringApproach(BaseApproach[_O, _X, _U]):
    """A learning approach for skeleton gen inspired by collaborative filtering."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        skeleton_batch_size: int = 100,
        num_training_skeletons_per_problem: int = 10,
        training_planning_timeout: float = 5,
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._skeleton_batch_size = skeleton_batch_size
        self._num_training_skeletons_per_problem = num_training_skeletons_per_problem
        self._training_planning_timeout = training_planning_timeout

        # Create the planning components.

        # Create the sampler.
        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the abstract plan generator.
        self._base_abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
            )
        )
        self._batched_abstract_plan_generator: AbstractPlanGenerator = (
            BatchRankingAbstractPlanGenerator(
                self._base_abstract_plan_generator,
                score_fn=self._score_skeleton,
                batch_size=self._skeleton_batch_size,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        self._planner = SesamePlanner(
            self._batched_abstract_plan_generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

        # Use the same refiner at training time that we will use at test time. Do this
        # by stealing the refiner from the planner.
        self._refiner = self._planner._refiner  # pylint: disable=protected-access

        # Store data.
        self._data: list[dict[FrozenSkeleton, bool]] = []

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        # Collect data for problem by generating a certain number of training
        # skeletons and attempting to refine each one. We could parallelize this but
        # I'm not sure if it would super help.
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        problem_data: dict[FrozenSkeleton, bool] = {}

        for skeleton in self._base_abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            self._training_planning_timeout,
            bpg,
        ):

            # Uncomment to debug.
            # print("Training on skeleton")
            # for a in skeleton[1]:
            #     print(a.short_str)

            plan = self._refiner(
                x0, skeleton[0], skeleton[1], self._training_planning_timeout, bpg
            )
            label = plan is not None
            frozen_skeleton = (tuple(skeleton[0]), tuple(skeleton[1]))
            problem_data[frozen_skeleton] = label

        self._data.append(problem_data)

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:

        # Run the planner.
        plan, _ = self._planner.run(problem, timeout=timeout)
        if plan is None:
            raise TimeoutError("No plan found")

        return plan

    def _score_skeleton(
        self, skeleton: Skeleton, failed_skeletons: list[Skeleton]
    ) -> float:
        """Score skeletons; higher is better."""

        # This should be rewritten completely very soon.
        total_sim = 0.0
        total_sim_pos = 0.0

        frozen_skeleton = (tuple(skeleton[0]), tuple(skeleton[1]))
        for problem_data in self._data:
            problem_sim = 0
            if frozen_skeleton not in problem_data:
                continue
            for failed_skeleton in failed_skeletons:
                failed_frozen_skeleton = (
                    tuple(failed_skeleton[0]),
                    tuple(failed_skeleton[1]),
                )
                if failed_frozen_skeleton in problem_data:
                    if not problem_data[failed_frozen_skeleton]:
                        problem_sim += 1
            total_sim += problem_sim
            if problem_data[frozen_skeleton]:
                total_sim_pos += 1

        # No data so score these low.
        if total_sim == 0.0:
            return -float("inf")

        score = total_sim_pos / total_sim

        return score


class BatchRankingAbstractPlanGenerator(
    AbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """Generates batches of abstract plans and then ranks them using a score function,
    where higher scores are considered better."""

    def __init__(
        self,
        base_generator: AbstractPlanGenerator[
            _X, RelationalAbstractState, GroundOperator
        ],
        score_fn: Callable[[Skeleton, list[Skeleton]], float],
        batch_size: int,
        seed: int,
    ) -> None:
        self._base_generator = base_generator
        self._score_fn = score_fn
        self._batch_size = batch_size
        # In the future, make this public or find another workaround.
        abstract_successor_fn = (
            self._base_generator._abstract_successor_function  # pylint: disable=protected-access
        )
        super().__init__(abstract_successor_fn, seed)

    def __call__(
        self,
        x0: _X,
        s0: RelationalAbstractState,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, RelationalAbstractState, GroundOperator],
    ) -> Iterator[Skeleton]:

        # This should be refactored soon.
        iterator = self._base_generator(x0, s0, goal, timeout, bpg)
        prev: list[Skeleton] = []
        while batch := list(islice(iterator, self._batch_size)):
            # NOTE: we need to reorder after every failed attempt because of prev.
            while batch:
                tiebreaking_score_fn = lambda x: (
                    self._score_fn(x, prev),
                    -len(x[1]),
                    self._rng.uniform(),
                )
                batch.sort(key=tiebreaking_score_fn)
                skeleton = batch.pop()

                # Uncomment to debug.
                # print("YIELDING")
                # for a in skeleton[1]:
                #     print(a.short_str)
                # print()

                yield skeleton

                # NOTE: assuming that every previous skeleton failed.
                prev.append(skeleton)
