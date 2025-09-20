"""Tests for sesame_planner.py."""

from typing import TypeAlias

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, Tuple
from numpy.typing import NDArray
from relational_structs import (
    LiftedOperator,
    Predicate,
    Type,
)

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    HeuristicSearchAbstractPlanGenerator,
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import (
    FunctionalGoal,
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    ParameterizedController,
    PlanningProblem,
    RelationalAbstractGoal,
    RelationalAbstractState,
    TransitionFailure,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

X: TypeAlias = NDArray[np.floating]  # 2D position of robot
U: TypeAlias = NDArray[np.floating]  # dx, dy


class MockController(ParameterizedController[X, U]):
    """A mock controller for testing.

    Represents a parameterized policy that moves in the direction of the abstract state,
    but with some small random offset.
    """

    def __init__(self, direction: tuple[int, int], offset_mag: float) -> None:
        super().__init__()
        self._direction = direction
        self._offset_mag = offset_mag
        self._current_params: tuple[float, float] = (0.0, 0.0)
        self._current_state: X | None = None
        self._current_action: U | None = None
        self._target_cell: tuple[int, int] | None = None

    def sample_parameters(self, x: X, rng: np.random.Generator) -> tuple[float, float]:
        offset_dx, offset_dy = rng.uniform(-self._offset_mag, self._offset_mag, size=2)
        return (offset_dx, offset_dy)

    def reset(self, x: X, params: tuple[float, float]) -> None:
        self._current_params = params
        self._current_state = x
        offset_dx, offset_dy = params
        dx = self._direction[0] / 2 + offset_dx
        dy = self._direction[1] / 2 + offset_dy
        self._current_action = np.array([dx, dy])
        self._target_cell = (
            round(x[0]) + self._direction[0],
            round(x[1]) + self._direction[1],
        )

    def terminated(self) -> bool:
        if self._current_state is None:
            return False
        # Terminate when we've reached the target.
        current_abstract_state = (
            round(self._current_state[0]),
            round(self._current_state[1]),
        )
        return current_abstract_state == self._target_cell

    def step(self) -> U:
        assert self._current_action is not None
        return self._current_action

    def observe(self, x: X) -> None:
        self._current_state = x


@pytest.mark.parametrize(
    "max_abstract_plans, samples_per_step, dependent_samples, obstacle, expected_out",
    [
        # # Easy mode: no dependent samples and no obstacle.
        (1, 1, False, False, True),
        (1, 10, False, False, True),
        (10, 10, False, False, True),
        # # Dependent samples, so naive planner will fail.
        (1, 1, True, False, False),
        (1, 10, True, False, True),
        (10, 10, True, False, True),
        # # Dependent samples and obsacles, so full sesame is required.
        (1, 1, False, True, False),
        (1, 10, False, True, False),
        (10, 10, False, True, True),
    ],
)
def test_sesame(
    max_abstract_plans, samples_per_step, dependent_samples, obstacle, expected_out
):
    """Test SesamePlanner and vary the need for multiple abstract plans and samples."""

    state_space = Box(0.0, 10.0, shape=(2,))
    action_space = Box(-0.5, 0.5, shape=(2,))
    initial_state = np.array([10.0, 5.0])

    def transition_fn(x, u):
        # Make samples dependent by preventing the accumulation of small y movement.
        if dependent_samples and x[0] < 8 and abs(x[1] - round(x[1])) > 0.2:
            raise TransitionFailure
        # Simulate obstacle.
        if obstacle and ((x[0] - 9.0) ** 2 + (x[1] - 5.0) ** 2 < 1.0):
            raise TransitionFailure
        return x + u

    def goal_check(x):
        return (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2 < 1

    def abstract_goal_check(s):
        return s == (5, 5)

    problem = PlanningProblem(
        state_space,
        action_space,
        initial_state,
        transition_fn,
        FunctionalGoal(goal_check, abstract_goal_check),
    )

    abstract_state_space = Tuple([Discrete(11), Discrete(11)])

    def state_abstractor(x):
        return (round(x[0]), round(x[1]))

    def abstract_successor_function(s):
        for a in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ns = (s[0] + a[0], s[1] + a[1])
            if not abstract_state_space.contains(ns):
                continue
            yield (a, ns)

    controllers = {}
    offset_mag = 0.25 if dependent_samples or obstacle else 0.1
    for abstract_action in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        controller = MockController(abstract_action, offset_mag)
        controllers[abstract_action] = controller

    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=controllers.get,
        transition_function=transition_fn,
        state_abstractor=state_abstractor,
        max_trajectory_steps=5,
    )

    abstract_plan_generator = HeuristicSearchAbstractPlanGenerator(
        heuristic_factory=lambda _1, _2: lambda _: 0,
        abstract_successor_function=abstract_successor_function,
        seed=123,
    )

    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        max_abstract_plans,
        samples_per_step,
        abstract_successor_function,
        state_abstractor,
        seed=123,
    )

    plan, bpg = planner.run(problem, timeout=10)
    success = plan is not None
    assert expected_out == success

    # Uncomment to make GIF.
    # if success:
    #     assert plan is not None
    #     final_state = plan.states[-1]
    # else:
    #     final_state = None
    # bpg.set_state_position_function(lambda x: x)
    # bpg.set_abstract_state_position_function(lambda x: x)
    # outfile = (
    #     f"sesame_{max_abstract_plans}_{samples_per_step}_"
    #     f"{dependent_samples}_{obstacle}.gif"
    # )
    # bpg.render_gif(save_path=outfile, final_state=final_state)

    del bpg


def test_relational_sesame():
    """Test SesamePlanner with relational abstractions."""

    # Hyperparameters for sesame.
    max_abstract_plans = 10
    samples_per_step = 10

    # Continuous planning domain.
    state_space = Box(0.0, 10.0, shape=(2,))
    action_space = Box(-0.5, 0.5, shape=(2,))
    initial_state = np.array([10.0, 5.0])

    def transition_fn(x, u):
        # Simulate obstacle.
        if (x[0] - 9.0) ** 2 + (x[1] - 5.0) ** 2 < 1.0:
            raise TransitionFailure
        return x + u

    # Relational abstractions.
    loc_type = Type("loc")
    at = Predicate("at", [loc_type])
    adjacent = Predicate("adjacent", [loc_type, loc_type])
    objects = {loc_type(f"loc-{x}-{y}") for x in range(11) for y in range(11)}
    loc_to_xy = lambda loc: map(int, loc.name[len("loc-") :].split("-"))

    # Create the abstract actions.
    start = loc_type("?start")
    end = loc_type("?end")
    move_operator = LiftedOperator(
        "move",
        [start, end],
        preconditions={at([start]), adjacent([start, end])},
        add_effects={at([end])},
        delete_effects={at([start])},
    )
    operators = {move_operator}

    # Create the controllers for the abstract actions.
    class GroundMoveController(GroundParameterizedController, MockController):
        """Need to change the inputs of MockController to accept objects."""

        def __init__(self, objects):
            start_loc, end_loc = objects
            start_x, start_y = loc_to_xy(start_loc)
            end_x, end_y = loc_to_xy(end_loc)
            direction = (end_x - start_x, end_y - start_y)
            MockController.__init__(self, direction, offset_mag=0.25)
            GroundParameterizedController.__init__(self, objects)

    move_controller = LiftedParameterizedController(
        [start, end],
        GroundMoveController,
    )

    move_skill = LiftedSkill(move_operator, move_controller)

    # Precompute the static adjacent ground atoms.
    adjacent_ground_atoms = set()
    for x in range(11):
        for y in range(11):
            loc1 = loc_type(f"loc-{x}-{y}")
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nx = x + dx
                ny = y + dy
                if not (0 <= nx <= 10 and 0 <= ny <= 10):
                    continue
                loc2 = loc_type(f"loc-{nx}-{ny}")
                adjacent_ground_atoms.add(adjacent([loc1, loc2]))

    def state_abstractor(state):
        x, y = round(state[0]), round(state[1])
        loc = loc_type(f"loc-{x}-{y}")
        atoms = {at([loc])} | adjacent_ground_atoms
        return RelationalAbstractState(atoms, objects)

    target_loc = loc_type("loc-5-5")
    goal = RelationalAbstractGoal({at([target_loc])}, state_abstractor)

    # Finish the planning problem.
    problem = PlanningProblem(
        state_space,
        action_space,
        initial_state,
        transition_fn,
        goal,
    )

    # Create the sampler.
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator({move_skill}),
        transition_function=transition_fn,
        state_abstractor=state_abstractor,
        max_trajectory_steps=5,
    )

    # Create the abstract plan generator.
    abstract_plan_generator = RelationalHeuristicSearchAbstractPlanGenerator(
        {loc_type},
        {at, adjacent},
        operators,
        "hff",
        seed=123,
    )

    # General successor function for relational abstractions. This isn't actually used
    # by sesame, because the successor generator is used in the abstract plan generator,
    # but all bilevel planners get access to a successor generator.
    abstract_successor_fn = RelationalAbstractSuccessorGenerator(operators)

    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        max_abstract_plans,
        samples_per_step,
        abstract_successor_fn,
        state_abstractor,
        seed=123,
    )

    plan, bpg = planner.run(problem, timeout=10)
    assert plan is not None

    # Uncomment to make GIF.
    # def abstact_state_to_position(abstract_state):
    #     at_atom = next(atom for atom in abstract_state.atoms if atom.predicate == at)
    #     loc = at_atom.objects[0]
    #     return loc_to_xy(loc)

    # def customize_fig_ax(fig, ax):
    #     ax.set_ylim((0, 10))

    # final_state = plan.states[-1]
    # bpg.set_state_position_function(lambda x: x)
    # bpg.set_abstract_state_position_function(abstact_state_to_position)
    # outfile = "sesame_relational.gif"
    # bpg.render_gif(
    #     save_path=outfile,
    #     final_state=final_state,
    #     customize_fig_ax=customize_fig_ax
    # )

    del bpg
