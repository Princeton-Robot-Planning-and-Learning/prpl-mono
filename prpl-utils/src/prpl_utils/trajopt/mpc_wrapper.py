"""Model-predictive controller."""

from prpl_utils.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)
from prpl_utils.trajopt.trajopt_solver import TrajOptSolver


class MPCWrapper:
    """Re-run a given trajectory optimization solver at every timestep."""

    def __init__(self, solver: TrajOptSolver, replan_interval: int = 1) -> None:
        self._solver = solver
        self._replan_interval = replan_interval
        self._problem: TrajOptProblem | None = None
        self._timestep = 0
        self._last_traj: TrajOptTraj | None = None
        self._steps_since_replan = 0

    def reset(self, problem: TrajOptProblem) -> None:
        """Reset the timestep to 0."""
        self._timestep = 0
        self._problem = problem
        self._last_traj = None
        self._steps_since_replan = 0
        self._solver.reset(problem)

    def step(self, state: TrajOptState) -> TrajOptAction:
        """Get the most recent state and return an action."""
        assert self._problem is not None, "Call reset() before step()"
        if self._last_traj is None or self._steps_since_replan >= self._replan_interval:
            self._last_traj = self._solver.solve(
                initial_state=state,
                horizon=self._problem.horizon,
            )
            self._steps_since_replan = 0
        action = self._last_traj.actions[self._steps_since_replan]
        self._steps_since_replan += 1
        self._timestep += 1
        return action
