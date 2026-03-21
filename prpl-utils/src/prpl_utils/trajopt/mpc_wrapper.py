"""Model-predictive controller."""

from prpl_utils.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
)
from prpl_utils.trajopt.trajopt_solver import TrajOptSolver


class MPCWrapper:
    """Re-run a given trajectory optimization solver at every timestep."""

    def __init__(self, solver: TrajOptSolver) -> None:
        self._solver = solver
        self._problem: TrajOptProblem | None = None
        self._timestep = 0

    def reset(self, problem: TrajOptProblem) -> None:
        """Reset the timestep to 0."""
        self._timestep = 0
        self._problem = problem
        self._solver.reset(problem)

    def step(self, state: TrajOptState) -> TrajOptAction:
        """Get the most recent state and return an action."""
        assert self._problem is not None, "Call reset() before step()"
        traj = self._solver.solve(
            initial_state=state,
            horizon=self._problem.horizon,
        )
        action = traj.actions[0]
        self._timestep += 1
        return action
