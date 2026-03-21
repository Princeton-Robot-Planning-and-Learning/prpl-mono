"""The simple trajopt method described in https://arxiv.org/abs/2212.00541"""

from dataclasses import dataclass

import numpy as np

from prpl_utils.trajopt.trajectory import Trajectory, point_sequence_to_trajectory
from prpl_utils.trajopt.trajopt_problem import TrajOptState, TrajOptTraj
from prpl_utils.trajopt.trajopt_solver import TrajOptSolver
from prpl_utils.trajopt.utils import (
    sample_standard_normal_spline,
    spline_to_trajopt_trajectory,
)


@dataclass(frozen=True)
class PredictiveSamplingHyperparameters:
    """Hyperparameters for predictive sampling."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 10


class PredictiveSamplingSolver(TrajOptSolver):
    """The simple method described in https://arxiv.org/abs/2212.00541"""

    def __init__(
        self,
        seed: int,
        config: PredictiveSamplingHyperparameters | None = None,
        warm_start: bool = True,
    ) -> None:
        self._config = config or PredictiveSamplingHyperparameters()
        super().__init__(seed, warm_start)

    def _solve(
        self,
        initial_state: TrajOptState,
        horizon: int,
    ) -> Trajectory:
        # Warm start by advancing the last solution by one step.
        sample_list: list[Trajectory] = []
        if self._warm_start and self._last_solution is not None:
            assert isinstance(self._last_solution, Trajectory)
            assert np.isclose(self._last_solution.duration, horizon + 1)
            nominal = self._last_solution.get_sub_trajectory(1, horizon + 1)
        else:
            nominal = self._get_initialization(horizon)
        sample_list.append(nominal)
        # Sample new candidates around the nominal trajectory.
        num_samples = self._config.num_rollouts - len(sample_list)
        new_samples = self._sample_from_nominal(nominal, num_samples)
        sample_list.extend(new_samples)
        # Pick the best one.
        return min(
            sample_list, key=lambda s: self._score_sample(s, initial_state, horizon)
        )

    def _get_initialization(self, horizon: int) -> Trajectory:
        assert self._problem is not None
        action_dim = self._problem.action_space.shape[0]
        return sample_standard_normal_spline(
            self._rng, self._config.num_control_points, horizon, action_dim
        )

    def _sample_from_nominal(
        self,
        nominal: Trajectory,
        num_samples: int,
    ) -> list[Trajectory]:
        assert self._problem is not None
        # Sample by adding Gaussian noise around the nominal trajectory.
        num_cp = self._config.num_control_points
        control_times = np.linspace(0, nominal.duration, num=num_cp, endpoint=True)
        # (num_cp, action_dim)
        nominal_cp = np.array([nominal(t) for t in control_times])
        # (num_samples, num_cp, action_dim)
        noise = self._rng.normal(
            loc=0,
            scale=self._config.noise_scale,
            size=(num_samples, num_cp, nominal_cp.shape[1]),
        )
        all_cp = nominal_cp + noise
        # Clip to obey bounds.
        low = self._problem.action_space.low
        high = self._problem.action_space.high
        all_cp = np.clip(all_cp, low, high)
        # Convert to trajectories.
        dt = control_times[1] - control_times[0]
        return [
            point_sequence_to_trajectory(
                [pt.astype(np.float64) for pt in sample_cp], dt=dt
            )
            for sample_cp in all_cp
        ]

    def _score_sample(
        self,
        sample: Trajectory,
        initial_state: TrajOptState,
        horizon: int,
    ) -> float:
        assert self._problem is not None
        traj = self._solution_to_trajectory(sample, initial_state, horizon)
        return self._problem.get_traj_cost(traj)

    def _solution_to_trajectory(
        self,
        solution: Trajectory,
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        assert self._problem is not None
        return spline_to_trajopt_trajectory(
            self._problem, solution, initial_state, horizon
        )
