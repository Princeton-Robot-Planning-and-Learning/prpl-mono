"""Utils for trajectory optimization."""

import numpy as np

from prpl_utils.trajopt.trajectory import Trajectory, point_sequence_to_trajectory
from prpl_utils.trajopt.trajopt_problem import (
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)


def spline_to_trajopt_trajectory(
    problem: TrajOptProblem,
    solution: Trajectory,
    initial_state: TrajOptState,
    horizon: int,
) -> TrajOptTraj:
    """Roll out a spline to create a trajectory."""
    state_list = [initial_state]
    state = initial_state
    action_list: list[TrajOptState] = []
    for t in range(horizon):
        action = solution(t)
        action_list.append(action)
        state = problem.get_next_state(state, action)
        state_list.append(state)
    state_arr = np.array(state_list)
    action_arr = np.array(action_list)
    return TrajOptTraj(state_arr, action_arr)


def sample_standard_normal_spline(
    rng: np.random.Generator, num_points: int, horizon: int
) -> Trajectory:
    """Sample a spline by sampling points and interpolating."""
    if num_points < 2:
        raise ValueError(
            "sample_standard_normal_spline requires num_points >= 2 to define a spline."
        )
    if horizon <= 0:
        raise ValueError(
            "sample_standard_normal_spline requires horizon > 0."
        )
    points = list(rng.standard_normal(size=(num_points, 1)))
    dt = horizon / (len(points) - 1)
    return point_sequence_to_trajectory(points, dt=dt)
