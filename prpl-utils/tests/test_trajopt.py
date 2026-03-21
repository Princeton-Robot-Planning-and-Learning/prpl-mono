"""Tests for trajopt."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from prpl_utils.structs import Image
from prpl_utils.trajopt.mpc_wrapper import MPCWrapper
from prpl_utils.trajopt.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)
from prpl_utils.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)


@dataclass(frozen=True)
class PendulumHyperparameters:
    """Hyperparameters for PendulumTrajOptProblem."""

    horizon: int = 200
    gravity: float = 10
    mass: float = 1.0
    length: float = 1.0
    dt: float = 0.05
    theta_cost_weight: float = 1.0
    theta_dot_cost_weight: float = 0.0
    torque_cost_weight: float = 0.0
    torque_lb: float = -np.inf
    torque_ub: float = np.inf


class PendulumTrajOptProblem(TrajOptProblem):
    """Classic pendulum swing-up trajopt problem."""

    def __init__(self, config: PendulumHyperparameters | None = None) -> None:
        self._config = config or PendulumHyperparameters()
        super().__init__()

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @cached_property
    def state_space(self) -> Box:
        return Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float64,
        )

    @cached_property
    def action_space(self) -> Box:
        return Box(
            low=np.array([self._config.torque_lb]),
            high=np.array([self._config.torque_ub]),
            dtype=np.float64,
        )

    @property
    def initial_state(self) -> TrajOptState:
        return np.array([np.pi, 1.0], dtype=np.float64)

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        theta, theta_dot = state
        u = action[0]
        g = self._config.gravity
        m = self._config.mass
        l = self._config.length  # noqa: E741
        dt = self._config.dt
        next_theta_dot = (
            theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
        )
        next_theta = theta + next_theta_dot * dt
        return np.array([next_theta, next_theta_dot], dtype=state.dtype)

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        thetas, theta_dots = traj.states.T
        theta_cost = float((thetas**2).sum())
        theta_dot_cost = float((theta_dots**2).sum())
        torque_cost = float((traj.actions**2).sum())
        return (
            self._config.theta_cost_weight * theta_cost
            + self._config.theta_dot_cost_weight * theta_dot_cost
            + self._config.torque_cost_weight * torque_cost
        )

    def render_state(self, state: TrajOptState) -> Image:
        raise NotImplementedError


def test_pendulum():
    """Test the pendulum environment."""
    env = PendulumTrajOptProblem()
    assert env.state_space.contains(env.initial_state)
    rng = np.random.default_rng(0)
    states: list[NDArray[np.float64]] = [env.initial_state]
    state = env.initial_state
    actions: list[NDArray[np.float64]] = []
    for _ in range(env.horizon):
        action = rng.standard_normal(env.action_space.shape).astype(np.float64)
        state = env.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = env.get_traj_cost(traj)
    assert cost > 0


def test_predictive_sampling():
    """Tests for predictive_sampling.py."""
    solver_config = PredictiveSamplingHyperparameters(
        num_rollouts=5, num_control_points=3
    )
    solver = PredictiveSamplingSolver(123, config=solver_config)
    mpc = MPCWrapper(solver)
    env_config = PendulumHyperparameters(horizon=10)
    env = PendulumTrajOptProblem(config=env_config)
    mpc.reset(env)
    initial_state = env.initial_state
    states = [initial_state]
    state = initial_state
    actions = []
    for _ in range(env.horizon):
        action = mpc.step(state)
        state = env.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = env.get_traj_cost(traj)
    assert cost > 0
