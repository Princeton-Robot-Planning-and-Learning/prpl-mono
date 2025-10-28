"""SAC agent implementation for PRBench environments.

This is heavily based on the implementation from
cleanrl:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

import dacite
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

from prbench_rl.agent import BaseRLAgent
from prbench_rl.gym_utils import make_env

_O = TypeVar("_O")
_U = TypeVar("_U")


# Default arguments for SAC
@dataclass
class SACArgs:
    """Arguments for the Soft Actor-Critic (SAC) algorithm."""

    seed: int = 0
    """Seed of the experiment."""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default."""
    capture_video: bool = True
    """Whether to capture videos of the self.agent performances (check out `videos`
    folder)"""
    save_trajectory: bool = False
    """Whether to save trajectory data into the `videos` folder."""
    save_model: bool = True
    """Whether to save model into the `runs/{run_name}` folder."""

    # Environment specific arguments
    num_envs: int = 1
    """The number of parallel environments."""
    num_eval_envs: int = 16
    """The number of parallel evaluation environments."""
    eval_freq: int = 10000
    """Evaluation frequency in terms of steps."""
    save_train_video_freq: Optional[int] = None
    """Frequency to save training videos in terms of iterations."""

    # Algorithm specific arguments
    hidden_size: int = 64
    """The hidden size of the neural networks."""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACAgent(BaseRLAgent[_O, _U]):
    """SAC agent for continuous control tasks."""

    def __init__(
        self,
        seed: int,
        env_id: str | None = None,
        max_episode_steps: int | None = None,
        cfg: DictConfig | None = None,
        observation_space: spaces.Box | None = None,
        action_space: spaces.Box | None = None,
    ) -> None:
        super().__init__(
            seed,
            env_id,
            max_episode_steps,
            cfg,
            observation_space,  # type: ignore
            action_space,  # type: ignore
        )

        # Ensure cfg is not None for SACAgent
        if cfg is None:
            cfg = DictConfig({})

        # Device setup
        cuda_enabled = cfg.get("cuda", False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cuda_enabled else "cpu"
        )

        # Load SAC arguments (with defaults if not provided)
        args_dict = cfg.get("args", cfg) if "args" in cfg else dict(cfg)
        self.args = dacite.from_dict(SACArgs, args_dict)

        # Setup tensorboard writer if logging is enabled
        if cfg.get("tf_log", True):
            exp_name = cfg.get("exp_name", "ppo_experiment")
            tb_log_dir = cfg.get("tb_log_dir", "runs")
            self.log_path = Path(tb_log_dir) / exp_name
            self.writer = SummaryWriter(self.log_path)  # type: ignore
            self.writer.add_text(  # type: ignore
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % (
                    "\n".join(
                        [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                    )
                ),
            )
        else:
            self.log_path = Path("runs/ppo_experiment")
            self.writer = None  # type: ignore

    def _get_action(self) -> _U:  # type: ignore
        """Get action from current observation (for base class compatibility)."""


    def get_action_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from observation tensor."""

    def evaluate(self, eval_episodes: int, render: bool = False) -> dict[str, Any]:
        """Evaluate the SAC agent."""


        eval_metrics = {
            "episodic_return": episodic_returns,
            "step_length": step_lengths,
        }
        return eval_metrics

    def train(self, render: bool = False) -> dict[str, Any]:  # type: ignore
        """Training the agent with an interactive batched environment."""
        # Initialize observation normalization variables
        # update the args with the environment-specific values
        # env setup
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    self.env_id,
                    i,
                    render,
                    self.cfg.exp_name + "_train",
                    self.max_episode_steps,
                )
                for i in range(self.args.num_envs)
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"


    def save(self, filepath: str) -> None:
        """Save agent parameters."""

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
