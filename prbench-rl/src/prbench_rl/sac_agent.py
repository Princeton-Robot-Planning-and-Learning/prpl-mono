"""SAC agent implementation for PRBench environments.

This is heavily based on the implementation from
cleanrl:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, cast

import dacite
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from omegaconf import DictConfig
from torch import nn, optim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

from prbench_rl.agent import BaseRLAgent
from prbench_rl.gym_utils import ReplayBuffer, make_env_sac

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
    save_model_freq: int = 50000
    """Frequency to save the model (in timesteps)."""

    # Environment specific arguments
    num_envs: int = 1
    """The number of parallel environments."""
    save_train_video_freq: Optional[int] = None
    """Frequency to save training videos in terms of iterations."""

    # Algorithm specific arguments
    hidden_size: int = 256
    """The hidden size of the neural networks."""
    total_timesteps: int = 1000000
    """Total timesteps of the experiments."""
    buffer_size: int = int(1e6)
    """The replay memory buffer size."""
    gamma: float = 0.99
    """The discount factor gamma."""
    tau: float = 0.005
    """Target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """The batch size of sample from the reply memory."""
    learning_starts: int = int(5e3)
    """Timestep to start learning."""
    policy_lr: float = 3e-4
    """The learning rate of the policy network optimizer."""
    q_lr: float = 1e-3
    """The learning rate of the Q network network optimizer."""
    policy_frequency: int = 2
    """The frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """The frequency of updates for the target nerworks."""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """Automatic tuning of the entropy coefficient."""


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """Soft Q-Network (Critic) for SAC."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_size: int = 256,
    ):
        super().__init__()
        obs_dim = int(np.array(observation_space.shape).prod())
        action_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for state-action pair."""
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Actor network (Policy) for SAC."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_size: int = 256,
    ):
        super().__init__()
        obs_dim = int(np.array(observation_space.shape).prod())
        action_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std for action distribution."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy, returning action, log_prob, and mean."""
        mean, log_std = self(x)
        if deterministic:
            # Return deterministic action (mean squashed through tanh)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, torch.zeros_like(action), action
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)  # type: ignore
        # Reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()  # type: ignore
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)  # type: ignore
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
            exp_name = cfg.get("exp_name", "sac_experiment")
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
            self.log_path = Path("runs/sac_experiment")
            self.writer = None  # type: ignore

        # Use spaces from base class
        assert isinstance(self.observation_space, spaces.Box)
        assert isinstance(self.action_space, spaces.Box)

        # Initialize actor and critics
        self.actor = Actor(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf1 = SoftQNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf2 = SoftQNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf1_target = SoftQNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf2_target = SoftQNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Initialize optimizers
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=self.args.q_lr,
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.args.policy_lr
        )

        # Automatic entropy tuning
        if self.args.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha

        # Replay buffer (will be initialized in train())
        self.rb: ReplayBuffer | None = None

    def _get_action(self) -> _U:  # type: ignore
        """Get action from current observation (for base class compatibility)."""
        assert self._last_observation is not None, "Must call reset() before step()"
        obs_tensor = torch.Tensor(self._last_observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(obs_tensor, deterministic=True)
        action_np = action.cpu().numpy()[0]
        # Clip action to be within action space bounds
        assert isinstance(self.action_space, spaces.Box)
        clipped_action = np.clip(
            action_np, self.action_space.low, self.action_space.high
        )
        return clipped_action  # type: ignore

    def get_action_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from observation tensor."""
        with torch.no_grad():
            action, _, _ = self.actor.get_action(obs, deterministic=True)
        return action

    def evaluate_on_env(
        self,
        train_envs: gym.vector.VectorEnv,
        eval_episodes: int,
        render_video: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the SAC agent with video recording.

        Wraps the training environments with RecordVideo to capture evaluation episodes.
        The environments retain their normalization statistics from training.

        Args:
            train_envs: Training environments to wrap with video recording
            eval_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Wrap the first training environment with RecordVideo for evaluation
        # This preserves the normalization statistics while enabling video recording
        if render_video:
            video_folder = f"videos/{self.cfg.exp_name}_eval"
            sync_envs = cast(gym.vector.SyncVectorEnv, train_envs)
            sync_envs.envs[0] = gym.wrappers.RecordVideo(
                sync_envs.envs[0],
                video_folder,
                episode_trigger=lambda x: True,  # Record all episodes
            )

        # Set agent to eval mode
        self.actor.eval()

        obs, _ = train_envs.reset()
        episodic_returns: list[float] = []
        step_lengths: list[int] = []
        step_length = 0

        while len(episodic_returns) < eval_episodes:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(self.device)
                action, _, _ = self.actor.get_action(obs_tensor, deterministic=True)
                actions = action.cpu().numpy()

            obs, _, _, _, infos = train_envs.step(actions)
            step_length += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None or "episode" not in info:
                        continue
                    logging.info(
                        f"eval_episode={len(episodic_returns)}, "
                        f"episodic_return={info['episode']['r']}"
                    )
                    episodic_returns.append(info["episode"]["r"])
                    step_lengths.append(step_length)
                    step_length = 0

        eval_metrics = {
            "episodic_return": episodic_returns,
            "step_length": step_lengths,
        }
        return eval_metrics

    def train(  # type: ignore[override]
        self, eval_episodes: int = 10, render_eval_video: bool = False
    ) -> dict[str, Any]:
        """Training the agent with an interactive batched environment."""
        # Seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        # env setup
        episodic_returns: list[float] = []
        # Create training environments (no video recording during training)
        envs = gym.vector.SyncVectorEnv(
            [
                make_env_sac(
                    self.env_id,
                    self.max_episode_steps,
                )
                for i in range(self.args.num_envs)
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        # Initialize replay buffer
        envs.single_observation_space.dtype = np.float32  # type: ignore
        self.rb = ReplayBuffer(
            self.args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            n_envs=self.args.num_envs,
            handle_timeout_termination=False,
        )

        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=self.args.seed)
        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.args.learning_starts:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                with torch.no_grad():
                    action_tensors, _, _ = self.actor.get_action(
                        torch.Tensor(obs).to(self.device)
                    )
                    actions = action_tensors.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_return = info["episode"]["r"]
                        logging.info(
                            f"global_step={global_step}, "
                            f"episodic_return={episode_return}"
                        )
                        episodic_returns.append(info["episode"]["r"])
                        if self.writer is not None:
                            self.writer.add_scalar(  # type: ignore[no-untyped-call]
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            self.writer.add_scalar(  # type: ignore[no-untyped-call]
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            # Convert vectorized infos dict to list of per-env info dicts
            infos_list = [
                {"TimeLimit.truncated": bool(truncations[i])}
                for i in range(self.args.num_envs)
            ]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos_list)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                data = self.rb.sample(self.args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations
                    )
                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()  # type: ignore
                self.q_optimizer.step()

                if global_step % self.args.policy_frequency == 0:
                    for _ in range(self.args.policy_frequency):
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()  # type: ignore
                        self.actor_optimizer.step()

                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (
                                -self.log_alpha.exp() * (log_pi + self.target_entropy)
                            ).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()  # type: ignore
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if (
                    global_step % self.args.save_model_freq == 0
                    and self.args.save_model
                ):
                    model_path = self.log_path / f"policies/ckpt_{global_step}.pt"
                    base_path = Path(self.log_path) / "policies"
                    base_path.mkdir(parents=True, exist_ok=True)
                    self.save(str(model_path))
                    logging.info(f"model saved to {model_path}")

                if global_step % 100 == 0 and self.writer is not None:
                    self.writer.add_scalar(  # type: ignore
                        "losses/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    self.writer.add_scalar(  # type: ignore
                        "losses/qf2_values", qf2_a_values.mean().item(), global_step
                    )
                    self.writer.add_scalar(  # type: ignore
                        "losses/qf1_loss", qf1_loss.item(), global_step
                    )
                    self.writer.add_scalar(  # type: ignore
                        "losses/qf2_loss", qf2_loss.item(), global_step
                    )
                    self.writer.add_scalar(  # type: ignore
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    self.writer.add_scalar(  # type: ignore
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    self.writer.add_scalar(  # type: ignore
                        "losses/alpha", self.alpha, global_step
                    )
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    self.writer.add_scalar(  # type: ignore
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar(  # type: ignore
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )

        if self.args.save_model:
            model_path = self.log_path / "final_ckpt.pt"
            self.save(str(model_path))
            logging.info(f"model saved to {model_path}")

        # Evaluate on the training environment (shares the same normalizer)
        logging.info(f"Starting evaluation for {eval_episodes} episodes...")
        eval_metrics = self.evaluate_on_env(
            envs, eval_episodes, render_video=render_eval_video
        )

        # Log evaluation results
        if eval_metrics["episodic_return"]:
            avg_return = np.mean(eval_metrics["episodic_return"])
            logging.info(f"Evaluation average return: {avg_return}")
            if self.writer is not None:
                self.writer.add_scalar(  # type: ignore
                    "eval/average_return", avg_return, global_step
                )

        # Close environments and writer
        envs.close()  # type: ignore
        if self.writer is not None:
            self.writer.close()  # type: ignore

        return {
            "train": {"episodic_return": episodic_returns},
            "eval": eval_metrics,
        }

    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        save_dict: dict[str, Any] = {
            "actor_state_dict": self.actor.state_dict(),
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }
        if self.args.autotune:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["a_optimizer_state_dict"] = self.a_optimizer.state_dict()
        torch.save(save_dict, filepath)

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.qf1.load_state_dict(checkpoint["qf1_state_dict"])
        self.qf2.load_state_dict(checkpoint["qf2_state_dict"])
        self.qf1_target.load_state_dict(checkpoint["qf1_target_state_dict"])
        self.qf2_target.load_state_dict(checkpoint["qf2_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        if self.args.autotune and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
            self.a_optimizer.load_state_dict(checkpoint["a_optimizer_state_dict"])
            self.alpha = self.log_alpha.exp().item()
