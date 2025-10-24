"""PPO agent implementation for PRBench environments."""

import os
import time
from typing import Any, TypeVar

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import Env
from omegaconf import DictConfig
from torch import nn, optim
from torch.distributions.normal import Normal

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

from prbench_rl.agent import BaseRLAgent

_O = TypeVar("_O")
_U = TypeVar("_U")


def layer_init(
    layer: nn.Linear, std: float = float(np.sqrt(2)), bias_const: float = 0.0
) -> nn.Linear:
    """Initialize layer weights with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPONetwork(nn.Module):
    """PPO actor-critic network."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        obs_shape = observation_space.shape
        action_shape = action_space.shape
        assert obs_shape is not None and action_shape is not None
        obs_dim = int(np.array(obs_shape).prod())
        action_dim = int(np.prod(action_shape))

        # Store action space bounds for bounded actions
        self.action_low = torch.tensor(action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        # Actor network (outputs raw values that will be scaled)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        # Learnable log standard deviation (in scaled space)
        self.actor_logstd = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass - delegates to get_action_and_value."""
        return self.get_action_and_value(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value estimate."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value, with optional action for evaluation."""
        # Get raw action mean in [-1, 1] range
        raw_action_mean = torch.tanh(self.actor_mean(x))
        action_logstd = self.actor_logstd.expand_as(raw_action_mean)
        action_std = torch.exp(action_logstd)

        # Scale to actual action space
        action_low = self.action_low.to(x.device)
        action_high = self.action_high.to(x.device)
        action_scale = self.action_scale.to(x.device)
        action_bias = self.action_bias.to(x.device)

        # Scale the mean to action space bounds
        scaled_action_mean = raw_action_mean * action_scale + action_bias
        # Scale the std to action space
        scaled_action_std = action_std * action_scale

        probs = Normal(scaled_action_mean, scaled_action_std)

        if action is None:
            action = probs.sample()  # type: ignore
            # Clamp to action bounds for safety
            action = torch.clamp(action, action_low, action_high)

        return (
            action,
            probs.log_prob(action).sum(1),  # type: ignore
            probs.entropy().sum(1),  # type: ignore
            self.critic(x),
        )


class PPOAgent(BaseRLAgent[_O, _U]):
    """PPO agent for continuous control tasks."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        seed: int,
        cfg: DictConfig,
    ) -> None:
        super().__init__(observation_space, action_space, seed, cfg)

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
        )

        # Set random seeds
        torch.manual_seed(seed)
        if cfg.torch_deterministic:
            torch.backends.cudnn.deterministic = False

        # Create network
        self.network = PPONetwork(observation_space, action_space, cfg.hidden_size).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate, eps=1e-5
        )

        # Initialize training attributes
        self.batch_size = 0
        self.minibatch_size = 0
        self.global_train_step = 0
        self.step_count = 0
        self.obs_buffer: torch.Tensor | None = None
        self.actions_buffer: torch.Tensor | None = None
        self.logprobs_buffer: torch.Tensor | None = None
        self.rewards_buffer: torch.Tensor | None = None
        self.dones_buffer: torch.Tensor | None = None
        self.values_buffer: torch.Tensor | None = None
        self.returns_buffer: torch.Tensor | None = None
        self.advantages_buffer: torch.Tensor | None = None

        # Tensorboard logging
        self.writer = None
        if cfg.get("tf_log", False):
            if not TENSORBOARD_AVAILABLE or SummaryWriter is None:
                print("Warning: tensorboard not available, skipping logging")
            else:
                tf_log_dir = cfg.get("tf_log_dir", "runs")
                run_name = f"ppo__{int(time.time())}__{seed}"
                log_path = os.path.join(tf_log_dir, run_name)
                self.writer = SummaryWriter(log_path)  # type: ignore

    def setup_storage(self) -> None:
        """Reset trajectory storage buffers."""
        cfg = self.cfg
        self.batch_size = int(cfg.num_envs * cfg.num_steps)
        self.minibatch_size = int(self.batch_size // cfg.num_minibatches)
        self.global_train_step = 0

        # Storage tensors
        obs_shape = self.observation_space.shape
        action_shape = self.action_space.shape
        assert obs_shape is not None and action_shape is not None
        self.obs_buffer = torch.zeros((cfg.num_steps, cfg.num_envs) + obs_shape).to(
            self.device
        )
        self.actions_buffer = torch.zeros(
            (cfg.num_steps, cfg.num_envs) + action_shape
        ).to(self.device)
        self.logprobs_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(
            self.device
        )
        self.rewards_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.dones_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.values_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.returns_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.advantages_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(
            self.device
        )

    def _collect_rollout(self, env: Env) -> list[dict[str, Any]]:
        """Collect a rollout of experience."""
        # Ensure buffers are initialized
        assert self.obs_buffer is not None
        assert self.actions_buffer is not None
        assert self.logprobs_buffer is not None
        assert self.rewards_buffer is not None
        assert self.dones_buffer is not None
        assert self.values_buffer is not None
        assert self.advantages_buffer is not None

        episode_metrics = []
        next_obs, _ = env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.cfg.num_envs).to(self.device)

        for step in range(0, self.cfg.num_steps):
            self.global_train_step += self.cfg.num_envs
            self.obs_buffer[step] = next_obs
            self.dones_buffer[step] = next_done
            with torch.no_grad():
                if next_obs.dim() == 1:
                    next_obs = next_obs.unsqueeze(0)
                action, logprob, _, value = self.network.get_action_and_value(next_obs)
                self.values_buffer[step] = value.flatten()
            self.actions_buffer[step] = action
            self.logprobs_buffer[step] = logprob

            next_obs, reward, terminated, truncated, infos = env.step(
                action.squeeze().cpu().numpy()
            )
            if isinstance(terminated, bool):
                assert (
                    self.cfg.num_envs == 1
                ), "num_envs must be 1 if terminated is bool"
                next_done = torch.zeros(self.cfg.num_envs).to(self.device)
                next_done[0] = float(terminated or truncated)
            self.rewards_buffer[step] = torch.tensor(reward).to(self.device)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(
                next_done
            ).to(self.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={self.global_train_step}, "
                            f"episodic_return={info['episode']['r']}"
                        )
                        episode_metrics.append(
                            {
                                "global_step": self.global_train_step,
                                "episodic_return": info["episode"]["r"],
                                "episodic_length": info["episode"]["l"],
                            }
                        )
                        # Log to tensorboard
                        if self.writer:
                            self.writer.add_scalar(  # type: ignore
                                "charts/episodic_return",
                                info["episode"]["r"],
                                self.global_train_step,
                            )
                            self.writer.add_scalar(  # type: ignore
                                "charts/episodic_length",
                                info["episode"]["l"],
                                self.global_train_step,
                            )

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = self.network.get_value(next_obs).reshape(1, -1)
            lastgaelam = 0
            for t in reversed(range(self.cfg.num_steps)):
                if t == self.cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones_buffer[t + 1]
                    nextvalues = self.values_buffer[t + 1]
                delta = (
                    self.rewards_buffer[t]
                    + self.cfg.gamma * nextvalues * nextnonterminal
                    - self.values_buffer[t]
                )
                self.advantages_buffer[t] = lastgaelam = (
                    delta
                    + self.cfg.gamma
                    * self.cfg.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            self.returns_buffer = self.advantages_buffer + self.values_buffer

        return episode_metrics

    def _get_action(self) -> _U:
        """Get action from policy."""
        if self._last_observation is None:
            return self.action_space.sample()

        obs_tensor = (
            torch.tensor(self._last_observation, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor)

        self._last_action = action.cpu().numpy()[0]
        return action.cpu().numpy()[0]

    def train_with_env(self, env: Env) -> list[dict[str, Any]]:
        """Train the PPO agent with environment interaction."""
        self.train()
        self.setup_storage()
        training_metrics = []
        num_iterations = self.cfg.total_timesteps // self.batch_size

        for iteration in range(1, num_iterations + 1):
            # 1. Collect rollout and store in buffer
            episode_metrics = self._collect_rollout(env)
            training_metrics.extend(episode_metrics)

            # 2. Update policy with the current buffer
            update_metrics = self._update_policy()

            # 3. Add update metrics to latest episode if available
            if training_metrics and update_metrics:
                training_metrics[-1].update(update_metrics)

            # Learning rate annealing
            if self.cfg.get("anneal_lr", False):
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

        return training_metrics

    def _update_policy(self) -> dict[str, Any]:
        """Update the policy using PPO."""
        cfg = self.cfg

        # Ensure buffers are initialized
        assert self.obs_buffer is not None
        assert self.actions_buffer is not None
        assert self.logprobs_buffer is not None
        assert self.rewards_buffer is not None
        assert self.dones_buffer is not None
        assert self.values_buffer is not None
        assert self.returns_buffer is not None
        assert self.advantages_buffer is not None

        # Flatten batch
        obs_shape = self.observation_space.shape
        action_shape = self.action_space.shape
        assert obs_shape is not None and action_shape is not None
        b_obs = self.obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = self.logprobs_buffer.reshape(-1)
        b_actions = self.actions_buffer.reshape((-1,) + action_shape)
        b_advantages = self.advantages_buffer.reshape(-1)
        b_returns = self.returns_buffer.reshape(-1)
        b_values = self.values_buffer.reshape(-1)

        # Optimize policy for multiple epochs
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # KL divergence approximation
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics = {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar(  # type: ignore
                "charts/learning_rate", metrics["learning_rate"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/value_loss", metrics["value_loss"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/policy_loss", metrics["policy_loss"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/entropy", metrics["entropy_loss"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/old_approx_kl", metrics["old_approx_kl"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/approx_kl", metrics["approx_kl"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/clipfrac", metrics["clipfrac"], self.global_train_step
            )
            self.writer.add_scalar(  # type: ignore
                "losses/explained_variance",
                metrics["explained_variance"],
                self.global_train_step,
            )

        return metrics

    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def close(self) -> None:
        """Close the tensorboard writer."""
        if self.writer:
            self.writer.close()  # type: ignore
