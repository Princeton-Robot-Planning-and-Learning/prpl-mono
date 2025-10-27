"""PPO agent implementation for PRBench environments.

This is heavily based on the implementation from
cleanrl:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

import dacite
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
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
from prbench_rl.gym_utils import make_env

_O = TypeVar("_O")
_U = TypeVar("_U")


# Default arguments for PPO
@dataclass
class PPOArgs:
    """Arguments for the Soft Actor-Critic (SAC) algorithm."""

    seed: int = 0
    """Seed of the experiment."""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default."""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project_name: str = "ManiSkill"
    """The wandb's project name."""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project."""
    wandb_group: str = "PPO"
    """The group of the run for wandb."""
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
    num_steps: int = 2048
    """The number of steps to run in each environment per policy rollout."""
    eval_freq: int = 25
    """Evaluation frequency in terms of iterations."""
    save_train_video_freq: Optional[int] = None
    """Frequency to save training videos in terms of iterations."""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """The control mode to use for the environment."""

    # Algorithm specific arguments
    hidden_size: int = 64
    """The hidden size of the MLP networks."""
    total_timesteps: int = 10_000_000
    """Total timesteps of the experiments."""
    learning_rate: float = 3e-4
    """The learning rate of the optimizer."""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks."""
    gamma: float = 0.99
    """The discount factor gamma."""
    gae_lambda: float = 0.95
    """The lambda for the general advantage estimation."""
    num_minibatches: int = 32
    """The number of mini-batches."""
    update_epochs: int = 10
    """The K epochs to update the policy."""
    norm_adv: bool = True
    """Toggles advantages normalization."""
    clip_coef: float = 0.2
    """The surrogate clipping coefficient."""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the
    paper."""
    ent_coef: float = 0.0
    """Coefficient of the entropy."""
    vf_coef: float = 0.5
    """Coefficient of the value function."""
    max_grad_norm: float = 0.5
    """The maximum norm for the gradient clipping."""
    target_kl: float | None = None
    """The target KL divergence threshold."""
    reward_scale: float = 1.0
    """Scale the reward by this factor."""

    # to be filled in runtime
    batch_size: int = 0
    """The batch size (computed in runtime)"""
    minibatch_size: int = 0
    """The mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """The number of iterations (computed in runtime)"""


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)  # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const)  # type: ignore
    return layer


class Agent(nn.Module):
    """PPO actor-critic network."""

    def __init__(
        self,
        single_observation_space: spaces.Box,
        single_action_space: spaces.Box,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    int(np.array(single_observation_space.shape).prod()), hidden_size
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    int(np.array(single_observation_space.shape).prod()), hidden_size
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_size, int(np.prod(single_action_space.shape))),
                std=0.01,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, int(np.prod(single_action_space.shape)))
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value estimate."""
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get an action from the policy."""
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)  # type: ignore
        return probs.sample()  # type: ignore

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()  # type: ignore[no-untyped-call]
        return (
            action,
            probs.log_prob(action).sum(1),  # type: ignore[no-untyped-call]
            probs.entropy().sum(1),  # type: ignore[no-untyped-call]
            self.critic(x),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the agent."""
        return self.get_action(x, deterministic=True)


class PPOAgent(BaseRLAgent[_O, _U]):
    """PPO agent for continuous control tasks."""

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

        # Ensure cfg is not None for PPOAgent
        if cfg is None:
            cfg = DictConfig({})

        # Device setup
        cuda_enabled = cfg.get("cuda", False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cuda_enabled else "cpu"
        )

        # Load PPO arguments (with defaults if not provided)
        args_dict = cfg.get("args", cfg) if "args" in cfg else dict(cfg)
        self.args = dacite.from_dict(PPOArgs, args_dict)

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

        # Use spaces from base class
        assert isinstance(self.observation_space, spaces.Box)
        assert isinstance(self.action_space, spaces.Box)

        self.agent = Agent(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

    def _get_action(self) -> _U:  # type: ignore
        """Get action from current observation (for base class compatibility)."""
        assert self._last_observation is not None, "Must call reset() before step()"
        obs_tensor = torch.Tensor(self._last_observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.agent.get_action(obs_tensor, deterministic=True)
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
            action = self.agent.get_action(obs, deterministic=True)
        return action

    def evaluate(self, eval_episodes: int, render: bool = False) -> dict[str, Any]:
        """Evaluate the PPO agent."""
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    self.env_id,
                    0,
                    render,
                    self.cfg.exp_name + "_eval",
                    self.max_episode_steps,
                )
            ]
        )

        # Set agent to eval mode
        self.agent.eval()

        obs, _ = envs.reset()
        episodic_returns: list[float] = []
        step_lengths: list[int] = []
        step_length = 0

        while len(episodic_returns) < eval_episodes:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(self.device)
                action = self.agent(obs_tensor)
                actions = action.cpu().numpy()

            obs, _, _, _, infos = envs.step(actions)
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
                obs, _ = envs.reset()

        envs.close()  # type: ignore

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
        episodic_returns = []
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

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(
            self.args.batch_size // self.args.num_minibatches
        )
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size

        # ALGO Logic: Storage setup
        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        assert obs_shape is not None and action_shape is not None
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + obs_shape).to(
            self.device
        )
        actions = torch.zeros(
            (self.args.num_steps, self.args.num_envs) + action_shape
        ).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        next_obs, _ = envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        global_step = 0

        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            logging.info(f"Epoch: {iteration}, global_step={global_step}")
            # self.agent.eval()
            # Evaluate episode performance
            # if iteration % self.args.eval_freq == 0:
            #     eval_metrics = self.evaluate(self.args.num_eval_envs)
            #     logging.info(
            #         f"Evaluated {self.args.num_eval_envs} episodes"
            #     )
            #     for k, v in eval_metrics.items():
            #         mean = np.stack(v).mean()
            #         if self.writer is not None:
            #             self.writer.add_scalar(f"eval/{k}", mean, global_step)
            #         logging.info(f"eval_{k}_mean={mean}")
            # if self.args.save_model and iteration % self.args.eval_freq == 1:
            #     model_path = self.log_path / f"policies/ckpt_{global_step}.pt"
            #     base_path = Path(self.log_path) / "policies"
            #     base_path.mkdir(parents=True, exist_ok=True)
            #     self.save(str(model_path))
            #     logging.info(f"model saved to {model_path}")
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            rollout_time = time.time()
            # ALGO LOGIC: collect data
            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done_np = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(next_done_np).to(self.device)

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

            rollout_time = time.time() - rollout_time

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam: torch.Tensor | float = 0.0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.args.gamma
                        * self.args.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + action_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # ALGO LOGIC: update the agent with the collected data
            self.agent.train()
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            update_time = time.time()
            for _ in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                    ) = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.args.ent_coef * entropy_loss
                        + v_loss * self.args.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()  # type: ignore
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.args.max_grad_norm
                    )
                    self.optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            update_time = time.time() - update_time  # type: ignore

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if self.writer is not None:
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/value_loss", v_loss.item(), global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/policy_loss", pg_loss.item(), global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/entropy", entropy_loss.item(), global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/old_approx_kl", old_approx_kl.item(), global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/approx_kl", approx_kl.item(), global_step
                )
                clipfracs_log = float(np.mean(clipfracs))
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/clipfrac", clipfracs_log, global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "losses/explained_variance", explained_var, global_step
                )
                elapsed_time = time.time() - start_time
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "charts/SPS", int(global_step / elapsed_time), global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "time/step", global_step, global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "time/update_time", update_time, global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "time/rollout_time", rollout_time, global_step
                )
                self.writer.add_scalar(  # type: ignore[no-untyped-call]
                    "time/rollout_fps",
                    self.args.num_envs * self.args.num_steps / rollout_time,
                    global_step,
                )

        if self.args.save_model:
            model_path = self.log_path / "final_ckpt.pt"
            self.save(str(model_path))
            logging.info(f"model saved to {model_path}")
        envs.close()  # type: ignore[no-untyped-call]
        if self.writer is not None:
            self.writer.close()  # type: ignore[no-untyped-call]

        train_metrics = {
            "episodic_return": episodic_returns,
        }
        return train_metrics

    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        torch.save(
            {
                "network_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agent.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
