"""PPO agent implementation for PRBench environments."""

from typing import TypeVar, Optional, Any
from dataclasses import dataclass

import time
import logging
import numpy as np
import torch
import dacite
from pathlib import Path
from gymnasium import spaces
from omegaconf import DictConfig
from collections import defaultdict
from torch import nn, optim
from torch.distributions.normal import Normal

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

from prbench_rl.agent import BaseRLAgent, Logger
from prbench_rl.gym_utils import MultiEnvWrapper

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
    evaluate: bool = False
    """If toggled, only runs evaluation with the given model checkpoint and saves the
    evaluation trajectories."""
    checkpoint: Optional[str] = None
    """Path to a pretrained checkpoint file to start evaluation/training from."""

    # Environment specific arguments
    num_envs: int = 512
    """The number of parallel environments."""
    num_eval_envs: int = 16
    """The number of parallel evaluation environments."""
    partial_reset: bool = True
    """Whether to let parallel environments reset upon termination instead of
    truncation."""
    eval_partial_reset: bool = False
    """Whether to let parallel evaluation environments reset upon termination instead of
    truncation."""
    num_steps: int = 50
    """The number of steps to run in each environment per policy rollout."""
    reconfiguration_freq: Optional[int] = None
    """How often to reconfigure the environment during training."""
    eval_reconfiguration_freq: Optional[int] = 1
    """For benchmarking purposes we want to reconfigure the eval environment each reset
    to ensure objects are randomized in some tasks."""
    eval_freq: int = 25
    """Evaluation frequency in terms of iterations."""
    save_train_video_freq: Optional[int] = None
    """Frequency to save training videos in terms of iterations."""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """The control mode to use for the environment."""

    # Algorithm specific arguments
    total_timesteps: int = 10_000_000
    """Total timesteps of the experiments."""
    learning_rate: float = 3e-4
    """The learning rate of the optimizer."""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks."""
    gamma: float = 0.8
    """The discount factor gamma."""
    gae_lambda: float = 0.9
    """The lambda for the general advantage estimation."""
    num_minibatches: int = 32
    """The number of mini-batches."""
    update_epochs: int = 4
    """The K epochs to update the policy."""
    norm_adv: bool = True
    """Toggles advantages normalization."""
    clip_coef: float = 0.2
    """The surrogate clipping coefficient."""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the
    paper."""
    ent_coef: float = 0.0
    """Coefficient of the entropy."""
    vf_coef: float = 0.5
    """Coefficient of the value function."""
    max_grad_norm: float = 0.5
    """The maximum norm for the gradient clipping."""
    target_kl: float = 0.1
    """The target KL divergence threshold."""
    reward_scale: float = 1.0
    """Scale the reward by this factor."""
    finite_horizon_gae: bool = False
    normalize_obs: bool = False
    """Whether to normalize observations using running mean and std."""

    # to be filled in runtime
    batch_size: int = 0
    """The batch size (computed in runtime)"""
    minibatch_size: int = 0
    """The mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """The number of iterations (computed in runtime)"""


def layer_init(layer: nn.Module, 
               std: float = np.sqrt(2), 
               bias_const: float = 0.0):
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
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        obs_shape = single_observation_space.shape
        action_shape = single_action_space.shape
        assert obs_shape is not None and action_shape is not None

        # Store action space bounds for bounded actions
        self.action_low = torch.tensor(single_action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(single_action_space.high, dtype=torch.float32)

        # Critic network
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(obs_shape).prod(), hidden_size)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1)),
        )

        # Actor network (outputs raw values that will be scaled)
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(obs_shape).prod(), hidden_size)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_size, np.prod(obs_shape)),  # type: ignore
                std=0.01 * np.sqrt(2),
            ),
        )

        # Learnable log standard deviation (in scaled space)
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(obs_shape)) * -0.5  # type: ignore
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
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an action and its value from the policy."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)  # type: ignore
        if action is None:
            action = probs.sample()  # type: ignore
        return (
            action,
            probs.log_prob(action).sum(1),  # type: ignore
            probs.entropy().sum(1),  # type: ignore
            self.critic(x),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the agent."""
        return self.get_action(x, deterministic=True)


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

        # Load PPO arguments
        self.args = dacite.from_dict(PPOArgs, cfg.args)
        log_path = Path(cfg.tb_log_dir) / f"{cfg.exp_name}"
        writer = SummaryWriter(log_path)  # type: ignore
        writer.add_text(  # type: ignore
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                )
            ),
        )
        self.logger = Logger(tensorboard=writer)

    def initialize(self, env: MultiEnvWrapper) -> None:
        """Initialize the PPO policy with the given environment."""
        # update the args with the environment-specific values
        num_envs = env.num_envs
        if num_envs != self.args.num_envs:
            logging.warning(
                f"Number of environments in the provided environment ({num_envs}) "
                f"does not match the configured number of environments ({self.args.num_envs}). "
                f"Using {num_envs} instead."
            )
            self.args.num_envs = num_envs

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(
            self.args.batch_size // self.args.num_minibatches
        )
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size

        self.agent = Agent(env).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.args.num_steps, self.args.num_envs)
            + (
                tuple(env.single_observation_space.shape)
                if env.single_observation_space.shape is not None
                else ()
            )
        ).to(self.device)
        self.actions = torch.zeros(
            (self.args.num_steps, self.args.num_envs)
            + (
                tuple(env.single_action_space.shape)
                if env.single_action_space.shape is not None
                else ()
            )
        ).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )

    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.agent.get_action(obs, deterministic=True)
        return action

    def train_with_env(
        self,
        env: MultiEnvWrapper,
        eval_env: Optional[MultiEnvWrapper] = None,
    ) -> list[dict[str, Any]]:
        """Training the agent with an interactive batched environment."""
        # Initialize observation normalization variables
        obs_shape = env.single_observation_space.shape
        if obs_shape is None:
            obs_shape = ()
        self.curr_obs_mean = torch.zeros(obs_shape, device=self.device)
        self.curr_obs_std = torch.ones(obs_shape, device=self.device)

        next_obs, _ = env.reset(seed=self.args.seed)
        if eval_env is not None:
            eval_obs, _ = eval_env.reset(seed=self.args.seed)
        next_done = torch.zeros(self.args.num_envs, device=self.device)
        global_step = 0

        action_space_low, action_space_high = torch.from_numpy(
            env.single_action_space.low  # type: ignore
        ).to(self.device), torch.from_numpy(
            env.single_action_space.high
        ).to(  # type: ignore
            self.device
        )

        def clip_action(action: torch.Tensor):
            return torch.clamp(action.detach(), action_space_low, action_space_high)

        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            logging.info(f"Epoch: {iteration}, global_step={global_step}")
            final_values = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            self.agent.eval()
            if iteration % self.args.eval_freq == 1 and eval_env is not None:
                logging.info("Evaluating")
                eval_obs, _ = eval_env.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(CFG.max_rl_steps):
                    with torch.no_grad():
                        normalized_eval_obs = self.normalize_obs(eval_obs)
                        eval_obs, _, _, _, eval_infos = eval_env.step(
                            clip_action(self.get_action(normalized_eval_obs))
                        )
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            num_episodes += mask.sum()
                            for k, v in eval_infos["final_info"]["episode"].items():
                                eval_metrics[k].append(v)
                evaluated_steps = CFG.max_rl_steps * self.args.num_eval_envs
                logging.info(
                    f"Evaluated {evaluated_steps} steps resulting in {num_episodes} episodes"
                )
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if self.logger is not None:
                        self.logger.add_scalar(f"eval/{k}", mean, global_step)
                    logging.info(f"eval_{k}_mean={mean}")
                if self.args.evaluate:
                    break
            if self.args.save_model and iteration % self.args.eval_freq == 1:
                model_path = (
                    Path(CFG.rl_policy_save_dir)
                    / f"runs/{CFG.exp_name}/ckpt_{global_step}.pt"
                )
                base_path = Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                base_path.mkdir(parents=True, exist_ok=True)
                self.save(model_path)
                logging.info(f"model saved to {model_path}")
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            rollout_time = time.time()
            # ALGO LOGIC: collect data
            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    normalized_obs = self.normalize_obs(next_obs)
                    action, logprob, _, value = self.agent.get_action_and_value(
                        normalized_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = env.step(
                    clip_action(action)
                )
                next_done = torch.logical_or(terminations, truncations).to(
                    torch.float32
                )
                self.rewards[step] = reward.view(-1) * self.args.reward_scale

                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for k, v in final_info["episode"].items():
                        self.logger.add_scalar(
                            f"train/{k}", v[done_mask].float().mean(), global_step
                        )
                    with torch.no_grad():
                        final_values[
                            step,
                            torch.arange(self.args.num_envs, device=self.device)[
                                done_mask
                            ],
                        ] = self.agent.get_value(
                            self.normalize_obs(infos["final_observation"][done_mask])
                        ).view(
                            -1
                        )
            rollout_time = time.time() - rollout_time

            # Update observation normalization statistics
            if self.args.normalize_obs:
                with torch.no_grad():
                    # Calculate mean and std from current rollout observations
                    batch_obs = self.obs.reshape(-1, *self.obs.shape[2:])
                    self.curr_obs_mean = batch_obs.mean(dim=0)
                    self.curr_obs_std = batch_obs.std(dim=0)

            # bootstrap value according to termination and truncation
            with torch.no_grad():
                normalized_next_obs = self.normalize_obs(next_obs)
                next_value = self.agent.get_value(normalized_next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    real_next_values = (
                        next_not_done * nextvalues + final_values[t]
                    )  # t instead of t+1
                    # next_not_done means nextvalues is computed from the correct next_obs
                    # if next_not_done is 1, final_values is always 0
                    # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                    if self.args.finite_horizon_gae:
                        if t == self.args.num_steps - 1:  # initialize
                            lam_coef_sum = torch.tensor(
                                0.0, device=self.device
                            )  # the sum of the first term
                            reward_term_sum = torch.tensor(
                                0.0, device=self.device
                            )  # the sum of the second term
                            value_term_sum = torch.tensor(
                                0.0, device=self.device
                            )  # the sum of the third term
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done

                        lam_coef_sum = 1 + self.args.gae_lambda * lam_coef_sum
                        reward_term_sum = (
                            self.args.gae_lambda * self.args.gamma * reward_term_sum
                            + lam_coef_sum * self.rewards[t]
                        )
                        value_term_sum = (
                            self.args.gae_lambda * self.args.gamma * value_term_sum
                            + self.args.gamma * real_next_values
                        )

                        advantages[t] = (
                            reward_term_sum + value_term_sum
                        ) / lam_coef_sum - self.values[t]
                    else:
                        delta = (
                            self.rewards[t]
                            + self.args.gamma * real_next_values
                            - self.values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.args.gamma
                            * self.args.gae_lambda
                            * next_not_done
                            * lastgaelam
                        )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                returns = advantages + self.values

            # Normalize observations before agent update
            if self.args.normalize_obs:
                self.obs = self.normalize_obs(self.obs)

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + env.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

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

                    if (
                        self.args.target_kl is not None
                        and approx_kl > self.args.target_kl
                    ):
                        break

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

            update_time = time.time() - update_time

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            self.logger.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.logger.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            clipfracs_log = float(np.mean(clipfracs))
            self.logger.add_scalar("losses/clipfrac", clipfracs_log, global_step)
            self.logger.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            elapsed_time = time.time() - start_time
            self.logger.add_scalar(
                "charts/SPS", int(global_step / elapsed_time), global_step
            )
            self.logger.add_scalar("time/step", global_step, global_step)
            self.logger.add_scalar("time/update_time", update_time, global_step)
            self.logger.add_scalar("time/rollout_time", rollout_time, global_step)
            self.logger.add_scalar(
                "time/rollout_fps",
                self.args.num_envs * self.args.num_steps / rollout_time,
                global_step,
            )
        if not self.args.evaluate:
            if self.args.save_model:
                model_path = (
                    Path(CFG.rl_policy_save_dir) / f"runs/{CFG.exp_name}/final_ckpt.pt"
                )
                self.save(model_path)
                logging.info(f"model saved to {model_path}")
            self.logger.close()


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
