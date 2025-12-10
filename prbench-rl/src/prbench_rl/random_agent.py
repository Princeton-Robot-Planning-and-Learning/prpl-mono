"""Random action baseline agent for PRBench environments."""

from typing import Any, TypeVar

import gymnasium as gym

from prbench_rl.agent import BaseRLAgent
from prbench_rl.gym_utils import make_env_ppo

_O = TypeVar("_O")
_U = TypeVar("_U")


class RandomAgent(BaseRLAgent[_O, _U]):
    """Random action baseline agent."""

    def _get_action(self) -> _U:
        """Sample a random action from the action space."""
        return self.action_space.sample()  # type: ignore

    def train(  # type: ignore[override]
        self, eval_episodes: int = 10, render_eval_video: bool = False
    ) -> dict[str, Any]:
        """Train does nothing for random agent."""
        del eval_episodes
        del render_eval_video
        return {}

    def evaluate(self, eval_episodes: int, render: bool = False) -> dict[str, Any]:
        """Evaluate the agent over a number of episodes."""
        del render
        envs = gym.vector.SyncVectorEnv(
            [make_env_ppo(self.env_id, self.max_episode_steps)]
        )

        _, _ = envs.reset()
        episodic_returns: list[float] = []
        step_lengths: list[int] = []
        step_length = 0
        while len(episodic_returns) < eval_episodes:
            actions = self._get_action()
            _, _, _, _, infos = envs.step(actions)
            step_length += 1
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    episode_return = info["episode"]["r"]
                    print(
                        f"eval_episode={len(episodic_returns)}, "
                        f"episodic_return={episode_return}"
                    )
                    episodic_returns += [info["episode"]["r"]]
                    step_lengths += [step_length]
                    step_length = 0

        eval_metrics = {
            "episodic_return": episodic_returns,
            "step_length": step_lengths,
        }
        return eval_metrics
