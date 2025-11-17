"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the
hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

You can also evaluate a model checkpoint
from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```
"""

import json
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import (
    close_envs,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from termcolor import colored

from prbench_imitation_learning.evaluate import eval_policy_all


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    """Evaluate a policy on an environment by running rollouts and computing metrics."""
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(  # pylint: disable=logging-not-lazy
        colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
    )

    logging.info("Making environment.")
    envs = make_env(
        cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware.
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )
    with (
        torch.no_grad(),
        (
            torch.autocast(device_type=device.type)
            if cfg.policy.use_amp
            else nullcontext()
        ),
    ):
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


def main() -> None:
    """Main function."""
    init_logging()
    eval_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
