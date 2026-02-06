#!/usr/bin/env python
"""Direct LeRobot training script.

for both lerobot and KinDER environments.
"""

from lerobot.utils.utils import init_logging

from kinder_imitation_learning.train import train


def main() -> None:
    """Main entry point for training."""
    init_logging()
    train()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
