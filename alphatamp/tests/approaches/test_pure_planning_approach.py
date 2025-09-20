"""Tests for pure_planning_approach.py."""

import prbench
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.pure_planning_approach import PurePlanningApproach


def test_pure_planning_approach():
    """Tests for PurePlanningApproach()."""

    # Test in simple PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=0
    )

    # Create the approach.
    approach = PurePlanningApproach(env_models, seed=123)

    # Create a problem.
    obs, _ = env.reset(seed=123)

    # Training should do nothing.
    approach.train(obs)

    # Evaluation should succeed because this is an easy problem.
    plan = approach.run_planning(obs, timeout=100)

    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    env.close()
