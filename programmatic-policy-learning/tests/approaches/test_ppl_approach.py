"""Tests for ppl_approach.py."""

import tempfile
from pathlib import Path

import gymnasium
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.ppl_approach import (
    ProgrammaticPolicyLearningApproach,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_ppl_approach():
    """Tests for ProgrammaticPolicyLearningApproach()."""
    env = gymnasium.make("LunarLander-v3")
    env.action_space.seed(123)
    constant_action = env.action_space.sample()
    environment_description = (
        "The well-known LunarLander in gymnasium, i.e., "
        'env = gymnasium.make("LunarLander-v3")'
    )

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    response = Response(
        f"""```python
def _policy(obs):
    return {constant_action}
```
""",
        {},
    )
    llm = OrderedResponseModel([response], cache)

    approach = ProgrammaticPolicyLearningApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    obs, info = env.reset()
    approach.reset(obs, info)
    for _ in range(5):
        action = approach.step()
        assert action == constant_action
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, reward, terminated, info)


@runllms
def test_ppl_approach_with_real_llm():
    """Tests for ProgrammaticPolicyLearningApproach() with real LLM."""
    env = gymnasium.make("LunarLander-v3")
    env.action_space.seed(123)
    environment_description = (
        "The well-known LunarLander in gymnasium, i.e., "
        'env = gymnasium.make("LunarLander-v3")'
    )

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-4o-mini", cache)

    approach = ProgrammaticPolicyLearningApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    obs, info = env.reset()
    approach.reset(obs, info)

    # Uncomment if curious.
    # print(approach._policy)

    for _ in range(5):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, reward, terminated, info)
