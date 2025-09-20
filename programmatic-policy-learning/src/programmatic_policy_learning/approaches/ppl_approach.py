"""An approach that synthesizes a programmatic policy using an LLM."""

import logging
from typing import Callable, TypeVar

from gymnasium.spaces import Space
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class ProgrammaticPolicyLearningApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that synthesizes a programmatic policy using an LLM."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        llm: PretrainedLargeModel,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._llm = llm
        # Wait to reset so that we have one example of an observation.
        self._policy: Callable[[_ObsType], _ActType] | None = None

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        assert self._last_observation is not None
        if self._policy is None:
            self._policy = synthesize_policy_from_environment_description(
                self._environment_description,
                self._llm,
                self._last_observation,
                self._action_space,
            )

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        return self._policy(self._last_observation)


def synthesize_policy_from_environment_description(
    environment_description: str,
    llm: PretrainedLargeModel,
    example_observation: _ObsType,
    action_space: Space[_ActType],
) -> Callable[[_ObsType], _ActType]:
    """Use the LLM to synthesize a programmatic policy."""

    function_name = "_policy"
    query = Query(
        f"""Generate a Python function policy of the form

```python
def _policy(obs):
    # your code here
```

The policy should do well in the environment described below.

{environment_description}

Here is the action space:

{action_space}

Here is an example observation:

{example_observation}

Return only the function; do not give example usages.
"""
    )
    reprompt_checks = [
        SyntaxRepromptCheck(),
        FunctionOutputRepromptCheck(
            function_name, [(example_observation,)], [action_space.contains]
        ),
    ]

    policy = synthesize_python_function_with_llm(
        function_name,
        llm,
        query,
        reprompt_checks=reprompt_checks,
    )

    logging.info("Synthesized new policy:")
    logging.info(str(policy))

    return policy
